"""
SFT Error Analysis: categorise failure modes in generated explanations.

Loads eval result files and classifies each generated explanation into one
or more failure categories:
  - URL_OUTPUT:        explanation contains a URL (hallucinated reference)
  - EMPTY_SHORT:       explanation is empty or <20 chars
  - WRONG_OPTION_REF:  explanation explicitly references a different option letter
                       than the correct one
  - VERBOSE_LOW_NLI:   explanation >400 chars but NLI < 0.05
  - SHORTHAND_HEAVY:   >30% of tokens are ≤2 chars (abbreviation-dominated text)
  - REPETITION:        any 6-word span repeated more than once in the text
  - GENERIC_FILLER:    explanation is boilerplate without domain-specific content

Usage:
    python scripts/analysis/error_analysis.py \
        --eval_files rl_eval_results/llama2_dpo_hybrid_cardiff_eval.json \
                     rl_eval_results/full_eval_new_metrics.json \
        --model_filter "SFT" \
        --output_path rl_eval_results/sft_error_analysis.json

    # Analyse all files in results dir, SFT model only:
    python scripts/analysis/error_analysis.py \
        --results_dir rl_eval_results \
        --model_filter "SFT" \
        --output_path rl_eval_results/sft_error_analysis.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


# ---- failure-mode detectors ------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
OPTION_RE = re.compile(r"\b(?:option|answer)\s+([A-E])\b", re.IGNORECASE)
GENERIC_PHRASES = [
    "the correct answer is",
    "as stated in the question",
    "as mentioned above",
    "refer to",
    "please refer",
    "see also",
    "it is important to note",
]


def has_url(text: str) -> bool:
    return bool(URL_RE.search(text))


def is_empty_short(text: str) -> bool:
    return len(text.strip()) < 20


def wrong_option_ref(text: str, correct_option: str) -> bool:
    """Return True if the explanation explicitly mentions a DIFFERENT option letter."""
    if not correct_option:
        return False
    mentions = set(m.upper() for m in OPTION_RE.findall(text))
    correct_option = correct_option.upper()
    if mentions and correct_option not in mentions:
        return True
    return False


def is_verbose_low_nli(text: str, nli_score: float, char_threshold: int = 400,
                        nli_threshold: float = 0.05) -> bool:
    return len(text) > char_threshold and nli_score < nli_threshold


def is_shorthand_heavy(text: str, ratio_threshold: float = 0.30) -> bool:
    tokens = text.split()
    if len(tokens) < 5:
        return False
    short = sum(1 for t in tokens if len(re.sub(r"[^a-zA-Z]", "", t)) <= 2)
    return short / len(tokens) > ratio_threshold


def has_repetition(text: str, ngram: int = 6) -> bool:
    words = text.lower().split()
    if len(words) < ngram * 2:
        return False
    spans = Counter(" ".join(words[i:i+ngram]) for i in range(len(words) - ngram + 1))
    return any(v > 1 for v in spans.values())


def is_generic_filler(text: str) -> bool:
    lower = text.lower()
    return sum(1 for p in GENERIC_PHRASES if p in lower) >= 2


def classify(text: str, nli_score: float, correct_option: str = "") -> list:
    labels = []
    if is_empty_short(text):
        labels.append("EMPTY_SHORT")
    if has_url(text):
        labels.append("URL_OUTPUT")
    if wrong_option_ref(text, correct_option):
        labels.append("WRONG_OPTION_REF")
    if is_verbose_low_nli(text, nli_score):
        labels.append("VERBOSE_LOW_NLI")
    if is_shorthand_heavy(text):
        labels.append("SHORTHAND_HEAVY")
    if has_repetition(text):
        labels.append("REPETITION")
    if is_generic_filler(text):
        labels.append("GENERIC_FILLER")
    if not labels:
        labels.append("OK")
    return labels


# ---- loading helpers -------------------------------------------------------

def load_eval_file(path: str, model_filter: str = None):
    """
    Returns list of dicts:
        {model, index, explanation, nli_score, bleu, acr, labels}
    """
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return []

    if isinstance(d, list):
        return []  # not an eval result file

    det = d.get("detailed_results", [])
    if det is None:
        return []
    if isinstance(det, dict):
        # old format: {bleu_scores: [...], generated_explanations: [...], ...}
        det = [det]

    rows = []
    for entry in det:
        model = entry.get("model", "unknown")
        if model_filter and model_filter.lower() not in model.lower():
            continue
        explanations = entry.get("generated_explanations", [])
        nli_scores   = entry.get("nli_entailment_scores", [0.0] * len(explanations))
        bleu_scores  = entry.get("bleu_scores",           [0.0] * len(explanations))
        acr_scores   = entry.get("acr_scores",            [0.0] * len(explanations))

        for i, exp in enumerate(explanations):
            nli = nli_scores[i] if i < len(nli_scores) else 0.0
            rows.append({
                "source_file": os.path.basename(path),
                "model": model,
                "index": i,
                "explanation": exp,
                "nli_score": round(float(nli), 4),
                "bleu": round(float(bleu_scores[i]) if i < len(bleu_scores) else 0.0, 4),
                "acr": round(float(acr_scores[i]) if i < len(acr_scores) else 0.0, 4),
                "char_len": len(exp),
                "labels": classify(exp, float(nli)),
            })
    return rows


# ---- reporting -------------------------------------------------------------

def print_report(rows: list, title: str = ""):
    if not rows:
        print("  (no entries)")
        return

    n = len(rows)
    n_ok = sum(1 for r in rows if r["labels"] == ["OK"])
    label_counts = Counter(lbl for r in rows for lbl in r["labels"] if lbl != "OK")

    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
    print(f"  Total explanations: {n}  |  Clean (OK): {n_ok} ({100*n_ok/n:.1f}%)")
    print(f"{'='*70}")
    print(f"  {'Failure type':<25} {'Count':>6}  {'%':>6}  {'Avg NLI':>8}")
    print(f"  {'-'*55}")
    for label, count in label_counts.most_common():
        affected = [r for r in rows if label in r["labels"]]
        avg_nli = sum(r["nli_score"] for r in affected) / len(affected)
        print(f"  {label:<25} {count:>6}  {100*count/n:>5.1f}%  {avg_nli:>8.4f}")

    # Show representative examples for each failure mode
    print(f"\n  --- Representative examples ---")
    shown = set()
    for label, _ in label_counts.most_common():
        examples = [r for r in rows if label in r["labels"] and label not in shown]
        if not examples:
            continue
        shown.add(label)
        # Pick lowest-NLI example as most representative failure
        ex = min(examples, key=lambda r: r["nli_score"])
        print(f"\n  [{label}] NLI={ex['nli_score']:.4f}, len={ex['char_len']}, "
              f"BLEU={ex['bleu']:.4f}, model={ex['model']}")
        print(f"  \"{ex['explanation'][:300]}{'...' if len(ex['explanation'])>300 else ''}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_files", nargs="*", default=[],
                        help="Specific eval JSON files to analyse.")
    parser.add_argument("--results_dir", default=None,
                        help="Scan all *.json in this directory.")
    parser.add_argument("--model_filter", default="SFT",
                        help="Only include rows whose model name contains this string. "
                             "Pass '' to include all models.")
    parser.add_argument("--output_path", default="rl_eval_results/sft_error_analysis.json")
    args = parser.parse_args()

    # Collect input files
    files = list(args.eval_files)
    if args.results_dir:
        files += [str(p) for p in Path(args.results_dir).glob("*.json")
                  if "calibration" not in p.name and "error_analysis" not in p.name]
    if not files:
        print("No input files specified. Use --eval_files or --results_dir.")
        return

    print(f"Loading from {len(files)} file(s), model_filter='{args.model_filter}'...")
    all_rows = []
    for f in sorted(files):
        rows = load_eval_file(f, model_filter=args.model_filter or None)
        if rows:
            all_rows.extend(rows)
            print(f"  {os.path.basename(f)}: {len(rows)} rows loaded")

    if not all_rows:
        print("No rows matched the filter.")
        return

    print(f"\nTotal rows: {len(all_rows)}")

    # Overall report
    print_report(all_rows, title=f"ALL FILES — model_filter='{args.model_filter}'")

    # Per-model breakdown
    models = sorted({r["model"] for r in all_rows})
    if len(models) > 1:
        for model in models:
            model_rows = [r for r in all_rows if r["model"] == model]
            print_report(model_rows, title=f"Model: {model} (n={len(model_rows)})")

    # Domain breakdown (inferred from source_file)
    domain_map = {
        "cardiff": "Cardiff", "sydney": "Sydney", "law": "Law",
        "med_y1": "Medicine Y1", "med_y2": "Medicine Y2",
    }
    domain_rows = defaultdict(list)
    for r in all_rows:
        fn = r["source_file"].lower()
        for key, label in domain_map.items():
            if key in fn:
                domain_rows[label].append(r)
                break
    if domain_rows:
        print(f"\n{'='*70}")
        print("  DOMAIN SUMMARY")
        print(f"  {'Domain':<20} {'n':>5}  {'URL%':>6}  {'Short%':>7}  {'Verbose%':>9}  {'OK%':>5}")
        print(f"  {'-'*60}")
        for domain_label in ["Cardiff", "Sydney", "Law", "Medicine Y1", "Medicine Y2"]:
            dr = domain_rows.get(domain_label, [])
            if not dr:
                continue
            n = len(dr)
            url_pct    = 100 * sum(1 for r in dr if "URL_OUTPUT" in r["labels"]) / n
            short_pct  = 100 * sum(1 for r in dr if "EMPTY_SHORT" in r["labels"]) / n
            verbose_pct= 100 * sum(1 for r in dr if "VERBOSE_LOW_NLI" in r["labels"]) / n
            ok_pct     = 100 * sum(1 for r in dr if r["labels"] == ["OK"]) / n
            print(f"  {domain_label:<20} {n:>5}  {url_pct:>5.1f}%  {short_pct:>6.1f}%  {verbose_pct:>8.1f}%  {ok_pct:>4.1f}%")

    # Failure co-occurrence matrix
    failure_types = ["URL_OUTPUT", "EMPTY_SHORT", "WRONG_OPTION_REF",
                     "VERBOSE_LOW_NLI", "SHORTHAND_HEAVY", "REPETITION", "GENERIC_FILLER"]
    print(f"\n  Failure co-occurrence (% of rows with both labels):")
    for a in failure_types:
        for b in failure_types:
            if a >= b:
                continue
            both = sum(1 for r in all_rows if a in r["labels"] and b in r["labels"])
            if both > 0:
                print(f"    {a} + {b}: {both} ({100*both/len(all_rows):.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    output = {
        "description": "SFT error analysis: failure mode categorisation of generated explanations",
        "model_filter": args.model_filter,
        "total_rows": len(all_rows),
        "global_label_counts": dict(
            Counter(lbl for r in all_rows for lbl in r["labels"]).most_common()
        ),
        "per_domain": {
            domain_label: {
                "n": len(dr),
                "label_counts": dict(Counter(
                    lbl for r in dr for lbl in r["labels"]
                ).most_common()),
            }
            for domain_label, dr in domain_rows.items()
        },
        "rows": [
            {k: v for k, v in r.items() if k != "explanation"}
            for r in all_rows
        ],
        "failure_examples": {
            label: [
                {"model": r["model"], "source": r["source_file"],
                 "nli": r["nli_score"], "text": r["explanation"][:400]}
                for r in sorted(
                    [r for r in all_rows if label in r["labels"]],
                    key=lambda x: x["nli_score"]
                )[:5]
            ]
            for label in failure_types
        },
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()

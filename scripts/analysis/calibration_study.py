"""
NLI Calibration Study: compare nli-deberta-v3-small vs nli-deberta-v3-large rankings.

Loads all eval result files that contain both avg_nli_entailment (small) and
avg_nli_entailment_large (large), computes Spearman/Kendall rank correlations,
and prints a summary table.

Usage:
    python scripts/analysis/calibration_study.py \
        --results_dir rl_eval_results \
        --output_path rl_eval_results/calibration_study.json
"""

import argparse
import json
import os
from pathlib import Path

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed; rank correlation will be skipped.")


DOMAIN_LABELS = {
    "cardiff": "Cardiff Biology",
    "sydney": "Sydney Biology",
    "law": "Law",
    "med_y1": "Medicine Y1",
    "med_y2": "Medicine Y2",
}


def load_result_file(path):
    """Return list of (model_name, nli_small, nli_large) from a result file."""
    try:
        d = json.load(open(path))
        results = d.get("results", [])
        if isinstance(results, dict):
            results = [results]
        rows = []
        for r in results:
            nli_small = r.get("avg_nli_entailment")
            nli_large = r.get("avg_nli_entailment_large")
            if nli_small is not None and nli_large is not None:
                rows.append({
                    "model": r["model"],
                    "nli_small": round(float(nli_small), 4),
                    "nli_large": round(float(nli_large), 4),
                    "bleu": round(float(r.get("avg_bleu", 0)), 4),
                    "bert_ans": round(float(r.get("avg_bert_score_f1_answer_anchored", 0)), 4),
                    "acr": round(float(r.get("avg_answer_coverage_rate", 0)), 4),
                })
        return rows
    except Exception:
        return []


def infer_domain(filename):
    fn = filename.lower()
    for key in ["cardiff", "sydney", "law", "med_y1", "med_y2"]:
        if key in fn:
            return key
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="rl_eval_results")
    parser.add_argument("--output_path", default="rl_eval_results/calibration_study.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_files = sorted(results_dir.glob("*.json"))

    # Collect all entries that have both small and large NLI
    by_domain = {k: [] for k in DOMAIN_LABELS}
    all_entries = []

    for path in all_files:
        rows = load_result_file(path)
        if not rows:
            continue
        domain = infer_domain(path.name)
        for row in rows:
            entry = dict(row, domain=domain, source_file=path.name)
            all_entries.append(entry)
            if domain in by_domain:
                by_domain[domain].append(entry)

    if not all_entries:
        print("No files with both nli_small and nli_large found.")
        return

    # Print per-domain tables
    print("=" * 90)
    print("NLI CALIBRATION STUDY: small vs large model")
    print("=" * 90)

    correlations = {}
    for domain, entries in by_domain.items():
        if not entries:
            continue
        # De-duplicate by model name (keep last occurrence per model per domain)
        seen = {}
        for e in entries:
            seen[e["model"]] = e
        entries = list(seen.values())
        entries.sort(key=lambda x: x["nli_small"], reverse=True)

        label = DOMAIN_LABELS[domain]
        print(f"\n--- {label} ---")
        print(f"{'Model':<40} {'NLI_small':>10} {'NLI_large':>10} {'ratio L/S':>10} {'source':>0}")
        print("-" * 80)
        for e in entries:
            ratio = (e["nli_large"] / e["nli_small"]) if e["nli_small"] > 0 else float("inf")
            print(f"{e['model']:<40} {e['nli_small']:>10.4f} {e['nli_large']:>10.4f} {ratio:>10.2f}x  ({e['source_file']})")

        if len(entries) >= 3 and HAS_SCIPY:
            small_vals = [e["nli_small"] for e in entries]
            large_vals = [e["nli_large"] for e in entries]
            spearman_r, spearman_p = stats.spearmanr(small_vals, large_vals)
            kendall_tau, kendall_p = stats.kendalltau(small_vals, large_vals)
            correlations[domain] = {
                "n": len(entries),
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "kendall_tau": round(kendall_tau, 4),
                "kendall_p": round(kendall_p, 4),
            }
            print(f"\n  Rank correlation (n={len(entries)}): "
                  f"Spearman r={spearman_r:.4f} (p={spearman_p:.3f}), "
                  f"Kendall τ={kendall_tau:.4f} (p={kendall_p:.3f})")

    # Global correlation across all entries
    if len(all_entries) >= 3 and HAS_SCIPY:
        seen_global = {}
        for e in all_entries:
            key = (e["domain"], e["model"])
            seen_global[key] = e
        deduped = list(seen_global.values())
        small_vals = [e["nli_small"] for e in deduped]
        large_vals = [e["nli_large"] for e in deduped]
        sr, sp = stats.spearmanr(small_vals, large_vals)
        kt, kp = stats.kendalltau(small_vals, large_vals)
        print(f"\n{'=' * 90}")
        print(f"GLOBAL rank correlation (n={len(deduped)} entries across all domains):")
        print(f"  Spearman r = {sr:.4f}  (p = {sp:.4f})")
        print(f"  Kendall  τ = {kt:.4f}  (p = {kp:.4f})")
        correlations["global"] = {
            "n": len(deduped),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 4),
            "kendall_tau": round(kt, 4),
            "kendall_p": round(kp, 4),
        }

    # Check whether relative ordering (SFT < RL) is preserved under large model
    print(f"\n{'=' * 90}")
    print("RANKING PRESERVATION CHECK: does 'RL > SFT' hold under both models?")
    print("-" * 90)
    sft_keywords = ["sft", "SFT", "Baseline-SFT", "baseline"]
    for domain, entries in by_domain.items():
        if not entries:
            continue
        seen = {}
        for e in entries:
            seen[e["model"]] = e
        entries = list(seen.values())
        sft_entries = [e for e in entries if any(kw in e["model"] for kw in sft_keywords)]
        rl_entries  = [e for e in entries if not any(kw in e["model"] for kw in sft_keywords)]
        if not sft_entries or not rl_entries:
            continue
        max_sft_small = max(e["nli_small"] for e in sft_entries)
        max_sft_large = max(e["nli_large"] for e in sft_entries)
        max_rl_small  = max(e["nli_small"] for e in rl_entries)
        max_rl_large  = max(e["nli_large"] for e in rl_entries)
        small_ok = max_rl_small > max_sft_small
        large_ok = max_rl_large > max_sft_large
        label = DOMAIN_LABELS.get(domain, domain)
        print(f"  {label:<22}: small RL({max_rl_small:.4f}) > SFT({max_sft_small:.4f}) → {'✓' if small_ok else '✗'}  |  "
              f"large RL({max_rl_large:.4f}) > SFT({max_sft_large:.4f}) → {'✓' if large_ok else '✗'}")

    # Save output
    output = {
        "description": "NLI calibration study: small vs large model ranking comparison",
        "total_entries_with_both": len(all_entries),
        "correlations": correlations,
        "entries": [
            {k: v for k, v in e.items() if k != "source_file"}
            for e in sorted(all_entries, key=lambda x: (x["domain"], -x["nli_small"]))
        ],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()

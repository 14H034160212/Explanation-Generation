"""Held-out NLI re-scoring: re-score saved generated explanations with an
INDEPENDENT, larger NLI model (default DeBERTa-v3-large) that was NOT used to
construct training preference pairs. Addresses the circular-evaluation concern.
Does not modify original eval JSONs; writes a summary JSON.

**Pairing.** Explanation j is NOT question j. rl_evaluation.py skips an item
whose generation came back empty, so a row with 188 explanations over 191 test
items is a subsequence, and zipping it against the full hypothesis list scores
every explanation after the first skip against the wrong question. 38 of the
eval JSONs in rl_eval_results have at least one short row, and the three cells
this previously reported as FLIPS were all of them -- LLaMA-2 AucklandLaw
(1 drop, at item 36), LLaMA-2 MedY1 (1 drop, at item 0, so the whole row was
shifted by one) and Qwen3 MedY1.

New eval JSONs carry `test_indices`. Older ones are realigned by
eval_alignment.recover_indices, which reconstructs the mapping from the stored
ACR scores. If neither is available the cell is REFUSED, not scored: a number
that might be misaligned is worse than no number, because it looks the same.
"""
import json, re, os, sys, argparse, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_training"))
from eval_alignment import recover_indices  # noqa: E402

def hypotheses_from_test(test_path):
    data = json.load(open(test_path, encoding="utf-8"))
    hyps = []
    for item in data:
        inp = item.get("input", "")
        m = re.search(r"The correct answer is Option ([A-Z])", inp)
        h = ""
        if m:
            letter = m.group(1)
            om = re.search(rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)", inp, re.DOTALL)
            h = om.group(1).strip() if om else ""
        hyps.append(h)
    return hyps

def get_exps(data, model_name):
    dr = data.get("detailed_results", {})
    if isinstance(dr, dict) and model_name in dr:
        return dr[model_name].get("generated_explanations", [])
    if isinstance(dr, list):
        for d in dr:
            if d.get("model") == model_name:
                return d.get("generated_explanations", [])
    for r in data.get("results", []):
        if r.get("model") == model_name and r.get("generated_explanations"):
            return r["generated_explanations"]
    return []

def detail_for(data, model_name):
    """The per-item block for one model, under either shape.

    detailed_results is a list of {"model": ...} in files written by
    rl_evaluation.py and a dict keyed by model name in the Qwen3 ones. Reading
    only the list shape made every Qwen3 cell look unalignable, when in fact
    those rows are complete.
    """
    dr = data.get("detailed_results")
    if isinstance(dr, dict):
        return dr.get(model_name)
    if isinstance(dr, list):
        return next(
            (d for d in dr if isinstance(d, dict) and d.get("model") == model_name), None
        )
    return None


def hypotheses_for_row(row, hyps, items):
    """The hypotheses belonging to this row's explanations, in its order.

    Returns None when the mapping cannot be established, and the caller must
    then refuse the cell rather than fall back to positional pairing -- that
    fallback is the bug this function exists to remove.
    """
    exps = row.get("generated_explanations") or []
    if not exps:
        return None
    # Written by rl_evaluation.py since the fix; authoritative when present.
    idx = row.get("test_indices")
    if not idx:
        if len(exps) == len(hyps):
            idx = list(range(len(exps)))      # nothing was skipped
        else:
            idx = recover_indices(exps, row.get("acr_scores") or [], items)
    if not idx or len(idx) != len(exps) or max(idx) >= len(hyps):
        return None
    return [hyps[i] for i in idx]


@torch.no_grad()
def score(exps, hyps, tok, model, device, eidx):
    # Same length by construction now: hyps comes from hypotheses_for_row.
    assert len(exps) == len(hyps), (len(exps), len(hyps))
    n = len(exps)
    sc = []
    for i in range(0, n, 8):
        e = exps[i:i+8]; h = hyps[i:i+8]
        pairs = [(x, y) for x, y in zip(e, h) if x and y]
        if not pairs:
            sc += [0.0]*len(e); continue
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                  truncation=True, max_length=512, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, dim=-1)[:, eidx].cpu().tolist()
        sc += p
        if len(pairs) < len(e): sc += [0.0]*(len(e)-len(pairs))
    return sum(sc)/len(sc) if sc else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-large")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--map", required=True, help="JSON: [[label, eval_json, test_json], ...]")
    ap.add_argument("--out", default="rl_eval_results/heldout_nli_summary.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.nli_model, use_fast=False, cache_dir="cache")
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model, cache_dir="cache").to(args.device).eval()
    eidx = next((k for k, v in model.config.id2label.items() if "entail" in v.lower()), 1)
    print(f"Loaded {args.nli_model}; entailment_idx={eidx}")

    cells = json.load(open(args.map))
    out = []
    for label, ej, tj in cells:
        if not (os.path.exists(ej) and os.path.exists(tj)):
            print(f"SKIP {label}: missing {ej if not os.path.exists(ej) else tj}"); continue
        data = json.load(open(ej, encoding="utf-8"))
        hyps = hypotheses_from_test(tj)
        items = json.load(open(tj, encoding="utf-8"))
        row = {"cell": label}
        for r in data.get("results", []):
            m = r.get("model", "?")
            exps = get_exps(data, m)
            if not exps: continue
            detail = detail_for(data, m)
            paired = hypotheses_for_row(detail, hyps, items) if detail else None
            if paired is None:
                print(f"REFUSE {label} / {m}: cannot align "
                      f"{len(exps)} explanations to {len(hyps)} test items")
                row.setdefault("refused", []).append(m)
                continue
            large = round(score(exps, paired, tok, model, args.device, eidx), 4)
            small = r.get("avg_nli_entailment")
            tag = "SFT" if "SFT" in m or "Baseline" in m else ("DPO" if ("DPO" in m or "RLearner" in m or "RL-" in m or "Hybrid" in m) else m)
            # Two tuned rows in one file both match the DPO branch -- "RL-DPO",
            # "RL-DPO-v2" and "RL-PPO" all do -- and this used to be a plain
            # assignment, so the last one written won and a PPO result could be
            # reported as DPO. Keep every row and say so.
            if tag in row:
                print(f"  NOTE {label}: {m} also maps to '{tag}' (kept as '{tag}:{m}'), "
                      f"first was {row[tag]['model']}")
                tag = f"{tag}:{m}"
            row[tag] = {"model": m, "small": small, "large": large}
        out.append(row)
        s = row.get("SFT", {}); d = row.get("DPO", {})
        if s and d:
            print(f"{label:34s} SMALL {s.get('small')}->{d.get('small')} | LARGE {s['large']}->{d['large']}  "
                  f"{'HOLDS' if (d['large']>s['large']) else 'FLIPS'}")
        else:
            print(f"{label:34s} {row}")
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out)

if __name__ == "__main__":
    main()

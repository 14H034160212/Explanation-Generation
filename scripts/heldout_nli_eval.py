"""Held-out NLI re-scoring: re-score saved generated explanations with an
INDEPENDENT, larger NLI model (default DeBERTa-v3-large) that was NOT used to
construct training preference pairs. Addresses the circular-evaluation concern.
Reads generated_explanations from eval JSONs; pairs them with the correct-option
hypothesis extracted from the matching test file; reports avg entailment per model.
Does not modify original eval JSONs; writes a summary JSON.
"""
import json, re, os, sys, argparse, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

@torch.no_grad()
def score(exps, hyps, tok, model, device, eidx):
    n = min(len(exps), len(hyps))
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
        row = {"cell": label}
        for r in data.get("results", []):
            m = r.get("model", "?")
            exps = get_exps(data, m)
            if not exps: continue
            large = round(score(exps, hyps, tok, model, args.device, eidx), 4)
            small = r.get("avg_nli_entailment")
            tag = "SFT" if "SFT" in m or "Baseline" in m else ("DPO" if ("DPO" in m or "RLearner" in m or "RL-" in m or "Hybrid" in m) else m)
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

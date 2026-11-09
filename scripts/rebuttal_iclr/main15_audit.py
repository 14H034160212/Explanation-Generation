"""ICLR-2027 rebuttal audit of the 15 main (architecture, domain) cells.

For every cell this re-reads the *exact* eval JSON that Table 1 reports the
RLearner number from, and pairs it with the SFT row OF THE SAME FILE (same
eval run, same test set).  Nothing is regenerated or retrained.

Per cell it reports
  * NLI (training scorer, deberta-v3-small) SFT -> RLearner, paired, with a
    95% bootstrap CI on the difference and a one-sided Wilcoxon p;
  * the same under the held-out scorer (deberta-v3-large) on these very files
    (the submitted held-out table used different per-domain runs for LLaMA-2
    and Qwen3 -- reviewer ew1Q's question);
  * mean words; verbatim containment of the correct-option text;
  * NLI after masking the correct-option text ("mask_full") and after masking
    every ACR key term of the option ("mask_terms"), under both scorers.

Alignment: explanations are re-aligned to test items with
eval_alignment.recover_indices (empty generations are skipped by the harness);
a cell that cannot be aligned is refused.
"""
import json, os, re, sys, argparse
import numpy as np
import torch
from scipy.stats import wilcoxon, pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = "/data/qbao775/Explanation-Generation"
sys.path.insert(0, os.path.join(ROOT, "scripts", "python_training"))
from eval_alignment import recover_indices, correct_option_text  # noqa: E402

R = f"{ROOT}/rl_eval_results"
T = {
    "Cardiff": f"{ROOT}/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",
    "Sydney": f"{ROOT}/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json",
    "Law": f"{ROOT}/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json",
    "MedY1": f"{ROOT}/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json",
    "MedY2": f"{ROOT}/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json",
}
# (arch, domain) -> (file, sft_row, rl_row)   -- the files Table 1 quotes RLearner from
L2 = ("Baseline-SFT (K=1)", "RL-DPO (K=1)")
Q3 = ("Qwen3-8B-SFT", "Qwen3-8B-DPO")
G4 = ("Gemma4-E4B-SFT", "Gemma4-E4B-DPO")
CELLS = {
    ("LLaMA-2", "Cardiff"): (f"{R}/llama2_dpo_hybrid_cross_domain_cardiff_eval.json", *L2),
    ("LLaMA-2", "Sydney"): (f"{R}/llama2_dpo_hybrid_cross_domain_sydney_eval.json", *L2),
    ("LLaMA-2", "Law"): (f"{R}/llama2_dpo_hybrid_cross_domain_law_eval.json", *L2),
    ("LLaMA-2", "MedY1"): (f"{R}/llama2_dpo_hybrid_cross_domain_med_y1_eval.json", *L2),
    ("LLaMA-2", "MedY2"): (f"{R}/llama2_dpo_hybrid_cross_domain_med_y2_eval.json", *L2),
    ("Qwen3", "Cardiff"): (f"{R}/qwen3_dpo_n5_cardiff_eval.json", *Q3),
    ("Qwen3", "Sydney"): (f"{R}/qwen3_dpo_n5_sydney_eval.json", *Q3),
    ("Qwen3", "Law"): (f"{R}/qwen3_dpo_hybrid_cross_domain_law_eval.json", *Q3),
    ("Qwen3", "MedY1"): (f"{R}/qwen3_dpo_hybrid_cross_domain_med_y1_eval.json", *Q3),
    ("Qwen3", "MedY2"): (f"{R}/qwen3_dpo_hybrid_cross_domain_med_y2_eval.json", *Q3),
    ("Gemma4", "Cardiff"): (f"{R}/gemma4_cardiff_merged_eval.json", *G4),
    ("Gemma4", "Sydney"): (f"{R}/gemma4_sydney_merged_eval.json", *G4),
    ("Gemma4", "Law"): (f"{R}/gemma4_auckland_merged_eval.json", *G4),
    ("Gemma4", "MedY1"): (f"{R}/gemma4_medicine_y1_merged_eval.json", *G4),
    ("Gemma4", "MedY2"): (f"{R}/gemma4_medicine_y2_merged_eval.json", *G4),
}


def row(data, name):
    dr = data.get("detailed_results")
    if isinstance(dr, dict):
        return dr.get(name)
    return next((d for d in dr if isinstance(d, dict) and d.get("model") == name), None)


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip(".").strip()


def contains(exp, opt):
    o = norm(opt)
    return bool(o) and o in norm(exp)


def mask_full(exp, opt):
    o = opt.strip().rstrip(".").strip()
    if not o:
        return exp
    return re.sub(re.escape(o), "[MASK]", exp, flags=re.IGNORECASE)


def mask_terms(exp, opt):
    out = mask_full(exp, opt)
    for t in sorted(set(re.findall(r"\b\w{4,}\b", opt.lower())), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(t)}\w*", "[MASK]", out, flags=re.IGNORECASE)
    return out


class Scorer:
    def __init__(self, name, device):
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.ix = next(k for k, v in self.m.config.id2label.items() if "entail" in v.lower())
        self.device = device

    @torch.no_grad()
    def __call__(self, prem, hyp, bs=16):
        out = []
        for i in range(0, len(prem), bs):
            enc = self.tok(prem[i:i + bs], hyp[i:i + bs], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            out += torch.softmax(self.m(**enc).logits, -1)[:, self.ix].float().cpu().tolist()
        return np.array(out)


def boot_ci(a, b, n=10000, seed=0):
    """CI of mean(b)-mean(a) with the two rows resampled independently
    (they cover slightly different item subsets when a generation was empty)."""
    rng = np.random.default_rng(seed)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/rebuttal_main15_audit.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--spec", default=None,
                    help="JSON list of [label, arch, domain, file, sft_row, rl_row]; default = the 15 Table-1 cells")
    args = ap.parse_args()
    if args.spec:
        cells = [(lab, a, d, f if f.startswith("/") else f"{ROOT}/{f}", s, r)
                 for lab, a, d, f, s, r in json.load(open(args.spec))]
    else:
        cells = [("table1", a, d, f, s, r) for (a, d), (f, s, r) in CELLS.items()]
    small = Scorer("cross-encoder/nli-deberta-v3-small", args.device)
    large = Scorer("cross-encoder/nli-deberta-v3-large", args.device)
    res = []
    for label, arch, dom, f, sft_name, rl_name in cells:
        data = json.load(open(f))
        items = json.load(open(T[dom]))
        opts = [correct_option_text(it.get("input") or "") for it in items]
        cell = {"label": label, "arch": arch, "domain": dom, "file": os.path.relpath(f, ROOT)}
        per = {}
        for tag, name in (("sft", sft_name), ("rl", rl_name)):
            r = row(data, name)
            exps = r["generated_explanations"]
            if len(exps) == len(items):
                # A complete row can only be the identity.  (The Qwen3 harness
                # computes ACR slightly differently, so ACR-based recovery
                # cannot be used to confirm it; the per-item NLI check below does.)
                idx = list(range(len(items)))
            else:
                idx = recover_indices(exps, r["acr_scores"], items)
            if idx is None:
                raise SystemExit(f"REFUSED: cannot align {arch} {dom} {tag} ({f})")
            hyp = [opts[i] for i in idx]
            stored = np.array(r["nli_entailment_scores"])
            s_small = small(exps, hyp)
            # Control: re-scoring with the training scorer must reproduce the
            # stored per-item NLI if (and only if) the alignment is right.
            agree = float(np.corrcoef(stored, s_small)[0, 1])
            if agree < 0.9:
                raise SystemExit(f"REFUSED: {arch} {dom} {tag} per-item NLI corr {agree:.2f} < 0.9")
            cell[f"{tag}_alignment_check_corr"] = agree
            per[tag] = {
                "idx": idx,
                "n": len(exps),
                "stored_nli_mean": float(stored.mean()),
                "rescored_small_mean": float(s_small.mean()),
                "small": s_small, "large": large(exps, hyp),
                "words": np.array([len(e.split()) for e in exps]),
                "contains": np.array([contains(e, o) for e, o in zip(exps, hyp)]),
                "small_maskfull": small([mask_full(e, o) for e, o in zip(exps, hyp)], hyp),
                "large_maskfull": large([mask_full(e, o) for e, o in zip(exps, hyp)], hyp),
                "small_maskterms": small([mask_terms(e, o) for e, o in zip(exps, hyp)], hyp),
                "large_maskterms": large([mask_terms(e, o) for e, o in zip(exps, hyp)], hyp),
            }
        s, r_ = per["sft"], per["rl"]
        common = sorted(set(s["idx"]) & set(r_["idx"]))
        ps = {i: k for k, i in enumerate(s["idx"])}
        pr = {i: k for k, i in enumerate(r_["idx"])}
        for key in ("small", "large", "small_maskfull", "large_maskfull", "small_maskterms", "large_maskterms"):
            a, b = s[key], r_[key]
            lo, hi = boot_ci(a, b)
            pa = np.array([a[ps[i]] for i in common]); pb = np.array([b[pr[i]] for i in common])
            try:
                p = float(wilcoxon(pb, pa, alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
            cell[key] = {"sft": float(a.mean()), "rl": float(b.mean()), "diff": float(b.mean() - a.mean()),
                         "ci95": [lo, hi], "wilcoxon_p_greater": p}
        for tag in ("sft", "rl"):
            cell[f"{tag}_n"] = per[tag]["n"]
            cell[f"{tag}_stored_nli"] = per[tag]["stored_nli_mean"]
            cell[f"{tag}_rescored_small"] = per[tag]["rescored_small_mean"]
            cell[f"{tag}_words"] = float(per[tag]["words"].mean())
            cell[f"{tag}_contains"] = float(per[tag]["contains"].mean())
            m = per[tag]["contains"]
            cell[f"{tag}_small_given_contains"] = float(per[tag]["small"][m].mean()) if m.any() else None
            cell[f"{tag}_small_given_not_contains"] = float(per[tag]["small"][~m].mean()) if (~m).any() else None
        res.append(cell)
        print(f"{label:14s}{arch:8s}{dom:8s} small {cell['small']['sft']:.3f}->{cell['small']['rl']:.3f} "
              f"[{cell['small']['ci95'][0]:+.3f},{cell['small']['ci95'][1]:+.3f}] p={cell['small']['wilcoxon_p_greater']:.1e} | "
              f"large {cell['large']['sft']:.3f}->{cell['large']['rl']:.3f} | "
              f"words {cell['sft_words']:.0f}->{cell['rl_words']:.0f} | contains {cell['sft_contains']:.0%}->{cell['rl_contains']:.0%} | "
              f"maskterms small {cell['small_maskterms']['sft']:.3f}->{cell['small_maskterms']['rl']:.3f} "
              f"large {cell['large_maskterms']['sft']:.3f}->{cell['large_maskterms']['rl']:.3f}", flush=True)
    if args.spec:
        json.dump({"cells": res}, open(args.out, "w"), indent=1); print("wrote", args.out); return
    # headroom: does SFT NLI predict post-DPO NLI, and is r(SFT, gain) more negative than chance?
    x = np.array([c["small"]["sft"] for c in res]); y = np.array([c["small"]["rl"] for c in res])
    rng = np.random.default_rng(0)
    perm = [pearsonr(x, rng.permutation(y) - x)[0] for _ in range(20000)]
    head = {"r_sft_post": pearsonr(x, y)[0], "p_sft_post": pearsonr(x, y)[1],
            "r_sft_gain": pearsonr(x, y - x)[0], "r_sft_relgain": pearsonr(x, (y - x) / x)[0],
            "perm_mean_r_sft_gain": float(np.mean(perm)),
            "perm_frac_le_observed": float(np.mean(np.array(perm) <= pearsonr(x, y - x)[0]))}
    print("headroom:", head)
    json.dump({"cells": res, "headroom_paired": head}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

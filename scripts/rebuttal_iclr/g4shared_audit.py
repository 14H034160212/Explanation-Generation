"""Audit of the Gemma 4 shared-pool variants (same metrics as Table 2).

SFT comes from rl_eval_results/g4shared_sft_<dom>_eval.json and each variant
from g4shared_<variant>[_s<seed>]_<dom>_eval.json.  Gemma 4 decodes greedily,
so the SFT outputs are the same in every run and pairing across files is exact.
Per variant: per-domain and pooled (all available domains) means of NLI,
held-out NLI, masked held-out NLI, words, copies, ACR, verifier, with a paired
bootstrap CI of the pooled change in NLI and masked held-out NLI vs SFT.
"""
import json, os, re, sys, glob
import numpy as np
ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/rebuttal_iclr")
sys.path.insert(0, f"{ROOT}/scripts/python_training")
from main15_audit import Scorer, mask_terms, contains, T  # noqa: E402
from eval_alignment import correct_option_text  # noqa: E402

DOMS = {"cardiff": "Cardiff", "sydney": "Sydney", "law": "Law", "medy1": "MedY1", "medy2": "MedY2"}
R = "rl_eval_results"
small = Scorer("cross-encoder/nli-deberta-v3-small", "cuda:0")
large = Scorer("cross-encoder/nli-deberta-v3-large", "cuda:0")


def load(path, name):
    d = json.load(open(path))
    r = d["detailed_results"][name]
    return r


def score(exps, hyp, r):
    exps = [e if e.strip() else "." for e in exps]
    return {"nli": small(exps, hyp), "large": large(exps, hyp),
            "masked": large([mask_terms(e, o) for e, o in zip(exps, hyp)], hyp),
            "words": np.array([len(e.split()) for e in exps], float),
            "copies": np.array([contains(e, o) for e, o in zip(exps, hyp)], float),
            "acr": np.array(r["acr_scores"], float),
            # reproduced verifier (the original is lost); NaN if a file was never rescored
            "ver": np.array(r.get("verifier_repro_scores", [np.nan] * len(exps)), float)}


cache = {}
sft = {}
for k, dom in DOMS.items():
    f = f"{R}/g4shared_sft_{k}_eval.json"
    if not os.path.exists(f):
        continue
    items = json.load(open(T[dom]))
    hyp = [correct_option_text(it["input"]) for it in items]
    r = load(f, "Gemma4-E4B-SFT")
    assert len(r["generated_explanations"]) == len(items)
    sft[k] = score(r["generated_explanations"], hyp, r)
    cache[k] = hyp

variants = sorted({re.match(rf"{R}/g4shared_(.+)_({'|'.join(DOMS)})_eval.json", f).group(1)
                   for f in glob.glob(f"{R}/g4shared_*_eval.json")} - {"sft"})
rng = np.random.default_rng(0)
out = {"sft": {k: {m: float(v.mean()) for m, v in s.items()} for k, s in sft.items()}, "variants": {}}
for v in variants:
    per, pooled = {}, {m: [] for m in ("nli", "large", "masked", "words", "copies", "acr", "ver")}
    psft = {m: [] for m in pooled}
    for k in DOMS:
        f = f"{R}/g4shared_{v}_{k}_eval.json"
        if not (os.path.exists(f) and k in sft):
            continue
        r = load(f, "Gemma4-E4B-DPO")
        if len(r["generated_explanations"]) != len(cache[k]):
            print("skip (incomplete row)", v, k); continue
        s = score(r["generated_explanations"], cache[k], r)
        per[k] = {m: float(x.mean()) for m, x in s.items()}
        for m in pooled:
            pooled[m].append(s[m]); psft[m].append(sft[k][m])
    if not per:
        continue
    P = {m: np.concatenate(x) for m, x in pooled.items()}
    S = {m: np.concatenate(x) for m, x in psft.items()}
    res = {"domains": per, "n_domains": len(per), "pooled": {m: float(P[m].mean()) for m in P},
           "pooled_sft": {m: float(S[m].mean()) for m in S}}
    for m in ("nli", "large", "masked"):
        d = P[m] - S[m]
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        res[f"ci_{m}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pairs_f = f"rl_preference_data_gemma4_shared/{v.split('_s')[0]}/preference_pairs.json"
    res["pairs"] = len(json.load(open(pairs_f))) if os.path.exists(pairs_f) else None
    out["variants"][v] = res
    print(f"{v:10s} doms={len(per)} pairs={res['pairs']} NLI {res['pooled_sft']['nli']:.3f}->{res['pooled']['nli']:.3f} "
          f"held-out {res['pooled_sft']['large']:.3f}->{res['pooled']['large']:.3f} masked {res['pooled_sft']['masked']:.3f}->{res['pooled']['masked']:.3f} "
          f"CI_masked {res['ci_masked'][0]:+.3f},{res['ci_masked'][1]:+.3f} words {res['pooled']['words']:.0f} copies {res['pooled']['copies']:.2f} ver {res['pooled']['ver']:.2f}", flush=True)
json.dump(out, open(f"{R}/rebuttal_g4shared_audit.json", "w"), indent=1)

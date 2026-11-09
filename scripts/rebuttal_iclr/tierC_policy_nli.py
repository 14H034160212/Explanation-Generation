"""NLI / held-out NLI / masked NLI / length / answer-containment for the greedy
Cardiff Tier-C generations of every policy in p3_full_generations.json
(SFT, SimPO, DPO_v2, OPD, HybridDPO).  Complements the reasoning judges:
shows how each policy scores on the paper's original metric."""
import json, os, sys, numpy as np
ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/rebuttal_iclr")
sys.path.insert(0, f"{ROOT}/scripts/python_training")
from main15_audit import Scorer, mask_terms, contains  # noqa: E402
from eval_alignment import correct_option_text  # noqa: E402

d = json.load(open("rl_eval_results/p3_full_generations.json"))
hyp = [correct_option_text(x) for x in d["test_inputs"]]
small = Scorer("cross-encoder/nli-deberta-v3-small", "cuda:0")
large = Scorer("cross-encoder/nli-deberta-v3-large", "cuda:0")
out = {}
for k, g in d["generations"].items():
    g = [x if x.strip() else "." for x in g]
    m = [mask_terms(e, o) for e, o in zip(g, hyp)]
    out[k] = {"n": len(g), "words": float(np.mean([len(x.split()) for x in g])),
              "contains": float(np.mean([contains(e, o) for e, o in zip(g, hyp)])),
              "nli_small": float(small(g, hyp).mean()), "nli_large": float(large(g, hyp).mean()),
              "masked_small": float(small(m, hyp).mean()), "masked_large": float(large(m, hyp).mean())}
    print(k, {a: round(b, 3) for a, b in out[k].items()}, flush=True)
json.dump(out, open("rl_eval_results/rebuttal_tierC_policy_nli.json", "w"), indent=1)

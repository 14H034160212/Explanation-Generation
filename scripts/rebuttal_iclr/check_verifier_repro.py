"""Does the reproduced verifier behave like the lost one?

(a) Held-out 1,000 training-format examples: MAE and Pearson r against the rating
    labels, next to the MAE of always predicting the training mean.
(b) The 164 Gemma 4 merged-pool pairs (328 texts) still carry the ORIGINAL
    verifier's scores: Pearson/Spearman between old and new scores, and how often
    old and new agree on which of chosen/rejected scores higher.
Uses the same prompt and parser as the pool builder. Output:
rl_eval_results/rebuttal_verifier_repro_check.json
"""
import json, os, re, sys
import numpy as np
import torch


def pearsonr(a, b):  # numpy only: the gemma4-rl env has no scipy
    return (float(np.corrcoef(a, b)[0, 1]),)


def spearmanr(a, b):
    def rk(x):  # average ranks, so tied scores (many 3.0s) get equal rank
        x = np.asarray(x, float); u, inv = np.unique(x, return_inverse=True)
        r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        return np.array([r[inv == i].mean() for i in range(len(u))])[inv]
    return pearsonr(rk(a), rk(b))

ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/python_training")
import rl_build_preference_data_nli as B  # noqa: E402

V = "models/verifier_way2_repro"
TMPL = ("Below is an instruction that describes a task, paired with an input that provides further "
        "context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
vm, vt = B.load_verifier(V, "cuda:0")


@torch.no_grad()
def gen(prompt):
    x = vt(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")
    o = vm.generate(**x, max_new_tokens=10, do_sample=False, pad_token_id=vt.eos_token_id)
    r = vt.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.findall(r"\d+(?:\.\d+)?", r)
    return float(m[0]) if m else None


out = {}
val = json.load(open(f"{V}/heldout_1000.json"))
train_mean = 2.8537  # mean of all 29,484 labels (train mean differs by <0.01)
pred = [gen(TMPL.format(**r)) for r in val]
ok = [(p, float(r["output"])) for p, r in zip(pred, val) if p is not None]
p, y = np.array(ok).T
out["heldout"] = {"n": len(val), "parsed": len(ok), "mae": float(np.abs(p - y).mean()),
                  "mae_const_mean": float(np.abs(train_mean - y).mean()),
                  "pearson": float(pearsonr(p, y)[0]), "spearman": float(spearmanr(p, y)[0])}
print(out["heldout"], flush=True)

P = json.load(open("rl_preference_data_gemma4_merged/preference_pairs.json"))
old, new, n_fail = [], [], 0
for q in P:
    for side in ("chosen", "rejected"):
        s = B.get_verifier_score(vm, vt, q["question_input"], q[side])
        old.append(q[f"{side}_verifier"]); new.append(s)
old, new = np.array(old), np.array(new)
oc, orj, nc, nr = old[0::2], old[1::2], new[0::2], new[1::2]
dif_old, dif_new = np.sign(oc - orj), np.sign(nc - nr)
m = dif_old != 0
out["vs_original"] = {"texts": len(old), "pearson": float(pearsonr(old, new)[0]),
                      "spearman": float(spearmanr(old, new)[0]),
                      "old_mean": float(old.mean()), "new_mean": float(new.mean()),
                      "old_sd": float(old.std()), "new_sd": float(new.std()),
                      "pair_order_agree_when_old_differs": float((dif_old[m] == dif_new[m]).mean()),
                      "pairs_old_differs": int(m.sum())}
print(out["vs_original"], flush=True)
json.dump(out, open("rl_eval_results/rebuttal_verifier_repro_check.json", "w"), indent=1)

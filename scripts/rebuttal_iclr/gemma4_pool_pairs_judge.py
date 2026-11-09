"""Judge-based variants on the SAME shared candidates (best vs worst, as for every variant).
  judge      : judge_wins/4                       (gap >= 0.25, i.e. at least one net win)
  judge_nli  : 0.5*judge_wins/4 + 0.5*NLI          (gap >= 0.05)
Both are trained with SimPO, like the paper's reasoning reward. NLI-only pairs are
reused for an NLI+SimPO control, so objective and reward can be told apart."""
import json, os
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
C = json.load(open("rl_preference_data_gemma4_shared/candidates.json"))
V = {"judge": (lambda c: c["judge_wins"] / 4, 0.25),
     "judge_nli": (lambda c: 0.5 * c["judge_wins"] / 4 + 0.5 * c["nli"], 0.05)}
stats = {}
for name, (f, gap) in V.items():
    pairs = []
    for q in C:
        cs = [c for c in q["cands"] if c["text"].strip()]
        if len(cs) < 2:
            continue
        a, b = max(cs, key=f), min(cs, key=f)
        if f(a) - f(b) < gap:
            continue
        pairs.append({"prompt": q["prompt"], "chosen": a["text"], "rejected": b["text"], "gap": f(a) - f(b),
                      "question_input": q["question_input"], "correct_option_text": q["correct_option_text"],
                      **{f"chosen_{k}": a[k] for k in ("nli", "masked_nli", "verifier", "acr", "words", "judge_wins")},
                      **{f"rejected_{k}": b[k] for k in ("nli", "masked_nli", "verifier", "acr", "words", "judge_wins")}})
    os.makedirs(f"rl_preference_data_gemma4_shared/{name}", exist_ok=True)
    json.dump(pairs, open(f"rl_preference_data_gemma4_shared/{name}/preference_pairs.json", "w"), ensure_ascii=False, indent=1)
    n = len(pairs) or 1
    stats[name] = {"pairs": len(pairs), **{f"{s}_{k}": sum(p[f"{s}_{k}"] for p in pairs) / n
                                           for s in ("chosen", "rejected") for k in ("nli", "masked_nli", "words", "judge_wins")}}
    print(name, {k: round(v, 3) for k, v in stats[name].items()})
json.dump(stats, open("rl_preference_data_gemma4_shared/pair_stats_judge.json", "w"), indent=1)

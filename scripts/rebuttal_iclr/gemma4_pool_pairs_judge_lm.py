"""Round 3 (pre-specified): LENGTH-MATCHED judge pairs on the shared pool.
Round 2 judge pairs were 8 words longer on the chosen side, and LLM judges favour length.
Here only candidate pairs whose lengths differ by <= 15 words are eligible (the rule used
for the Cardiff reasoning reward); among them we take the pair with the largest score gap.
  judge_lm     : judge_wins/4                     (gap >= 0.25)
  judgenli_lm  : 0.5*judge_wins/4 + 0.5*NLI       (gap >= 0.05)"""
import json, os, itertools
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
C = json.load(open("rl_preference_data_gemma4_shared/candidates.json"))
V = {"judge_lm": (lambda c: c["judge_wins"] / 4, 0.25),
     "judgenli_lm": (lambda c: 0.5 * c["judge_wins"] / 4 + 0.5 * c["nli"], 0.05)}
K = ("nli", "masked_nli", "verifier", "acr", "words", "judge_wins")
stats = {}
for name, (f, gap) in V.items():
    pairs = []
    for q in C:
        cs = [c for c in q["cands"] if c["text"].strip()]
        el = [(a, b) if f(a) >= f(b) else (b, a) for a, b in itertools.combinations(cs, 2)
              if abs(a["words"] - b["words"]) <= 15]
        if not el:
            continue
        a, b = max(el, key=lambda p: f(p[0]) - f(p[1]))
        if f(a) - f(b) < gap:
            continue
        pairs.append({"prompt": q["prompt"], "chosen": a["text"], "rejected": b["text"], "gap": f(a) - f(b),
                      "question_input": q["question_input"], "correct_option_text": q["correct_option_text"],
                      **{f"chosen_{k}": a[k] for k in K}, **{f"rejected_{k}": b[k] for k in K}})
    os.makedirs(f"rl_preference_data_gemma4_shared/{name}", exist_ok=True)
    json.dump(pairs, open(f"rl_preference_data_gemma4_shared/{name}/preference_pairs.json", "w"), ensure_ascii=False, indent=1)
    n = len(pairs) or 1
    stats[name] = {"pairs": len(pairs), **{f"{s}_{k}": sum(p[f"{s}_{k}"] for p in pairs) / n
                                           for s in ("chosen", "rejected") for k in ("nli", "masked_nli", "words", "judge_wins")}}
    print(name, {k: round(v, 3) for k, v in stats[name].items()})
json.dump(stats, open("rl_preference_data_gemma4_shared/pair_stats_judge_lm.json", "w"), indent=1)

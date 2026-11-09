"""Build every reward variant's DPO pairs from the SAME scored candidates.

Variants (score -> best-vs-worst pair, kept if gap >= 0.05, as in the paper):
  hm      : H_M  = (0.7*NLI * 0.3*Ver~ - 0.002*words/100) * 1[ACR>=0.5]   (Table 1 reward)
  ha      : H_A  = 0.5*NLI + 0.5*Ver~
  ha_w25  : 0.25*NLI + 0.75*Ver~          (weight sweep, reviewer 7g7h)
  ha_w75  : 0.75*NLI + 0.25*Ver~
  nli     : NLI only
  ver     : verifier only (raw 1-5)
  hm_dom  : H_M, but chosen must not lose on NLI or Ver (reviewer enkh)
  masknli : answer-masked NLI * 1[ACR>=0.5]  -- entailment must come from the
            explanation's own content, not from restating the answer (ew1Q W2)
Ver~ = (Ver-1)/4.
"""
import json, os, itertools, collections
ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
C = json.load(open("rl_preference_data_gemma4_shared/candidates.json"))
GAP = 0.05
HAS_VER = all("verifier" in c for q in C for c in q["cands"])
if not HAS_VER:  # verifier model unavailable: only the verifier-free variants can be built
    for q in C:
        for c in q["cands"]:
            c["verifier"] = float("nan")


def vn(c):
    return (c["verifier"] - 1) / 4


def hm(c):
    return (0.7 * c["nli"] * 0.3 * vn(c) - 0.002 * c["words"] / 100) * (1.0 if c["acr"] >= 0.5 else 0.0)


SCORES = {
    "hm": hm,
    "ha": lambda c: 0.5 * c["nli"] + 0.5 * vn(c),
    "ha_w25": lambda c: 0.25 * c["nli"] + 0.75 * vn(c),
    "ha_w75": lambda c: 0.75 * c["nli"] + 0.25 * vn(c),
    "nli": lambda c: c["nli"],
    "ver": lambda c: c["verifier"],
    "hm_dom": hm,
    "masknli": lambda c: c["masked_nli"] * (1.0 if c["acr"] >= 0.5 else 0.0),
}


def pick(cands, f, dom=False):
    best = None
    for i, j in itertools.permutations(range(len(cands)), 2):
        a, b = cands[i], cands[j]
        if dom and not (a["nli"] >= b["nli"] and a["verifier"] >= b["verifier"]):
            continue
        g = f(a) - f(b)
        if g >= GAP and (best is None or g > best[2]):
            best = (i, j, g)
    return best


stats = {}
for name, f in SCORES.items():
    if not HAS_VER and name not in ("nli", "masknli"):
        continue
    pairs = []
    for q in C:
        cs = [c for c in q["cands"] if c["text"].strip()]
        if len(cs) < 2:
            continue
        p = pick(cs, f, dom=(name == "hm_dom"))
        if not p:
            continue
        a, b = cs[p[0]], cs[p[1]]
        pairs.append({"prompt": q["prompt"], "chosen": a["text"], "rejected": b["text"], "gap": p[2],
                      "question_input": q["question_input"], "correct_option_text": q["correct_option_text"],
                      **{f"chosen_{k}": a[k] for k in ("nli", "masked_nli", "verifier", "acr", "words")},
                      **{f"rejected_{k}": b[k] for k in ("nli", "masked_nli", "verifier", "acr", "words")}})
    d = f"rl_preference_data_gemma4_shared/{name}"
    os.makedirs(d, exist_ok=True)
    json.dump(pairs, open(f"{d}/preference_pairs.json", "w"), ensure_ascii=False, indent=1)
    n = len(pairs) or 1
    m = lambda k: sum(p[k] for p in pairs) / n
    stats[name] = {"pairs": len(pairs),
                   "chosen_nli": m("chosen_nli"), "rejected_nli": m("rejected_nli"),
                   "chosen_masked_nli": m("chosen_masked_nli"), "rejected_masked_nli": m("rejected_masked_nli"),
                   "chosen_ver": m("chosen_verifier"), "rejected_ver": m("rejected_verifier"),
                   "chosen_words": m("chosen_words"), "rejected_words": m("rejected_words"),
                   "chosen_lower_nli": sum(p["chosen_nli"] < p["rejected_nli"] for p in pairs) / n,
                   "chosen_lower_ver": sum(p["chosen_verifier"] < p["rejected_verifier"] for p in pairs) / n}
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in stats[name].items()})
json.dump(stats, open("rl_preference_data_gemma4_shared/pair_stats.json", "w"), indent=1)

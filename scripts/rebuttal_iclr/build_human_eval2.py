"""Second blinded human study, on the shared Gemma 4 test sets (five courses).

Comparisons (4 items per course each, 40 total, no question used twice):
  <SYS> (length-matched judge+NLI reward, SimPO) vs NLI-DPO, and <SYS> vs SFT.
Eligible items: both outputs non-empty, not identical, lengths within 15 words of each
other (the length-matching rule of the first study). Items are drawn with a fixed seed,
the order is shuffled and A/B is randomised per item.

Writes the PUBLIC blinded items (no system names) and a LOCAL key that is never published:
  human_eval2/items.json
  rl_eval_results/human_eval2_KEY.json
Usage: python build_human_eval2.py judge_lm
"""
import json, os, random, sys
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
SYS = sys.argv[1] if len(sys.argv) > 1 else "judgenli_lm"
TEST = {"cardiff": "preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",
        "sydney": "preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json",
        "law": "preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json",
        "medy1": "preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json",
        "medy2": "preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"}
COURSE = {"cardiff": "Biology (Cardiff)", "sydney": "Biology (Sydney)", "law": "Law (Auckland)",
          "medy1": "Medicine, year 1", "medy2": "Medicine, year 2"}
NAMES = {"sft": "Gemma4-E4B-SFT"}


def gens(tag, dom):
    r = json.load(open(f"rl_eval_results/g4shared_{tag}_{dom}_eval.json"))["detailed_results"]
    name = NAMES.get(tag, "Gemma4-E4B-DPO")
    return r[name]["generated_explanations"]


def ok(a, b):
    return a and b and a.strip() and b.strip() and a.strip() != b.strip() \
        and abs(len(a.split()) - len(b.split())) <= 15


rng = random.Random(2027)
# (label, system A tag, system B tag, items per course); each item's pair is length-matched
COMPS = [("judgenli_vs_nli", SYS, "nli", 2)]
items, key, used = [], [], {d: set() for d in TEST}
for dom, tp in TEST.items():
    test = json.load(open(tp))
    for comp, x, y, k in COMPS:
        X, Y = gens(x, dom), gens(y, dom)
        elig = [i for i in range(len(test)) if i not in used[dom] and ok(X[i], Y[i])]
        idx = rng.sample(elig, k); used[dom].update(idx)
        flips = [n % 2 == 0 for n in range(k)]; rng.shuffle(flips)  # balanced A/B per comparison
        for i, flip in zip(idx, flips):
            a, b = (Y[i], X[i]) if flip else (X[i], Y[i])
            items.append({"course": COURSE[dom], "question": test[i]["input"].strip(), "A": a.strip(), "B": b.strip()})
            key.append({"comparison": comp, "domain": dom, "test_index": i,
                        "A": y if flip else x, "B": x if flip else y,
                        "words_A": len(a.split()), "words_B": len(b.split())})
order = list(range(len(items))); rng.shuffle(order)
items = [dict(items[k], id=f"q{n + 1}") for n, k in enumerate(order)]
key = [dict(key[k], id=f"q{n + 1}", system=SYS) for n, k in enumerate(order)]
os.makedirs("human_eval2", exist_ok=True)
json.dump(items, open("human_eval2/items.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open("rl_eval_results/human_eval2_KEY.json", "w"), indent=1)
import collections; print(len(items), "items", collections.Counter((k["comparison"], k["A"]) for k in key))

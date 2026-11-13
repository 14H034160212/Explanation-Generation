"""M2 (China arm): length-match the Chinese preference pairs before SimPO.

The pairwise judge, told to ignore length, still prefers the longer explanation
~79% of the time (chosen is +50 CJK chars longer on average). Training SimPO on
that as-is teaches "longer = better", not reasoning -- the exact confound the
English line removed with a length-matched subset (preference_pairs_lenmatched).

This reproduces that control for the ZH arm, on CJK-character length:
  1. ratio window: keep pairs whose chosen/rejected CJK-length ratio is within
     [1/R, R] (default R=1.5), dropping the grossly length-confounded pairs;
  2. balance: among the survivors, equalise the count of chosen-longer vs
     rejected-longer pairs by dropping the largest-gap excess on the heavier
     side, so the kept set is ~50/50 and mean(chosen-rejected) ~ 0.

Usage:
    python scripts/python_training/cn_lenmatch_pairs.py \\
        --in preference_data/cn/CMExam_zh_pref_pairs.json \\
        --out preference_data/cn/CMExam_zh_pref_pairs_lenmatched.json --ratio 1.5
"""
from __future__ import annotations

import argparse
import json
import statistics as st


def _report(pairs, label):
    if not pairs:
        print(f"{label}: EMPTY")
        return
    diffs = [p["chosen_cjk"] - p["rejected_cjk"] for p in pairs]
    longer = sum(1 for x in diffs if x > 0)
    print(f"{label}: n={len(pairs)}  chosen_longer={longer} ({longer/len(pairs):.0%})  "
          f"mean(chosen-rejected)={st.mean(diffs):+.1f} CJK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=1.5, help="keep |len ratio| within [1/R, R]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = json.load(open(args.inp, encoding="utf-8"))
    _report(pairs, "ALL")

    R = args.ratio
    win = [p for p in pairs
           if 1.0 / R <= (p["chosen_cjk"] + 1) / (p["rejected_cjk"] + 1) <= R]
    _report(win, f"ratio[{1/R:.2f},{R:.2f}]")

    # Balance: make #(chosen longer) == #(rejected longer). Keep every tie and
    # every pair on the lighter side; from the heavier side keep only the ones
    # with the SMALLEST length gap (closest to balanced), dropping the rest.
    longer = sorted((p for p in win if p["chosen_cjk"] > p["rejected_cjk"]),
                    key=lambda p: p["chosen_cjk"] - p["rejected_cjk"])
    shorter = sorted((p for p in win if p["chosen_cjk"] < p["rejected_cjk"]),
                     key=lambda p: p["rejected_cjk"] - p["chosen_cjk"])
    ties = [p for p in win if p["chosen_cjk"] == p["rejected_cjk"]]
    k = min(len(longer), len(shorter))
    balanced = ties + longer[:k] + shorter[:k]
    _report(balanced, "BALANCED (lenmatched)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(balanced, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(balanced)} length-matched pairs -> {args.out}")


if __name__ == "__main__":
    main()

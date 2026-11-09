"""Decode the second human study (human_eval2) with the local key.

Inputs (any mix):
  --dir D    JSON documents saved by ArtifactData list --out_dir (one file per rater)
  --codes F  text file with one result code per line ("HEV2|name|field|q1=A,q2=Tie,...")
Prints, per comparison, wins for the first-named system / the other / ties / skips per rater
and pooled, with a two-sided sign test on non-tied judgements.
"""
import argparse, glob, json, os
from collections import defaultdict
from scipy.stats import binomtest

ROOT = "/data/qbao775/Explanation-Generation"
ap = argparse.ArgumentParser(); ap.add_argument("--dir"); ap.add_argument("--codes"); a = ap.parse_args()
KEY = {k["id"]: k for k in json.load(open(f"{ROOT}/rl_eval_results/human_eval2_KEY.json"))}

raters = []
if a.dir:
    for f in glob.glob(os.path.join(a.dir, "**", "*.json"), recursive=True):
        d = json.load(open(f)); d = d.get("data", d)
        if "answers" in d:
            raters.append((d.get("rater") or os.path.basename(f), d.get("field", "?"), d["answers"], d.get("submitted")))
if a.codes:
    for line in open(a.codes):
        if line.strip().startswith("HEV2|"):
            _, who, field, ans = line.strip().split("|", 3)
            raters.append((who, field, dict(x.split("=") for x in ans.split(",")), True))

tot = defaultdict(lambda: [0, 0, 0, 0])
for who, field, ans, sub in raters:
    per = defaultdict(lambda: [0, 0, 0, 0])
    for q, v in ans.items():
        k = KEY.get(q)
        if not k or v == "?":
            continue
        sysA = k["system"]  # the system under test is the first in the comparison label
        if v == "Tie":
            i = 2
        elif v == "Skip":
            i = 3
        else:
            i = 0 if k[v] == sysA else 1
        per[k["comparison"]][i] += 1; tot[k["comparison"]][i] += 1
    print(f"{who} ({field}){'' if sub else ' [not submitted]'}: " +
          "; ".join(f"{c}: {w}/{l}/tie {t}/skip {s}" for c, (w, l, t, s) in sorted(per.items())))
print()
for c, (w, l, t, s) in sorted(tot.items()):
    p = binomtest(w, w + l).pvalue if w + l else float("nan")
    print(f"POOLED {c}: system wins {w}, other wins {l}, ties {t}, skips {s} -> "
          f"{w / max(w + l, 1):.0%} of non-tied, sign test p={p:.3g}")

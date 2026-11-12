"""M1 (China arm): turn CMExam into the ZH_HANS SFT surface form.

CMExam (github.com/williamliujl/CMExam, Apache-2.0) is the one Chinese medical
MCQ set with a native per-question 解析 at training scale -- so unlike M0 this
needs no translation and no teacher-distilled rationale. Each CSV row is
{Question, Options, Answer, Explanation}; this reshapes it into the exact
ZH_HANS surface denny-bao-lab serves, keeping only rows with a real explanation.

    python scripts/python_training/cn_prepare_cmexam.py \\
        --csv /data/qbao775/cn_datasets/CMExam/data/train.csv \\
        --out preference_data/cn/CMExam_zh_generator_train.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

from cn_translate_pool import ZH_INSTRUCTION, render_zh_input

# Options come as a block: "A ...\nB ...\n...". Capture each letter's text up to
# the next letter-at-line-start or end of field.
_OPT_RE = re.compile(r"(?:^|\n)\s*([A-Z])[\s\.、:：]+(.*?)(?=\n\s*[A-Z][\s\.、:：]|\Z)", re.S)


def parse_options(block: str):
    """ordered [(letter, text)] or None if the block is not parseable."""
    pairs = [(m.group(1), " ".join(m.group(2).split())) for m in _OPT_RE.finditer(block or "")]
    pairs = [(l, t) for l, t in pairs if t]
    return pairs or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out, skipped = [], 0
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("Question") or "").strip()
            ans = (row.get("Answer") or "").strip()
            expl = (row.get("Explanation") or "").strip()
            pairs = parse_options(row.get("Options", ""))
            # Keep only single-answer MCQs with a real explanation and options
            # whose letters include the answer; the ZH_HANS answer sentence is
            # one letter, and multi-answer rows would need a different surface.
            if not (q and expl and pairs and len(ans) == 1):
                skipped += 1
                continue
            letters = [l for l, _ in pairs]
            if ans not in letters:
                skipped += 1
                continue
            options = [t for _, t in pairs]
            out.append({
                "instruction": ZH_INSTRUCTION,
                "input": render_zh_input(q, options, ans),
                "output": expl,
            })
            if args.limit and len(out) >= args.limit:
                break

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {len(out)} items to {p} ({skipped} skipped)")


if __name__ == "__main__":
    main()

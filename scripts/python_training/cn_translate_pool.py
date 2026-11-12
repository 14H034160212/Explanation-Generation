"""M0 (China arm): translate an English learnersourced generator pool into the
Simplified-Chinese ``ZH_HANS`` surface form, so a Qwen2.5-7B SFT produces a
*demonstrable* Chinese explanation generator for denny-bao-lab's profile=cn.

This is the stopgap data path in docs/china-model-plan.md §三①: fast, good
enough to prove the UI+model path end to end, NOT a credible A/B corpus (the
content is a UK medical course and the Chinese is machine-translated). The
honest caveats live in that doc.

What it does, per item {instruction, input, output} (English surface form):
  1. parse the English ``input`` into stem / options / correct letter;
  2. translate stem, each option, and the explanation to Chinese with a local
     Qwen2.5-7B-Instruct (no API, no key, GPU pinned by the caller);
  3. re-render ``input`` in the exact ZH_HANS surface the served template uses
     (题目：… 选项 A: … 正确答案是选项 X。) and set the Chinese instruction.

Run (GPU 3 only, per machine policy):
    CUDA_VISIBLE_DEVICES=3 conda run -n qwen3-rl python \\
        scripts/python_training/cn_translate_pool.py \\
        --infile preference_data/Paul_new_data/Cardiff_tierC_generator_train.json \\
        --outfile preference_data/cn/Cardiff_zh_generator_train.json \\
        --limit 0            # 0 = all; a small N for a dry run
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

# ZH_HANS surface form, mirrored from denny-bao-lab backend/app/llm/prompts.py.
# The SFT target must match what the server renders, byte for byte, or the
# aligned model sees an off-distribution prompt at serving time.
ZH_INSTRUCTION = "你是一位习题解析专家，请为下面的题目生成解析，逐个说明每个选项为什么正确或错误。"
ZH_QUESTION_PREFIX = "题目："
ZH_OPTION_PREFIX = "选项 "
ZH_ANSWER_SENTENCE = "正确答案是选项 {letter}。"
LETTERS = "ABCDEFGH"

_ANSWER_RE = re.compile(r"The correct answer is Option\s+([A-H])\.?\s*$")
_OPTION_SPLIT_RE = re.compile(r"\s*Option\s+([A-H]):\s*")


def parse_english_input(text: str):
    """(stem, [options], correct_letter) or None if the surface form is off."""
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    letter = m.group(1)
    body = text[: m.start()].strip()
    parts = _OPTION_SPLIT_RE.split(body)
    # parts = [prefix+stem, 'A', optA, 'B', optB, ...]
    if len(parts) < 3:
        return None
    stem = parts[0]
    for lead in ("Given question:", "Question:"):
        if stem.startswith(lead):
            stem = stem[len(lead):]
    stem = stem.strip()
    options = [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]
    if not stem or not options:
        return None
    return stem, options, letter


def render_zh_input(stem_zh: str, options_zh: list[str], letter: str) -> str:
    parts = [f"{ZH_QUESTION_PREFIX}{stem_zh}"]
    for i, opt in enumerate(options_zh):
        parts.append(f"{ZH_OPTION_PREFIX}{LETTERS[i]}: {opt}")
    parts.append(ZH_ANSWER_SENTENCE.format(letter=letter))
    return " ".join(parts)


class Translator:
    def __init__(self, model_path: str, cache_dir: str, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True, padding_side="left"
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map={"": 0},
        ).eval()

    def _prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "你是专业的中英医学翻译，译文准确、通顺、术语规范。"},
            {"role": "user", "content":
                "把下面的英文翻译成简体中文。数字保持阿拉伯数字形式（如 14、34，不要写成中文数字），"
                "保留必要的医学英文缩写。只输出译文本身，不要添加解释、标注、括号说明或引号：\n\n" + text},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def translate(self, texts: list[str]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            prompts = [self._prompt(t) for t in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=640, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            for i in range(len(chunk)):
                new = gen[i][enc["input_ids"].shape[1]:]
                out.append(self.tok.decode(new, skip_special_tokens=True).strip())
            print(f"  translated {min(start + self.batch_size, len(texts))}/{len(texts)} units", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    raw = json.load(open(args.infile, encoding="utf-8"))
    if args.limit:
        raw = raw[: args.limit]

    # Parse first; collect every translation unit across all items so one
    # batched pass covers the whole pool, options kept aligned by index.
    parsed, skipped = [], 0
    units: list[str] = []
    for item in raw:
        p = parse_english_input(item.get("input", ""))
        expl = item.get("output", "").replace("</s>", "").strip()
        if p is None or not expl:
            skipped += 1
            continue
        stem, options, letter = p
        base = len(units)
        units.append(stem)
        units.extend(options)
        units.append(expl)
        parsed.append({"base": base, "n_opt": len(options), "letter": letter})

    print(f"parsed {len(parsed)} items ({skipped} skipped), {len(units)} translation units", flush=True)

    translator = Translator(args.model, args.cache_dir, args.batch_size)
    zh = translator.translate(units)

    out = []
    for rec in parsed:
        b, n = rec["base"], rec["n_opt"]
        stem_zh = zh[b]
        options_zh = zh[b + 1: b + 1 + n]
        expl_zh = zh[b + 1 + n]
        out.append({
            "instruction": ZH_INSTRUCTION,
            "input": render_zh_input(stem_zh, options_zh, rec["letter"]),
            "output": expl_zh,
        })

    outpath = pathlib.Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {len(out)} Chinese items to {outpath}", flush=True)


if __name__ == "__main__":
    main()

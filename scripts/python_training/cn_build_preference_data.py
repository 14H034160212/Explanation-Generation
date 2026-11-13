"""M2 (China arm): build REASONING-aware Chinese preference pairs for SimPO.

Same recipe as rl_build_preference_data_judge.py (sample K candidates from the
SFT'd generator, round-robin PAIRWISE judging, chosen = most wins / rejected =
fewest, keep only strict wins), adapted for the Chinese arm in three ways that
the English script cannot do:

  1. Surface parser reads the ZH_HANS form (题目：… 选项 A: … 正确答案是选项 X。),
     not the English "Option A:/The correct answer is Option X" surface.
  2. The judge is a LOCAL HF model pinned to the caller's GPU (judge_hf.py style),
     NOT the ollama path -- on this machine ollama lives on GPU 0 (off-limits),
     so the judge must run on the same GPU 3/4 the generator is pinned to.
  3. Length filter / word counts use CJK CHARACTER counts. The English code uses
     len(text.split()); Chinese has no spaces, so split() collapses a whole
     explanation to ~1 "word" and would discard every candidate.

Generator (M1) and judge (default microsoft/phi-4, a different family) share one
GPU; a 7B generator + bf16 14B judge fit in 80 GB, else pass --judge_load4.

Run (GPU 3 only, per machine policy):
    CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      python scripts/python_training/cn_build_preference_data.py \\
        --generator_path <Qwen2.5-7B snapshot> \\
        --lora_adapter_path rl_sft_qwen25_7b_cmexam_zh \\
        --data_path preference_data/cn/CMExam_zh_generator_train.json \\
        --judge_path <phi-4 snapshot> --judge_load4 \\
        --max_questions 300 --num_samples 4 \\
        --output_path preference_data/cn/CMExam_zh_pref_pairs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_build_preference_data_nli import (  # noqa: E402  (reuse the live generator path)
    load_qwen3_generator,
    generate_qwen3,
    make_qwen3_dpo_prompt,
    answer_coverage_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- ZH_HANS surface parser (mirror of cn_translate_pool.render_zh_input) -----
_ZH_CORR = re.compile(r"正确答案是选项\s*([A-H])")
_ZH_OPT = re.compile(r"选项\s*([A-H])[:：]\s*(.*?)(?=\s*选项\s*[A-H][:：]|\s*正确答案是选项|$)", re.S)
_ZH_STEM = re.compile(r"题目[:：]\s*(.*?)\s*选项\s*A[:：]", re.S)


def parse_zh_options(input_text: str):
    """(stem, {letter: text}, correct_letter) from the ZH_HANS surface, or (.., None)."""
    cm = _ZH_CORR.search(input_text)
    corr = cm.group(1) if cm else None
    opts = {m.group(1): " ".join(m.group(2).split()) for m in _ZH_OPT.finditer(input_text)}
    sm = _ZH_STEM.search(input_text)
    stem = sm.group(1).strip() if sm else input_text
    return stem, opts, corr


def cjk_len(text: str) -> int:
    """Count CJK characters -- the Chinese stand-in for a word count."""
    return sum(1 for ch in (text or "") if "\u4e00" <= ch <= "\u9fff")


# --- local HF pairwise judge (judge_hf.py style, Chinese prompt) --------------
JUDGE_SYS = (
    "比较同一道选择题（正确答案已给出）的两份解析。哪一份对「为什么该答案正确、其余选项为什么错误」"
    "给出了逻辑上更站得住脚的说明？只评判推理质量；忽略长度、流畅度和排版；只复述答案而不给理由的更差。"
    "严格用 JSON 回复 {\"better\":1|2|0}（1=第一份更好，2=第二份更好，0=二者相当）。"
)


def _last_json(text: str):
    for m in reversed(re.findall(r"\{[^{}]*\}", text or "", flags=re.DOTALL)):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


class HTTPJudge:
    """Pairwise judge over an OpenAI-compatible endpoint (e.g. a vLLM server).

    Lets the judge be a big quantized model (Qwen3-32B-AWQ) served once, with its
    GPU memory capped by the server (--gpu-memory-utilization), so the shared GPU
    keeps headroom for other users instead of one process grabbing the whole card.
    """

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model

    def better(self, q: str, ans: str, e1: str, e2: str) -> int:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 40,
            # Qwen3 defaults to a <think> block that would eat the whole budget
            # before any JSON; turn it off so the verdict comes out directly.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content":
                    f"题目：\n{q}\n正确答案：{ans}\n\n解析1：\n{e1}\n\n解析2：\n{e2}\n"},
            ],
        }
        data = json.dumps(body).encode()
        for _ in range(3):
            try:
                req = urllib.request.Request(
                    self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    txt = json.loads(r.read())["choices"][0]["message"]["content"]
                d = _last_json(txt)
                v = int(d["better"])
                return v if v in (0, 1, 2) else 0
            except Exception:
                time.sleep(2)
        return 0  # unreachable/unparsable -> tie, never crash a multi-hour run


class HFJudge:
    def __init__(self, path: str, device: str, load4: bool):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if load4:
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=qc, device_map={"": int(device.split(":")[-1])},
                trust_remote_code=True).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    def better(self, q: str, ans: str, e1: str, e2: str) -> int:
        msgs = [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content":
                f"题目：\n{q}\n正确答案：{ans}\n\n解析1：\n{e1}\n\n解析2：\n{e2}\n"},
        ]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = self.tok(p, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.model.generate(**x, max_new_tokens=40, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        d = _last_json(self.tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
        try:
            v = int(d["better"])
            return v if v in (0, 1, 2) else 0
        except Exception:
            return 0  # unparsable -> treat as tie, never crash a multi-hour run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator_path", required=True, help="Qwen2.5-7B base snapshot")
    p.add_argument("--lora_adapter_path", required=True, help="M1 SFT adapter dir")
    p.add_argument("--data_path", required=True, help="ZH_HANS items {instruction,input,output}")
    p.add_argument("--judge_path", default=None, help="local HF judge snapshot (mutually exclusive with --judge_url)")
    p.add_argument("--judge_load4", action="store_true")
    p.add_argument("--judge_url", default=None, help="OpenAI-compatible judge endpoint, e.g. http://127.0.0.1:8041 (vLLM)")
    p.add_argument("--judge_model", default="judge", help="served model name at --judge_url")
    p.add_argument("--output_path", default="preference_data/cn/CMExam_zh_pref_pairs.json")
    p.add_argument("--num_samples", type=int, default=4, help="candidates per question")
    p.add_argument("--max_questions", type=int, default=300)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--min_cjk", type=int, default=15, help="drop candidates shorter than this many CJK chars")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if not args.judge_url and not args.judge_path:
        p.error("need one of --judge_url (served judge) or --judge_path (local HF judge)")
    gen_model, gen_tok = load_qwen3_generator(args.generator_path, args.lora_adapter_path, args.device)
    if args.judge_url:
        judge = HTTPJudge(args.judge_url, args.judge_model)
        logger.info(f"judge via HTTP {args.judge_url} (model={args.judge_model})")
    else:
        judge = HFJudge(args.judge_path, args.device, args.judge_load4)

    with open(args.data_path, encoding="utf-8") as f:
        raw = json.load(f)
    random.shuffle(raw)
    questions = raw[args.start_index: args.start_index + args.max_questions]
    judge_name = args.judge_model if args.judge_url else os.path.basename(args.judge_path)
    logger.info(f"ZH pairwise-judge | judge={judge_name} | "
                f"{len(questions)} q x {args.num_samples} candidates")

    pairs = []
    skip_noopt = skip_dup = skip_notie = 0
    for i, item in enumerate(questions):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue
        q, opts, corr = parse_zh_options(input_text)
        if not corr or corr not in opts:
            skip_noopt += 1
            continue
        ans = opts[corr]
        try:
            cands = generate_qwen3(gen_model, gen_tok, instruction, input_text,
                                   args.num_samples, args.device, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"Q{i}: gen error {e}")
            continue
        # CJK-char length filter + de-dup (NOT .split(): Chinese has no spaces)
        cands = [c for c in cands if c and cjk_len(c) >= args.min_cjk]
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        cands = uniq
        if len(cands) < 2:
            skip_dup += 1
            continue

        K = len(cands)
        wins = [0] * K
        comps = 0
        for a in range(K):
            for b in range(a + 1, K):
                comps += 1
                swap = (a + b + i) % 2 == 1  # deterministic per-comparison order flip
                first, second = (b, a) if swap else (a, b)
                res = judge.better(q, ans, cands[first], cands[second])
                if res == 1:
                    wins[first] += 1
                elif res == 2:
                    wins[second] += 1
                # res == 0 -> tie, no win
        best = max(range(K), key=lambda x: wins[x])
        worst = min(range(K), key=lambda x: wins[x])
        if best == worst or wins[best] <= wins[worst]:
            skip_notie += 1
            continue

        pairs.append({
            "prompt": make_qwen3_dpo_prompt(gen_tok, instruction, input_text),
            "chosen": cands[best],
            "rejected": cands[worst],
            "chosen_wins": wins[best],
            "rejected_wins": wins[worst],
            "num_candidates": K,
            "num_comparisons": comps,
            "chosen_cjk": cjk_len(cands[best]),
            "rejected_cjk": cjk_len(cands[worst]),
            "chosen_acr": round(answer_coverage_rate(cands[best], ans), 4),
            "rejected_acr": round(answer_coverage_rate(cands[worst], ans), 4),
            "correct_option_text": ans,
            "question_input": input_text,
        })

        if (i + 1) % 10 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"Q{i + 1}/{len(questions)} | kept {len(pairs)} pairs | "
                        f"skip noopt={skip_noopt} dup={skip_dup} notie={skip_notie}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"DONE: {len(pairs)} preference pairs -> {args.output_path} "
                f"(skip noopt={skip_noopt} dup={skip_dup} notie={skip_notie})")


if __name__ == "__main__":
    main()

"""M3 (China arm): A/B eval of SFT (M1) vs SFT+SimPO (M2) on held-out CMExam.

For each held-out ZH question, generate one explanation GREEDILY from each model
(so the A/B difference is the model, not sampling noise), then have an
INDEPENDENT, different-family judge (Gemma-4-31B-it -- NOT the Qwen3-32B reward
judge, to avoid circularity) say which explanation reasons better, with per-item
order flip to blunt position bias. Report SimPO win/loss/tie and non-tie win rate.

Memory-frugal on a shared GPU: exactly one model is resident at a time -- generate
all SFT, free it; generate all SimPO, free it; then load the 4-bit judge.

    CUDA_VISIBLE_DEVICES=3 python scripts/python_training/cn_eval_ab.py \\
        --base <Qwen2.5-7B snapshot> \\
        --sft_adapter rl_sft_qwen25_7b_cmexam_zh \\
        --simpo_adapter rl_simpo_qwen25_7b_cmexam_zh \\
        --judge_path <gemma-4-31B-it snapshot> \\
        --data preference_data/cn/CMExam_zh_generator_test50.json \\
        --out rl_eval_results/cn_ab_sft_vs_simpo.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cn_build_preference_data import parse_zh_options, JUDGE_SYS, _last_json  # noqa: E402

try:
    from transformers import Gemma4ForConditionalGeneration
except Exception:
    Gemma4ForConditionalGeneration = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def gen_all(base, adapters, items, device, max_new_tokens):
    """Greedy-generate one explanation per item from base + composed adapters."""
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    for ad in adapters:                       # base -> SFT -> (SimPO), each merged
        model = PeftModel.from_pretrained(model, ad)
        model = model.merge_and_unload()
    model = model.to(device).eval()

    outs = []
    for it in items:
        msgs = [{"role": "user", "content": f"{it['instruction']}\n\n{it['input']}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        x = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    del model
    gc.collect(); torch.cuda.empty_cache()
    return outs


def load_gemma_judge(path, device):
    if Gemma4ForConditionalGeneration is None:
        raise RuntimeError("transformers>=5.5 with Gemma4ForConditionalGeneration required")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        path, quantization_config=qc, device_map={"": int(device.split(":")[-1])},
        trust_remote_code=True).eval()
    return model, tok


def judge_better(model, tok, device, q, ans, e1, e2):
    msgs = [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content":
                f"题目：\n{q}\n正确答案：{ans}\n\n解析1：\n{e1}\n\n解析2：\n{e2}\n"}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tok(p, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model.generate(**x, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    d = _last_json(tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
    try:
        v = int(d["better"]);  return v if v in (0, 1, 2) else 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--simpo_adapter", required=True)
    ap.add_argument("--judge_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="rl_eval_results/cn_ab_sft_vs_simpo.json")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    items = json.load(open(args.data, encoding="utf-8"))
    logger.info(f"A/B on {len(items)} held-out items")

    logger.info("generating SFT explanations ...")
    sft = gen_all(args.base, [args.sft_adapter], items, args.device, args.max_new_tokens)
    logger.info("generating SFT+SimPO explanations ...")
    simpo = gen_all(args.base, [args.sft_adapter, args.simpo_adapter], items, args.device, args.max_new_tokens)

    logger.info("loading Gemma-4-31B judge (4-bit) ...")
    jm, jt = load_gemma_judge(args.judge_path, args.device)

    simpo_win = sft_win = tie = 0
    per = []
    for i, it in enumerate(items):
        q, opts, corr = parse_zh_options(it["input"])
        ans = opts.get(corr, "") if corr else ""
        swap = (i % 2 == 1)                    # flip presentation order per item
        e1, e2 = (sft[i], simpo[i]) if swap else (simpo[i], sft[i])
        v = judge_better(jm, jt, args.device, q, ans, e1, e2)
        # map verdict back to which model (position 1 is simpo when not swapped)
        if v == 0:
            tie += 1; winner = "tie"
        elif (v == 1 and not swap) or (v == 2 and swap):
            simpo_win += 1; winner = "simpo"
        else:
            sft_win += 1; winner = "sft"
        per.append({"i": i, "winner": winner, "sft": sft[i], "simpo": simpo[i],
                    "sft_cjk": sum(1 for c in sft[i] if "一" <= c <= "鿿"),
                    "simpo_cjk": sum(1 for c in simpo[i] if "一" <= c <= "鿿")})
        if (i + 1) % 10 == 0:
            logger.info(f"{i+1}/{len(items)} | simpo {simpo_win} / sft {sft_win} / tie {tie}")

    nt = simpo_win + sft_win
    summary = {"n": len(items), "simpo_win": simpo_win, "sft_win": sft_win, "tie": tie,
               "simpo_nontie_winrate": round(simpo_win / max(nt, 1), 4),
               "judge": os.path.basename(args.judge_path.rstrip("/")),
               "mean_sft_cjk": round(sum(p["sft_cjk"] for p in per) / len(per), 1),
               "mean_simpo_cjk": round(sum(p["simpo_cjk"] for p in per) / len(per), 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_item": per}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logger.info(f"DONE {summary}")


if __name__ == "__main__":
    main()

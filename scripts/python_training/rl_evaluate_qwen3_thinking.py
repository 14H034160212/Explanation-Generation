"""
Qwen3 thinking mode evaluation (enable_thinking=True).

Same pipeline as rl_evaluate_qwen3.py but with chain-of-thought enabled.
The <think>...</think> block is stripped before metric computation so that
only the final answer text is scored.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n qwen3-rl python3 \
        scripts/python_training/rl_evaluate_qwen3_thinking.py \
        --test_data_path preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path models/rl_sft_qwen3_8b_generator \
        --output_path rl_eval_results/qwen3_thinking_cardiff_eval.json \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu
"""

import re
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from bert_score import score as bert_score_fn
from peft import PeftModel
from sacrebleu.metrics import BLEU
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As an explanation evaluation expert, can you evaluate the quality of the following "
    "explanation for the given exam question and provide a score from 1 to 5?"
)
VERIFIER_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block and return the remaining answer text."""
    cleaned = THINK_RE.sub("", text).strip()
    # Also strip any leading newlines left after removal
    return cleaned.lstrip("\n").strip()


@dataclass
class EvalResult:
    model: str
    bleu_scores: List[float] = field(default_factory=list)
    bert_scores_stu: List[float] = field(default_factory=list)
    bert_scores_ans: List[float] = field(default_factory=list)
    acr_scores: List[float] = field(default_factory=list)
    nli_scores: List[float] = field(default_factory=list)
    verifier_scores: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    generated_explanations: List[str] = field(default_factory=list)
    raw_outputs: List[str] = field(default_factory=list)  # includes <think> block

    def summary(self):
        n = len(self.bleu_scores) or 1
        think_count = sum(1 for r in self.raw_outputs if "<think>" in r)
        return {
            "model": self.model,
            "avg_bleu": round(sum(self.bleu_scores) / n, 4),
            "avg_bert_score_f1": round(sum(self.bert_scores_stu) / n, 4),
            "avg_bert_score_f1_answer_anchored": round(sum(self.bert_scores_ans) / n, 4),
            "avg_answer_coverage_rate": round(sum(self.acr_scores) / n, 4),
            "avg_nli_entailment": round(sum(self.nli_scores) / n, 4),
            "avg_verifier_score": round(sum(self.verifier_scores) / n, 4),
            "avg_inference_time_s": round(sum(self.inference_times) / n, 3),
            "n_examples": n,
            "thinking_activated_pct": round(100 * think_count / n, 1),
        }


def load_test_data(path):
    return json.load(open(path, "r", encoding="utf-8"))


def load_generator(base_path, lora_path, device):
    logger.info(f"Loading Qwen3-8B from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, tokenizer


def load_verifier(verifier_path, device):
    logger.info(f"Loading verifier from {verifier_path}...")
    tok = AutoTokenizer.from_pretrained(verifier_path, trust_remote_code=True, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        verifier_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return mdl, tok


def load_nli_model(model_name, device, cache_dir):
    logger.info(f"Loading NLI model {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to(device).eval()
    id2label = mdl.config.id2label or {}
    ent_idx = next((k for k, v in id2label.items() if "entail" in v.lower()), 1)
    return mdl, tok, ent_idx


def extract_correct_option_text(input_text):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_m = re.search(
        rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)",
        input_text, re.DOTALL
    )
    return letter, (opt_m.group(1).strip() if opt_m else None)


def generate_explanation_thinking(model, tokenizer, instruction, input_text,
                                   device, max_new_tokens=512):
    """Generate with enable_thinking=True; returns (raw_output, stripped_explanation)."""
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,   # KEY DIFFERENCE
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    stripped = strip_thinking(raw)
    return raw, stripped


def get_verifier_score(ver_model, ver_tok, question_input, explanation):
    prompt = VERIFIER_TEMPLATE.format(
        instruction=VERIFIER_INSTRUCTION,
        input=f"{question_input}\n\nExplanation: {explanation}",
    )
    inputs = ver_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(next(ver_model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = ver_model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=ver_tok.eos_token_id,
        )
    resp = ver_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    nums = re.findall(r"\d+(?:\.\d+)?", resp)
    return min(max(float(nums[0]), 1.0), 5.0) if nums else 3.0


def compute_acr(explanation, correct_option):
    if not correct_option:
        return 0.0
    keywords = [w for w in correct_option.lower().split() if len(w) >= 4]
    if not keywords:
        return 0.0
    exp_lower = explanation.lower()
    return sum(1 for kw in keywords if kw in exp_lower) / len(keywords)


@torch.no_grad()
def compute_nli_batch(nli_model, nli_tok, premises, hypotheses, ent_idx, device, batch_size=16):
    scores = []
    for i in range(0, len(premises), batch_size):
        enc = nli_tok(
            premises[i:i+batch_size], hypotheses[i:i+batch_size],
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, ent_idx].cpu().tolist())
    return scores


def evaluate_model(model_name, gen_model, gen_tok, ver_model, ver_tok,
                   nli_model, nli_tok, nli_ent_idx,
                   test_data, device, nli_device, max_new_tokens, bleu_metric):
    result = EvalResult(model=model_name)
    generated, refs_stu, refs_ans, correct_options = [], [], [], []

    logger.info(f"Evaluating {model_name} on {len(test_data)} examples (thinking mode ON)...")
    for i, item in enumerate(test_data):
        instruction = item.get("instruction", "").strip()
        input_text  = item.get("input", "").strip()
        ref_stu     = item.get("output", item.get("Explanation", "")).strip()
        correct_opt = item.get("correct_option_text", item.get("correct_option", "")).strip()
        if not correct_opt:
            _, correct_opt = extract_correct_option_text(input_text)
            correct_opt = correct_opt or ""
        if not input_text:
            continue

        t0 = time.time()
        try:
            raw, expl = generate_explanation_thinking(
                gen_model, gen_tok, instruction, input_text, device, max_new_tokens
            )
        except Exception as e:
            logger.warning(f"[{i}] generation error: {e}")
            raw, expl = "", ""
        t1 = time.time()

        result.inference_times.append(t1 - t0)
        result.generated_explanations.append(expl)
        result.raw_outputs.append(raw)
        generated.append(expl)
        refs_stu.append(ref_stu)
        refs_ans.append(correct_opt)
        correct_options.append(correct_opt)

        try:
            b = bleu_metric.sentence_score(expl, [ref_stu]).score / 100.0
        except Exception:
            b = 0.0
        result.bleu_scores.append(b)
        result.acr_scores.append(compute_acr(expl, correct_opt))

        try:
            vs = get_verifier_score(ver_model, ver_tok, input_text, expl)
        except Exception:
            vs = 3.0
        result.verifier_scores.append(vs)

        if (i + 1) % 10 == 0:
            think_count = sum(1 for r in result.raw_outputs if "<think>" in r)
            logger.info(f"  [{model_name}] {i+1}/{len(test_data)} done | "
                        f"thinking activated: {think_count}/{i+1}")

    if generated:
        try:
            _, _, F1 = bert_score_fn(generated, refs_stu, lang="en", verbose=False, device=device)
            result.bert_scores_stu = F1.tolist()
        except Exception as e:
            logger.warning(f"BERTScore(stu) error: {e}")
            result.bert_scores_stu = [0.0] * len(generated)

        valid = [(g, a) for g, a in zip(generated, refs_ans) if a]
        if valid:
            g_v, a_v = zip(*valid)
            try:
                _, _, F1a = bert_score_fn(list(g_v), list(a_v), lang="en", verbose=False, device=device)
                f1a = F1a.tolist()
            except Exception:
                f1a = [0.0] * len(valid)
            idx = 0
            for a in refs_ans:
                result.bert_scores_ans.append(f1a[idx] if a else 0.0)
                if a:
                    idx += 1
        else:
            result.bert_scores_ans = [0.0] * len(generated)

        if nli_model is not None:
            valid_nli = [(g, c) for g, c in zip(generated, correct_options) if c]
            if valid_nli:
                g_nli, h_nli = zip(*valid_nli)
                nli_vals = compute_nli_batch(
                    nli_model, nli_tok, list(g_nli), list(h_nli),
                    nli_ent_idx, nli_device
                )
                idx = 0
                for c in correct_options:
                    result.nli_scores.append(nli_vals[idx] if c else 0.0)
                    if c:
                        idx += 1
            else:
                result.nli_scores = [0.0] * len(generated)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--base_model_path", default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--sft_lora_path", default="models/rl_sft_qwen3_8b_generator")
    parser.add_argument("--verifier_path",
                        default="models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    parser.add_argument("--output_path", default="rl_eval_results/qwen3_thinking_eval.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier_device", default="cuda:0")
    parser.add_argument("--nli_device", default="cpu")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--cache_dir", default="scripts/python_training/cache")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    test_data = load_test_data(args.test_data_path)
    logger.info(f"Test data: {len(test_data)} examples | thinking mode: ON")

    bleu_metric = BLEU(effective_order=True)
    ver_model, ver_tok = load_verifier(args.verifier_path, args.verifier_device)

    nli_model = nli_tok = None
    nli_ent_idx = 1
    try:
        nli_model, nli_tok, nli_ent_idx = load_nli_model(
            args.nli_model, args.nli_device, args.cache_dir
        )
    except Exception as e:
        logger.warning(f"NLI load failed: {e}")

    gen_model, gen_tok = load_generator(args.base_model_path, args.sft_lora_path, args.device)
    result = evaluate_model(
        "Qwen3-8B-SFT-Thinking", gen_model, gen_tok, ver_model, ver_tok,
        nli_model, nli_tok, nli_ent_idx,
        test_data, args.device, args.nli_device, args.max_new_tokens, bleu_metric,
    )
    del gen_model
    torch.cuda.empty_cache()

    s = result.summary()
    logger.info(
        f"[Thinking] BLEU={s['avg_bleu']:.4f} BERT(Stu)={s['avg_bert_score_f1']:.4f} "
        f"BERT(Ans)={s['avg_bert_score_f1_answer_anchored']:.4f} "
        f"ACR={s['avg_answer_coverage_rate']:.4f} NLI={s['avg_nli_entailment']:.4f} "
        f"Ver={s['avg_verifier_score']:.4f} Time={s['avg_inference_time_s']:.2f}s "
        f"Thinking%={s['thinking_activated_pct']:.1f}%"
    )

    output = {
        "test_data_path": args.test_data_path,
        "num_test_examples": len(test_data),
        "thinking_mode": True,
        "results": [s],
        "detailed_results": [{
            "model": result.model,
            "bleu_scores": result.bleu_scores,
            "bert_scores_vs_student": result.bert_scores_stu,
            "bert_scores_vs_answer": result.bert_scores_ans,
            "acr_scores": result.acr_scores,
            "nli_entailment_scores": result.nli_scores,
            "verifier_scores": result.verifier_scores,
            "inference_times": result.inference_times,
            "generated_explanations": result.generated_explanations,
            "raw_outputs_with_thinking": result.raw_outputs,
        }],
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()

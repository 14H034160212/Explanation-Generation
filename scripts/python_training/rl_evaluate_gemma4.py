"""
RLearner-LLM Evaluation (Gemma 4 E4B-it)
gemma4-rl conda env (transformers 5.5.4, TRL 0.28)

Evaluates Gemma 4 E4B-it SFT and DPO models on PeerWise test data.
Uses Gemma 4 chat template (enable_thinking=False).
Metrics: BLEU, BERTScore(Stu), BERTScore(Ans), ACR, NLI, Verifier, Inference time.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n gemma4-rl python3 rl_evaluate_gemma4.py \\
        --test_data_path ./preference_data/Paul_new_data/Cardiff_all_generator_test_avg_3_lenexp_10.json \\
        --base_model_path google/gemma-4-E4B-it \\
        --sft_lora_path ./rl_sft_gemma4_e4b_cardiff_generator \\
        --dpo_lora_path ./rl_dpo_gemma4_e4b_cardiff_generator \\
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \\
        --output_path ./rl_eval_results/gemma4_cardiff_eval.json \\
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
    Gemma4ForConditionalGeneration,
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

    def summary(self):
        n = len(self.bleu_scores) or 1
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
        }


def load_test_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_generator(base_path: str, lora_paths, device: str, cache_dir: str):
    """Load base + one or more LoRA adapters, merging each in sequence.

    `lora_paths` is either a single path (SFT eval) or a list of paths
    (e.g. [sft_path, dpo_path] for DPO eval — DPO was trained ON TOP of SFT,
    so we must first merge SFT into base weights before applying the DPO
    adapter, otherwise the DPO delta is applied to the wrong reference and
    the base-IT prior (markdown/meta-format) leaks through.
    """
    logger.info(f"Loading Gemma 4 E4B-it from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, padding_side="left", cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Gemma4ForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=cache_dir,
    )

    if isinstance(lora_paths, (str, type(None))):
        lora_paths = [lora_paths] if lora_paths else []

    for lp in lora_paths:
        if lp and os.path.exists(lp):
            logger.info(f"Loading LoRA adapter from {lp} (will be merged into base)...")
            model = PeftModel.from_pretrained(model, lp)
            model = model.merge_and_unload()

    model = model.to(device).eval()
    return model, tokenizer


def load_verifier(verifier_path: str, device: str):
    logger.info(f"Loading verifier from {verifier_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        verifier_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        verifier_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, tokenizer


def load_nli_model(model_name: str, device: str, cache_dir: str):
    logger.info(f"Loading NLI model {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to(device).eval()
    id2label = mdl.config.id2label or {}
    ent_idx = next((k for k, v in id2label.items() if "entail" in v.lower()), 1)
    return mdl, tok, ent_idx


def extract_correct_option_text(input_text: str):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    opt_text = opt_m.group(1).strip() if opt_m else None
    return letter, opt_text


def generate_explanation(model, tokenizer, instruction: str, input_text: str,
                          device: str, max_new_tokens: int = 300) -> str:
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def get_verifier_score(ver_model, ver_tok, question_input: str, explanation: str) -> float:
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


def compute_acr(explanation: str, correct_option: str) -> float:
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


def evaluate_model(model_name: str, gen_model, gen_tok, ver_model, ver_tok,
                   nli_model, nli_tok, nli_ent_idx,
                   test_data, device: str, nli_device: str,
                   max_new_tokens: int, bleu_metric) -> EvalResult:
    result = EvalResult(model=model_name)
    generated, refs_stu, refs_ans, correct_options = [], [], [], []

    logger.info(f"Evaluating {model_name} on {len(test_data)} examples...")
    for i, item in enumerate(test_data):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        ref_stu = item.get("output", item.get("Explanation", "")).strip()
        correct_opt = item.get("correct_option_text", item.get("correct_option", "")).strip()
        if not correct_opt:
            _, correct_opt = extract_correct_option_text(input_text)
            correct_opt = correct_opt or ""

        if not input_text:
            continue

        t0 = time.time()
        try:
            expl = generate_explanation(gen_model, gen_tok, instruction, input_text,
                                         device, max_new_tokens)
        except Exception as e:
            logger.warning(f"[{i}] generation error: {e}")
            expl = ""
        t1 = time.time()

        result.inference_times.append(t1 - t0)
        result.generated_explanations.append(expl)
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
            logger.info(f"  [{model_name}] {i+1}/{len(test_data)} done")

    if generated:
        try:
            _, _, F1 = bert_score_fn(generated, refs_stu, lang="en", verbose=False,
                                      device=device)
            result.bert_scores_stu = F1.tolist()
        except Exception as e:
            logger.warning(f"BERTScore(stu) error: {e}")
            result.bert_scores_stu = [0.0] * len(generated)

        valid = [(g, a) for g, a in zip(generated, refs_ans) if a]
        if valid:
            g_valid, a_valid = zip(*valid)
            try:
                _, _, F1a = bert_score_fn(list(g_valid), list(a_valid), lang="en",
                                           verbose=False, device=device)
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
                nli_vals = compute_nli_batch(nli_model, nli_tok, list(g_nli), list(h_nli),
                                              nli_ent_idx, nli_device)
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
    parser.add_argument("--base_model_path", default="google/gemma-4-E4B-it")
    parser.add_argument("--sft_lora_path", default="./rl_sft_gemma4_e4b_cardiff_generator")
    parser.add_argument("--dpo_lora_path", default=None)
    parser.add_argument("--verifier_path",
                        default="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    parser.add_argument("--output_path", default="./rl_eval_results/gemma4_cardiff_eval.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier_device", default="cuda:0")
    parser.add_argument("--nli_device", default="cpu")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--skip_nli", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N examples (for quick sanity)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    test_data = load_test_data(args.test_data_path)
    if args.limit is not None:
        test_data = test_data[: args.limit]
    logger.info(f"Test data: {len(test_data)} examples")

    bleu_metric = BLEU(effective_order=True)
    ver_model, ver_tok = load_verifier(args.verifier_path, args.verifier_device)

    nli_model = nli_tok = None
    nli_ent_idx = 1
    if not args.skip_nli:
        try:
            nli_model, nli_tok, nli_ent_idx = load_nli_model(
                args.nli_model, args.nli_device, args.cache_dir
            )
        except Exception as e:
            logger.warning(f"NLI load failed: {e}. Skipping NLI metric.")

    all_results = []
    # For SFT eval, apply only SFT adapter. For DPO eval, apply SFT first then DPO
    # because the DPO adapter was trained on top of the SFT-merged base.
    models_to_eval = [("Gemma4-E4B-SFT", [args.sft_lora_path])]
    if args.dpo_lora_path and os.path.exists(args.dpo_lora_path):
        models_to_eval.append(("Gemma4-E4B-DPO", [args.sft_lora_path, args.dpo_lora_path]))

    for model_name, lora_paths in models_to_eval:
        gen_model, gen_tok = load_generator(args.base_model_path, lora_paths, args.device, args.cache_dir)
        result = evaluate_model(
            model_name, gen_model, gen_tok, ver_model, ver_tok,
            nli_model, nli_tok, nli_ent_idx,
            test_data, args.device, args.nli_device,
            args.max_new_tokens, bleu_metric,
        )
        all_results.append(result)
        del gen_model
        torch.cuda.empty_cache()

        s = result.summary()
        logger.info(
            f"[{model_name}] BLEU={s['avg_bleu']:.4f} BERT(Stu)={s['avg_bert_score_f1']:.4f} "
            f"BERT(Ans)={s['avg_bert_score_f1_answer_anchored']:.4f} "
            f"ACR={s['avg_answer_coverage_rate']:.4f} NLI={s['avg_nli_entailment']:.4f} "
            f"Verifier={s['avg_verifier_score']:.4f} Time={s['avg_inference_time_s']:.3f}s"
        )

    out = {
        "results": [r.summary() for r in all_results],
        "detailed_results": {
            r.model: {
                "bleu_scores": r.bleu_scores,
                "bert_scores_vs_student": r.bert_scores_stu,
                "bert_scores_vs_answer": r.bert_scores_ans,
                "acr_scores": r.acr_scores,
                "nli_entailment_scores": r.nli_scores,
                "verifier_scores": r.verifier_scores,
                "inference_times": r.inference_times,
                "generated_explanations": r.generated_explanations,
            }
            for r in all_results
        },
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()

"""
RLearner-LLM Step 2 (Pareto-Dominant): Build DPO Preference Dataset
LLaMA-2-13B / LLaMA-3-8B variant (llm-tuning conda env)

KEY DIFFERENCE from the original multiplicative-ACR script:
  The original approach selects (best_verifier, worst_verifier) pairs, which can
  create training signal that trades ACR for NLI or vice versa.

  This script uses PARETO-DOMINANT pair selection:
    chosen Pareto-dominates rejected iff:
      chosen_acr  >= rejected_acr   (ACR never decreases)
      chosen_nli  >= rejected_nli   (NLI never decreases)
      AND at least one is STRICTLY better

  Every DPO gradient therefore pushes BOTH ACR and NLI upward simultaneously,
  eliminating the conflicting signals that caused per-architecture metric trade-offs.

Usage:
    CUDA_VISIBLE_DEVICES=4 conda run -n llm-tuning python3 \
        scripts/python_training/rl_build_preference_data_pareto.py \
        --generator_path /data/shared/llama2/llama-2-13b-hf \
        --lora_adapter_path ./rl_sft_llama2_13b_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_path ./rl_preference_data_pareto_llama2/preference_pairs.json \
        --num_samples 5 --max_questions 500 \
        --generator_device cuda:0 --verifier_device cuda:0 --nli_device cuda:0
"""

import argparse
import json
import logging
import os
import re
import random
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As a question rating verifier expert, can you generate the question rating score "
    "for the given input?"
)
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


# ---------------------------------------------------------------------------
# Metric helpers (identical to rl_evaluation.py)
# ---------------------------------------------------------------------------

def extract_correct_option_text(input_text: str) -> Optional[str]:
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None
    letter = m.group(1)
    opt_m = re.search(
        rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)",
        input_text, re.DOTALL
    )
    return opt_m.group(1).strip() if opt_m else None


def answer_coverage_rate(explanation: str, correct_option_text: str) -> float:
    if not explanation or not correct_option_text:
        return 0.0
    exp_lower = explanation.lower()
    key_terms = re.findall(r"\b\w{4,}\b", correct_option_text.lower())
    if not key_terms:
        return 0.0
    matched = sum(1 for t in key_terms if t in exp_lower)
    return matched / len(key_terms)


@torch.no_grad()
def nli_entailment_score(
    nli_model, nli_tokenizer, entailment_idx: int,
    explanation: str, correct_option: str, device: str
) -> float:
    """DeBERTa cross-encoder: P(explanation ENTAILS correct_option)."""
    if not explanation or not correct_option:
        return 0.0
    enc = nli_tokenizer(
        explanation, correct_option,
        truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0, entailment_idx].item()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_generator(model_path: str, lora_path: Optional[str], device: str):
    logger.info(f"Loading generator from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if lora_path and os.path.isdir(lora_path):
        logger.info(f"Loading LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    logger.info("Generator ready.")
    return model, tokenizer


def load_verifier(model_path: str, device: str):
    logger.info(f"Loading verifier from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    logger.info("Verifier ready.")
    return model, tokenizer


def load_nli_model(device: str, cache_dir: str = "cache"):
    logger.info(f"Loading NLI model: {NLI_MODEL_NAME} ...")
    nli_tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL_NAME, cache_dir=cache_dir, use_fast=False
    )
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME, cache_dir=cache_dir
    ).to(device).eval()
    id2label = nli_model.config.id2label or {}
    entailment_idx = next(
        (k for k, v in id2label.items() if "entail" in v.lower()), 1
    )
    logger.info(f"NLI model ready. entailment_idx={entailment_idx}")
    return nli_model, nli_tokenizer, entailment_idx


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_explanations(
    model, tokenizer, instruction: str, input_text: str,
    num_samples: int, device: str, max_new_tokens: int = 300
) -> List[str]:
    prompt = ALPACA_PROMPT.format(instruction=instruction, input=input_text)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    temperatures = [0.4, 0.6, 0.7, 0.8, 1.0]
    explanations = []
    for i in range(num_samples):
        temp = temperatures[i % len(temperatures)]
        out = model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temp, top_p=0.95, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        response = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
        if response:
            explanations.append(response)
    return explanations


@torch.no_grad()
def get_verifier_score(model, tokenizer, question_input: str, explanation: str) -> float:
    fulltext = (
        f"Below is an instruction that describes a task, paired with an input that "
        f"provides further context. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{VERIFIER_INSTRUCTION}\n\n"
        f"### Input:\n{question_input} Explanation: {explanation}\n\n"
        f"### Response:\n"
    )
    tokens = tokenizer(fulltext, return_tensors="pt").input_ids.to(
        next(model.parameters()).device
    )
    out = model.generate(
        tokens, max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    after = text.split("### Response:")[-1]
    nums = re.findall(r"\d+(?:\.\d+)?", after)
    return float(nums[0]) if nums else 3.0


# ---------------------------------------------------------------------------
# Pareto-dominant pair selection
# ---------------------------------------------------------------------------

def pareto_dominant_pairs(
    candidates: List[Tuple[str, float, float, float]],  # (text, acr, nli, verifier)
    min_acr_delta: float = 0.0,
    min_nli_delta: float = 0.0,
    min_combined_delta: float = 0.05,
) -> Optional[Tuple[dict, dict]]:
    """
    Find the best Pareto-dominant pair from candidates.

    chosen Pareto-dominates rejected iff:
      chosen_acr  >= rejected_acr + min_acr_delta
      chosen_nli  >= rejected_nli + min_nli_delta
      combined improvement (Δacr + Δnli) >= min_combined_delta

    Returns (chosen_dict, rejected_dict) for the best pair found, or None.
    """
    best_pair = None
    best_combined = -1.0

    n = len(candidates)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            text_i, acr_i, nli_i, ver_i = candidates[i]
            text_j, acr_j, nli_j, ver_j = candidates[j]

            # Pareto dominance: i must be >= j on BOTH metrics
            if acr_i < acr_j - 1e-6 or nli_i < nli_j - 1e-6:
                continue

            # At least one must be strictly better
            if acr_i <= acr_j + 1e-6 and nli_i <= nli_j + 1e-6:
                continue

            delta_acr = acr_i - acr_j
            delta_nli = nli_i - nli_j
            combined = delta_acr + delta_nli

            if combined < min_combined_delta:
                continue

            if combined > best_combined:
                best_combined = combined
                best_pair = (
                    {"text": text_i, "acr": acr_i, "nli": nli_i, "verifier": ver_i},
                    {"text": text_j, "acr": acr_j, "nli": nli_j, "verifier": ver_j},
                )

    return best_pair


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pareto-dominant DPO preference dataset builder.")
    parser.add_argument("--generator_path", required=True)
    parser.add_argument("--lora_adapter_path", default=None)
    parser.add_argument("--verifier_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--min_combined_delta", type=float, default=0.05,
                        help="Min (Δacr + Δnli) to accept a pair.")
    parser.add_argument("--generator_device", default="cuda:0")
    parser.add_argument("--verifier_device", default="cuda:0")
    parser.add_argument("--nli_device", default="cuda:0")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    gen_model, gen_tokenizer = load_generator(
        args.generator_path, args.lora_adapter_path, args.generator_device
    )
    ver_model, ver_tokenizer = load_verifier(args.verifier_path, args.verifier_device)
    nli_model, nli_tokenizer, entailment_idx = load_nli_model(args.nli_device, args.cache_dir)

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    random.shuffle(raw_data)
    questions = raw_data[:args.max_questions]
    logger.info(f"Processing {len(questions)} questions × {args.num_samples} samples each ...")

    pairs = []
    skipped_no_pareto = 0
    skipped_no_option = 0

    for i, item in enumerate(tqdm(questions, desc="Pareto pair building")):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue

        # Extract correct option text (required for ACR and NLI)
        correct_option = extract_correct_option_text(input_text)
        if not correct_option:
            skipped_no_option += 1
            continue

        # Generate N candidate explanations
        try:
            explanations = generate_explanations(
                gen_model, gen_tokenizer, instruction, input_text,
                args.num_samples, args.generator_device, args.max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Q{i}: generation error: {e}")
            continue

        if len(explanations) < 2:
            continue

        # Score each candidate on all three metrics
        candidates = []
        for exp in explanations:
            if not exp.strip():
                continue
            acr = answer_coverage_rate(exp, correct_option)
            nli = nli_entailment_score(
                nli_model, nli_tokenizer, entailment_idx,
                exp, correct_option, args.nli_device
            )
            ver = get_verifier_score(ver_model, ver_tokenizer, input_text, exp)
            candidates.append((exp, acr, nli, ver))

        if len(candidates) < 2:
            continue

        # Pareto-dominant pair selection
        result = pareto_dominant_pairs(
            candidates,
            min_combined_delta=args.min_combined_delta,
        )
        if result is None:
            skipped_no_pareto += 1
            continue

        chosen, rejected = result

        # Build DPO-format prompt (Alpaca template, ends at ### Response:)
        prompt_text = ALPACA_PROMPT.format(instruction=instruction, input=input_text)

        pairs.append({
            "prompt": prompt_text,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "chosen_acr": chosen["acr"],
            "chosen_nli": chosen["nli"],
            "chosen_verifier": chosen["verifier"],
            "rejected_acr": rejected["acr"],
            "rejected_nli": rejected["nli"],
            "rejected_verifier": rejected["verifier"],
            "delta_acr": chosen["acr"] - rejected["acr"],
            "delta_nli": chosen["nli"] - rejected["nli"],
        })

        if (i + 1) % 50 == 0:
            logger.info(
                f"Q{i+1}/{len(questions)} | pairs={len(pairs)} "
                f"| skipped_no_pareto={skipped_no_pareto}"
            )

    logger.info(
        f"\nDone! {len(pairs)} Pareto-dominant pairs built.\n"
        f"  Skipped (no Pareto pair found): {skipped_no_pareto}\n"
        f"  Skipped (no correct option parsed): {skipped_no_option}"
    )

    if pairs:
        avg_delta_acr = sum(p["delta_acr"] for p in pairs) / len(pairs)
        avg_delta_nli = sum(p["delta_nli"] for p in pairs) / len(pairs)
        logger.info(f"  Avg Δacr = {avg_delta_acr:.4f}, Avg Δnli = {avg_delta_nli:.4f}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()

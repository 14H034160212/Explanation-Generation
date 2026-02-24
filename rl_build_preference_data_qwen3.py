"""
RL-ILearner Step 2 (Qwen3-8B): Build DPO Preference Pairs
TRL >=0.12 + Transformers >=4.51.0 (qwen3-rl conda env)

Generates N explanations per question using Qwen3-8B + SFT LoRA adapter,
scores them with the Alpaca-7B verifier, and creates (chosen, rejected) pairs.
Uses Qwen3 chat template with enable_thinking=False.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 conda run -n qwen3-rl python3 rl_build_preference_data_qwen3.py \\
        --generator_path /data/shared/qwen3/Qwen3-8B \\
        --lora_adapter_path ./rl_sft_qwen3_8b_generator \\
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \\
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \\
        --output_path ./rl_preference_data_qwen3/preference_pairs.json \\
        --num_samples 3 --min_score_gap 0.1 --max_questions 500 \\
        --generator_device cuda:0 --verifier_device cuda:1
"""

import argparse
import json
import logging
import os
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Alpaca template for verifier (unchanged — verifier is Alpaca-7B)
VERIFIER_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
VERIFIER_INSTRUCTION = (
    "As an explanation evaluation expert, can you evaluate the quality of the following "
    "explanation for the given exam question and provide a score from 1 to 5?"
)


def load_generator(generator_path: str, lora_path: str, device: str):
    """Load Qwen3-8B generator with SFT LoRA adapter."""
    logger.info(f"Loading Qwen3-8B generator from {generator_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        generator_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        generator_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading SFT LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged into base model.")
    model = model.to(device).eval()
    return model, tokenizer


def load_verifier(verifier_path: str, device: str):
    """Load Alpaca-7B verifier (use_fast=False to avoid LlamaTokenizerFast recursion)."""
    logger.info(f"Loading verifier from {verifier_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        verifier_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        verifier_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer


def generate_explanations(
    model, tokenizer, instruction: str, input_text: str,
    num_samples: int, device: str, max_new_tokens: int = 300
):
    """Generate N explanations using Qwen3 chat template (enable_thinking=False)."""
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    explanations = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            explanations.append(text)
    return explanations


def get_verifier_score(
    verifier_model, verifier_tokenizer,
    question_input: str, explanation: str, device: str
) -> float:
    """Score explanation with Alpaca-7B verifier. Returns float 1-5."""
    prompt = VERIFIER_TEMPLATE.format(
        instruction=VERIFIER_INSTRUCTION,
        input=f"{question_input}\n\nExplanation: {explanation}",
    )
    inputs = verifier_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    # Move to verifier device (NOT generator device — avoids cuda:0 vs cuda:1 mismatch)
    inputs = {k: v.to(next(verifier_model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = verifier_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=verifier_tokenizer.eos_token_id,
        )
    response = verifier_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Parse score from response (e.g. "4", "Score: 3/5", "4.5")
    import re
    nums = re.findall(r"\d+(?:\.\d+)?", response)
    if nums:
        score = float(nums[0])
        return min(max(score, 1.0), 5.0)
    return 3.0  # default if parsing fails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--lora_adapter_path", default="./rl_sft_qwen3_8b_generator")
    parser.add_argument("--verifier_path", default="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    parser.add_argument("--data_path", default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json")
    parser.add_argument("--output_path", default="./rl_preference_data_qwen3/preference_pairs.json")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--min_score_gap", type=float, default=0.1)
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--generator_device", default="cuda:0")
    parser.add_argument("--verifier_device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    gen_model, gen_tokenizer = load_generator(
        args.generator_path, args.lora_adapter_path, args.generator_device
    )
    ver_model, ver_tokenizer = load_verifier(args.verifier_path, args.verifier_device)

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    random.shuffle(raw_data)
    questions = raw_data[:args.max_questions]
    logger.info(f"Processing {len(questions)} questions × {args.num_samples} samples...")

    pairs = []
    for i, item in enumerate(questions):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue

        try:
            explanations = generate_explanations(
                gen_model, gen_tokenizer,
                instruction, input_text,
                args.num_samples, args.generator_device, args.max_new_tokens,
            )
            scores = [
                get_verifier_score(ver_model, ver_tokenizer, input_text, exp, args.verifier_device)
                for exp in explanations
            ]
        except Exception as e:
            logger.warning(f"Q{i}: generation/scoring error: {e}")
            continue

        best_idx = max(range(len(scores)), key=lambda x: scores[x])
        worst_idx = min(range(len(scores)), key=lambda x: scores[x])

        if best_idx == worst_idx or (scores[best_idx] - scores[worst_idx]) < args.min_score_gap:
            logger.info(f"Q{i}: gap={scores[best_idx]-scores[worst_idx]:.2f} < {args.min_score_gap} → skipped")
            continue

        # Build DPO pair using Qwen3 chat template (prompt-only for chosen/rejected)
        prompt_messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
        prompt_text = gen_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        pairs.append({
            "prompt": prompt_text,
            "chosen": explanations[best_idx],
            "rejected": explanations[worst_idx],
            "chosen_score": scores[best_idx],
            "rejected_score": scores[worst_idx],
            "question_input": input_text,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(questions)} | Pairs so far: {len(pairs)}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"Done! {len(pairs)} preference pairs saved to {args.output_path}")
    if pairs:
        avg_gap = sum(p["chosen_score"] - p["rejected_score"] for p in pairs) / len(pairs)
        logger.info(f"Average score gap: {avg_gap:.3f}")


if __name__ == "__main__":
    main()

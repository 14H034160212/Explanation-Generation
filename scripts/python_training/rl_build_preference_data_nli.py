"""
RLearner-LLM: NLI-Guided Preference Data Builder
Supports LLaMA-2-13B and Qwen3-8B via --model_type.

Replaces the Alpaca-7B verifier with NLI entailment probability as the
preference ranking signal, directly optimising the metric that matters most:
does the explanation logically entail the correct answer option?

Scoring modes (--score_method):
  nli              — rank by NLI entailment probability only (default)
  hybrid           — rank by 0.5 * NLI + 0.5 * normalised_verifier
  verifier         — rank by Alpaca-7B verifier only (legacy behaviour)
  multiplicative_acr — Multiplicative Gate (matches methodology figure):
                       R_i = (w_nli * S_nli * S_ver_norm - gamma * len_norm(E_i))
                             * I[acr(E_i) >= acr_threshold]
                       Explanations failing ACR gate get score=0 (become rejected).

Usage — Qwen3-8B NLI (qwen3-rl env):
    CUDA_VISIBLE_DEVICES=6 conda run -n qwen3-rl python3 rl_build_preference_data_nli.py \\
        --model_type qwen3 --score_method nli \\
        --generator_path /data/shared/qwen3/Qwen3-8B \\
        --lora_adapter_path ./rl_sft_qwen3_8b_generator \\
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \\
        --output_path ./rl_preference_data_nli_qwen3/preference_pairs.json \\
        --num_samples 3 --min_score_gap 0.05 --max_questions 500 \\
        --generator_device cuda:0 --nli_device cpu

Usage — Qwen3-8B Hybrid 0.5NLI+0.5Ver (qwen3-rl env):
    CUDA_VISIBLE_DEVICES=6 conda run -n qwen3-rl python3 rl_build_preference_data_nli.py \\
        --model_type qwen3 --score_method hybrid \\
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \\
        --generator_path /data/shared/qwen3/Qwen3-8B \\
        --lora_adapter_path ./rl_sft_qwen3_8b_generator \\
        --output_path ./rl_preference_data_hybrid_qwen3/preference_pairs.json \\
        --num_samples 3 --min_score_gap 0.05 --max_questions 500 \\
        --generator_device cuda:0 --nli_device cpu --verifier_device cuda:1

Usage — LLaMA-2-13B NLI (llm-tuning env):
    CUDA_VISIBLE_DEVICES=4,5 conda run -n llm-tuning python3 rl_build_preference_data_nli.py \\
        --model_type llama2 --score_method nli \\
        --generator_path /data/shared/llama2/llama-2-13b-hf \\
        --lora_adapter_path ./rl_sft_llama2_13b_generator \\
        --output_path ./rl_preference_data_nli_llama2/preference_pairs.json \\
        --num_samples 3 --min_score_gap 0.05 --max_questions 500 \\
        --generator_device cuda:0 --nli_device cpu
"""

import argparse
import json
import logging
import os
import random
import re
from typing import List, Tuple, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:
    Gemma4ForConditionalGeneration = None  # only required when --model_type gemma4

# Verifier constants (reused from rl_build_preference_data_qwen3.py)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca template (LLaMA-2 generator prompt + DPO prompt)
# ---------------------------------------------------------------------------
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


# ---------------------------------------------------------------------------
# ACR (Answer Coverage Rate) — mirrored from gpt_generate_and_eval.py
# ---------------------------------------------------------------------------

def answer_coverage_rate(explanation: str, correct_option_text: str) -> float:
    """Fraction of ≥4-char keywords from correct option text found in explanation."""
    if not explanation or not correct_option_text:
        return 0.0
    key_terms = re.findall(r"\b\w{4,}\b", correct_option_text.lower())
    if not key_terms:
        return 0.0
    exp_lower = explanation.lower()
    return sum(1 for t in key_terms if t in exp_lower) / len(key_terms)


# ---------------------------------------------------------------------------
# NLI utilities (copied verbatim from rl_evaluation.py)
# ---------------------------------------------------------------------------

def extract_correct_option_text(input_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the correct answer letter and option text from the question input.
    Format: "Option A: <text> ... The correct answer is Option X."
    Returns (correct_letter, option_text) or (None, None) if parsing fails.
    """
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    opt_text = opt_m.group(1).strip() if opt_m else None
    return letter, opt_text


def load_nli_model(model_name: str, device: str = "cpu", cache_dir: str = "cache"):
    """
    Load DeBERTa-based NLI model.
    Returns (model, tokenizer, entailment_idx).
    use_fast=False required for tokenizers 0.13.x compatibility.
    """
    logger.info(f"Loading NLI model: {model_name} on {device} ...")
    nli_tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=False
    )
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to(device)
    nli_model.eval()
    id2label = nli_model.config.id2label or {}
    entailment_idx = next(
        (k for k, v in id2label.items() if "entail" in v.lower()), 1
    )
    logger.info(f"  NLI loaded. id2label={id2label}, entailment_idx={entailment_idx}")
    return nli_model, nli_tokenizer, entailment_idx


@torch.no_grad()
def score_by_nli(
    nli_model,
    nli_tokenizer,
    explanations: List[str],
    correct_option_text: str,
    entailment_idx: int,
    device: str = "cpu",
) -> List[float]:
    """
    Compute NLI entailment probability for each explanation.
    Premise = explanation, Hypothesis = correct option text.
    Returns list of probabilities in [0, 1].
    """
    hypotheses = [correct_option_text] * len(explanations)
    enc = nli_tokenizer(
        explanations, hypotheses,
        padding=True, truncation=True, max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, entailment_idx].cpu().tolist()


# ---------------------------------------------------------------------------
# Verifier (Alpaca-7B) — used only when score_method == "hybrid" or "verifier"
# ---------------------------------------------------------------------------

def load_verifier(verifier_path: str, device: str):
    """Load Alpaca-7B verifier. use_fast=False avoids LlamaTokenizerFast recursion.
    Uses float32 on CPU (bfloat16/float16 matmul not supported on CPU)."""
    logger.info(f"Loading verifier from {verifier_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        verifier_path, trust_remote_code=True, use_fast=False
    )
    verifier_dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        verifier_path, torch_dtype=verifier_dtype, trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer


def get_verifier_score(ver_model, ver_tokenizer, question_input: str, explanation: str) -> float:
    """Score explanation with Alpaca-7B verifier. Returns float in [1, 5]."""
    prompt = VERIFIER_TEMPLATE.format(
        instruction=VERIFIER_INSTRUCTION,
        input=f"{question_input}\n\nExplanation: {explanation}",
    )
    inputs = ver_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(next(ver_model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = ver_model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=ver_tokenizer.eos_token_id,
        )
    response = ver_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    nums = re.findall(r"\d+(?:\.\d+)?", response)
    if nums:
        return min(max(float(nums[0]), 1.0), 5.0)
    return 3.0


# ---------------------------------------------------------------------------
# Generator: LLaMA-2
# ---------------------------------------------------------------------------

def load_llama2_generator(generator_path: str, lora_path: str, device: str):
    """Load LLaMA-2-13B + SFT LoRA adapter, merge, move to device."""
    logger.info(f"Loading LLaMA-2 generator from {generator_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        generator_path, trust_remote_code=True, use_fast=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        generator_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading SFT LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged.")
    model = model.to(device).eval()
    return model, tokenizer


def generate_llama2(
    model, tokenizer, instruction: str, input_text: str,
    num_samples: int, device: str, max_new_tokens: int = 300,
) -> List[str]:
    """Generate N explanations with LLaMA-2 using Alpaca template."""
    prompt = ALPACA_PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)

    temperatures = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    explanations = []
    with torch.no_grad():
        for i in range(num_samples):
            temp = temperatures[i % len(temperatures)]
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            explanations.append(text)
    return explanations


def make_llama2_dpo_prompt(tokenizer, instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)


# ---------------------------------------------------------------------------
# Generator: Qwen3-8B
# ---------------------------------------------------------------------------

def load_qwen3_generator(generator_path: str, lora_path: str, device: str):
    """Load Qwen3-8B + SFT LoRA adapter, merge, move to device."""
    logger.info(f"Loading Qwen3-8B generator from {generator_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        generator_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        generator_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading SFT LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged.")
    model = model.to(device).eval()
    return model, tokenizer


def generate_qwen3(
    model, tokenizer, instruction: str, input_text: str,
    num_samples: int, device: str, max_new_tokens: int = 300,
) -> List[str]:
    """Generate N explanations with Qwen3 chat template (enable_thinking=False)."""
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
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


def make_qwen3_dpo_prompt(tokenizer, instruction: str, input_text: str) -> str:
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Generator: Gemma 4 E4B-it (multimodal checkpoint; text tower only for SFT)
# ---------------------------------------------------------------------------

def load_gemma4_generator(generator_path: str, lora_path: str, device: str):
    """Load Gemma 4 E4B-it + SFT LoRA adapter, merge, move to device.

    The checkpoint ships as Gemma4ForConditionalGeneration (multimodal).
    Loading via AutoModelForCausalLM silently random-inits the text tower
    because text weights live under `model.language_model.*`. We must use
    Gemma4ForConditionalGeneration directly; the vision/audio towers sit
    idle during text-only candidate generation.
    """
    if Gemma4ForConditionalGeneration is None:
        raise RuntimeError(
            "transformers>=5.5 is required for --model_type gemma4; "
            "upgrade with `pip install -U transformers`."
        )
    logger.info(f"Loading Gemma 4 E4B-it generator from {generator_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        generator_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Gemma4ForConditionalGeneration.from_pretrained(
        generator_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading SFT LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA merged.")
    model = model.to(device).eval()
    return model, tokenizer


def generate_gemma4(
    model, tokenizer, instruction: str, input_text: str,
    num_samples: int, device: str, max_new_tokens: int = 300,
) -> List[str]:
    """Generate N explanations with Gemma 4 chat template (enable_thinking=False)."""
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
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


def make_gemma4_dpo_prompt(tokenizer, instruction: str, input_text: str) -> str:
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build NLI-guided DPO preference pairs.")
    parser.add_argument("--model_type", choices=["llama2", "llama3", "qwen3", "gemma4"], default="qwen3",
                        help="Generator model type")
    parser.add_argument("--generator_path", default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--lora_adapter_path", default="./rl_sft_qwen3_8b_generator")
    parser.add_argument("--data_path",
                        default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/"
                                "generator_merged_avg_3_lenexp_10.json")
    parser.add_argument("--output_path",
                        default="./rl_preference_data_nli_qwen3/preference_pairs.json")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Candidate explanations per question")
    parser.add_argument("--min_score_gap", type=float, default=0.05,
                        help="Minimum NLI gap between chosen and rejected to keep pair")
    parser.add_argument("--max_questions", type=int, default=1000, help="Limit questions to process")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing questions from")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--score_method",
                        choices=["nli", "hybrid", "verifier", "multiplicative_acr", "pareto_normalized"],
                        default="nli",
                        help="nli=NLI only | hybrid=0.5*NLI+0.5*ver | verifier=ver only | "
                             "multiplicative_acr=S_nli*S_ver_norm*I[acr>=thresh]-gamma*len | "
                             "pareto_normalized=within-question normalized NLI+ACR+Ver with Pareto filter")
    # Multiplicative-ACR specific hyperparameters
    parser.add_argument("--w_nli", type=float, default=0.7,
                        help="[multiplicative_acr] Weight for NLI in product score (default 0.7)")
    parser.add_argument("--w_ver", type=float, default=0.3,
                        help="[multiplicative_acr] Weight for verifier in product score (default 0.3)")
    parser.add_argument("--w_len_penalty", type=float, default=0.002,
                        help="[multiplicative_acr] gamma: length penalty per word (default 0.002)")
    parser.add_argument("--acr_threshold", type=float, default=0.5,
                        help="[multiplicative_acr] Min ACR to pass gate (default 0.5)")
    # pareto_normalized specific
    parser.add_argument("--w_acr", type=float, default=0.3,
                        help="[pareto_normalized] Weight for ACR in combined score (default 0.3)")
    parser.add_argument("--pareto_min_wins", type=int, default=2,
                        help="[pareto_normalized] Min metrics where chosen > rejected (default 2)")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli_device", default="cpu",
                        help="Device for NLI model (cpu recommended to save GPU VRAM)")
    parser.add_argument("--verifier_path",
                        default="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2",
                        help="Alpaca-7B verifier path (required for hybrid/verifier modes)")
    parser.add_argument("--verifier_device", default="cuda:1",
                        help="Device for verifier model (only used in hybrid/verifier modes)")
    parser.add_argument("--generator_device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    logger.info(f"Score method: {args.score_method}")

    # Load NLI model (needed for nli, hybrid, multiplicative_acr, pareto_normalized)
    nli_model = nli_tokenizer = entailment_idx = None
    if args.score_method in ("nli", "hybrid", "multiplicative_acr", "pareto_normalized"):
        nli_model, nli_tokenizer, entailment_idx = load_nli_model(
            args.nli_model, args.nli_device, args.cache_dir
        )

    # Load verifier (needed for hybrid, verifier, multiplicative_acr, pareto_normalized)
    ver_model = ver_tokenizer = None
    if args.score_method in ("hybrid", "verifier", "multiplicative_acr", "pareto_normalized"):
        ver_model, ver_tokenizer = load_verifier(args.verifier_path, args.verifier_device)

    # Load generator
    if args.model_type in ("llama2", "llama3"):
        gen_model, gen_tokenizer = load_llama2_generator(
            args.generator_path, args.lora_adapter_path, args.generator_device
        )
        generate_fn = generate_llama2
        make_prompt_fn = make_llama2_dpo_prompt
    elif args.model_type == "gemma4":
        gen_model, gen_tokenizer = load_gemma4_generator(
            args.generator_path, args.lora_adapter_path, args.generator_device
        )
        generate_fn = generate_gemma4
        make_prompt_fn = make_gemma4_dpo_prompt
    else:
        gen_model, gen_tokenizer = load_qwen3_generator(
            args.generator_path, args.lora_adapter_path, args.generator_device
        )
        generate_fn = generate_qwen3
        make_prompt_fn = make_qwen3_dpo_prompt

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    random.shuffle(raw_data)

    # Limit and slice questions
    questions = raw_data[args.start_index : args.start_index + args.max_questions]
    
    logger.info(f"Processing {len(questions)} questions (from index {args.start_index}) "
                f"× {args.num_samples} samples "
                f"(model_type={args.model_type}, min_gap={args.min_score_gap}) ...")

    pairs = []
    skipped_no_option = 0
    skipped_low_gap = 0

    for i, item in enumerate(questions):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue

        # Extract correct option text (needed for nli, hybrid, multiplicative_acr)
        _, correct_option_text = extract_correct_option_text(input_text)
        if args.score_method in ("nli", "hybrid", "multiplicative_acr", "pareto_normalized") and not correct_option_text:
            skipped_no_option += 1
            logger.debug(f"Q{i}: could not extract correct option text — skipped")
            continue

        try:
            explanations = generate_fn(
                gen_model, gen_tokenizer,
                instruction, input_text,
                args.num_samples, args.generator_device, args.max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Q{i}: generation error: {e}")
            continue

        # --- Compute scores per scoring method ---
        if args.score_method == "nli":
            raw_scores = score_by_nli(
                nli_model, nli_tokenizer,
                explanations, correct_option_text,
                entailment_idx, args.nli_device,
            )
            score_label = "nli"

        elif args.score_method == "verifier":
            raw_scores = [
                get_verifier_score(ver_model, ver_tokenizer, input_text, exp)
                for exp in explanations
            ]
            score_label = "verifier"

        elif args.score_method == "hybrid":
            # Additive: 0.5 * NLI + 0.5 * normalised verifier
            nli_s = score_by_nli(
                nli_model, nli_tokenizer,
                explanations, correct_option_text,
                entailment_idx, args.nli_device,
            )
            ver_s_raw = [
                get_verifier_score(ver_model, ver_tokenizer, input_text, exp)
                for exp in explanations
            ]
            ver_s = [(v - 1.0) / 4.0 for v in ver_s_raw]
            raw_scores = [0.5 * n + 0.5 * v for n, v in zip(nli_s, ver_s)]
            score_label = "hybrid"

        elif args.score_method == "multiplicative_acr":
            # R_i = (w_nli*S_nli * w_ver*S_ver_norm - gamma*len_norm(E_i))
            #        * I[acr(E_i) >= acr_threshold]
            nli_s = score_by_nli(
                nli_model, nli_tokenizer,
                explanations, correct_option_text,
                entailment_idx, args.nli_device,
            )
            ver_s_raw = [
                get_verifier_score(ver_model, ver_tokenizer, input_text, exp)
                for exp in explanations
            ]
            ver_s = [(v - 1.0) / 4.0 for v in ver_s_raw]   # normalise [1,5]→[0,1]
            raw_scores = []
            acr_flags = []
            for exp, ns, vs in zip(explanations, nli_s, ver_s):
                acr = answer_coverage_rate(exp, correct_option_text)
                acr_pass = acr >= args.acr_threshold
                acr_flags.append(acr_pass)
                len_norm = len(exp.split()) / 100.0   # ~1.0 for 100-word explanation
                product = args.w_nli * ns * args.w_ver * vs - args.w_len_penalty * len_norm
                raw_scores.append(product * (1.0 if acr_pass else 0.0))
            score_label = "multiplicative_acr"

        elif args.score_method == "pareto_normalized":
            # Within-question normalized multi-signal reward + Pareto filter
            # Step 1: compute all three signals
            nli_s = score_by_nli(
                nli_model, nli_tokenizer,
                explanations, correct_option_text,
                entailment_idx, args.nli_device,
            )
            ver_s_raw = [
                get_verifier_score(ver_model, ver_tokenizer, input_text, exp)
                for exp in explanations
            ]
            ver_s = [(v - 1.0) / 4.0 for v in ver_s_raw]
            acr_s = [answer_coverage_rate(exp, correct_option_text) for exp in explanations]

            # Step 2: within-question min-max normalization per signal
            def _norm(vals):
                lo, hi = min(vals), max(vals)
                if hi == lo:
                    return [0.5] * len(vals)
                return [(v - lo) / (hi - lo) for v in vals]

            nli_norm = _norm(nli_s)
            acr_norm = _norm(acr_s)
            ver_norm = _norm(ver_s)

            # Step 3: weighted linear combination (NLI dominant)
            w_nli_p = args.w_nli          # default 0.7 → used as NLI weight here too
            w_acr_p = args.w_acr          # default 0.3 → new arg
            w_ver_p = max(0.0, 1.0 - w_nli_p - w_acr_p)  # remainder to verifier
            raw_scores = [
                w_nli_p * n + w_acr_p * a + w_ver_p * v - args.w_len_penalty * (len(exp.split()) / 100.0)
                for n, a, v, exp in zip(nli_norm, acr_norm, ver_norm, explanations)
            ]
            score_label = "pareto_normalized"

        # --- Pair selection ---
        if args.score_method == "pareto_normalized":
            # Pareto-dominant pair selection: find best pair where chosen wins on
            # at least pareto_min_wins metrics vs rejected
            best_pair = None
            best_gap = -1.0
            for ci in range(len(explanations)):
                for ri in range(len(explanations)):
                    if ci == ri:
                        continue
                    # Count wins
                    wins = sum([
                        nli_s[ci] > nli_s[ri],
                        acr_s[ci] > acr_s[ri],
                        ver_s[ci] > ver_s[ri],
                    ])
                    combined_gap = raw_scores[ci] - raw_scores[ri]
                    if wins >= args.pareto_min_wins and combined_gap > best_gap:
                        best_gap = combined_gap
                        best_pair = (ci, ri)

            if best_pair is None or best_gap < args.min_score_gap:
                skipped_low_gap += 1
                logger.debug(f"Q{i}: pareto filter — no valid pair (best_gap={best_gap:.3f})")
                continue
            best_idx, worst_idx = best_pair
            gap = best_gap
        else:
            best_idx = max(range(len(raw_scores)), key=lambda x: raw_scores[x])
            worst_idx = min(range(len(raw_scores)), key=lambda x: raw_scores[x])
            gap = raw_scores[best_idx] - raw_scores[worst_idx]

            if best_idx == worst_idx or gap < args.min_score_gap:
                skipped_low_gap += 1
                logger.debug(f"Q{i}: {score_label} gap={gap:.3f} < {args.min_score_gap} → skipped")
                continue

        # Build DPO prompt (model-specific)
        prompt_text = make_prompt_fn(gen_tokenizer, instruction, input_text)

        pair = {
            "prompt": prompt_text,
            "chosen": explanations[best_idx],
            "rejected": explanations[worst_idx],
            f"chosen_{score_label}": round(raw_scores[best_idx], 4),
            f"rejected_{score_label}": round(raw_scores[worst_idx], 4),
            f"{score_label}_gap": round(gap, 4),
            "question_input": input_text,
        }
        if correct_option_text:
            pair["correct_option_text"] = correct_option_text
        # Store per-component scores for hybrid, multiplicative_acr, pareto_normalized
        if args.score_method in ("hybrid", "multiplicative_acr", "pareto_normalized"):
            pair["chosen_nli"] = round(nli_s[best_idx], 4)
            pair["rejected_nli"] = round(nli_s[worst_idx], 4)
            pair["chosen_verifier"] = round(ver_s_raw[best_idx], 4)
            pair["rejected_verifier"] = round(ver_s_raw[worst_idx], 4)
        if args.score_method == "pareto_normalized":
            pair["chosen_acr"] = round(acr_s[best_idx], 4)
            pair["rejected_acr"] = round(acr_s[worst_idx], 4)
            pair["pareto_wins"] = sum([
                nli_s[best_idx] > nli_s[worst_idx],
                acr_s[best_idx] > acr_s[worst_idx],
                ver_s[best_idx] > ver_s[worst_idx],
            ])
        if args.score_method == "multiplicative_acr":
            pair["chosen_acr_pass"] = acr_flags[best_idx]
            pair["rejected_acr_pass"] = acr_flags[worst_idx]

        pairs.append(pair)

        if (i + 1) % 50 == 0:
            logger.info(f"Progress {i+1}/{len(questions)} | Pairs: {len(pairs)} "
                        f"| Skipped (no opt): {skipped_no_option} "
                        f"| Skipped (low gap): {skipped_low_gap}")
            # Periodic save
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            logger.info(f"  Periodic save: {len(pairs)} pairs saved to {args.output_path}")

    # Save
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Done! {len(pairs)} preference pairs saved to {args.output_path}")
    logger.info(f"  Questions processed : {len(questions)}")
    logger.info(f"  Skipped (no option) : {skipped_no_option}")
    logger.info(f"  Skipped (low gap)   : {skipped_low_gap}")
    if pairs:
        gap_key = f"{args.score_method}_gap"
        if gap_key in pairs[0]:
            avg_gap = sum(p[gap_key] for p in pairs) / len(pairs)
            logger.info(f"  Avg {args.score_method} gap       : {avg_gap:.4f}")
        if "chosen_nli" in pairs[0]:
            avg_c = sum(p["chosen_nli"] for p in pairs) / len(pairs)
            avg_r = sum(p["rejected_nli"] for p in pairs) / len(pairs)
            logger.info(f"  Avg chosen NLI      : {avg_c:.4f}")
            logger.info(f"  Avg rejected NLI    : {avg_r:.4f}")
        if "chosen_verifier" in pairs[0]:
            avg_cv = sum(p["chosen_verifier"] for p in pairs) / len(pairs)
            avg_rv = sum(p["rejected_verifier"] for p in pairs) / len(pairs)
            logger.info(f"  Avg chosen verifier : {avg_cv:.4f}")
            logger.info(f"  Avg rejected verif. : {avg_rv:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

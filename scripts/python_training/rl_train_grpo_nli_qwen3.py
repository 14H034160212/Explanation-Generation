"""
Two-stage NLI-GRPO for Qwen3-8B.

Initialises from a Hybrid-DPO LoRA adapter (or SFT) and fine-tunes with
GRPO using DeBERTa NLI entailment probability as the sole reward signal.
This is the TRL 0.29 replacement for the old PPO-based NLI pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u scripts/python_training/rl_train_grpo_nli_qwen3.py \
        --init_adapter_path rl_dpo_multiplicative_qwen3_8b_generator \
        --output_dir rl_grpo_nli_twostage_qwen3_8b_generator \
        --data_path "preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
        --max_questions 500 --max_steps 125 \
        --num_generations 4 --per_device_train_batch_size 2
"""
import sys
import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    base_model_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    init_adapter_path: Optional[str] = field(
        default="./rl_dpo_multiplicative_qwen3_8b_generator",
        metadata={"help": "DPO or SFT LoRA adapter to initialise the policy. "
                          "Leave None to start from bare base model."},
    )
    data_path: str = field(
        default="./preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
    )
    max_questions: int = field(default=500)
    nli_model: str = field(default="cross-encoder/nli-deberta-v3-small")
    nli_device: str = field(default="cpu")
    cache_dir: Optional[str] = field(default="cache")
    # LoRA config for new adapter on top of merged init
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def build_dataset(data_path: str, max_questions: Optional[int] = None):
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    if max_questions:
        raw = raw[:max_questions]
    rows = []
    for item in raw:
        prompt = ALPACA_PROMPT.format(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
        )
        correct_opt = item.get("output", {})
        if isinstance(correct_opt, dict):
            correct_opt = correct_opt.get("correct_option_text", "")
        rows.append({"prompt": prompt, "correct_option": str(correct_opt)})
    return rows


# ---------------------------------------------------------------------------
# NLI reward function
# ---------------------------------------------------------------------------
def make_nli_reward_fn(nli_model_name: str, device: str, cache_dir: Optional[str]):
    """Returns a reward function compatible with GRPOTrainer."""
    from transformers import AutoTokenizer as ATok, AutoModelForSequenceClassification as AMSC

    logger.info(f"Loading NLI model {nli_model_name} on {device} ...")
    nli_tok = ATok.from_pretrained(
        nli_model_name, use_fast=False,
        cache_dir=cache_dir,
    )
    nli_model = AMSC.from_pretrained(
        nli_model_name,
        cache_dir=cache_dir,
    ).to(device)
    nli_model.eval()

    # Label order: contradiction=0, entailment=1, neutral=2 (deberta-v3-small)
    try:
        id2label = nli_model.config.id2label
        entailment_id = next(
            k for k, v in id2label.items() if "entail" in v.lower()
        )
    except Exception:
        entailment_id = 1  # default

    logger.info(f"NLI entailment label id: {entailment_id}")

    def reward_fn(completions: List[str], correct_option: List[str], **kwargs) -> List[float]:
        rewards = []
        for comp, opt in zip(completions, correct_option):
            # Strip <think>...</think> if present
            comp_clean = re.sub(r"<think>.*?</think>", "", comp, flags=re.DOTALL).strip()
            if not comp_clean or not opt:
                rewards.append(0.0)
                continue
            try:
                enc = nli_tok(
                    comp_clean, opt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.no_grad():
                    logits = nli_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                rewards.append(float(probs[0, entailment_id].item()))
            except Exception as e:
                logger.warning(f"NLI reward error: {e}")
                rewards.append(0.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, grpo_config = parser.parse_args_into_dataclasses()

    logger.info(f"Base model: {script_args.base_model_path}")
    logger.info(f"Init adapter: {script_args.init_adapter_path}")
    logger.info(f"Output: {grpo_config.output_dir}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model_path,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=script_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Base model + merge init adapter ────────────────────────────────────
    logger.info("Loading base model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=script_args.cache_dir,
    )

    if script_args.init_adapter_path and os.path.exists(script_args.init_adapter_path):
        logger.info(f"Merging init adapter from {script_args.init_adapter_path} ...")
        base_model = PeftModel.from_pretrained(base_model, script_args.init_adapter_path)
        base_model = base_model.merge_and_unload()
        logger.info("Adapter merged.")

    # ── New LoRA on top of merged model ────────────────────────────────────
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    logger.info("Building dataset ...")
    rows = build_dataset(script_args.data_path, script_args.max_questions)
    from datasets import Dataset
    dataset = Dataset.from_list(rows)
    logger.info(f"Dataset size: {len(dataset)}")

    # ── NLI reward ─────────────────────────────────────────────────────────
    reward_fn = make_nli_reward_fn(
        script_args.nli_model,
        script_args.nli_device,
        script_args.cache_dir,
    )

    # ── GRPO Trainer ───────────────────────────────────────────────────────
    os.makedirs(grpo_config.output_dir, exist_ok=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO-NLI training ...")
    trainer.train()

    logger.info(f"Saving model to {grpo_config.output_dir} ...")
    trainer.save_model(grpo_config.output_dir)
    tokenizer.save_pretrained(grpo_config.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()

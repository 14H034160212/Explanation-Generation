"""
RL-ILearner Step 3 (Recommended): DPO Fine-tuning with LoRA

Direct Preference Optimization (DPO) trains the model to prefer high-quality
explanations (chosen) over low-quality ones (rejected) — without needing a
separate reward model during training.

Advantages over PPO:
  - ~50% lower VRAM: No Critic/Value model needed, only Actor + Reference.
  - Training stability: No reward hacking, KL divergence implicitly controlled.
  - With synthetic hard negatives: Often beats PPO in explanation quality.

Usage (8x A100, recommended DPO path):
    deepspeed --num_gpus 8 rl_train_dpo.py \
        --model_name_or_path Qwen/Qwen3-14B-Instruct \
        --sft_adapter_path ./rl_sft_qwen3_14b_generator \
        --preference_data_path ./rl_preference_data/preference_pairs.json \
        --output_dir ./rl_dpo_qwen3_14b_generator \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --beta 0.1 \
        --deepspeed ./rl_configs/ds_zero3.json
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert educator specializing in generating high-quality explanations "
    "for exam questions. Your explanations should be clear, accurate, educational, "
    "and written in a style similar to how knowledgeable students explain answers."
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-14B-Instruct",
        metadata={"help": "Base model HF ID or local path."},
    )
    sft_adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to SFT LoRA adapter (from rl_train_sft.py). "
                "If provided, will be loaded and merged into the base model "
                "before DPO training begins."
            )
        },
    )
    use_4bit: bool = field(default=False, metadata={"help": "Use 4-bit QLoRA."})
    use_8bit: bool = field(default=False, metadata={"help": "Use 8-bit quantization."})
    cache_dir: Optional[str] = field(default="cache")
    use_flash_attention: bool = field(default=True)


@dataclass
class DataArguments:
    preference_data_path: str = field(
        default="./rl_preference_data/preference_pairs.json",
        metadata={"help": "Preference pairs JSON (fields: prompt, chosen, rejected)."},
    )
    eval_data_path: Optional[str] = field(default=None)
    max_length: int = field(
        default=1024,
        metadata={"help": "Max total sequence length (prompt + response)."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Max prompt length. Prompts longer than this are truncated."},
    )
    min_score_gap_filter: float = field(
        default=0.0,
        metadata={"help": "Skip preference pairs with score_gap below this threshold."},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated LoRA target modules."},
    )


def load_preference_dataset(data_path: str, min_score_gap: float = 0.0) -> Dataset:
    """
    Load preference pairs and format them as chat conversations.

    Each item becomes:
      prompt   -> [{"role": "system", ...}, {"role": "user", ...}]
      chosen   -> [{"role": "assistant", "content": chosen_explanation}]
      rejected -> [{"role": "assistant", "content": rejected_explanation}]

    This conversational format is what TRL's DPOTrainer expects for modern chat models.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    skipped = 0
    for item in raw_data:
        prompt_text = item.get("prompt", "").strip()
        chosen_text = item.get("chosen", "").strip()
        rejected_text = item.get("rejected", "").strip()
        score_gap = item.get("score_gap", 1.0)

        if not prompt_text or not chosen_text or not rejected_text:
            skipped += 1
            continue
        if score_gap < min_score_gap:
            skipped += 1
            continue

        # Conversational format expected by TRL DPOTrainer
        formatted.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
        })

    logger.info(
        f"Loaded {len(formatted)} preference pairs from {data_path} "
        f"(skipped {skipped} invalid/low-gap pairs)."
    )
    return Dataset.from_list(formatted)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, DPOConfig)
    )
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # ---------- Quantization ----------
    bnb_config = None
    if model_args.use_4bit:
        logger.info("Using 4-bit QLoRA quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        logger.info("Using 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",  # DPO prefers left-padding for decoder-only models
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Model ----------
    attn_impl = "flash_attention_2" if model_args.use_flash_attention else "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if bnb_config is not None else None,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception:
        logger.warning("flash_attention_2 unavailable, falling back to eager.")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if bnb_config is not None else None,
            trust_remote_code=True,
        )

    # ---------- Merge SFT LoRA adapter (if provided) ----------
    if model_args.sft_adapter_path and os.path.isdir(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA adapter from {model_args.sft_adapter_path} ...")
        model = PeftModel.from_pretrained(model, model_args.sft_adapter_path)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged into base model.")

    # ---------- LoRA config for DPO ----------
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(f"DPO LoRA: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")

    # ---------- DPO Config ----------
    # The beta parameter controls how much the model diverges from the reference.
    # Lower beta (e.g., 0.05) = more aggressive learning; higher (0.5) = conservative.
    training_args.max_length = data_args.max_length
    training_args.max_prompt_length = data_args.max_prompt_length

    # ---------- Dataset ----------
    train_dataset = load_preference_dataset(
        data_args.preference_data_path,
        min_score_gap=data_args.min_score_gap_filter,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = load_preference_dataset(data_args.eval_data_path)

    # ---------- DPO Trainer ----------
    # The reference model is automatically created as a frozen copy of the
    # initial model (before LoRA is applied). This ensures KL divergence
    # is measured against the SFT checkpoint, preventing reward hacking.
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting DPO training...")
    logger.info(f"  Beta (KL penalty): {training_args.beta}")
    logger.info(f"  Training examples: {len(train_dataset)}")
    logger.info(f"  Max length: {data_args.max_length}")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"DPO LoRA adapter saved to {training_args.output_dir}")

    # Save training metrics
    metrics_path = os.path.join(training_args.output_dir, "dpo_train_metrics.json")
    if hasattr(trainer, "state") and trainer.state.log_history:
        with open(metrics_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

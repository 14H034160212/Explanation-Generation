"""
Qwen3-8B KTO Training
基于 TRL 0.29.0
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    sft_adapter_path: Optional[str] = field(default="models/rl_sft_qwen3_8b_generator")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    kto_data_path: str = field(default="./rl_preference_data_verifier_qwen3/kto_data.json")


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


def load_kto_data(data_path: str) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} KTO records from {data_path}")
    return Dataset.from_list(data)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, KTOConfig, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if model_args.sft_adapter_path and os.path.exists(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA from {model_args.sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, model_args.sft_adapter_path)
        model = model.merge_and_unload()

    model.config.use_cache = False

    # ── Apply LoRA ─────────────────────────────────────────────
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # ── Dataset ────────────────────────────────────────────────
    train_dataset = load_kto_data(data_args.kto_data_path)

    # ── KTO Training ───────────────────────────────────────────
    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting Qwen3-8B KTO training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"KTO LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()

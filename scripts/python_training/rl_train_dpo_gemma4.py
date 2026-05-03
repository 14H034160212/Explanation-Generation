"""
RLearner-LLM Step 3 (Gemma 4 E4B-it): Hybrid-DPO Training with LoRA
TRL 0.28 + Transformers 5.5.4 (gemma4-rl conda env)

Gemma-4 adaptations vs rl_train_dpo_qwen3.py:
  * Uses Gemma4ForConditionalGeneration (multimodal checkpoint; text-only SFT
    loads full model, LoRA restricted to language tower).
  * LoRA targets are a FULL-PATH REGEX so vision/audio towers (which wrap
    q/k/v/o_proj inside Gemma4ClippableLinear that PEFT cannot wrap) are
    excluded. Gemma 4 has no in-attention QK-norm, so q/k_proj are safe
    targets inside the text tower.
  * The SFT adapter (a `PeftModel` saved by rl_train_sft_gemma4.py) is
    merged back via merge_and_unload() before the DPO LoRA is applied.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n gemma4-rl python3 rl_train_dpo_gemma4.py \\
        --model_name_or_path google/gemma-4-E4B-it \\
        --sft_adapter_path ./rl_sft_gemma4_e4b_cardiff_generator \\
        --preference_data_path ./rl_preference_data_gemma4_cardiff/preference_pairs.json \\
        --output_dir ./rl_dpo_gemma4_e4b_cardiff_generator \\
        --num_train_epochs 5 \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 8 \\
        --bf16 True --report_to none > rl_dpo_gemma4_training.log 2>&1
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
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="google/gemma-4-E4B-it")
    sft_adapter_path: Optional[str] = field(default="./rl_sft_gemma4_e4b_cardiff_generator")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    preference_data_path: str = field(
        default="./rl_preference_data_gemma4_cardiff/preference_pairs.json"
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default=r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    )


def load_preference_data(data_path: str) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} DPO preference pairs from {data_path}")
    return Dataset.from_list(pairs)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DPOConfig, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {model_args.model_name_or_path}")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if model_args.sft_adapter_path and os.path.exists(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA from {model_args.sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, model_args.sft_adapter_path)
        model = model.merge_and_unload()
        logger.info("SFT LoRA merged into base model.")

    model.config.use_cache = False

    tm_raw = lora_args.lora_target_modules
    if any(c in tm_raw for c in r"\^$()|"):
        target_modules = tm_raw
    else:
        target_modules = [m.strip() for m in tm_raw.split(",")]
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
    model.print_trainable_parameters()

    train_dataset = load_preference_data(data_args.preference_data_path)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting Gemma 4 E4B-it DPO training...")
    logger.info(f"  Preference pairs: {len(train_dataset)}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Beta: {training_args.beta}")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"DPO LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()

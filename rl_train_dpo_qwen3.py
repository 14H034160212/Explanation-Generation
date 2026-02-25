"""
RL-ILearner Step 3 (Qwen3-8B): DPO Training with LoRA
TRL 0.28 + Transformers 5.2 (qwen3-rl conda env)

Usage (single GPU 7):
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_dpo_qwen3.py \\
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \\
        --sft_adapter_path ./rl_sft_qwen3_8b_generator \\
        --preference_data_path ./rl_preference_data_qwen3/preference_pairs.json \\
        --output_dir ./rl_dpo_qwen3_8b_generator \\
        --num_train_epochs 5 \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 8 \\
        --bf16 True \\
        --report_to none > rl_dpo_qwen3_training.log 2>&1
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# -- Monkey Patch for TRL 0.8.6 + Transformers 5.2+ --
import transformers
original_trainer_init = transformers.Trainer.__init__
def patched_trainer_init(self, *args, **kwargs):
    if "tokenizer" in kwargs and "processing_class" not in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    elif "tokenizer" in kwargs and "processing_class" in kwargs:
        kwargs.pop("tokenizer")
    original_trainer_init(self, *args, **kwargs)
transformers.Trainer.__init__ = patched_trainer_init

from trl import DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    sft_adapter_path: Optional[str] = field(default="./rl_sft_qwen3_8b_generator")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    preference_data_path: str = field(default="./rl_preference_data_qwen3/preference_pairs.json")


@dataclass
class DPOArguments:
    beta: float = field(default=0.1)
    max_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        # Qwen3 applies q_norm/k_norm after q_proj/k_proj reshape → exclude from LoRA
        default="v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


def load_preference_data(data_path: str, tokenizer, max_length: int, max_prompt_length: int) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} DPO preference pairs from {data_path}")

    def tokenize_function(example):
        # TRL 0.8.6 DPODataCollatorWithPadding expects:
        # chosen_input_ids, rejected_input_ids, prompt_input_ids
        # And their corresponding attention masks.
        
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Tokenize prompt
        prompt_tokens = tokenizer(
            prompt, truncation=True, max_length=max_prompt_length, add_special_tokens=True
        )
        
        # Tokenize chosen (full sequence)
        chosen_tokens = tokenizer(
            prompt + chosen, truncation=True, max_length=max_length, add_special_tokens=True
        )
        
        # Tokenize rejected (full sequence)
        rejected_tokens = tokenizer(
            prompt + rejected, truncation=True, max_length=max_length, add_special_tokens=True
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "chosen_labels": chosen_tokens["input_ids"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
            "rejected_labels": rejected_tokens["input_ids"],
        }

    dataset = Dataset.from_list(pairs)
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=[c for c in dataset.column_names if c not in ["prompt","chosen","rejected"]])
    return tokenized_dataset


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DPOArguments, LoraArguments, TrainingArguments)
    )
    model_args, data_args, dpo_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (merge SFT LoRA first) ───────────────────────────────────────────
    logger.info(f"Loading base model: {model_args.model_name_or_path}")
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
        logger.info("SFT LoRA merged into base model.")

    model.config.use_cache = False

    # ── Apply DPO LoRA ──────────────────────────────────────────────────────────
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
    model.print_trainable_parameters()

    # ── Dataset ─────────────────────────────────────────────────────────────────
    train_dataset = load_preference_data(
        data_args.preference_data_path, 
        tokenizer, 
        dpo_args.max_length, 
        dpo_args.max_prompt_length
    )

    # ── DPO Training (TRL 0.8.6 API) ─────────────────────────────────────────────
    # Manual collation to fix TRL 0.8.6 + Transformers 5.2 collision
    from trl.trainer.utils import DPODataCollatorWithPadding
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=dpo_args.beta,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_length=dpo_args.max_length,
        max_prompt_length=dpo_args.max_prompt_length,
    )

    logger.info("Starting Qwen3-8B DPO training...")
    logger.info(f"  Preference pairs: {len(train_dataset)}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Beta: {dpo_args.beta}")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"DPO LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()

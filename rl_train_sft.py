"""
RL-ILearner Step 1: SFT Fine-tuning with LoRA

Fine-tunes modern LLMs (Qwen3.5 / Qwen3 / Llama-3) on PeerWise explanation data
using LoRA (PEFT) for parameter-efficient training.

Supports 8x A100 via DeepSpeed ZeRO-3. Replaces the legacy full-parameter
fine-tuning in train.py with a LoRA-based approach that allows hot-swap deployment.

Usage (8x A100, Qwen3-14B):
    deepspeed --num_gpus 8 rl_train_sft.py \
        --model_name_or_path Qwen/Qwen3-14B-Instruct \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_dir ./rl_sft_qwen3_14b_generator \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --deepspeed ./rl_configs/ds_zero3.json
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

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
        metadata={
            "help": (
                "HuggingFace model ID or local path. "
                "Supported: Qwen/Qwen3.5-14B-Instruct, Qwen/Qwen3-14B-Instruct, "
                "Qwen/Qwen3-32B-Instruct, meta-llama/Llama-3.3-70B-Instruct, "
                "meta-llama/Llama-3.1-8B-Instruct"
            )
        },
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization (QLoRA). Recommended for 70B+ models."},
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization. Alternative to 4-bit."},
    )
    cache_dir: Optional[str] = field(
        default="cache",
        metadata={"help": "Directory to cache downloaded models."},
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 for faster training (requires flash-attn)."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json",
        metadata={"help": "Path to training data JSON (fields: instruction, input, output)."},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation data path."},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum tokenized sequence length. Sequences will be truncated."},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank (higher = more capacity, more VRAM)."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA scaling factor (usually 2x rank)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout probability for LoRA layers."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated LoRA target modules. Works for Qwen3/Llama-3."},
    )


def load_peerwise_data(data_path: str) -> Dataset:
    """Load PeerWise JSON data and convert to chat-format for modern models."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    skipped = 0
    for item in raw_data:
        instruction = item.get("instruction", "").replace("</s>", "").strip()
        input_text = item.get("input", "").replace("</s>", "").strip()
        output_text = item.get("output", "").replace("</s>", "").strip()

        if not input_text or not output_text:
            skipped += 1
            continue

        # Build user message: merge instruction + question/options input
        user_content = f"{instruction}\n\n{input_text}" if instruction else input_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]
        formatted.append({"messages": messages})

    logger.info(f"Loaded {len(formatted)} examples from {data_path} (skipped {skipped} empty).")
    return Dataset.from_list(formatted)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, SFTConfig)
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
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token.")

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
        logger.warning("flash_attention_2 unavailable, falling back to eager attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if bnb_config is not None else None,
            trust_remote_code=True,
        )

    # ---------- LoRA ----------
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, "
                f"target_modules={target_modules}")

    # ---------- Dataset ----------
    train_dataset = load_peerwise_data(data_args.data_path)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = load_peerwise_data(data_args.eval_data_path)

    # ---------- SFT Training ----------
    training_args.max_seq_length = data_args.max_seq_length

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()

"""
RLearner-LLM Step 1 (Qwen3-8B): SFT Fine-tuning with LoRA
TRL 0.28 + Transformers 5.2 (qwen3-rl conda env)

Fine-tunes Qwen3-8B on PeerWise explanation data using LoRA (PEFT).
Uses Qwen3 chat template with enable_thinking=False (no <think> blocks).

Usage (single GPU 7, 81GB free):
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_sft_qwen3.py \\
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \\
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \\
        --output_dir ./rl_sft_qwen3_8b_generator \\
        --num_train_epochs 3 \\
        --per_device_train_batch_size 2 \\
        --gradient_accumulation_steps 8 \\
        --learning_rate 2e-4 \\
        --bf16 True \\
        --report_to none > rl_sft_qwen3_training.log 2>&1

Usage (2 GPUs, e.g. GPU 6+7):
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port 29501 \\
        rl_train_sft_qwen3.py [same args]
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    data_path: str = field(
        default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
    )
    eval_data_path: Optional[str] = field(default=None)
    max_seq_length: int = field(default=1024)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        # Qwen3 applies q_norm/k_norm after q_proj/k_proj reshape → LoRA on those
        # causes CUBLAS_STATUS_INVALID_VALUE. Exclude q_proj and k_proj from targets.
        default="v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


def format_qwen3_example(tokenizer, instruction: str, input_text: str, output_text: str) -> str:
    """Format training example using Qwen3 chat template (enable_thinking=False)."""
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": output_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def load_peerwise_data(data_path: str, tokenizer) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    examples, skipped = [], 0
    for item in raw_data:
        instruction = item.get("instruction", "").replace("</s>", "").strip()
        input_text = item.get("input", "").replace("</s>", "").strip()
        output_text = item.get("output", "").replace("</s>", "").strip()
        if not input_text or not output_text:
            skipped += 1
            continue
        text = format_qwen3_example(tokenizer, instruction, input_text, output_text)
        examples.append({"text": text})

    logger.info(f"Loaded {len(examples)} examples (skipped {skipped} empty).")
    return Dataset.from_list(examples)


def main():
    # SFTConfig handles both TrainingArguments and SFT-specific args in TRL 0.28
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, SFTConfig))
    model_args, data_args, lora_args, sft_config = parser.parse_args_into_dataclasses()

    # Override SFT-specific fields from DataArguments
    sft_config.dataset_text_field = "text"
    sft_config.max_seq_length = data_args.max_seq_length

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ───────────────────────────────────────────────────────────────────
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

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_dataset = load_peerwise_data(data_args.data_path, tokenizer)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = load_peerwise_data(data_args.eval_data_path, tokenizer)

    # ── SFT Training (TRL 0.28 API) ────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting Qwen3-8B SFT training...")
    logger.info(f"  Model: {model_args.model_name_or_path}")
    logger.info(f"  Training examples: {len(train_dataset)}")
    logger.info(f"  Epochs: {sft_config.num_train_epochs}")
    logger.info(f"  Per-device batch size: {sft_config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {sft_config.gradient_accumulation_steps}")

    trainer.train()
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    logger.info(f"LoRA adapter saved to {sft_config.output_dir}")


if __name__ == "__main__":
    main()

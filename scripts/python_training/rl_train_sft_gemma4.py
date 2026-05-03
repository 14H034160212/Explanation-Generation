"""
RLearner-LLM Step 1 (Gemma 4 E4B-it): SFT Fine-tuning with LoRA
TRL 0.28 + Transformers 5.2 (shared env with qwen3-rl; Gemma4 support lands in
transformers >= 5.2).

Fine-tunes google/gemma-4-E4B-it (8B total / ~4.5B effective params, dense,
PLE-equipped) on PeerWise explanation data using LoRA (PEFT). Small-scale
starter configuration: Cardiff Biology domain only (~5.5K training examples).

Usage (single GPU, 1x A100 80GB):
    CUDA_VISIBLE_DEVICES=0 conda run -n qwen3-rl python3 rl_train_sft_gemma4.py \\
        --model_name_or_path google/gemma-4-E4B-it \\
        --data_path ./preference_data/Paul_new_data/Cardiff_all_generator_train_avg_3_lenexp_10.json \\
        --output_dir ./rl_sft_gemma4_e4b_cardiff_generator \\
        --num_train_epochs 3 \\
        --per_device_train_batch_size 2 \\
        --gradient_accumulation_steps 8 \\
        --learning_rate 2e-4 \\
        --bf16 True \\
        --report_to none > rl_sft_gemma4_training.log 2>&1

Notes for Gemma 4:
  * Chat template uses standard system/user/assistant roles. Thinking mode is
    triggered by placing <|think|> at the start of the system prompt; we keep
    enable_thinking=False to match the Qwen3 baseline's no-<think> setup.
  * Attention is hybrid (interleaved local sliding window + global layers). No
    QK-norm in-attention, so LoRA on q_proj/k_proj is safe unlike Qwen3.
  * Gemma 4 is multimodal; AutoModelForCausalLM loads only the text tower for
    text-only SFT. If the auto-class resolution fails on a future release,
    swap to `Gemma4ForCausalLM` explicitly.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="google/gemma-4-E4B-it")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    data_path: str = field(
        default="./preference_data/Paul_new_data/Cardiff_all_generator_train_avg_3_lenexp_10.json"
    )
    eval_data_path: Optional[str] = field(default=None)
    max_seq_length: int = field(default=1024)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    # Regex over FULL dotted module names. Restricts LoRA to plain nn.Linear
    # projections inside the text tower (model.language_model.layers.*). Vision
    # and audio towers wrap their q/k/v/o_proj inside a Gemma4ClippableLinear
    # class that PEFT cannot LoRA-wrap, so we must exclude them by path.
    lora_target_modules: str = field(
        default=r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    )


def format_gemma4_example(tokenizer, instruction: str, input_text: str, output_text: str) -> str:
    """Format training example using Gemma 4 chat template (enable_thinking=False)."""
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
        text = format_gemma4_example(tokenizer, instruction, input_text, output_text)
        examples.append({"text": text})

    logger.info(f"Loaded {len(examples)} examples (skipped {skipped} empty).")
    return Dataset.from_list(examples)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, SFTConfig))
    model_args, data_args, lora_args, sft_config = parser.parse_args_into_dataclasses()

    sft_config.dataset_text_field = "text"
    sft_config.max_seq_length = data_args.max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma 4 E4B-it ships as a multimodal Gemma4ForConditionalGeneration
    # checkpoint with text weights under `model.language_model.*`. Loading
    # with Gemma4ForCausalLM silently random-initialises these layers (the
    # prefixes don't match), so we must load the full multimodal class and
    # restrict LoRA to the language-tower Linear modules. Vision/audio
    # towers are held in memory but not activated during text-only SFT.
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("Loaded Gemma4ForConditionalGeneration (multimodal; LoRA will target text tower only).")
    model.config.use_cache = False

    # If the target spec contains regex metachars we pass it as a single
    # string (PEFT interprets this as a full-path regex via re.fullmatch);
    # otherwise split on comma for leaf-name matching (Qwen3-style).
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

    train_dataset = load_peerwise_data(data_args.data_path, tokenizer)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = load_peerwise_data(data_args.eval_data_path, tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting Gemma 4 E4B-it SFT training...")
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

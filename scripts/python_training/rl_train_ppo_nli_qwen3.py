"""
RLearner-LLM PPO with NLI Reward (Qwen3-8B)
TRL 0.28 + Transformers 5.2 (qwen3-rl conda env)

Replaces verifier reward with NLI entailment probability:
  reward = P(entailment | premise=explanation, hypothesis=correct_option_text)

Two-stage pipeline: initialise from NLI-DPO adapter via --sft_adapter_path
to combine offline DPO alignment with online NLI-guided RL.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=6 conda run -n qwen3-rl python3 rl_train_ppo_nli_qwen3.py \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_dpo_qwen3_nli_generator \
        --output_dir ./rl_ppo_nli_qwen3_generator \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --batch_size 4 --mini_batch_size 1 --ppo_epochs 4 \
        --learning_rate 1e-5 --max_questions 500 \
        --num_epochs 1 --max_global_steps 500 --save_every_steps 100
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    sft_adapter_path: Optional[str] = field(
        default="./rl_sft_qwen3_8b_generator",
        metadata={"help": "SFT or DPO-NLI LoRA adapter to initialise the Actor."},
    )
    output_dir: str = field(default="./rl_ppo_nli_qwen3_generator")
    nli_model: str = field(default="cross-encoder/nli-deberta-v3-small")
    nli_device: str = field(default="cpu")
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    data_path: str = field(
        default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
    )
    max_questions: Optional[int] = field(default=500)
    num_epochs: int = field(default=1)
    max_global_steps: Optional[int] = field(default=500)
    save_every_steps: int = field(default=100)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    # Exclude q_proj/k_proj for Qwen3 (q_norm/k_norm CUBLAS bug)
    lora_target_modules: str = field(
        default="v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def extract_correct_option_text(input_text: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    return letter, opt_m.group(1).strip() if opt_m else None


# ── NLI Reward Model ──────────────────────────────────────────────────────────

class NLIRewardModel:
    """NLI entailment probability as PPO reward (runs on CPU)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small",
                 device: str = "cpu", cache_dir: str = "cache"):
        logger.info(f"Loading NLI reward model: {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        self.model.eval()
        self.device = device
        id2label = self.model.config.id2label
        self.entailment_idx = next(
            k for k, v in id2label.items() if v.lower() == "entailment"
        )
        logger.info(f"NLI reward loaded. entailment_idx={self.entailment_idx}")

    @torch.no_grad()
    def get_rewards(
        self, correct_options: List[str], explanations: List[str]
    ) -> List[torch.Tensor]:
        enc = self.tokenizer(
            explanations, correct_options,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        entailment_probs = probs[:, self.entailment_idx]
        return [torch.tensor(p.item(), dtype=torch.float32) for p in entailment_probs]


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_ppo_dataset(
    data_path: str, tokenizer, max_questions: Optional[int] = None
) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if max_questions:
        raw_data = raw_data[:max_questions]

    examples = []
    skipped = 0
    for item in raw_data:
        instruction = item.get("instruction", "").replace("</s>", "").strip()
        input_text = item.get("input", "").replace("</s>", "").strip()
        if not input_text:
            continue

        _, correct_opt = extract_correct_option_text(input_text)
        if not correct_opt:
            skipped += 1
            continue

        prompt = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        ids = [int(x) for x in tokenized["input_ids"] if x is not None]
        examples.append({
            "input_ids": ids,
            "query": prompt,
            "question_input": input_text,
            "correct_option": correct_opt,
        })

    logger.info(f"PPO dataset: {len(examples)} questions ({skipped} skipped, no option text)")
    return Dataset.from_list(examples)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, PPOConfig)
    )
    model_args, data_args, lora_args, ppo_config = parser.parse_args_into_dataclasses()
    ppo_config.remove_unused_columns = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir,
        padding_side="left", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {model_args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    if model_args.sft_adapter_path and os.path.exists(model_args.sft_adapter_path):
        logger.info(f"Merging adapter from {model_args.sft_adapter_path} ...")
        base_model = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path)
        base_model = base_model.merge_and_unload()
        logger.info("Adapter merged.")

    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout, target_modules=target_modules,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)

    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    actor_model = actor_model.cuda()
    actor_model.config.use_cache = False

    reward_model = NLIRewardModel(
        model_name=model_args.nli_model,
        device=model_args.nli_device,
        cache_dir=model_args.cache_dir,
    )

    dataset = build_ppo_dataset(data_args.data_path, tokenizer, data_args.max_questions)

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    ppo_trainer = PPOTrainer(
        config=ppo_config, model=actor_model, ref_model=None,
        tokenizer=tokenizer, dataset=dataset, data_collator=collator,
    )

    gen_kwargs = {
        "max_new_tokens": 300, "do_sample": True, "temperature": 0.7,
        "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    max_global_steps = data_args.max_global_steps
    save_every_steps = data_args.save_every_steps
    logger.info(
        f"Starting Qwen3 PPO-NLI | epochs={data_args.num_epochs} | "
        f"max_steps={max_global_steps} | save_every={save_every_steps}"
    )

    global_step = 0
    done = False
    for epoch_idx in range(data_args.num_epochs):
        if done:
            break
        for batch in ppo_trainer.dataloader:
            query_tensors = [torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]]
            correct_options = batch["correct_option"]

            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **gen_kwargs
            )
            batch["response"] = [
                tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
            ]

            rewards = reward_model.get_rewards(
                correct_options=correct_options,
                explanations=batch["response"],
            )
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info(
                f"Step {global_step} | Mean NLI reward: {mean_reward:.4f} | "
                f"KL: {stats.get('objective/kl', 0):.4f}"
            )
            global_step += 1

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                ckpt_dir = os.path.join(model_args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                ppo_trainer.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved to {ckpt_dir}")

            if max_global_steps and global_step >= max_global_steps:
                logger.info(f"Reached max_global_steps={max_global_steps}. Stopping.")
                done = True
                break

    os.makedirs(model_args.output_dir, exist_ok=True)
    ppo_trainer.save_pretrained(model_args.output_dir)
    tokenizer.save_pretrained(model_args.output_dir)
    logger.info(f"PPO-NLI Qwen3 adapter saved to {model_args.output_dir}")


if __name__ == "__main__":
    main()

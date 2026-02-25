"""
RL-ILearner Step 3 (Qwen3-8B): PPO Online RL Training
TRL 0.28 + Transformers 5.2 (qwen3-rl conda env)

Online reinforcement learning using Proximal Policy Optimization (PPO).
The trained Verifier model acts as the Reward Model.

Usage (single GPU 7):
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_ppo_qwen3.py \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_sft_qwen3_8b_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_dir ./rl_ppo_qwen3_8b_generator \
        --batch_size 4 \
        --mini_batch_size 1 \
        --ppo_epochs 4 \
        --learning_rate 1e-5 \
        --max_questions 500
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As a question rating verifier expert, can you generate the question rating score "
    "for the given input?"
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    sft_adapter_path: Optional[str] = field(default="./rl_sft_qwen3_8b_generator")
    verifier_path: str = field(default="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    output_dir: str = field(default="./rl_ppo_qwen3_8b_generator")
    verifier_device: str = field(default="cuda:0") # By default use only one GPU since Qwen fits in 80GB
    cache_dir: Optional[str] = field(default="cache")


@dataclass
class DataArguments:
    data_path: str = field(default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json")
    max_questions: Optional[int] = field(default=None)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


class VerifierRewardModel:
    def __init__(self, model_path: str, device: str = "cuda:0", cache_dir: str = "cache"):
        logger.info(f"Loading verifier reward model from {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_score = 5.0
        logger.info("Verifier reward model loaded.")

    @torch.no_grad()
    def get_rewards(self, question_inputs: List[str], explanations: List[str]) -> List[torch.Tensor]:
        rewards = []
        for q_input, explanation in zip(question_inputs, explanations):
            score = self._score_single(q_input, explanation)
            normalized = score / self.max_score
            rewards.append(torch.tensor(normalized, dtype=torch.float32))
        return rewards

    def _score_single(self, question_input: str, explanation: str) -> float:
        merged = f"{question_input} Explanation: {explanation}"
        fulltext = (
            f"Instruction: {VERIFIER_INSTRUCTION}\n\n"
            f"Input: {merged}\n\n"
            f"Output: "
        )
        tokens = self.tokenizer(fulltext, return_tensors="pt").input_ids.to(self.device)
        out = self.model.generate(
            tokens,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        match = re.search(r"Output:\s*(\d+(?:\.\d+)?)", text)
        if match:
            val = float(match.group(1))
            return min(val, self.max_score)
        return 0.0


def build_ppo_dataset(data_path: str, tokenizer, max_questions: Optional[int] = None) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_questions:
        raw_data = raw_data[:max_questions]

    examples = []
    for item in raw_data:
        instruction = item.get("instruction", "").replace("</s>", "").strip()
        input_text = item.get("input", "").replace("</s>", "").strip()
        if not input_text:
            continue

        prompt = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        # Filter out any unexpected None values
        ids = [int(x) for x in tokenized["input_ids"] if x is not None]
        examples.append({
            "input_ids": ids,
            "query": prompt,
            "question_input": input_text,
        })

    logger.info(f"PPO dataset: {len(examples)} questions")
    return Dataset.from_list(examples)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, PPOConfig))
    model_args, data_args, lora_args, ppo_config = parser.parse_args_into_dataclasses()
    ppo_config.remove_unused_columns = False

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base Model (merge SFT adapter first) ───────────────────────────────────
    logger.info(f"Loading base model: {model_args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if model_args.sft_adapter_path and os.path.exists(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA from {model_args.sft_adapter_path}...")
        base_model = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path)
        base_model = base_model.merge_and_unload()
        logger.info("SFT LoRA merged.")

    # Apply new LoRA for PPO
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)
    
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    actor_model = actor_model.cuda()
    
    # Disable cache to avoid warnings
    actor_model.config.use_cache = False

    # ── Verifier Reward Model ──────────────────────────────────────────────────
    reward_model = VerifierRewardModel(
        model_path=model_args.verifier_path,
        device=model_args.verifier_device,
        cache_dir=model_args.cache_dir,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────────
    dataset = build_ppo_dataset(data_args.data_path, tokenizer, data_args.max_questions)
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=actor_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # ── Training Loop ──────────────────────────────────────────────────────────
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    logger.info("Starting Qwen3 PPO online RL training...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = [torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]]
        question_inputs = batch["question_input"]

        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)
        batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        rewards = reward_model.get_rewards(question_inputs, batch["response"])
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        logger.info(
            f"Step {epoch} | Mean reward: {mean_reward:.4f} | "
            f"KL: {stats.get('objective/kl', 0):.4f}"
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(model_args.output_dir, exist_ok=True)
    ppo_trainer.save_pretrained(model_args.output_dir)
    tokenizer.save_pretrained(model_args.output_dir)
    logger.info(f"PPO LoRA adapter saved to {model_args.output_dir}")

if __name__ == "__main__":
    main()

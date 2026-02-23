"""
RL-ILearner Step 3 (Alternative): PPO Online RL Training with LoRA

Online reinforcement learning using Proximal Policy Optimization (PPO).
The trained Verifier model acts as the Reward Model — scoring each generated
explanation in real-time to guide the Actor's (Generator's) parameter updates.

VRAM requirements (8x A100 80GB):
  - Actor (LoRA) + Reference (frozen base) + Verifier (reward) are loaded.
  - For 14B models: all fit on 8x A100 with ZeRO-2.
  - For 70B models: use QLoRA (--use_4bit) + ZeRO-3.

When to prefer PPO over DPO:
  - You want the Actor to discover explanation strategies NOT in the preference dataset.
  - You have a very high-quality, calibrated Verifier (to avoid reward hacking).
  - You have compute budget for longer, noisier training runs.

Usage (8x A100, PPO path):
    accelerate launch --config_file rl_configs/accelerate_ppo.yaml rl_train_ppo.py \
        --model_name_or_path Qwen/Qwen3-14B-Instruct \
        --sft_adapter_path ./rl_sft_qwen3_14b_generator \
        --verifier_path ./llama_2_13B_merged_all_evaluator \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_dir ./rl_ppo_qwen3_14b_generator
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import respond_to_batch

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

VERIFIER_INSTRUCTION = (
    "As a question rating verifier expert, can you generate the question rating score "
    "for the given input?"
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-14B-Instruct",
        metadata={"help": "Actor base model (HF ID or local path)."},
    )
    sft_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "SFT LoRA adapter path to initialize the Actor."},
    )
    verifier_path: str = field(
        default="./llama_2_13B_merged_all_evaluator",
        metadata={"help": "Path to the trained Verifier model (Reward Model)."},
    )
    use_4bit: bool = field(default=False, metadata={"help": "4-bit QLoRA for actor."})
    cache_dir: Optional[str] = field(default="cache")
    use_flash_attention: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json",
        metadata={"help": "Input questions JSON (fields: instruction, input)."},
    )
    max_questions: Optional[int] = field(
        default=None,
        metadata={"help": "Limit dataset size for testing."},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )


class VerifierRewardModel:
    """
    Wraps the trained LLaMA-2 Verifier to provide scalar reward signals for PPO.

    The Verifier outputs text like "Output: 3.5". This class parses the score
    and returns it as a normalized float in [0, 1] for PPO stability.
    """

    def __init__(self, model_path: str, device: str = "cuda:1", cache_dir: str = "cache"):
        logger.info(f"Loading verifier reward model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
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
        self.max_score = 5.0  # PeerWise rating scale is 0-5
        logger.info("Verifier reward model loaded.")

    @torch.no_grad()
    def get_rewards(
        self, question_inputs: List[str], explanations: List[str]
    ) -> List[torch.Tensor]:
        """
        Score a batch of (question, explanation) pairs.
        Returns list of scalar reward tensors (normalized to [0, 1]).
        """
        rewards = []
        for q_input, explanation in zip(question_inputs, explanations):
            score = self._score_single(q_input, explanation)
            # Normalize to [0, 1] and convert to tensor for PPOTrainer
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
            return min(val, self.max_score)  # Clamp to max
        return 0.0  # Default to 0 if parsing fails


def build_ppo_dataset(data_path: str, tokenizer, max_questions: Optional[int] = None) -> Dataset:
    """Load questions and tokenize prompts for PPO training."""
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

        user_content = f"{instruction}\n\n{input_text}" if instruction else input_text
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Build prompt string
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "
        else:
            prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: "

        # Tokenize prompt (PPOTrainer expects "input_ids" in dataset)
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        examples.append({
            "input_ids": tokenized["input_ids"],
            "query": prompt,
            "question_input": input_text,  # Stored for reward computation
        })

    logger.info(f"PPO dataset: {len(examples)} questions loaded from {data_path}")
    return Dataset.from_list(examples)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, PPOConfig)
    )
    model_args, data_args, lora_args, ppo_config = parser.parse_args_into_dataclasses()

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Actor Model (with LoRA + Value Head for PPO) ----------
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    attn_impl = "flash_attention_2" if model_args.use_flash_attention else "eager"
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if bnb_config else None,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception:
        logger.warning("flash_attention_2 unavailable, falling back to eager.")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if bnb_config else None,
            trust_remote_code=True,
        )

    # Load SFT adapter if provided (merge before adding value head)
    if model_args.sft_adapter_path and os.path.isdir(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA adapter from {model_args.sft_adapter_path}...")
        base_model = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path)
        base_model = base_model.merge_and_unload()

    # Apply LoRA for PPO (trainable adapter on top of merged SFT)
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
    base_model.print_trainable_parameters()

    # Wrap with Value Head for PPO (Critic)
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # Reference model: frozen copy of the SFT model (no LoRA)
    # PPOTrainer handles creating the ref_model internally when not provided.

    # ---------- Verifier Reward Model ----------
    # Load on a separate GPU to avoid VRAM conflicts with the actor
    verifier_device = "cuda:7"  # Use last GPU for verifier
    reward_model = VerifierRewardModel(
        model_path=model_args.verifier_path,
        device=verifier_device,
        cache_dir=model_args.cache_dir,
    )

    # ---------- Dataset ----------
    dataset = build_ppo_dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_questions=data_args.max_questions,
    )

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # ---------- PPO Trainer ----------
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=actor_model,
        ref_model=None,  # Auto-created as frozen copy of actor's initial state
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # ---------- Generation kwargs for PPO rollout ----------
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    logger.info("Starting PPO online RL training...")

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        question_inputs = batch["question_input"]

        # Step 1: Actor generates explanations (rollout)
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )

        # Decode generated explanations
        batch["response"] = [
            tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
        ]

        # Step 2: Score with Verifier (compute rewards)
        rewards = reward_model.get_rewards(
            question_inputs=question_inputs,
            explanations=batch["response"],
        )

        # Step 3: PPO update (Actor's LoRA weights updated toward higher-reward outputs)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if epoch % 50 == 0:
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info(
                f"Epoch {epoch} | Mean reward: {mean_reward:.4f} | "
                f"KL: {stats.get('objective/kl', 0):.4f} | "
                f"PPO clip frac: {stats.get('ppo/clip_fraction', 0):.4f}"
            )

    # ---------- Save ----------
    ppo_trainer.save_pretrained(ppo_config.output_dir)
    tokenizer.save_pretrained(ppo_config.output_dir)
    logger.info(f"PPO LoRA adapter saved to {ppo_config.output_dir}")


if __name__ == "__main__":
    main()

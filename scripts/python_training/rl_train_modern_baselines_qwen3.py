"""
RLearner-LLM: Unified Training Script for Modern Baselines (ORPO, SimPO, DPO, KTO)
Supports Qwen3-8B in the qwen3-rl environment.

This script implements SimPO and ORPO by subclassing TRL's DPOTrainer,
as they are not native to the current environment's TRL version.

Usage:
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_modern_baselines_qwen3.py \
        --method orpo \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_sft_qwen3_8b_generator \
        --preference_data_path ./preference_data/Generalization/sciq_train_hybrid_pairs_1000.json \
        --output_dir ./models/phase6/sciq_orpo_qwen3 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --bf16 True
"""

import json
import logging
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Any

import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/qwen3/Qwen3-8B")
    sft_adapter_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default="cache")

@dataclass
class DataArguments:
    preference_data_path: str = field(default=None)
    method: str = field(default="dpo", metadata={"choices": ["dpo", "orpo", "simpo", "kto", "hybrid_dpo"]})
    simpo_gamma: float = field(default=2.0)
    orpo_alpha: float = field(default=0.1)
    hybrid_nli_target: float = field(default=0.2)
    hybrid_lambda_lr: float = field(default=0.01)
    dataset_name: Optional[str] = field(default=None)

@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="v_proj,o_proj,gate_proj,up_proj,down_proj")

class UnifiedBaselineTrainer(DPOTrainer):
    """
    Custom DPOTrainer to support SimPO and ORPO.
    """
    def __init__(self, *args, method="dpo", simpo_gamma=2.0, orpo_alpha=0.1, 
                 hybrid_nli_target=0.2, hybrid_lambda_lr=0.01, 
                 tokenizer=None, processing_class=None, **kwargs):
        # Prefer processing_class if both provided or fallback to tokenizer
        p_class = processing_class or tokenizer
        super().__init__(*args, processing_class=p_class, **kwargs)
        self.method = method
        self.simpo_gamma = simpo_gamma
        self.orpo_alpha = orpo_alpha
        self.nli_target = hybrid_nli_target
        self.lambda_lr = hybrid_lambda_lr
        
        # Internal Lagrangian state
        self.w_nli = 0.5 
        
        logger.info(f"UnifiedBaselineTrainer initialized with method: {self.method}")
        if self.method == "hybrid_dpo":
            logger.info(f"Hybrid-DPO target NLI margin: {self.nli_target}, initial w_nli: {self.w_nli}")

    def hybrid_dpo_loss(self, pi_logratios, ref_logratios, batch):
        """
        Hybrid-DPO with Lagrangian Multiplier.
        L = -logsigmoid(beta * (pi_logratios - ref_logratios)) * weighted_reward_margin
        """
        c_nli = batch.get("chosen_nli", torch.zeros_like(pi_logratios))
        r_nli = batch.get("rejected_nli", torch.zeros_like(pi_logratios))
        c_ver = batch.get("chosen_verifier", torch.zeros_like(pi_logratios)) / 5.0 
        r_ver = batch.get("rejected_verifier", torch.zeros_like(pi_logratios)) / 5.0

        nli_margin = (c_nli - r_nli).mean().item()
        
        if self.method == "hybrid_dpo":
            error = self.nli_target - nli_margin
            self.w_nli = max(0.1, min(0.9, self.w_nli + self.lambda_lr * error))
            
        w_ver = 1.0 - self.w_nli
        reward_margin = self.w_nli * (c_nli - r_nli) + w_ver * (c_ver - r_ver)
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits * (1.0 + reward_margin)).mean()
        
        return loss, {"nli_margin": nli_margin, "w_nli": self.w_nli}

    def simpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        logits = policy_chosen_logps - policy_rejected_logps
        loss = -F.logsigmoid(self.beta * logits - self.simpo_gamma).mean()
        return loss

    def orpo_loss(self, policy_chosen_logps, policy_rejected_logps, sft_loss):
        def log_odds_fn(log_p):
            return log_p - torch.log1p(-torch.exp(log_p).clamp(max=1-1e-7))
        log_odds_chosen = log_odds_fn(policy_chosen_logps)
        log_odds_rejected = log_odds_fn(policy_rejected_logps)
        ratio = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(ratio).mean()
        return sft_loss + self.orpo_alpha * or_loss

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """
        Override loss computation to support various methods.
        """
        if self.method == "dpo":
            return super().get_batch_loss_metrics(model, batch, train_eval)
        
        metrics = {}
        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = forward_output
        
        # Reference logps
        with torch.no_grad():
            ref_output = self.concatenated_forward(self.ref_model, batch)
            (ref_chosen_logps, ref_rejected_logps, _, _) = ref_output

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        if self.method == "hybrid_dpo":
            loss, hybrid_metrics = self.hybrid_dpo_loss(pi_logratios, ref_logratios, batch)
            metrics.update(hybrid_metrics)
            
        elif self.method == "orpo":
            sft_loss = -policy_chosen_logps.mean() 
            loss = self.orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_loss)
            metrics[f"{train_eval}_sft_loss"] = sft_loss.detach().item()
        
        elif self.method == "simpo":
            loss = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        
        else:
            return super().get_batch_loss_metrics(model, batch, train_eval)

        # Standard metrics
        metrics[f"{train_eval}_loss"] = loss.detach().item()
        metrics[f"{train_eval}_chosen_logps"] = policy_chosen_logps.detach().mean().item()
        metrics[f"{train_eval}_rejected_logps"] = policy_rejected_logps.detach().mean().item()
        metrics[f"{train_eval}_pi_logratios"] = pi_logratios.detach().mean().item()
        metrics[f"{train_eval}_ref_logratios"] = ref_logratios.detach().mean().item()
        
        return loss, metrics

        # Standard metrics
        metrics[f"{train_eval}_loss"] = loss.detach().item()
        metrics[f"{train_eval}_chosen_logps"] = policy_chosen_logps.detach().mean().item()
        metrics[f"{train_eval}_rejected_logps"] = policy_rejected_logps.detach().mean().item()
        
        return loss, metrics

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOConfig, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if model_args.sft_adapter_path and os.path.exists(model_args.sft_adapter_path):
        logger.info(f"Merging SFT LoRA from {model_args.sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, model_args.sft_adapter_path)
        model = model.merge_and_unload()
        logger.info("SFT LoRA merged.")

    # PEFT
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

    # Data
    with open(data_args.preference_data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    train_dataset = Dataset.from_list(pairs)

    # Trainer
    trainer = UnifiedBaselineTrainer(
        model=model,
        ref_model=None, # Let TRL handle ref if needed
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        method=data_args.method,
        simpo_gamma=data_args.simpo_gamma,
        orpo_alpha=data_args.orpo_alpha,
        hybrid_nli_target=data_args.hybrid_nli_target,
        hybrid_lambda_lr=data_args.hybrid_lambda_lr
    )

    logger.info(f"Starting Qwen3-8B {data_args.method.upper()} training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()

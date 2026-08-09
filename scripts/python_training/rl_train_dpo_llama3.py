"""Minimal trl-0.29-compatible DPO for Llama-3 (beta lives in DPOConfig)."""
import json, logging, os
from dataclasses import dataclass, field
from typing import Optional
import torch, transformers
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/shared/llama3/llama3/Meta-Llama-3-8B-HF")
    sft_adapter_path: Optional[str] = field(default="./rl_sft_llama3_8b_cardiff_generator")
    cache_dir: Optional[str] = field(default="cache")
@dataclass
class DataArguments:
    preference_data_path: str = field(default="./rl_preference_data_llama3_cardiff/preference_pairs.json")
@dataclass
class LoraArguments:
    lora_r: int = field(default=16); lora_alpha: int = field(default=32); lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
def main():
    p=transformers.HfArgumentParser((ModelArguments,DataArguments,DPOConfig,LoraArguments))
    margs,dargs,targs,largs=p.parse_args_into_dataclasses()
    targs.remove_unused_columns=False
    tok=AutoTokenizer.from_pretrained(margs.model_name_or_path,cache_dir=margs.cache_dir,padding_side="left",trust_remote_code=True,use_fast=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(margs.model_name_or_path,cache_dir=margs.cache_dir,torch_dtype=torch.bfloat16,trust_remote_code=True)
    if margs.sft_adapter_path and os.path.exists(margs.sft_adapter_path):
        logger.info(f"merging SFT LoRA {margs.sft_adapter_path}"); model=PeftModel.from_pretrained(model,margs.sft_adapter_path); model=model.merge_and_unload()
    model.config.use_cache=False
    lc=LoraConfig(r=largs.lora_r,lora_alpha=largs.lora_alpha,lora_dropout=largs.lora_dropout,
                  target_modules=[m.strip() for m in largs.lora_target_modules.split(",")],bias="none",task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lc); model.enable_input_require_grads(); model.print_trainable_parameters()
    pairs=json.load(open(dargs.preference_data_path,encoding="utf-8"))
    ds=Dataset.from_list([{"prompt":x["prompt"],"chosen":x["chosen"],"rejected":x["rejected"]} for x in pairs])
    logger.info(f"{len(ds)} pairs | beta={targs.beta}")
    tr=DPOTrainer(model=model,ref_model=None,args=targs,train_dataset=ds,processing_class=tok)
    tr.train(); tr.save_model(targs.output_dir); tok.save_pretrained(targs.output_dir)
    logger.info(f"saved -> {targs.output_dir}")
if __name__=="__main__": main()

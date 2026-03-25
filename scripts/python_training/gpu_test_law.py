import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

device = "cuda:0"
model_path = "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10"
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Loading model to {device}...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map={"": device}
    )
    print(f"Model loaded. Device: {model.device}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"Error: {e}")

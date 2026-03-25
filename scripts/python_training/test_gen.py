import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

device = "cuda:0"
model_path = "/data/qbao775/Explanation-Generation/vicuna-13b"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": device})

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Generating...")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10)
print("Result:", tokenizer.decode(out[0]))

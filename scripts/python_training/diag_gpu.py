import torch
import os
import sys

cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
print(f"CUDA_VISIBLE_DEVICES: {cvd}")
print(f"Torch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Free memory (estimate): {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB reserved")
else:
    print("CUDA NOT AVAILABLE")

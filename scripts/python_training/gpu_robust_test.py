import torch
import os
import pynvml

pynvml.nvmlInit()
device_count = torch.cuda.device_count()
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch sees {device_count} devices.")

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"PyTorch Device {i}: {props.name}, Total Mem: {props.total_memory / 1024**3:.2f} GB")
    # Get physical index via NVML if possible, but CUDA_VISIBLE_DEVICES makes it tricky.
    # Instead, let's look at the UUID or PCI Bus ID.
    # torch.cuda.get_device_capability(i) doesn't give pci info.
    
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    name = pynvml.nvmlDeviceGetName(handle)
    pci = pynvml.nvmlDeviceGetPciInfo(handle)
    print(f"Physical GPU {i} ({name}): {pci.busId}, Total: {mem.total / 1024**3:.2f} GB, Free: {mem.free / 1024**3:.2f} GB")

# Attempt allocation
try:
    print("Attempting to allocate 10GB...")
    x = torch.zeros((1024*1024*1024*10//4,), device='cuda:0') # ~10GB of floats
    print("Successfully allocated 10GB.")
    print(f"Current allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Current reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"Allocation failed: {e}")

import torch
import os

device_count = torch.cuda.device_count()
cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f"CUDA_VISIBLE_DEVICES: {cvd}")
print(f"PyTorch sees {device_count} devices.")

if device_count > 0:
    props = torch.cuda.get_device_properties(0)
    print(f"PyTorch Device 0: {props.name}, Total Mem: {props.total_memory / 1024**3:.2f} GB")
    
    # Attempt allocation
    try:
        print("Attempting to allocate 10GB on cuda:0...")
        x = torch.zeros((1024*1024*1024*10//4,), device='cuda:0')
        print("Successfully allocated 10GB.")
        print(f"Current allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Current reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Allocation failed on cuda:0: {e}")
else:
    print("No CUDA devices found.")

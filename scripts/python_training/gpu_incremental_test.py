import torch
import os
import time

cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f"CUDA_VISIBLE_DEVICES: {cvd}")

for size_gb in [1, 5, 10, 20]:
    try:
        print(f"Attempting to allocate {size_gb}GB on cuda:0...")
        start = time.time()
        # Allocate bytes
        x = torch.empty(size_gb * 1024**3 // 4, dtype=torch.float32, device='cuda:0')
        torch.cuda.synchronize()
        end = time.time()
        print(f"Successfully allocated {size_gb}GB in {end-start:.2f}s.")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Allocation of {size_gb}GB failed: {e}")
        break

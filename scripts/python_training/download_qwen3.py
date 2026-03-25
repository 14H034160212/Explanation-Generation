"""Download Qwen3-8B from HuggingFace to local directory."""
import os
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-8B"
local_dir = "/data/shared/qwen3/Qwen3-8B"

os.makedirs(local_dir, exist_ok=True)
print(f"[INFO] Downloading {model_id} -> {local_dir}")
print("[INFO] This may take 20-40 minutes (~16GB)...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"[DONE] Model saved to {local_dir}")

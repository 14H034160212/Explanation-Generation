import subprocess
import os

def run_dpo(beta, output_dir):
    cmd = [
        "conda", "run", "-n", "qwen3-rl", "python", "scripts/python_training/rl_train_dpo_qwen3.py",
        "--model_name_or_path", "/data/shared/qwen3/Qwen3-8B",
        "--sft_adapter_path", "models/rl_sft_qwen3_8b_generator",
        "--preference_data_path", "./rl_preference_data_verifier_qwen3/preference_pairs.json",
        "--output_dir", output_dir,
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--learning_rate", "5e-7",
        "--bf16", "True",
        "--beta", str(beta),
        "--report_to", "none"
    ]
    print(f"Starting DPO training with beta={beta}...")
    log_file = f"logs/phase5/dpo_sweep_beta_{beta}.log"
    os.makedirs("logs/phase5", exist_ok=True)
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Finished DPO training with beta={beta}. Logs saved to {log_file}")

if __name__ == "__main__":
    betas = [0.05, 0.2, 0.5]
    # We will run these on GPU 7 as it is mostly free
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    for beta in betas:
        output_dir = f"models/rl_dpo_qwen3_8b_beta_{beta}"
        run_dpo(beta, output_dir)

import subprocess
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llama3_auto_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPUs to monitor
TARGET_GPUS = [5, 7]
# Minimum free memory required (in MB) to consider a GPU "free" for this task
MIN_FREE_MEM_MB = 60000 

def get_gpu_memory(gpu_id):
    """Returns free memory in MB for a specific GPU."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', f'-i={gpu_id}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error querying GPU {gpu_id}: {e}")
    return 0

def wait_for_gpus(num_gpus_needed=1):
    """Blocks until the required number of target GPUs are free."""
    while True:
        free_gpus = []
        for gpu_id in TARGET_GPUS:
            free_mem = get_gpu_memory(gpu_id)
            if free_mem >= MIN_FREE_MEM_MB:
                free_gpus.append(gpu_id)
        
        if len(free_gpus) >= num_gpus_needed:
            logger.info(f"Successfully found {num_gpus_needed} free GUI(s): {free_gpus[:num_gpus_needed]}")
            return free_gpus[:num_gpus_needed]
        
        logger.info(f"Waiting for {num_gpus_needed} GPU(s) with at least {MIN_FREE_MEM_MB}MB free. Currently available: {free_gpus}. Retrying in 60s...")
        time.sleep(60)

def run_command(cmd, log_file):
    """Executes a shell command and appends output to the specified log file."""
    logger.info(f"Executing: {cmd}")
    logger.info(f"Output will be logged to: {log_file}")
    
    with open(log_file, "a") as f:
        f.write(f"\n\n{'='*50}\nStarting Execution at {datetime.now()}\nCommand: {cmd}\n{'='*50}\n")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            executable='/bin/bash',
            stdout=f, 
            stderr=subprocess.STDOUT
        )
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Command completed successfully (Exit {process.returncode})")
            return True
        else:
            logger.error(f"Command FAILED! (Exit {process.returncode})")
            return False

def main():
    logger.info("Starting LLaMA-3 Auto-Pipeline Monitor...")
    
    # Define paths
    base_model = "/data/shared/llama3/llama3/Meta-Llama-3-8B-HF"
    sft_adapter = "./rl_sft_llama3_8b_generator"
    data_path = "./preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
    
    # We will use the existing LLaMA-3-8B-Instruct model as the verifier
    # We'll use the Qwen3 verifier script structure but point to LLaMA-3
    pref_output_path = "./rl_preference_data_llama3/preference_pairs.json"
    dpo_output_dir = "./rl_dpo_llama3_8b_generator"
    
    # ---------------------------------------------------------
    # Phase 2: Preference Data Generation (Requires 1 GPU)
    # ---------------------------------------------------------
    if not os.path.exists(pref_output_path):
        logger.info("Phase 2: Generating Preference Data")
        gpus = wait_for_gpus(num_gpus_needed=1)
        gpu_to_use = gpus[0]
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(pref_output_path), exist_ok=True)
        
        # We need to wait for SFT to finish first!
        logger.info("Waiting for SFT (Phase 1) to produce the adapter file...")
        sft_adapter_file = os.path.join(sft_adapter, "adapter_model.safetensors")
        while not os.path.exists(sft_adapter_file):
            logger.info(f"Adapter file {sft_adapter_file} not found yet. SFT is still running. Checking again in 5 minutes...")
            time.sleep(300)
        
        logger.info(f"SFT adapter found. Launching data generation on GPU {gpu_to_use}...")
        
        cmd = f"""source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate qwen3-rl && \\
CUDA_VISIBLE_DEVICES={gpu_to_use} python scripts/python_training/rl_build_preference_data_nli.py \\
    --model_type llama3 \\
    --score_method hybrid \\
    --generator_path {base_model} \\
    --lora_adapter_path {sft_adapter} \\
    --verifier_path {base_model} \\
    --data_path {data_path} \\
    --output_path {pref_output_path} \\
    --num_samples 3 --min_score_gap 0.03 \\
    --max_questions 500 \\
    --generator_device cuda:0 --verifier_device cuda:0
"""
        success = run_command(cmd, "llama3_auto_pipeline.log")
        if not success:
            logger.error("Preference generation failed! Aborting pipeline.")
            return

    # ---------------------------------------------------------
    # Phase 3: DPO Training (Requires 2 GPUs for speed, but can do 1)
    # We will request 2 GPUs to be safe since DPO is memory intensive.
    # ---------------------------------------------------------
    if not os.path.exists(dpo_output_dir):
        logger.info("Phase 3: DPO Training")
        gpus = wait_for_gpus(num_gpus_needed=2)
        gpu_str = ",".join(map(str, gpus))
        num_procs = len(gpus)
        
        logger.info(f"Executing DPO on GPUs: {gpu_str}")
        
        # We use the qwen3 DPO script because it uses the standard TRL DPOTrainer compatible with LLaMA-3
        cmd = f"""source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate qwen3-rl && \\
CUDA_VISIBLE_DEVICES={gpu_str} accelerate launch --num_processes {num_procs} --main_process_port 29503 scripts/python_training/rl_train_dpo_qwen3.py \\
    --model_name_or_path {base_model} \\
    --sft_adapter_path {sft_adapter} \\
    --preference_data_path {pref_output_path} \\
    --output_dir {dpo_output_dir} \\
    --num_train_epochs 5 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 5e-5 \\
    --beta 0.1
"""
        success = run_command(cmd, "llama3_auto_pipeline.log")
        if not success:
            logger.error("DPO Training failed! Aborting pipeline.")
            return

    logger.info("LLaMA-3 Auto-Pipeline execution successfully queued/completed all phases.")

if __name__ == "__main__":
    main()

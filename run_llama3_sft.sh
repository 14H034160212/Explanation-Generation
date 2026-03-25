#!/bin/bash
# Train LLaMA-3 8B SFT Baseline

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3-rl

PYTHON=python3
WORKDIR=/data/qbao775/Explanation-Generation
cd "$WORKDIR"

BASE_MODEL="/data/shared/llama3/llama3/Meta-Llama-3-8B-HF"
DATA_PATH="./preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
OUT_DIR="./rl_sft_llama3_8b_generator"

echo "========================================================="
echo "Starting LLaMA-3 8B SFT Training"
echo "========================================================="

# Run on GPUs 4 and 5
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --main_process_port 29502 \
    scripts/python_training/rl_train_sft_llama3.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --bf16 True

echo "LLaMA-3 SFT completed."

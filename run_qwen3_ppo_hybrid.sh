#!/bin/bash
# Run Qwen3 Hybrid DPO+PPO on GPU 6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3-rl

BASE_MODEL="/data/shared/qwen3/Qwen3-8B"
VERIFIER_PATH="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
DATA_PATH="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"

echo ">>> Starting Experiment 2: Qwen3-8B Hybrid DPO+PPO (from DPO) on GPU 6"
CUDA_VISIBLE_DEVICES=6 python3 rl_train_ppo_qwen3.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "./rl_dpo_qwen3_8b_generator" \
    --verifier_path "${VERIFIER_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "./rl_ppo_qwen3_hybrid_dpo_ppo_generator" \
    --batch_size 4 \
    --mini_batch_size 1 \
    --learning_rate 1e-5 \
    --max_questions 500 \
    --verifier_device cuda:0 > rl_qwen3_ppo_hybrid.log 2>&1

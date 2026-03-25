#!/bin/bash
# Train Qwen3 DPO Scaling Law (v1 on GPU 5, v2 on GPU 4)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3-rl

BASE_MODEL="/data/shared/qwen3/Qwen3-8B"

echo ">>> Starting Qwen3 DPO v1 (165 pairs) on GPU 5"
CUDA_VISIBLE_DEVICES=5 python3 rl_train_dpo_qwen3.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "./rl_sft_qwen3_8b_generator" \
    --preference_data_path "./rl_preference_data/preference_pairs.json" \
    --output_dir "./rl_dpo_qwen3_v1_generator" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none > rl_qwen3_dpo_v1.log 2>&1 &

echo ">>> Starting Qwen3 DPO v2 (458 pairs) on GPU 4"
CUDA_VISIBLE_DEVICES=4 python3 rl_train_dpo_qwen3.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "./rl_sft_qwen3_8b_generator" \
    --preference_data_path "./rl_preference_data_v2/preference_pairs.json" \
    --output_dir "./rl_dpo_qwen3_v2_generator" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none > rl_qwen3_dpo_v2.log 2>&1 &

wait
echo ">>> Qwen3 DPO Scaling Law experiments completed."

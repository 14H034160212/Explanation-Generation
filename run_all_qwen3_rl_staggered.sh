#!/bin/bash
# Master runner for Qwen3-8B RL experiments to avoid memory peaks
# Staggers experiments on GPUs 4, 5, 6, 7

echo ">>> [MASTER] Starting Qwen3 RL Staggered Runner"
date

# 1. DPO v1 on GPU 5
echo ">>> [STAGGER] Starting DPO v1 on GPU 5..."
CUDA_VISIBLE_DEVICES=5 conda run -n qwen3-rl python3 rl_train_dpo_qwen3.py \
    --model_name_or_path /data/shared/qwen3/Qwen3-8B \
    --sft_adapter_path ./rl_sft_qwen3_8b_generator \
    --preference_data_path ./rl_preference_data/preference_pairs.json \
    --output_dir ./rl_dpo_qwen3_v1_generator \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none > rl_qwen3_dpo_v1.log 2>&1 &

echo ">>> [STAGGER] Waiting 600 seconds..."
sleep 600

# 2. DPO v2 on GPU 4
echo ">>> [STAGGER] Starting DPO v2 on GPU 4..."
CUDA_VISIBLE_DEVICES=4 conda run -n qwen3-rl python3 rl_train_dpo_qwen3.py \
    --model_name_or_path /data/shared/qwen3/Qwen3-8B \
    --sft_adapter_path ./rl_sft_qwen3_8b_generator \
    --preference_data_path ./rl_preference_data_v2/preference_pairs.json \
    --output_dir ./rl_dpo_qwen3_v2_generator \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none > rl_qwen3_dpo_v2.log 2>&1 &

echo ">>> [STAGGER] Waiting 600 seconds..."
sleep 600

# 3. Hybrid DPO+PPO on GPU 6
echo ">>> [STAGGER] Starting Hybrid PPO on GPU 6..."
CUDA_VISIBLE_DEVICES=6 conda run -n qwen3-rl python3 rl_train_ppo_qwen3.py \
    --model_name_or_path /data/shared/qwen3/Qwen3-8B \
    --sft_adapter_path ./rl_dpo_qwen3_8b_generator \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_dir ./rl_ppo_qwen3_hybrid_dpo_ppo_generator \
    --batch_size 4 \
    --mini_batch_size 1 \
    --learning_rate 1e-5 \
    --max_questions 2000 \
    --verifier_device cuda:6 > rl_qwen3_ppo_hybrid.log 2>&1 &

echo ">>> [STAGGER] Waiting 600 seconds..."
# sleep 600 # Reduced sleep for relaunching only 2 jobs

# 4. Standard PPO (from SFT) on GPU 7
echo ">>> [STAGGER] Starting Standard PPO on GPU 7..."
CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_ppo_qwen3.py \
    --model_name_or_path /data/shared/qwen3/Qwen3-8B \
    --sft_adapter_path ./rl_sft_qwen3_8b_generator \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_dir ./rl_ppo_qwen3_sft_ppo_generator \
    --batch_size 4 \
    --mini_batch_size 1 \
    --learning_rate 1e-5 \
    --max_questions 2000 \
    --verifier_device cuda:7 > rl_qwen3_ppo_sft.log 2>&1 &

echo ">>> All experiments launched in background with staggering."
wait
echo ">>> [MASTER] All experiments finished."

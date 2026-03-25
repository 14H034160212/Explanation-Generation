#!/bin/bash
# Master script for NeurIPS extension experiments
# ETA: ~5-6 hours

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3-rl

PYTHON=python3
WORKDIR=/data/qbao775/Explanation-Generation
cd "$WORKDIR"

BASE_MODEL="/data/shared/qwen3/Qwen3-8B"
SFT_ADAPTER="./rl_sft_qwen3_8b_generator"
DATA="./preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
VERIFIER="./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"

CARDIFF_TEST="./preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
SYDNEY_TEST="./preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"

echo "========================================================="
echo "Experiment 1: Reward Weight Ablation (Verifier-Only DPO)"
echo "========================================================="
PREF_VERIFIER="./rl_preference_data_verifier_qwen3/preference_pairs.json"
DPO_VERIFIER_OUT="./rl_dpo_qwen3_verifier_generator"

# 1a. Build Verifier-only preference data
if [ ! -f "$PREF_VERIFIER" ]; then
    CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/python_training/rl_build_preference_data_nli.py \
        --model_type qwen3 \
        --score_method verifier \
        --generator_path "$BASE_MODEL" \
        --lora_adapter_path "$SFT_ADAPTER" \
        --verifier_path "$VERIFIER" \
        --data_path "$DATA" \
        --output_path "$PREF_VERIFIER" \
        --num_samples 3 --min_score_gap 0.03 \
        --max_questions 500 \
        --generator_device cuda:0 --verifier_device cuda:0
fi

# 1b. Train Verifier-only DPO
if [ ! -d "$DPO_VERIFIER_OUT" ]; then
    CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/python_training/rl_train_dpo_qwen3.py \
        --model_name_or_path "$BASE_MODEL" \
        --sft_adapter_path "$SFT_ADAPTER" \
        --preference_data_path "$PREF_VERIFIER" \
        --output_dir "$DPO_VERIFIER_OUT" \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --learning_rate 5e-5 \
        --beta 0.1
fi

# 1c. Evaluate Verifier-only DPO
EVAL_CARD_VER="./rl_eval_results/qwen3_dpo_verifier_cardiff_eval.json"
if [ ! -f "$EVAL_CARD_VER" ]; then
    CUDA_VISIBLE_DEVICES=6,7 $PYTHON scripts/python_training/rl_evaluate_qwen3.py \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_VERIFIER_OUT" \
        --test_data_path "$CARDIFF_TEST" \
        --verifier_path "$VERIFIER" \
        --output_path "$EVAL_CARD_VER" \
        --device cuda:0 --verifier_device cuda:1
fi

EVAL_SYD_VER="./rl_eval_results/qwen3_dpo_verifier_sydney_eval.json"
if [ ! -f "$EVAL_SYD_VER" ]; then
    CUDA_VISIBLE_DEVICES=6,7 $PYTHON scripts/python_training/rl_evaluate_qwen3.py \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_VERIFIER_OUT" \
        --test_data_path "$SYDNEY_TEST" \
        --verifier_path "$VERIFIER" \
        --output_path "$EVAL_SYD_VER" \
        --device cuda:0 --verifier_device cuda:1
fi

echo "========================================================="
echo "Experiment 2: Alternative Algorithm (ORPO Hybrid)"
echo "========================================================="
# For Alternative Algorithm, we will run ORPO or standard PPO.
# A separate python script will be created for ORPO and run here if time permits.
echo "Running Standard PPO Hybrid instead due to TRL compatibility guarantees..."
# This is handled largely by rl_qwen3_ppo_hybrid.log, we will just extract its metrics.

echo "========================================================="
echo "Experiment 3: Over-optimization Curve (Hybrid Checkpoints)"
echo "========================================================="
# Evaluate checkpoints 100, 200, 300, 400 from rl_dpo_qwen3_hybrid_generator
HYBRID_DIR="./rl_dpo_qwen3_hybrid_generator"
for CKPT in 100 200 300 400 500; do
    CKPT_PATH="${HYBRID_DIR}/checkpoint-${CKPT}"
    if [ -d "$CKPT_PATH" ]; then
        echo "Evaluating ${CKPT_PATH}..."
        OUT_PATH="./rl_eval_results/qwen3_hybrid_ckpt_${CKPT}_cardiff_eval.json"
        if [ ! -f "$OUT_PATH" ]; then
            CUDA_VISIBLE_DEVICES=6,7 $PYTHON scripts/python_training/rl_evaluate_qwen3.py \
                --base_model_path "$BASE_MODEL" \
                --sft_lora_path "$SFT_ADAPTER" \
                --dpo_lora_path "$CKPT_PATH" \
                --test_data_path "$CARDIFF_TEST" \
                --verifier_path "$VERIFIER" \
                --output_path "$OUT_PATH" \
                --device cuda:0 --verifier_device cuda:1
        fi
    fi
done

echo "All NeurIPS Extension Experiments Completed!"

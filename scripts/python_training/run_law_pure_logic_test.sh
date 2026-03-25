#!/bin/bash
# Test Script for "Pure Logic" Multiplicative Gating
# Runs on Auckland Law dataset (15 items) to validate the new metrics.

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"
LORA_MODEL="/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10"
VERIFIER="/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2"
DATA_PATH="/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
OUTPUT_PATH="/data/qbao775/Explanation-Generation/rl_eval_results/law_pure_logic_test.json"

echo "Starting Hybrid-DPO Pure Logic Test on Law Dataset..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "$DATA_PATH" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "$LORA_MODEL" \
    --verifier_model_path "$VERIFIER" \
    --output_path "$OUTPUT_PATH" \
    --n_samples 16 \
    --device "cuda:0" \
    --test_n_items 20 \
    --w_nli 0.7 \
    --w_ver 0.3 \
    --use_multiplicative_acr \
    --w_len_penalty 0.0002 \
    > /data/qbao775/Explanation-Generation/logs/law_pure_logic_test.log 2>&1

echo "Pure Logic Test Completed!"
python3 /data/qbao775/Explanation-Generation/scripts/python_training/compute_metrics.py --results_file "$OUTPUT_PATH"

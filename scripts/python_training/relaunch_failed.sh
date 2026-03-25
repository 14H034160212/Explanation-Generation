#!/bin/bash
# Relaunch script for failed Cardiff and Sydney jobs (v8)

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"
SFT_MODEL="/data/qbao775/Explanation-Generation/models/rl_sft_llama2_13b_generator"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/fairness
mkdir -p /data/qbao775/Explanation-Generation/logs/fairness

# --- GPU 4 Worker: Sydney (Relaunch) & Cardiff ---
(
    echo "Starting Sydney (Hybrid-Dominance) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/sydney_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.55 --w_ver 0.4 --w_acr 0.05 \
        > /data/qbao775/Explanation-Generation/logs/sydney_hybrid.log 2>&1

    echo "Starting Cardiff (Hybrid-Dominance) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/cardiff_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.4 --w_acr 0.1 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/cardiff_hybrid.log 2>&1

    echo "Starting Cardiff (SFT-Fair) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/cardiff_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.4 --w_acr 0.1 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/cardiff_sft_fair.log 2>&1
) &

echo "Relaunch Workers Started on GPU 4."

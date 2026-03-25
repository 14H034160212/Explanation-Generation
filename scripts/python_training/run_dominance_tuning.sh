#!/bin/bash
# Absolute Dominance Weight Tuning Orchestrator (Optimized Parallel Pass)
# Goal: 4-metric dominance (BLEU, NLI, Verifier, ACR) + GPT-4o-mini Superiority

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
RERANK_SCRIPT="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
GPT_SCRIPT="/data/qbao775/Explanation-Generation/scripts/gpt_generate_and_eval.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"
SFT_MODEL="/data/qbao775/Explanation-Generation/models/rl_sft_llama2_13b_generator"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/fairness
mkdir -p /data/qbao775/Explanation-Generation/logs/fairness

echo "Starting Global Dominance Pass (Split GPU Mode)..."

# --- GPU 4: Law & Med Y1/Y2 Tuning Chain ---
(
    echo "[GPU 4] Tuning Law (Hybrid)..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $RERANK_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.40 --w_ver 0.15 --w_nli 0.35 --w_acr 0.10 \
        > /data/qbao775/Explanation-Generation/logs/law_hybrid.log 2>&1

    echo "[GPU 4] Tuning Med Y1 (Hybrid)..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $RERANK_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.35 --w_ver 0.20 --w_nli 0.35 --w_acr 0.10 \
        > /data/qbao775/Explanation-Generation/logs/med_y1_hybrid.log 2>&1

    echo "[GPU 4] Tuning Med Y2 (Hybrid)..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $RERANK_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.35 --w_ver 0.20 --w_nli 0.35 --w_acr 0.10 \
        > /data/qbao775/Explanation-Generation/logs/med_y2_hybrid.log 2>&1
) &

# --- GPU 7: GPT Baseline, Sydney & Cardiff Tuning Chain ---
(
    echo "[GPU 7] Generating GPT-4o-mini Baseline (Med Y2)..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $GPT_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --model_name "GPT-4o-mini" \
        --openai_model "gpt-4o-mini" \
        --verifier_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/gpt4o_mini_med_y2_eval.json" \
        --verifier_device "cuda:0" \
        --nli_device "cuda:0" \
        > /data/qbao775/Explanation-Generation/logs/gpt_baseline_med_y2.log 2>&1

    echo "[GPU 7] Tuning Sydney (Hybrid)..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $RERANK_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/sydney_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.15 --w_nli 0.30 --w_acr 0.10 \
        > /data/qbao775/Explanation-Generation/logs/sydney_hybrid.log 2>&1

    echo "[GPU 7] Tuning Cardiff (Hybrid)..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $RERANK_SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/cardiff_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.40 --w_ver 0.25 --w_nli 0.15 --w_acr 0.20 \
        > /data/qbao775/Explanation-Generation/logs/cardiff_hybrid.log 2>&1
) &

wait
echo "Sequential tuning on GPU 4 & 7 complete."

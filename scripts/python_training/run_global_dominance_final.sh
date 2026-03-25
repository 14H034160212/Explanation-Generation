#!/bin/bash
# Universal Global Dominance & Fairness Orchestrator FINAL (v9 - Fixed Metrics & Paths)
# Runs Hybrid-DPO and SFT-Baseline reranking sequentially on GPUs 4, 6, 7.

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"
SFT_MODEL="/data/qbao775/Explanation-Generation/models/rl_sft_llama2_13b_generator"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/fairness
mkdir -p /data/qbao775/Explanation-Generation/logs/fairness

# --- GPU 4 Worker: Law & Med Y1 ---
(
    echo "Starting Law (Hybrid) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.6 --w_ver 0.15 --w_nli 0.2 --w_acr 0.05 \
        > /data/qbao775/Explanation-Generation/logs/law_hybrid.log 2>&1

    echo "Starting Law (SFT-Fair) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/law_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.6 --w_ver 0.15 --w_nli 0.2 --w_acr 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/law_sft_fair.log 2>&1

    echo "Starting Med Y1 (Hybrid) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.45 --w_acr 0.05 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/med_y1_hybrid.log 2>&1

    echo "Starting Med Y1 (SFT-Fair) on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/med_y1_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.45 --w_acr 0.05 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/med_y1_sft_fair.log 2>&1
) &

# --- GPU 7 Worker: Med Y2, Sydney, Cardiff ---
(
    echo "Starting Med Y2 (Hybrid) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.5 --w_ver 0.4 --w_acr 0.05 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/med_y2_hybrid.log 2>&1

    echo "Starting Med Y2 (SFT-Fair) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/med_y2_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.5 --w_ver 0.4 --w_acr 0.05 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/med_y2_sft_fair.log 2>&1

    echo "Starting Sydney (Hybrid) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/sydney_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.55 --w_ver 0.4 --w_acr 0.05 \
        > /data/qbao775/Explanation-Generation/logs/sydney_hybrid.log 2>&1

    echo "Starting Sydney (SFT-Fair) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/sydney_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.55 --w_ver 0.4 --w_acr 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/sydney_sft_fair.log 2>&1

    echo "Starting Cardiff (Hybrid) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/cardiff_hybrid_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.4 --w_acr 0.1 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/cardiff_hybrid.log 2>&1

    echo "Starting Cardiff (SFT-Fair) on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" \
        --lora_model_path "$SFT_MODEL" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/cardiff_sft_best_of_32.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.45 --w_ver 0.4 --w_acr 0.1 --w_nli 0.05 \
        > /data/qbao775/Explanation-Generation/logs/fairness/cardiff_sft_fair.log 2>&1
) &

echo "All Workers Launched. Hybrid-DPO followed by Fair SFT Reranking."
wait
echo "All Fairness and Dominance runs COMPLETE."

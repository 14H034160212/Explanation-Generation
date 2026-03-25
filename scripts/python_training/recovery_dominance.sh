#!/bin/bash
# Recovery script for failed Dominance Tuning jobs
# Sequentially running to avoid OOM

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"
SFT_MODEL="/data/qbao775/Explanation-Generation/models/rl_sft_llama2_13b_generator"

# Use GPU 4 and 7 sequentially to avoid OOM
# We will run these one by one or in very small batches.

# Target: Med Y2 (Re-run for Dominance)
echo "Starting Med Y2 Aggressive NLI Tuning..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_hybrid_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.20 --w_ver 0.25 --w_nli 0.45 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/med_y2_hybrid.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "$SFT_MODEL" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/med_y2_sft_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.20 --w_ver 0.25 --w_nli 0.45 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/fairness/med_y2_sft_fair.log 2>&1 &

wait

# Target: Law (Recovery)
echo "Starting Law Aggressive NLI Tuning..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_hybrid_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.25 --w_ver 0.15 --w_nli 0.50 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/law_hybrid.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "$SFT_MODEL" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/law_sft_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.25 --w_ver 0.15 --w_nli 0.50 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/fairness/law_sft_fair.log 2>&1 &

wait

# Target: Sydney (Recovery/Re-run)
echo "Starting Sydney Aggressive NLI Tuning..."
CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/sydney_hybrid_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.30 --w_ver 0.20 --w_nli 0.40 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/sydney_hybrid.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "$SFT_MODEL" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/fairness/sydney_sft_best_of_32.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.30 --w_ver 0.20 --w_nli 0.40 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/fairness/sydney_sft_fair.log 2>&1 &

wait
echo "Aggressive Tuning Completed."

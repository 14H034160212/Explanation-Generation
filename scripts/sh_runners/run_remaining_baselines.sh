#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-tuning

BASE_MODEL="/data/shared/llama2/llama-2-13b-hf"
VERIFIER_PATH="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"

DOMAINS=("law" "med_y1")
TEST_FILES=(
    "./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
    "./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"
)

for i in "${!DOMAINS[@]}"; do
    LABEL="${DOMAINS[$i]}"
    TEST_FILE="${TEST_FILES[$i]}"

    echo ">>> Evaluating Missing Baselines for Domain: ${LABEL}"
    CUDA_VISIBLE_DEVICES=7,6 python3 rl_evaluation.py \
        --test_data_path "${TEST_FILE}" \
        --verifier_path "${VERIFIER_PATH}" \
        --output_path "./rl_eval_results/baselines_${LABEL}_remaining_eval.json" \
        --dpo_model_path "${BASE_MODEL}" \
        --dpo_lora_path "./rl_dpo_llama2_13b_generator" \
        --dpo_v2_model_path "${BASE_MODEL}" \
        --dpo_v2_lora_path "./rl_dpo_v2_llama2_13b_generator" \
        --ppo_model_path "${BASE_MODEL}" \
        --ppo_lora_path "./rl_ppo_llama2_13b_generator" \
        --device cuda:0 --verifier_device cuda:1 \
        --nli_device cpu \
        --cache_dir cache
done

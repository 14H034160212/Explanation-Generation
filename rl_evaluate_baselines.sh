#!/bin/bash
# Evaluate baseline models (DPO v1, v2, PPO, ILearner) on all domains sequentially to avoid GPU OOM.
# GPU 5 for Generator, GPU 7 for Verifier (sharing with existing v3 evals)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-tuning

BASE_MODEL="/data/shared/llama2/llama-2-13b-hf"
VERIFIER_PATH="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
ILEA_MODEL="./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back"

DOMAINS=("Cardiff" "Sydney" "Law" "Medicine/Medicine_year1" "Medicine/Medicine_year2")
DOMAIN_LABELS=("cardiff" "sydney" "law" "med_y1" "med_y2")

for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    LABEL="${DOMAIN_LABELS[$i]}"
    
    # Map domain to specific file name format
    if [[ "$LABEL" == "cardiff" ]]; then
        TEST_FILE="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
    elif [[ "$LABEL" == "sydney" ]]; then
        TEST_FILE="./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
    elif [[ "$LABEL" == "law" ]]; then
        TEST_FILE="./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
    elif [[ "$LABEL" == "med_y1" ]]; then
        TEST_FILE="./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"
    elif [[ "$LABEL" == "med_y2" ]]; then
        TEST_FILE="./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
    fi

    echo ">>> Evaluating Baselines for Domain: ${LABEL}"
    CUDA_VISIBLE_DEVICES=5,7 python3 rl_evaluation.py \
        --test_data_path "${TEST_FILE}" \
        --verifier_path "${VERIFIER_PATH}" \
        --output_path "./rl_eval_results/baselines_${LABEL}_eval.json" \
        --sft_model_path "${BASE_MODEL}" \
        --sft_lora_path "./rl_sft_llama2_13b_generator" \
        --dpo_model_path "${BASE_MODEL}" \
        --dpo_lora_path "./rl_dpo_llama2_13b_generator" \
        --dpo_v2_model_path "${BASE_MODEL}" \
        --dpo_v2_lora_path "./rl_dpo_v2_llama2_13b_generator" \
        --ppo_model_path "${BASE_MODEL}" \
        --ppo_lora_path "./rl_ppo_llama2_13b_generator" \
        --ilearner_model_path "${ILEA_MODEL}" \
        --ilearner_k 5 \
        --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:1 \
        --cache_dir cache
done

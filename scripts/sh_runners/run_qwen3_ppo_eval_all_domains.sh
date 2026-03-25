#!/bin/bash
# Qwen3 PPO full cross-domain eval on GPU 4
# Evaluates the best PPO model (dpo_v1_beta05 init, most stable KL) 
# across all 4 domains sequentially.
# Runs concurrently with med_y1/y2 evals already on GPU 5/7

source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p ./models/rl_eval_results

PPO_ADAPTER="./models/rl_ppo_qwen3_dpo_v1_beta05_generator"
GPU=4

echo ">>> [$(date)] Starting Qwen3 PPO cross-domain eval on GPU $GPU"
echo ">>> PPO adapter: $PPO_ADAPTER"

DOMAINS=("cardiff" "sydney" "law" "med_y1" "med_y2")
TEST_FILES=(
    "./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
    "./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
    "./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
    "./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"
    "./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
)

for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    test_file="${TEST_FILES[$i]}"
    out="./models/rl_eval_results/qwen3_ppo_${domain}_eval.json"
    log="rl_qwen3_ppo_${domain}_eval.log"

    echo ">>> [$(date)] Evaluating domain: $domain"
    CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n qwen3-rl python3 rl_evaluate_qwen3.py \
        --test_data_path "$test_file" \
        --sft_lora_path "./models/rl_sft_qwen3_8b_generator" \
        --dpo_lora_path "$PPO_ADAPTER" \
        --output_path "$out" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > "$log" 2>&1
    echo ">>> [$(date)] Domain $domain DONE. Exit: $?"
done

echo ">>> [$(date)] All Qwen3 PPO domain evals completed."

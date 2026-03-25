#!/bin/bash
# Pure Logic Scaling Launcher v1
# Implements Multiplicative ACR Gating & Zero-BLEU Logic across all benchmarks.

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/pure_logic
mkdir -p /data/qbao775/Explanation-Generation/logs/pure_logic

# Standard Weights for Pure Logic
W_NLI=0.7
W_VER=0.3
LEN_PEN=0.0002

# 1. Auckland Law - GPU 4
echo "Launching Law (Pure Logic) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic/law_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic/law_pure_logic.log 2>&1 &

# 2. Sydney Biology - GPU 5
echo "Launching Sydney (Pure Logic) on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic/sydney_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic/sydney_pure_logic.log 2>&1 &

# 3. Medicine Year 1 - GPU 6
echo "Launching Med Y1 (Pure Logic) on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic/med1_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic/med1_pure_logic.log 2>&1 &

# 4. Medicine Year 2 - GPU 7
echo "Launching Med Y2 (Pure Logic) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic/med2_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic/med2_pure_logic.log 2>&1 &

# 5. Cardiff Biology - GPU 3
echo "Launching Cardiff (Pure Logic) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic/cardiff_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic/cardiff_pure_logic.log 2>&1 &

echo "All Pure Logic Scaling Experiments launched."

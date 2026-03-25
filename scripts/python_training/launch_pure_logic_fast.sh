#!/bin/bash
# High-Speed Pure Logic Scaling (Batch Size 4, FP16)
PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast
mkdir -p /data/qbao775/Explanation-Generation/logs/pure_logic_fast

W_NLI=0.7
W_VER=0.3
LEN_PEN=0.0002

# 1. Cardiff - GPU 0
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast/cardiff_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --batch_size 4 \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic_fast/cardiff_pure_logic.log 2>&1 &

# 2. Law - GPU 1
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast/law_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --batch_size 4 \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic_fast/law_pure_logic.log 2>&1 &

# 3. Sydney - GPU 2
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast/sydney_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --batch_size 4 \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic_fast/sydney_pure_logic.log 2>&1 &

# 4. Med2 - GPU 3
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast/med2_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --batch_size 4 \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic_fast/med2_pure_logic.log 2>&1 &

# 5. Med1 - GPU 4
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_fast/med1_pure_logic_100.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --batch_size 4 \
    --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
    > /data/qbao775/Explanation-Generation/logs/pure_logic_fast/med1_pure_logic.log 2>&1 &

echo "All high-speed Pure Logic jobs launched."

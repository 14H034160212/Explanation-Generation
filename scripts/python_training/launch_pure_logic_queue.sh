#!/bin/bash
# Hardware-Safe Sequential Pure Logic Scaling
# Runs at full speed (Batch 4, FP16) by queueing on the two completely free GPUs (4 and 6).

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue
mkdir -p /data/qbao775/Explanation-Generation/logs/pure_logic_queue

W_NLI=0.7
W_VER=0.3
LEN_PEN=0.0002
BATCH_SIZE=4

# -------------------------------------------------------------
# GPU 4 Queue: Cardiff -> Law -> Sydney
# -------------------------------------------------------------
(
    echo "Starting Cardiff on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue/cardiff_pure_logic_100.json" \
        --n_samples 32 --device "cuda:0" --batch_size $BATCH_SIZE \
        --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
        > /data/qbao775/Explanation-Generation/logs/pure_logic_queue/cardiff_pure_logic.log 2>&1
        
    echo "Cardiff done. Starting Law on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue/law_pure_logic_100.json" \
        --n_samples 32 --device "cuda:0" --batch_size $BATCH_SIZE \
        --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
        > /data/qbao775/Explanation-Generation/logs/pure_logic_queue/law_pure_logic.log 2>&1

    echo "Law done. Starting Sydney on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue/sydney_pure_logic_100.json" \
        --n_samples 32 --device "cuda:0" --batch_size $BATCH_SIZE \
        --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
        > /data/qbao775/Explanation-Generation/logs/pure_logic_queue/sydney_pure_logic.log 2>&1
) &

# -------------------------------------------------------------
# GPU 6 Queue: Med1 -> Med2
# -------------------------------------------------------------
(
    echo "Starting Med1 on GPU 6..."
    CUDA_VISIBLE_DEVICES=6 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue/med1_pure_logic_100.json" \
        --n_samples 32 --device "cuda:0" --batch_size $BATCH_SIZE \
        --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
        > /data/qbao775/Explanation-Generation/logs/pure_logic_queue/med1_pure_logic.log 2>&1

    echo "Med1 done. Starting Med2 on GPU 6..."
    CUDA_VISIBLE_DEVICES=6 $PYTHON_PATH -u $SCRIPT_PATH \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "$BASE_MODEL" --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/pure_logic_queue/med2_pure_logic_100.json" \
        --n_samples 32 --device "cuda:0" --batch_size $BATCH_SIZE \
        --use_multiplicative_acr --w_nli $W_NLI --w_ver $W_VER --w_len_penalty $LEN_PEN \
        > /data/qbao775/Explanation-Generation/logs/pure_logic_queue/med2_pure_logic.log 2>&1
) &

echo "Queues launched safely on GPU 4 and GPU 6."

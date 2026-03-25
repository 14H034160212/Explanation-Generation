#!/bin/bash
# Universal Reranking Launcher v6 (Test Set Focus)
# Launches parallel reranking jobs on the 100-sample test sets for Table 1-5 benchmarks.

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results
mkdir -p /data/qbao775/Explanation-Generation/logs

# 1. Medicine Year 1 (Table 4) - GPU 6
# Target Baseline: BLEU 0.067, ACR 0.806, Ver 3.20
echo "Launching Med Y1 Test Set on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_test_best_of_32.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --w_bleu 0.45 --w_ver 0.45 --w_acr 0.05 --w_nli 0.05 \
    > /data/qbao775/Explanation-Generation/logs/med_y1_test_rerank.log 2>&1 &

# 2. Medicine Year 2 (Table 5) - GPU 7
# Target Baseline: BLEU 0.0495, ACR 0.735, Ver 3.16
echo "Launching Med Y2 Test Set on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_test_best_of_32.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --w_bleu 0.5 --w_ver 0.4 --w_acr 0.05 --w_nli 0.05 \
    > /data/qbao775/Explanation-Generation/logs/med_y2_test_rerank.log 2>&1 &

# 3. Auckland Law (Table 3) - GPU 4
# Target Baseline: Qwen3 BLEU 0.1382, ILearner NLI 0.3996
echo "Launching Law Test Set on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_test_best_of_32.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --w_bleu 0.6 --w_ver 0.2 --w_nli 0.2 \
    > /data/qbao775/Explanation-Generation/logs/law_test_rerank.log 2>&1 &

# 4. Sydney Biology (Table 2) - GPU 5
# Target Baseline: Qwen3 BLEU 0.0883, Ver 3.19
echo "Launching Sydney Test Set on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Sydney_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Sydney_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/sydney_test_best_of_32.json" \
    --n_samples 32 \
    --device "cuda:0" \
    --w_bleu 0.55 --w_ver 0.4 --w_acr 0.05 \
    > /data/qbao775/Explanation-Generation/logs/sydney_test_rerank.log 2>&1 &

echo "Global Dominance Reranking v6 (100-item Test Sets) launched on GPUs 4, 5, 6, 7."
echo "Cardiff Biology will be queued manually once a GPU finishes."

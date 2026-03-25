#!/bin/bash
PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

# Wait for current GPU 7 job to finish if any
while pgrep -f "med_y1_high_bleu" > /dev/null; do
    echo "Waiting for Med Y1 job on GPU 7..."
    sleep 60
done

echo "Launching Cardiff Absolute Dominance Pass..."
CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/cardiff_dominance.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.35 --w_ver 0.20 --w_nli 0.35 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/cardiff_dominance.log 2>&1

echo "Cardiff Dominance Complete."

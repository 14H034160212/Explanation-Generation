#!/bin/bash
PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

while pgrep -f "law_high_bleu" > /dev/null; do
    echo "Waiting for Law job on GPU 4..."
    sleep 60
done

echo "Launching Med Y2 Final Dominance Pass..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_dominance.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.40 --w_ver 0.20 --w_nli 0.30 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/med_y2_dominance.log 2>&1

echo "Med Y2 Dominance Complete."

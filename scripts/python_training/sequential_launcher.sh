#!/bin/bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-your_key_here}"

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"
BASE_MODEL="/data/qbao775/Explanation-Generation/vicuna-13b"

echo "[09:10] Launching Law on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_high_bleu.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.60 --w_ver 0.10 --w_nli 0.20 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/law_high_bleu.log 2>&1 &

echo "Waiting 3 minutes for Law to load..."
sleep 180

echo "[09:13] Launching Med Y1 on GPU 7..."
CUDA_VISIBLE_DEVICES=7 $PYTHON_PATH -u $SCRIPT_PATH \
    --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
    --base_model_path "$BASE_MODEL" \
    --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
    --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
    --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_high_bleu.json" \
    --n_samples 32 --device "cuda:0" \
    --w_bleu 0.60 --w_ver 0.10 --w_nli 0.20 --w_acr 0.10 \
    > /data/qbao775/Explanation-Generation/logs/med_y1_high_bleu.log 2>&1 &

echo "Sequential jobs launched."

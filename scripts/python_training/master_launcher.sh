#!/bin/bash
# Master Sequential Launcher for Absolute Dominance (v3 - Robust ENV)

export OPENAI_API_KEY="${OPENAI_API_KEY:-your_key_here}"
PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT="/data/qbao775/Explanation-Generation/scripts/python_training/rl_rerank_universal.py"

# --- Chain 1 on GPU 6 ---
(
    export CUDA_VISIBLE_DEVICES=6
    echo "[$(date)] GPU 6 Visible Devices: $CUDA_VISIBLE_DEVICES"
    $PYTHON -u $SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "/data/qbao775/Explanation-Generation/vicuna-13b" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Auckland_law_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/law_high_bleu.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.60 --w_ver 0.10 --w_nli 0.20 --w_acr 0.10 \
        >> /data/qbao775/Explanation-Generation/logs/law_high_bleu.log 2>&1

    echo "[$(date)] Starting Medicine Y2 Dominance on GPU 6"
    $PYTHON -u $SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "/data/qbao775/Explanation-Generation/vicuna-13b" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y2_dominance.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.40 --w_ver 0.20 --w_nli 0.30 --w_acr 0.10 \
        >> /data/qbao775/Explanation-Generation/logs/med_y2_dominance.log 2>&1
) &

# --- Chain 2 on GPU 7 ---
(
    export CUDA_VISIBLE_DEVICES=7
    echo "[$(date)] GPU 7 Visible Devices: $CUDA_VISIBLE_DEVICES"
    $PYTHON -u $SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "/data/qbao775/Explanation-Generation/vicuna-13b" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_high_bleu.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.60 --w_ver 0.10 --w_nli 0.20 --w_acr 0.10 \
        >> /data/qbao775/Explanation-Generation/logs/med_y1_high_bleu.log 2>&1

    echo "[$(date)] Starting Cardiff High-Dominance on GPU 7"
    $PYTHON -u $SCRIPT \
        --test_data_path "/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Biology/Cardiff_Biology_vicuna_13b_finetuned_random_100.json" \
        --base_model_path "/data/qbao775/Explanation-Generation/vicuna-13b" \
        --lora_model_path "/data/qbao775/Explanation-Generation/models/vicuna_13B_Cardiff_biology_all_generator_avg_3_lenexp_10" \
        --verifier_model_path "/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_Cardiff_biology_merged_verifier_way_2" \
        --output_path "/data/qbao775/Explanation-Generation/rl_eval_results/cardiff_dominance.json" \
        --n_samples 32 --device "cuda:0" \
        --w_bleu 0.35 --w_ver 0.20 --w_nli 0.35 --w_acr 0.10 \
        >> /data/qbao775/Explanation-Generation/logs/cardiff_dominance.log 2>&1
) &

echo "Master sequential pipelines (v3) launched on GPUs 6 and 7."

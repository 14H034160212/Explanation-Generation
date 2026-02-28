#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-tuning

echo ">>> Starting Cardiff K=5 Baseline"
CUDA_VISIBLE_DEVICES=6,7 python3 rl_evaluation.py     --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json     --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2     --output_path ./rl_eval_results/baselines_cardiff_k5_eval.json     --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back     --ilearner_k 5     --ilearner_is_legacy     --device cuda:0 --verifier_device cuda:1     --cache_dir cache

echo ">>> Starting Sydney K=5 Baseline"
CUDA_VISIBLE_DEVICES=6,7 python3 rl_evaluation.py     --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json     --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2     --output_path ./rl_eval_results/baselines_sydney_k5_eval.json     --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back     --ilearner_k 5     --ilearner_is_legacy     --device cuda:0 --verifier_device cuda:1     --cache_dir cache

#!/bin/bash
set -e
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
export CUDA_VISIBLE_DEVICES=3
BASE=/data/shared/llama3/llama3/Meta-Llama-3-8B-HF
SFT=./rl_sft_llama3_8b_cardiff_generator
DPO=./rl_dpo_llama3_8b_cardiff_generator
PREF=./rl_preference_data_llama3_cardiff/preference_pairs.json
TEST=./preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
echo "=== [3/4] DPO ==="
$PY scripts/python_training/rl_train_dpo_llama3.py \
  --model_name_or_path $BASE --sft_adapter_path $SFT \
  --preference_data_path $PREF --output_dir $DPO \
  --num_train_epochs 5 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 5e-5 --beta 0.1 \
  --logging_steps 5 --save_strategy no --bf16 True --report_to none
touch .llama3_dpo_DONE
echo "=== [4/4] Eval NLI (SFT vs DPO) ==="
$PY scripts/python_training/rl_evaluate_llama3.py \
  --test_data_path $TEST --base_model_path $BASE \
  --sft_lora_path $SFT --dpo_lora_path $DPO --verifier_path "" \
  --output_path ./rl_eval_results/llama3_cardiff_control_eval.json \
  --device cuda:0 --nli_device cpu
touch .llama3_control_DONE
echo "=== DONE ==="

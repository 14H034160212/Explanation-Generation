#!/bin/bash
set -e
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
export CUDA_VISIBLE_DEVICES=0
VERIFIER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
SFT=./rl_sft_gemma4_e4b_sydney_tierC_generator
DPO=./rl_dpo_gemma4_e4b_sydney_tierC_generator
PREF=./rl_preference_data_gemma4_sydney_tierC

echo "=== [1/4] SFT on Sydney Tier-C ==="
$PY scripts/python_training/rl_train_sft_gemma4.py \
  --model_name_or_path google/gemma-4-E4B-it \
  --data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_train.json \
  --output_dir $SFT \
  --num_train_epochs 3 --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 --learning_rate 2e-4 \
  --bf16 True --report_to none

echo "=== [2/4] build hybrid preference data ==="
$PY scripts/python_training/rl_build_preference_data_nli.py \
  --model_type gemma4 --score_method hybrid \
  --verifier_path $VERIFIER \
  --generator_path google/gemma-4-E4B-it \
  --lora_adapter_path $SFT \
  --data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_train.json \
  --output_path $PREF/preference_pairs.json \
  --num_samples 3 --min_score_gap 0.05 --max_questions 500 \
  --generator_device cuda:0 --nli_device cpu --verifier_device cuda:0

echo "=== [3/4] DPO (lr=5e-5) ==="
$PY scripts/python_training/rl_train_dpo_gemma4.py \
  --model_name_or_path google/gemma-4-E4B-it \
  --sft_adapter_path $SFT \
  --preference_data_path $PREF/preference_pairs.json \
  --output_dir $DPO \
  --num_train_epochs 5 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 5e-5 \
  --bf16 True --report_to none

echo "=== [4/4] eval ==="
$PY scripts/python_training/rl_evaluate_gemma4.py \
  --test_data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_test.json \
  --base_model_path google/gemma-4-E4B-it \
  --sft_lora_path $SFT \
  --dpo_lora_path $DPO \
  --verifier_path $VERIFIER \
  --output_path ./rl_eval_results/gemma4_sydney_tierC_eval.json \
  --device cuda:0 --verifier_device cuda:0 --nli_device cpu

touch /data/qbao775/Explanation-Generation/.tierC_gemma4_sydney_DONE
echo "=== DONE ==="

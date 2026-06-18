#!/bin/bash
set -e
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/llm-tuning/bin/python3
export CUDA_VISIBLE_DEVICES=__GPU__
BASE=/data/shared/llama2/llama-2-13b-hf
VERIFIER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
SFT=./rl_sft_llama2_13b_sydney_tierC_generator
DPO=./rl_dpo_llama2_13b_sydney_tierC_generator
PREF=./rl_preference_data_llama2_sydney_tierC

echo "=== [1/4] SFT (LoRA, single-GPU, grad ckpt) ==="
$PY scripts/python_training/rl_train_sft.py \
  --model_name_or_path $BASE \
  --data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_train.json \
  --output_dir $SFT \
  --num_train_epochs 3 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 2e-4 \
  --bf16 True --gradient_checkpointing True --report_to none

echo "=== [2/4] build hybrid pref (llama2, min_gap 0.03) ==="
$PY scripts/python_training/rl_build_preference_data_nli.py \
  --model_type llama2 --score_method hybrid \
  --generator_path $BASE --lora_adapter_path $SFT \
  --verifier_path $VERIFIER \
  --data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_train.json \
  --output_path $PREF/preference_pairs.json \
  --num_samples 3 --min_score_gap 0.03 --max_questions 500 \
  --generator_device cuda:0 --nli_device cpu --verifier_device cuda:0 --cache_dir ./cache

echo "=== [3/4] DPO (lr 5e-5) ==="
$PY scripts/python_training/rl_train_dpo.py \
  --model_name_or_path $BASE \
  --sft_adapter_path $SFT \
  --preference_data_path $PREF/preference_pairs.json \
  --output_dir $DPO \
  --num_train_epochs 5 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 5e-5 \
  --lora_r 16 --lora_alpha 32 --bf16 True

echo "=== [4/4] eval ==="
$PY scripts/python_training/rl_evaluation.py \
  --test_data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_test.json \
  --sft_model_path $BASE --sft_lora_path $SFT \
  --dpo_model_path $BASE --dpo_lora_path $DPO \
  --verifier_path $VERIFIER \
  --output_path ./rl_eval_results/llama2_sydney_tierC_eval.json \
  --device cuda:0 --verifier_device cuda:0 --nli_device cpu

touch /data/qbao775/Explanation-Generation/.tierC_llama2_sydney_DONE
echo "=== DONE llama2 sydney tierC ==="

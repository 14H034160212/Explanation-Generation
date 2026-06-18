#!/bin/bash
# Resume Gemma-4 Cardiff Tier-C: build_pref from index 250 -> merge -> DPO -> eval
set -e
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
export CUDA_VISIBLE_DEVICES=7
VERIFIER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
SFT=./rl_sft_gemma4_e4b_cardiff_tierC_generator
DPO=./rl_dpo_gemma4_e4b_cardiff_tierC_generator
PREF=./rl_preference_data_gemma4_cardiff_tierC

echo "=== [1/4] build remaining pref pairs (questions 250-500) ==="
$PY scripts/python_training/rl_build_preference_data_nli.py \
  --model_type gemma4 --score_method hybrid \
  --verifier_path $VERIFIER \
  --generator_path google/gemma-4-E4B-it \
  --lora_adapter_path $SFT \
  --data_path ./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json \
  --output_path $PREF/preference_pairs_part2.json \
  --num_samples 3 --min_score_gap 0.05 --max_questions 250 --start_index 250 \
  --generator_device cuda:0 --nli_device cpu --verifier_device cuda:0

echo "=== [2/4] merge part1 + part2 ==="
$PY - << 'MERGE'
import json
p1 = json.load(open('rl_preference_data_gemma4_cardiff_tierC/preference_pairs_part1.json'))
p2 = json.load(open('rl_preference_data_gemma4_cardiff_tierC/preference_pairs_part2.json'))
merged = p1 + p2
json.dump(merged, open('rl_preference_data_gemma4_cardiff_tierC/preference_pairs.json','w'), indent=2)
print(f"merged: {len(p1)} + {len(p2)} = {len(merged)} pairs")
MERGE

echo "=== [3/4] DPO ==="
$PY scripts/python_training/rl_train_dpo_gemma4.py \
  --model_name_or_path google/gemma-4-E4B-it \
  --sft_adapter_path $SFT \
  --preference_data_path $PREF/preference_pairs.json \
  --output_dir $DPO \
  --num_train_epochs 5 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --bf16 True --report_to none

echo "=== [4/4] eval (SFT vs DPO on Tier-C test set) ==="
$PY scripts/python_training/rl_evaluate_gemma4.py \
  --test_data_path ./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json \
  --base_model_path google/gemma-4-E4B-it \
  --sft_lora_path $SFT \
  --dpo_lora_path $DPO \
  --verifier_path $VERIFIER \
  --output_path ./rl_eval_results/gemma4_cardiff_tierC_eval.json \
  --device cuda:0 --verifier_device cuda:0 --nli_device cpu

echo "=== DONE gemma4 cardiff tierC (resumed) ==="

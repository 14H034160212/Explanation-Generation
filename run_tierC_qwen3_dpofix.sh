#!/bin/bash
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/qwen3-rl/bin/python
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VERIFIER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2

for CORPUS in cardiff sydney; do
  Cap=$(echo $CORPUS | sed 's/./\U&/')
  SFT=./rl_sft_qwen3_8b_${CORPUS}_tierC_generator
  DPO=./rl_dpo_qwen3_8b_${CORPUS}_tierC_generator
  PREF=./rl_preference_data_qwen3_${CORPUS}_tierC/preference_pairs.json
  echo "===== $CORPUS : DPO (lr=5e-5) ====="
  $PY scripts/python_training/rl_train_dpo_qwen3.py \
    --model_name_or_path /data/shared/qwen3/Qwen3-8B \
    --sft_adapter_path $SFT \
    --preference_data_path $PREF \
    --output_dir $DPO \
    --num_train_epochs 5 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 --learning_rate 5e-5 \
    --bf16 True --report_to none || { echo "DPO FAILED $CORPUS"; continue; }
  echo "===== $CORPUS : eval ====="
  $PY scripts/python_training/rl_evaluate_qwen3.py \
    --test_data_path ./preference_data/Paul_new_data/${Cap}_tierC_generator_test.json \
    --base_model_path /data/shared/qwen3/Qwen3-8B \
    --sft_lora_path $SFT --dpo_lora_path $DPO \
    --verifier_path $VERIFIER \
    --output_path ./rl_eval_results/qwen3_${CORPUS}_tierC_eval.json \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu || echo "EVAL FAILED $CORPUS"
done
touch /data/qbao775/Explanation-Generation/.tierC_qwen3_dpofix_DONE
echo "=== QWEN3 DPOFIX DONE ==="

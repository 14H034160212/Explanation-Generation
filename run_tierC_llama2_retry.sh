#!/bin/bash
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/llm-tuning/bin/python3
BASE=/data/shared/llama2/llama-2-13b-hf
VERIFIER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[retry $(date '+%m-%d %H:%M')] $*"; }

pick_gpu(){ # $1=min MiB ; waits until a gpu has >= min free, echoes idx
  while true; do
    read idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr ',' ' ')
    if [ "${free:-0}" -ge "$1" ]; then echo "$idx"; return 0; fi
    log "no GPU >=${1}MiB free (best=${free}MiB), waiting..."; sleep 150
  done
}

run_retry(){ # $1=minMiB $2=label ; rest=command
  local min=$1 label=$2; shift 2
  for attempt in $(seq 1 12); do
    local g=$(pick_gpu $min)
    log "$label attempt $attempt on GPU $g"
    CUDA_VISIBLE_DEVICES=$g "$@" && { log "$label OK"; return 0; }
    log "$label FAILED attempt $attempt; wait 240s"; sleep 240
  done
  log "$label GAVE UP"; return 1
}

# ---- Sydney: DPO done, needs EVAL ----
if [ ! -f rl_eval_results/llama2_sydney_tierC_eval.json ]; then
  run_retry 42000 "sydney-eval" $PY scripts/python_training/rl_evaluation.py \
    --test_data_path ./preference_data/Paul_new_data/Sydney_tierC_generator_test.json \
    --sft_model_path $BASE --sft_lora_path ./rl_sft_llama2_13b_sydney_tierC_generator \
    --dpo_model_path $BASE --dpo_lora_path ./rl_dpo_llama2_13b_sydney_tierC_generator \
    --verifier_path $VERIFIER --output_path ./rl_eval_results/llama2_sydney_tierC_eval.json \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu
fi
[ -f rl_eval_results/llama2_sydney_tierC_eval.json ] && touch .tierC_llama2_sydney_DONE

# ---- Cardiff: pref -> dpo -> eval ----
if [ ! -f rl_preference_data_llama2_cardiff_tierC/preference_pairs.json ]; then
  run_retry 42000 "cardiff-pref" $PY scripts/python_training/rl_build_preference_data_nli.py \
    --model_type llama2 --score_method hybrid --generator_path $BASE \
    --lora_adapter_path ./rl_sft_llama2_13b_cardiff_tierC_generator --verifier_path $VERIFIER \
    --data_path ./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json \
    --output_path ./rl_preference_data_llama2_cardiff_tierC/preference_pairs.json \
    --num_samples 3 --min_score_gap 0.03 --max_questions 500 \
    --generator_device cuda:0 --nli_device cpu --verifier_device cuda:0 --cache_dir ./cache
fi
if [ -f rl_preference_data_llama2_cardiff_tierC/preference_pairs.json ] && [ ! -f rl_dpo_llama2_13b_cardiff_tierC_generator/adapter_model.bin ]; then
  run_retry 30000 "cardiff-dpo" $PY scripts/python_training/rl_train_dpo.py \
    --model_name_or_path $BASE --sft_adapter_path ./rl_sft_llama2_13b_cardiff_tierC_generator \
    --preference_data_path ./rl_preference_data_llama2_cardiff_tierC/preference_pairs.json \
    --output_dir ./rl_dpo_llama2_13b_cardiff_tierC_generator \
    --num_train_epochs 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 --lora_r 16 --lora_alpha 32 --bf16 True
fi
if [ -f rl_dpo_llama2_13b_cardiff_tierC_generator/adapter_model.bin ] && [ ! -f rl_eval_results/llama2_cardiff_tierC_eval.json ]; then
  run_retry 42000 "cardiff-eval" $PY scripts/python_training/rl_evaluation.py \
    --test_data_path ./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json \
    --sft_model_path $BASE --sft_lora_path ./rl_sft_llama2_13b_cardiff_tierC_generator \
    --dpo_model_path $BASE --dpo_lora_path ./rl_dpo_llama2_13b_cardiff_tierC_generator \
    --verifier_path $VERIFIER --output_path ./rl_eval_results/llama2_cardiff_tierC_eval.json \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu
fi
[ -f rl_eval_results/llama2_cardiff_tierC_eval.json ] && touch .tierC_llama2_cardiff_DONE
touch .tierC_llama2_retry_FINISHED
log "retry launcher finished"

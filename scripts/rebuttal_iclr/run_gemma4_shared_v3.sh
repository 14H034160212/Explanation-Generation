#!/bin/bash
# Gemma 4 shared-pool ablation, verifier-free version. The Alpaca-7B verifier is
# no longer on disk, so only NLI-only, masked-NLI and a size-matched NLI control
# (nli_n93: 93 random NLI pairs = masknli's pair count) are run. Verifier scores
# are recorded as NaN. Each GPU step runs on GPU 3 or GPU 4 when it has room.
cd /data/qbao775/Explanation-Generation
export HF_HUB_OFFLINE=1
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
SFT=./rl_sft_gemma4_e4b_merged_generator
declare -A TEST=(
 [cardiff]=./preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
 [sydney]=./preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
 [law]=./preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json
 [medy1]=./preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json
 [medy2]=./preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json)
DOMS="cardiff medy2 medy1 law sydney"
NEED=30000
free_mb() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits; }
pick_gpu() {
  while true; do
    [ "$(free_mb 3)" -ge "$1" ] && { echo 3; return; }
    [ "$(free_mb 4)" -ge "$(( $1 + 5000 ))" ] && { echo 4; return; }
    sleep 120
  done
}
run_gpu() {
  local need=$1; shift
  for attempt in 1 2 3 4 5 6; do
    local g=$(pick_gpu $need)
    echo "[$(date)] GPU $g (attempt $attempt): $*" | cut -c1-220
    CUDA_VISIBLE_DEVICES=$g "$@" && return 0
    echo "[$(date)] failed on GPU $g; waiting before retry"; sleep 300
  done
  return 1
}
EV="--verifier_path none --device cuda:0 --verifier_device cuda:0 --nli_device cpu"
for d in $DOMS; do
  out=rl_eval_results/g4shared_sft_${d}_eval.json
  [ -f $out ] || run_gpu $NEED $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} \
     --sft_lora_path $SFT --output_path $out $EV
done
train_eval() {  # $1 variant, $2 seed ("" = default 42)
  local v=$1 s=$2 tag=$1; [ -n "$s" ] && tag=${1}_s$2
  local DPO=./rl_dpo_gemma4_shared_${tag}
  if [ ! -f $DPO/adapter_model.safetensors ]; then
    run_gpu $NEED $PY scripts/python_training/rl_train_dpo_gemma4.py --model_name_or_path google/gemma-4-E4B-it \
      --sft_adapter_path $SFT --preference_data_path rl_preference_data_gemma4_shared/$v/preference_pairs.json \
      --output_dir $DPO --num_train_epochs 5 --per_device_train_batch_size 1 ${s:+--seed $s} \
      --gradient_accumulation_steps 8 --learning_rate 5e-5 --bf16 True --report_to none || return 1
  fi
  for d in $DOMS; do
    out=rl_eval_results/g4shared_${tag}_${d}_eval.json
    [ -f $out ] || run_gpu $NEED $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} --skip_sft \
      --sft_lora_path $SFT --dpo_lora_path $DPO --output_path $out $EV
  done
  touch .g4shared_${tag}_DONE
}
for v in masknli nli nli_n93; do train_eval $v ""; done
touch .g4shared_ALL_DONE
for s in 2 3; do for v in masknli nli_n93; do train_eval $v $s; done; done
touch .g4shared_SEEDS_DONE
echo "[$(date)] ALL DONE"

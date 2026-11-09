#!/bin/bash
# Verifier-dependent half of the Gemma 4 shared-pool ablation, using the REPRODUCED
# verifier (models/verifier_way2_repro). Runs alongside run_gemma4_shared_v3.sh;
# prefers GPU 4, falls back to GPU 3, never any other GPU.
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
    [ "$(free_mb 4)" -ge "$(( $1 + 5000 ))" ] && { echo 4; return; }
    [ "$(free_mb 3)" -ge "$1" ] && { echo 3; return; }
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
until [ -f models/verifier_way2_repro/model.safetensors.index.json ] || [ -f models/verifier_way2_repro/model.safetensors ]; do
  grep -q Traceback logs/verifier_repro.log && { echo "verifier training FAILED"; exit 1; }; sleep 60; done
sleep 60   # let save_pretrained finish writing
[ -f rl_eval_results/rebuttal_verifier_repro_check.json ] || run_gpu 18000 $PY scripts/rebuttal_iclr/check_verifier_repro.py || exit 1
run_gpu 18000 $PY scripts/rebuttal_iclr/gemma4_pool_candidates.py ver || exit 1
python3 scripts/rebuttal_iclr/gemma4_pool_pairs.py || exit 1
EV="--verifier_path none --device cuda:0 --verifier_device cuda:0 --nli_device cpu"
train_eval() {
  local v=$1 tag=$1 DPO=./rl_dpo_gemma4_shared_$1
  if [ ! -f $DPO/adapter_model.safetensors ]; then
    run_gpu $NEED $PY scripts/python_training/rl_train_dpo_gemma4.py --model_name_or_path google/gemma-4-E4B-it \
      --sft_adapter_path $SFT --preference_data_path rl_preference_data_gemma4_shared/$v/preference_pairs.json \
      --output_dir $DPO --num_train_epochs 5 --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 --learning_rate 5e-5 --bf16 True --report_to none || return 1
  fi
  for d in $DOMS; do
    out=rl_eval_results/g4shared_${tag}_${d}_eval.json
    [ -f $out ] || run_gpu $NEED $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} --skip_sft \
      --sft_lora_path $SFT --dpo_lora_path $DPO --output_path $out $EV
  done
  touch .g4shared_${tag}_DONE
}
for v in hm hm_dom ver ha ha_w25 ha_w75; do train_eval $v; done
touch .g4shared_VER_DONE
# wait for the verifier-free half, then add reproduced-verifier scores to every eval file
until [ -f .g4shared_SEEDS_DONE ]; do sleep 300; done
run_gpu 18000 $PY scripts/rebuttal_iclr/rescore_verifier_repro.py
echo "[$(date)] VER HALF DONE"

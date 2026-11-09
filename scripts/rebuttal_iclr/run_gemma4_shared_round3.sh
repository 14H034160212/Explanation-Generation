#!/bin/bash
# Round 3 (pre-specified): length-matched judge reward (|len diff|<=15) with SimPO, 3 seeds;
# length-matched judge+NLI. Judges: Qwen2.5 / Llama-3.1 / Gemma-4-31B. GPU 3/4 only.
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
EV="--verifier_path none --device cuda:0 --verifier_device cuda:0 --nli_device cpu"
evals() {  # $1 tag
  local DPO=./rl_dpo_gemma4_shared_$1
  for d in $DOMS; do
    out=rl_eval_results/g4shared_$1_${d}_eval.json
    [ -f $out ] || run_gpu $NEED $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} --skip_sft \
      --sft_lora_path $SFT --dpo_lora_path $DPO --output_path $out $EV
  done
}
simpo() {  # $1 tag, $2 pool, $3 seed
  local DPO=./rl_dpo_gemma4_shared_$1
  [ -f $DPO/adapter_model.safetensors ] || run_gpu $NEED $PY scripts/python_training/rl_train_simpo_gemma4.py \
    --sft_adapter $SFT --pref rl_preference_data_gemma4_shared/$2/preference_pairs.json --out $DPO \
    --epochs 3 --lr 5e-5 --beta 2.0 --gamma 1.0 --seed $3 --device cuda:0 || return 1
  evals $1
}
( simpo judge_lm judge_lm 42; simpo judge_lm_s2 judge_lm 2 ) &
sleep 240
( simpo judgenli_lm judgenli_lm 42; simpo judge_lm_s3 judge_lm 3 ) &
wait
touch .g4shared_ROUND3_TRAIN_DONE
python3 scripts/rebuttal_iclr/g4shared_judge_input.py || exit 1
NEW="JudgeLM:SFT,JudgeLMs2:SFT,JudgeLMs3:SFT,JudgeNLILM:SFT,JudgeNLILM:NLI,JudgeLM:JudgeSimPO"
for J in Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct; do
  run_gpu 22000 $PY scripts/rebuttal_iclr/judge_hf.py --judge $J --pairs $NEW \
    --gens rl_eval_results/g4shared_judge_generations.json --out rl_eval_results/rebuttal_g4shared_judge.json
done
run_gpu 26000 $PY scripts/rebuttal_iclr/judge_hf.py --judge google/gemma-4-31B-it --load4 --pairs $NEW \
  --gens rl_eval_results/g4shared_judge_generations.json --out rl_eval_results/rebuttal_g4shared_judge.json
touch .g4shared_ROUND3_DONE
echo "[$(date)] ROUND3 DONE"

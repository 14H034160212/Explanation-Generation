#!/bin/bash
# Gemma 4 shared-pool ablation (rebuttal). GPU 3 ONLY.
# gen (already launched separately) -> verifier -> pairs -> [wait for GPU 3 room]
# -> SFT eval once -> per variant: DPO train + 5-domain eval (DPO only).
cd /data/qbao775/Explanation-Generation
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
VER=./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
SFT=./rl_sft_gemma4_e4b_merged_generator
declare -A TEST=(
 [cardiff]=./preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
 [sydney]=./preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
 [law]=./preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json
 [medy1]=./preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json
 [medy2]=./preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json)
DOMS="cardiff medy2 medy1 law sydney"

echo "[$(date)] wait for candidate generation"
until grep -q "gen done" logs/g4pool_gen.log; do
  if grep -q Traceback logs/g4pool_gen.log; then echo "gen FAILED"; exit 1; fi; sleep 60; done
echo "[$(date)] verifier pass"
$PY scripts/rebuttal_iclr/gemma4_pool_candidates.py ver || exit 1
python3 scripts/rebuttal_iclr/gemma4_pool_pairs.py || exit 1

echo "[$(date)] waiting for >=60GB free on GPU 3 (LayoutLM run)"
until [ "$(nvidia-smi -i 3 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 60000 ]; do sleep 120; done

for d in $DOMS; do
  out=rl_eval_results/g4shared_sft_${d}_eval.json
  [ -f $out ] || $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} \
     --sft_lora_path $SFT --verifier_path $VER --output_path $out --device cuda:0 --verifier_device cuda:0 --nli_device cpu
done

for v in hm nli masknli ver hm_dom ha ha_w25 ha_w75; do
  DPO=./rl_dpo_gemma4_shared_${v}
  if [ ! -f $DPO/adapter_model.safetensors ]; then
    echo "[$(date)] train $v"
    $PY scripts/python_training/rl_train_dpo_gemma4.py --model_name_or_path google/gemma-4-E4B-it \
      --sft_adapter_path $SFT --preference_data_path rl_preference_data_gemma4_shared/$v/preference_pairs.json \
      --output_dir $DPO --num_train_epochs 5 --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 --learning_rate 5e-5 --bf16 True --report_to none || { echo "train $v FAILED"; continue; }
  fi
  for d in $DOMS; do
    out=rl_eval_results/g4shared_${v}_${d}_eval.json
    [ -f $out ] && continue
    echo "[$(date)] eval $v $d"
    $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} --skip_sft \
      --sft_lora_path $SFT --dpo_lora_path $DPO --verifier_path $VER --output_path $out \
      --device cuda:0 --verifier_device cuda:0 --nli_device cpu
  done
  touch .g4shared_${v}_DONE
done
touch .g4shared_ALL_DONE
echo "[$(date)] ALL DONE"

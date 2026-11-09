#!/bin/bash
# Extra training seeds for the key Gemma 4 shared-pool variants. Starts after the main queue.
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
until [ -f .g4shared_ALL_DONE ]; do sleep 120; done
for seed in 2 3; do for v in masknli hm nli; do
  DPO=./rl_dpo_gemma4_shared_${v}_s${seed}
  [ -f $DPO/adapter_model.safetensors ] || $PY scripts/python_training/rl_train_dpo_gemma4.py --model_name_or_path google/gemma-4-E4B-it \
      --sft_adapter_path $SFT --preference_data_path rl_preference_data_gemma4_shared/$v/preference_pairs.json \
      --output_dir $DPO --num_train_epochs 5 --per_device_train_batch_size 1 --seed $seed \
      --gradient_accumulation_steps 8 --learning_rate 5e-5 --bf16 True --report_to none || continue
  for d in cardiff medy2 medy1 law sydney; do
    out=rl_eval_results/g4shared_${v}_s${seed}_${d}_eval.json
    [ -f $out ] || $PY scripts/python_training/rl_evaluate_gemma4.py --test_data_path ${TEST[$d]} --skip_sft \
      --sft_lora_path $SFT --dpo_lora_path $DPO --verifier_path $VER --output_path $out \
      --device cuda:0 --verifier_device cuda:0 --nli_device cpu
  done
done; done
touch .g4shared_SEEDS_DONE

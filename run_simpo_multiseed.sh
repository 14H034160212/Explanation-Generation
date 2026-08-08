#!/bin/bash
set -e
cd /data/qbao775/Explanation-Generation
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
export CUDA_VISIBLE_DEVICES=2
PREF=./rl_preference_data_gemma4_cardiff_judge/preference_pairs_lenmatched.json
rm -f rl_eval_results/p3_multiseed_generations.json
for s in 1 2 3; do
  echo "=== SEED $s: train ==="
  $PY scripts/python_training/rl_train_simpo_gemma4.py \
    --pref $PREF --seed $s --device cuda:0 \
    --epochs 3 --lr 5e-5 --beta 2.0 --gamma 1.0 \
    --out ./rl_dpo_gemma4_e4b_cardiff_simpo_seed$s
  echo "=== SEED $s: generate 191 ==="
  SEED=$s $PY scripts/simpo_multiseed_gen.py
done
touch .simpo_multiseed_traingen_DONE
echo "=== ALL SEEDS TRAINED + GENERATED ==="

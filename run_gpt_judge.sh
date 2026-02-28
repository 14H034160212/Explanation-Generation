#!/bin/bash
# GPT-4o-mini pairwise judge: compare SFT vs RL models on Cardiff
# Runs 3 comparisons: SFT vs DPO-v2, SFT vs PPO, DPO-v2 vs PPO

PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
WORKDIR="/data/qbao775/Explanation-Generation"
cd "$WORKDIR"

# Set OPENAI_API_KEY in your environment before running this script.
# e.g.: export OPENAI_API_KEY="sk-..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Please export it before running."
    exit 1
fi

TEST_DATA="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
FULL_EVAL="./rl_eval_results/full_eval_new_metrics_nli_large.json"

echo "[$(date '+%H:%M:%S')] === GPT-4o-mini Judge: Cardiff (LLaMA-2 models) ==="

echo "[$(date '+%H:%M:%S')] Running SFT vs DPO-v2..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$FULL_EVAL" --model_a_name "Baseline-SFT (K=1)" \
    --model_b_json "$FULL_EVAL" --model_b_name "RL-DPO-v2 (K=1)" \
    --test_data_json "$TEST_DATA" \
    --output_path ./rl_eval_results/gpt_judge_sft_vs_dpov2_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] SFT vs DPO-v2 done."

echo "[$(date '+%H:%M:%S')] Running SFT vs PPO..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$FULL_EVAL" --model_a_name "Baseline-SFT (K=1)" \
    --model_b_json "$FULL_EVAL" --model_b_name "RL-PPO (K=1)" \
    --test_data_json "$TEST_DATA" \
    --output_path ./rl_eval_results/gpt_judge_sft_vs_ppo_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] SFT vs PPO done."

echo "[$(date '+%H:%M:%S')] Running DPO-v2 vs PPO..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$FULL_EVAL" --model_a_name "RL-DPO-v2 (K=1)" \
    --model_b_json "$FULL_EVAL" --model_b_name "RL-PPO (K=1)" \
    --test_data_json "$TEST_DATA" \
    --output_path ./rl_eval_results/gpt_judge_dpov2_vs_ppo_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] DPO-v2 vs PPO done."

echo "[$(date '+%H:%M:%S')] === All GPT judge comparisons complete ==="

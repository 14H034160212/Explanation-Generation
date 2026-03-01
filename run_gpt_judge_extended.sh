#!/bin/bash
# Extended GPT-4o-mini pairwise judge:
#   - Sydney: SFT vs DPO-v2, SFT vs PPO, DPO-v2 vs PPO (LLaMA-2)
#   - Cardiff Qwen3: Qwen3-SFT vs Qwen3-DPO

PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
WORKDIR="/data/qbao775/Explanation-Generation"
cd "$WORKDIR"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

SYDNEY_DATA="./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
SYDNEY_EVAL="./rl_eval_results/sydney_eval_nli.json"
CARDIFF_DATA="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
QWEN3_EVAL="./rl_eval_results/qwen3_ppo_cardiff_eval.json"

echo "[$(date '+%H:%M:%S')] === GPT-4o-mini Judge: Sydney (LLaMA-2) ==="

echo "[$(date '+%H:%M:%S')] Sydney: SFT vs DPO-v2..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$SYDNEY_EVAL" --model_a_name "Baseline-SFT (K=1)" \
    --model_b_json "$SYDNEY_EVAL" --model_b_name "RL-DPO-v2 (K=1)" \
    --test_data_json "$SYDNEY_DATA" \
    --output_path ./rl_eval_results/gpt_judge_sft_vs_dpov2_sydney.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Sydney SFT vs DPO-v2 done."

echo "[$(date '+%H:%M:%S')] Sydney: SFT vs PPO..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$SYDNEY_EVAL" --model_a_name "Baseline-SFT (K=1)" \
    --model_b_json "$SYDNEY_EVAL" --model_b_name "RL-PPO (K=1)" \
    --test_data_json "$SYDNEY_DATA" \
    --output_path ./rl_eval_results/gpt_judge_sft_vs_ppo_sydney.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Sydney SFT vs PPO done."

echo "[$(date '+%H:%M:%S')] Sydney: DPO-v2 vs PPO..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$SYDNEY_EVAL" --model_a_name "RL-DPO-v2 (K=1)" \
    --model_b_json "$SYDNEY_EVAL" --model_b_name "RL-PPO (K=1)" \
    --test_data_json "$SYDNEY_DATA" \
    --output_path ./rl_eval_results/gpt_judge_dpov2_vs_ppo_sydney.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Sydney DPO-v2 vs PPO done."

echo "[$(date '+%H:%M:%S')] === GPT-4o-mini Judge: Cardiff Qwen3 ==="

echo "[$(date '+%H:%M:%S')] Qwen3: SFT vs DPO..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$QWEN3_EVAL" --model_a_name "Qwen3-8B-SFT" \
    --model_b_json "$QWEN3_EVAL" --model_b_name "Qwen3-8B-DPO" \
    --test_data_json "$CARDIFF_DATA" \
    --output_path ./rl_eval_results/gpt_judge_qwen3_sft_vs_dpo_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Qwen3 SFT vs DPO done."

echo "[$(date '+%H:%M:%S')] === All extended GPT judge comparisons complete ==="

#!/bin/bash
# GPT-4o-mini judge: compare our best RL model (DPO v2) vs commercial GPT models
# Also evaluates GPT-3.5 on Sydney domain

PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
WORKDIR="/data/qbao775/Explanation-Generation"
cd "$WORKDIR"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

CARDIFF_DATA="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
SYDNEY_DATA="./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
LLAMA2_EVAL="./rl_eval_results/full_eval_new_metrics_nli_large.json"
GPT4OMINI_CARDIFF="./rl_eval_results/gpt4o_mini_cardiff_eval.json"
GPT4OMINI_SYDNEY="./rl_eval_results/gpt4o_mini_sydney_eval.json"
GPT35_CARDIFF="./rl_eval_results/gpt35_cardiff_eval.json"

echo "[$(date '+%H:%M:%S')] === GPT judge: DPO-v2 vs commercial models ==="

echo "[$(date '+%H:%M:%S')] Cardiff: DPO-v2 vs GPT-4o-mini..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$LLAMA2_EVAL"      --model_a_name "RL-DPO-v2 (K=1)" \
    --model_b_json "$GPT4OMINI_CARDIFF" --model_b_name "GPT-4o-mini" \
    --test_data_json "$CARDIFF_DATA" \
    --output_path ./rl_eval_results/gpt_judge_dpov2_vs_gpt4omini_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Cardiff DPO-v2 vs GPT-4o-mini done."

echo "[$(date '+%H:%M:%S')] Cardiff: DPO-v2 vs GPT-3.5..."
$PYTHON scripts/llm_judge.py \
    --model_a_json "$LLAMA2_EVAL"   --model_a_name "RL-DPO-v2 (K=1)" \
    --model_b_json "$GPT35_CARDIFF" --model_b_name "GPT-3.5" \
    --test_data_json "$CARDIFF_DATA" \
    --output_path ./rl_eval_results/gpt_judge_dpov2_vs_gpt35_cardiff.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Cardiff DPO-v2 vs GPT-3.5 done."

echo "[$(date '+%H:%M:%S')] Sydney: DPO-v2 vs GPT-4o-mini..."
SYDNEY_LLAMA2="./rl_eval_results/sydney_eval_nli.json"
$PYTHON scripts/llm_judge.py \
    --model_a_json "$SYDNEY_LLAMA2"    --model_a_name "RL-DPO-v2 (K=1)" \
    --model_b_json "$GPT4OMINI_SYDNEY" --model_b_name "GPT-4o-mini" \
    --test_data_json "$SYDNEY_DATA" \
    --output_path ./rl_eval_results/gpt_judge_dpov2_vs_gpt4omini_sydney.json \
    --limit 100
echo "[$(date '+%H:%M:%S')] Sydney DPO-v2 vs GPT-4o-mini done."

echo "[$(date '+%H:%M:%S')] === All RL vs GPT judge comparisons complete ==="

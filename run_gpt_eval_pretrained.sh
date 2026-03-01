#!/bin/bash
# Evaluate pre-generated GPT-4/GPT-3.5 explanations + generate with GPT-4o-mini
# Run on GPU 7 after Law K5 finishes (PPO v2 verifier ~14GB, room for 41GB more)

PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
WORKDIR="/data/qbao775/Explanation-Generation"
cd "$WORKDIR"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

VERIFIER="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
DEVICE="cuda:0"

echo "[$(date '+%H:%M:%S')] === Evaluating pre-generated GPT-4 explanations ==="

CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/eval_pretrained_explanations.py \
    --input_path ./Paul_new_data/Cardiff/Cardiff_gpt4_random_100.json \
    --model_name "GPT-4" \
    --verifier_path "$VERIFIER" \
    --output_path ./rl_eval_results/gpt4_cardiff_eval.json \
    --verifier_device "$DEVICE"
echo "[$(date '+%H:%M:%S')] GPT-4 Cardiff done."

CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/eval_pretrained_explanations.py \
    --input_path ./Paul_new_data/Cardiff/Cardiff_gpt35_random_100.json \
    --model_name "GPT-3.5" \
    --verifier_path "$VERIFIER" \
    --output_path ./rl_eval_results/gpt35_cardiff_eval.json \
    --verifier_device "$DEVICE"
echo "[$(date '+%H:%M:%S')] GPT-3.5 Cardiff done."

CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/eval_pretrained_explanations.py \
    --input_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
    --model_name "GPT-4" \
    --verifier_path "$VERIFIER" \
    --output_path ./rl_eval_results/gpt4_sydney_eval.json \
    --verifier_device "$DEVICE"
echo "[$(date '+%H:%M:%S')] GPT-4 Sydney done."

echo "[$(date '+%H:%M:%S')] === Generating with GPT-4o-mini and evaluating ==="

CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/gpt_generate_and_eval.py \
    --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
    --model_name "GPT-4o-mini" \
    --openai_model gpt-4o-mini \
    --verifier_path "$VERIFIER" \
    --output_path ./rl_eval_results/gpt4o_mini_cardiff_eval.json \
    --verifier_device "$DEVICE"
echo "[$(date '+%H:%M:%S')] GPT-4o-mini Cardiff done."

CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/gpt_generate_and_eval.py \
    --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
    --model_name "GPT-4o-mini" \
    --openai_model gpt-4o-mini \
    --verifier_path "$VERIFIER" \
    --output_path ./rl_eval_results/gpt4o_mini_sydney_eval.json \
    --verifier_device "$DEVICE"
echo "[$(date '+%H:%M:%S')] GPT-4o-mini Sydney done."

echo "[$(date '+%H:%M:%S')] === All GPT eval tasks complete ==="

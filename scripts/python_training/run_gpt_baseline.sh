#!/bin/bash
# Generate GPT-4o-mini baseline for UK Medicine Year 2
export CUDA_VISIBLE_DEVICES=7

PYTHON_PATH="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
SCRIPT_PATH="/data/qbao775/Explanation-Generation/scripts/gpt_generate_and_eval.py"

# Domain: Med Year 2
DOMAIN="Medicine_year2"
TEST_DATA="/data/qbao775/Explanation-Generation/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
VERIFIER="/data/qbao775/Explanation-Generation/models/qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2"
OUTPUT="/data/qbao775/Explanation-Generation/rl_eval_results/gpt4o_mini_med_y2_eval.json"

echo "Launching GPT-4o-mini generation for $DOMAIN..."
$PYTHON_PATH $SCRIPT_PATH \
    --test_data_path "$TEST_DATA" \
    --model_name "GPT-4o-mini" \
    --openai_model "gpt-4o-mini" \
    --verifier_path "$VERIFIER" \
    --output_path "$OUTPUT" \
    --verifier_device "cuda:0" \
    --nli_device "cuda:0" \
    

echo "GPT-4o-mini Baseline Generation Complete."

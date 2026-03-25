#!/bin/bash
# ============================================================
# DPO + Thinking Mode Evaluation for Qwen3-8B
#
# Evaluates the Hybrid-DPO Qwen3-8B model with chain-of-thought
# thinking mode enabled (enable_thinking=True). Compares SFT+Thinking
# vs DPO+Thinking to measure whether DPO alignment is additive with
# chain-of-thought reasoning.
#
# Motivation:
#   - SFT+Thinking improves Sydney NLI by +3.3% vs SFT standard
#   - DPO+Standard improves NLI by ~0-20% (architecture-dependent)
#   - DPO+Thinking may compound both gains if reasoning helps
#     ground the answer more precisely
#
# Usage:
#   bash scripts/sh_runners/run_dpo_thinking_eval_qwen3.sh [GPU_ID] [DPO_ADAPTER_PATH]
#
# Default DPO adapter: models/rl_dpo_multiplicative_acr_qwen3_8b_generator
# Override: bash run_dpo_thinking_eval_qwen3.sh 7 models/rl_dpo_n5_qwen3_8b_generator
#
# Outputs:
#   rl_eval_results/qwen3_dpo_thinking_{domain}_eval.json
# ============================================================
set -euo pipefail

GPU=${1:-6}
BASE=/data/qbao775/Explanation-Generation
DPO_ADAPTER=${2:-$BASE/rl_dpo_multiplicative_qwen3_8b_generator}
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/dpo_thinking_eval
mkdir -p "$LOG_DIR"

BASE_MODEL=/data/shared/qwen3/Qwen3-8B
SFT_ADAPTER=$BASE/models/rl_sft_qwen3_8b_generator
VERIFIER_PATH=$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2

# Derive a short tag from the DPO adapter path for result filenames
DPO_TAG=$(basename "$DPO_ADAPTER" | sed 's/rl_dpo_//' | sed 's/_qwen3_8b_generator//')

echo "======================================================"
echo " DPO + Thinking Mode Evaluation  $(date)"
echo " GPU=$GPU"
echo " DPO adapter: $DPO_ADAPTER"
echo " Tag: $DPO_TAG"
echo "======================================================"

declare -A TEST_FILES=(
    [cardiff]="$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
    [sydney]="$BASE/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
    [law]="$BASE/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
    [med_y1]="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"
    [med_y2]="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
)
# Only evaluate Cardiff and Sydney by default (fastest; add others as needed)
DOMAINS=(cardiff sydney)

for DOMAIN in "${DOMAINS[@]}"; do
    TEST_FILE="${TEST_FILES[$DOMAIN]}"
    EVAL_OUT=$BASE/rl_eval_results/qwen3_dpo_thinking_${DPO_TAG}_${DOMAIN}_eval.json
    echo ""
    echo "--- Evaluating $DOMAIN (SFT+Thinking vs DPO+Thinking) ---"

    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_evaluate_qwen3_dpo_thinking.py" \
        --test_data_path "$TEST_FILE" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_ADAPTER" \
        --verifier_path "$VERIFIER_PATH" \
        --output_path "$EVAL_OUT" \
        --device cuda:0 \
        --verifier_device cuda:0 \
        --nli_device cpu \
        --max_new_tokens 512 \
        2>&1 | tee "$LOG_DIR/${DOMAIN}.log"

    echo "  Results:"
    python3 -c "
import json
d = json.load(open('$EVAL_OUT'))
for r in d['results']:
    print(f\"  [{r['model']}] NLI={r['avg_nli_entailment']:.4f} ACR={r['avg_answer_coverage_rate']:.4f} Ver={r['avg_verifier_score']:.2f} Thinking={r['thinking_activated_pct']:.0f}%\")
"
done

# ---- Comparison summary: Standard vs Thinking ----
echo ""
echo "======================================================"
echo " DPO Standard vs DPO+Thinking Comparison — Cardiff & Sydney"
echo "======================================================"
echo "NOTE: Compare against standard DPO eval files:"
for DOMAIN in cardiff sydney; do
    STD_EVAL=$BASE/rl_eval_results/qwen3_dpo_${DPO_TAG}_${DOMAIN}_eval.json
    THINK_EVAL=$BASE/rl_eval_results/qwen3_dpo_thinking_${DPO_TAG}_${DOMAIN}_eval.json

    if [ -f "$STD_EVAL" ] && [ -f "$THINK_EVAL" ]; then
        echo ""
        echo "  $DOMAIN:"
        python3 -c "
import json
std_d  = json.load(open('$STD_EVAL'))
think_d = json.load(open('$THINK_EVAL'))

std_r   = next((r for r in std_d['results'] if 'DPO' in r['model'] and 'Thinking' not in r['model']), None)
think_r = next((r for r in think_d['results'] if 'DPO' in r['model'] and 'Thinking' in r['model']), None)

if std_r and think_r:
    delta_nli = think_r['avg_nli_entailment'] - std_r['avg_nli_entailment']
    delta_acr = think_r['avg_answer_coverage_rate'] - std_r['avg_answer_coverage_rate']
    print(f\"    DPO Standard:  NLI={std_r['avg_nli_entailment']:.4f} ACR={std_r['avg_answer_coverage_rate']:.4f}\")
    print(f\"    DPO+Thinking:  NLI={think_r['avg_nli_entailment']:.4f} ACR={think_r['avg_answer_coverage_rate']:.4f}\")
    print(f\"    Delta:         NLI={delta_nli:+.4f} ACR={delta_acr:+.4f}\")
else:
    print('    (standard eval file not found for comparison)')
"
    else
        echo "  $DOMAIN: standard eval file not available for comparison"
    fi
done

echo ""
echo "Logs in: $LOG_DIR"
echo "Done!  $(date)"

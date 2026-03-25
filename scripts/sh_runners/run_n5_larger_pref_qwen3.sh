#!/bin/bash
# ============================================================
# N=5 Larger Preference Data Pipeline for Qwen3-8B
#
# Key difference from standard multiplicative_acr (N=3, gap=0.1):
#   - N=5 candidate explanations per question → more diverse pool
#   - min_score_gap=0.2 → cleaner, more discriminative pairs
#   - Same multiplicative_acr reward formula (w_nli=0.7, w_ver=0.3)
#
# Motivation: with N=5, the best/worst pair spans a larger NLI
# range, producing harder and more informative DPO pairs.
# A stricter gap=0.2 removes near-tie pairs that add noise.
#
# Usage:
#   bash scripts/sh_runners/run_n5_larger_pref_qwen3.sh [GPU_ID]
#
# Outputs:
#   rl_preference_data_n5_qwen3/preference_pairs.json
#   models/rl_dpo_n5_qwen3_8b_generator/
#   rl_eval_results/qwen3_dpo_n5_{domain}_eval.json  (all 5 domains)
# ============================================================
set -euo pipefail

GPU=${1:-7}
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/n5_larger_pref
mkdir -p "$LOG_DIR"

BASE_MODEL=/data/shared/qwen3/Qwen3-8B
SFT_ADAPTER=$BASE/models/rl_sft_qwen3_8b_generator
VERIFIER_PATH=$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA_PATH=$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json

PREF_OUT=$BASE/rl_preference_data_n5_qwen3/preference_pairs.json
DPO_OUT=$BASE/rl_dpo_n5_qwen3_8b_generator
mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT"

echo "======================================================"
echo " N=5 Larger Preference Data Pipeline  $(date)"
echo " GPU=$GPU  N=5  gap=0.2  w_nli=0.7  w_ver=0.3"
echo "======================================================"

# ---- Stage 1: Build N=5 preference data ----
echo ""
echo "--- Stage 1: Building N=5 preference data (gap=0.2, multiplicative_acr) ---"
CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
    python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
    --model_type qwen3 \
    --score_method multiplicative_acr \
    --generator_path "$BASE_MODEL" \
    --lora_adapter_path "$SFT_ADAPTER" \
    --data_path "$DATA_PATH" \
    --output_path "$PREF_OUT" \
    --verifier_path "$VERIFIER_PATH" \
    --num_samples 5 \
    --min_score_gap 0.2 \
    --max_questions 500 \
    --w_nli 0.7 \
    --w_ver 0.3 \
    --w_len_penalty 0.002 \
    --acr_threshold 0.5 \
    --generator_device cuda:0 \
    --verifier_device cuda:0 \
    --nli_device cpu \
    2>&1 | tee "$LOG_DIR/build_pref.log"

N_PAIRS=$(python3 -c "import json; d=json.load(open('$PREF_OUT')); print(len(d))")
echo "  -> $N_PAIRS preference pairs generated"
echo "  -> Compare: N=3/gap=0.1 typically yields ~430-480 pairs"
echo "              N=5/gap=0.2 should yield ~400-500 pairs with higher quality"

python3 -c "
import json, statistics
d = json.load(open('$PREF_OUT'))
gaps = [p['chosen_score'] - p['rejected_score'] for p in d]
print(f'  -> Score gap: mean={statistics.mean(gaps):.3f} median={statistics.median(gaps):.3f} min={min(gaps):.3f} max={max(gaps):.3f}')
"

# ---- Stage 2: DPO training ----
echo ""
echo "--- Stage 2: DPO training on N=5 pairs ---"
CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
    python3 "$SCRIPTS/rl_train_dpo_qwen3.py" \
    --model_name_or_path "$BASE_MODEL" \
    --sft_adapter_path "$SFT_ADAPTER" \
    --preference_data_path "$PREF_OUT" \
    --output_dir "$DPO_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none \
    2>&1 | tee "$LOG_DIR/dpo_train.log"

# ---- Stage 3: Evaluate on all 5 domains ----
echo ""
echo "--- Stage 3: Evaluating on all 5 domains ---"
CARDIFF_TEST=$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
SYDNEY_TEST=$BASE/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
LAW_TEST=$BASE/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json
MEDY1_TEST=$BASE/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json
MEDY2_TEST=$BASE/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json

for DOMAIN in cardiff sydney law med_y1 med_y2; do
    case $DOMAIN in
        cardiff) TEST_FILE="$CARDIFF_TEST" ;;
        sydney)  TEST_FILE="$SYDNEY_TEST" ;;
        law)     TEST_FILE="$LAW_TEST" ;;
        med_y1)  TEST_FILE="$MEDY1_TEST" ;;
        med_y2)  TEST_FILE="$MEDY2_TEST" ;;
    esac
    EVAL_OUT=$BASE/rl_eval_results/qwen3_dpo_n5_${DOMAIN}_eval.json
    echo "  Evaluating $DOMAIN..."
    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "$TEST_FILE" \
        --output_path "$EVAL_OUT" \
        --verifier_path "$VERIFIER_PATH" \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/eval_${DOMAIN}.log"

    python3 -c "
import json
d = json.load(open('$EVAL_OUT'))
for r in d['results']:
    if 'DPO' in r['model']:
        print(f\"    [{r['model']}] NLI={r['avg_nli_entailment']:.4f} ACR={r['avg_answer_coverage_rate']:.4f} Ver={r['avg_verifier_score']:.2f}\")
"
done

echo ""
echo "======================================================"
echo " N=5 Pipeline Complete!  $(date)"
echo " Preference pairs: $PREF_OUT"
echo " DPO model: $DPO_OUT"
echo " Results: rl_eval_results/qwen3_dpo_n5_*"
echo "======================================================"

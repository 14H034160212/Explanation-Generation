#!/bin/bash
# ============================================================
# NLI Reward Weight Sweep for Qwen3-8B (multiplicative_acr)
#
# Sweeps w_nli over {0.3, 0.5, 0.7, 0.9} (w_ver = 1 - w_nli)
# to find the optimal NLI weight that maximises Cardiff NLI
# without degrading ACR or verifier score.
#
# Hypothesis: the alignment tax on Qwen3/LLaMA-3 is caused by
# the verifier's fluency bias. Increasing w_nli should reduce
# the tax and push NLI scores toward (or above) LLaMA-2 levels.
#
# Usage:
#   bash scripts/sh_runners/run_nli_weight_sweep_qwen3.sh [GPU_ID]
#
# Outputs:
#   rl_preference_data_w{W}_qwen3/preference_pairs.json
#   models/rl_dpo_w{W}_qwen3_8b_generator/
#   rl_eval_results/qwen3_dpo_wnli{W}_cardiff_eval.json
# ============================================================
set -euo pipefail

GPU=${1:-7}
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/nli_weight_sweep
mkdir -p "$LOG_DIR"

BASE_MODEL=/data/shared/qwen3/Qwen3-8B
SFT_ADAPTER=$BASE/models/rl_sft_qwen3_8b_generator
VERIFIER_PATH=$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA_PATH=$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json
CARDIFF_TEST=$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json

# Sweep values: w_nli ∈ {0.3, 0.5, 0.7, 0.9}
for W_NLI in 0.3 0.5 0.7 0.9; do
    W_VER=$(python3 -c "print(round(1.0 - $W_NLI, 1))")
    W_TAG="wnli${W_NLI//./}"   # e.g. 0.7 → wnli07

    PREF_OUT=$BASE/rl_preference_data_${W_TAG}_qwen3/preference_pairs.json
    DPO_OUT=$BASE/models/rl_dpo_${W_TAG}_qwen3_8b_generator
    EVAL_OUT=$BASE/rl_eval_results/qwen3_dpo_${W_TAG}_cardiff_eval.json

    mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT"

    echo ""
    echo "======================================================"
    echo " w_nli=$W_NLI  w_ver=$W_VER  tag=$W_TAG  GPU=$GPU"
    echo "======================================================"

    # ---- Stage 1: Build preference data ----
    echo "--- Stage 1: Building preference data (N=3, gap=0.1, w_nli=$W_NLI) ---"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
        --model_type qwen3 \
        --score_method multiplicative_acr \
        --generator_path "$BASE_MODEL" \
        --lora_adapter_path "$SFT_ADAPTER" \
        --data_path "$DATA_PATH" \
        --output_path "$PREF_OUT" \
        --verifier_path "$VERIFIER_PATH" \
        --num_samples 3 \
        --min_score_gap 0.1 \
        --max_questions 500 \
        --w_nli "$W_NLI" \
        --w_ver "$W_VER" \
        --w_len_penalty 0.002 \
        --acr_threshold 0.5 \
        --generator_device cuda:0 \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/${W_TAG}_build_pref.log"

    N_PAIRS=$(python3 -c "import json; d=json.load(open('$PREF_OUT')); print(len(d))")
    echo "  -> $N_PAIRS preference pairs generated"

    # ---- Stage 2: DPO training ----
    echo "--- Stage 2: DPO training (w_nli=$W_NLI) ---"
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
        2>&1 | tee "$LOG_DIR/${W_TAG}_dpo_train.log"

    # ---- Stage 3: Evaluate on Cardiff ----
    echo "--- Stage 3: Cardiff evaluation (w_nli=$W_NLI) ---"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "$CARDIFF_TEST" \
        --output_path "$EVAL_OUT" \
        --verifier_path "$VERIFIER_PATH" \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/${W_TAG}_cardiff_eval.log"

    echo ""
    echo "  -> Results: $EVAL_OUT"
    python3 -c "
import json
d = json.load(open('$EVAL_OUT'))
for r in d['results']:
    print(f\"  [{r['model']}] NLI={r['avg_nli_entailment']:.4f} ACR={r['avg_answer_coverage_rate']:.4f} Ver={r['avg_verifier_score']:.2f} Time={r['avg_inference_time_s']:.2f}s\")
"
done

# ---- Summary ----
echo ""
echo "======================================================"
echo " NLI Weight Sweep Summary — Cardiff Biology"
echo "======================================================"
printf "%-10s %-10s %-10s %-10s %-10s\n" "w_nli" "w_ver" "NLI" "ACR" "Ver"
printf "%-10s %-10s %-10s %-10s %-10s\n" "-----" "-----" "---" "---" "---"
for W_NLI in 0.3 0.5 0.7 0.9; do
    W_VER=$(python3 -c "print(round(1.0 - $W_NLI, 1))")
    W_TAG="wnli${W_NLI//./}"
    EVAL_OUT=$BASE/rl_eval_results/qwen3_dpo_${W_TAG}_cardiff_eval.json
    if [ -f "$EVAL_OUT" ]; then
        python3 -c "
import json
d = json.load(open('$EVAL_OUT'))
r = [x for x in d['results'] if 'DPO' in x['model']][0]
print(f\"$W_NLI       $W_VER      {r['avg_nli_entailment']:.4f}    {r['avg_answer_coverage_rate']:.4f}    {r['avg_verifier_score']:.2f}\")
"
    fi
done

echo ""
echo "Logs in: $LOG_DIR"
echo "Done!  $(date)"

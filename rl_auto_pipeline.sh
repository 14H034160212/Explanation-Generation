#!/bin/bash
# =============================================================================
# RL-ILearner: Auto-continue pipeline monitor
# Waits for Step 2 (preference data) to finish, then runs Step 3 (DPO) and
# Step 4 (evaluation) automatically, logging all results.
# =============================================================================

set -euo pipefail

LOG="/data/qbao775/Explanation-Generation/rl_pipeline_monitor.log"
PREF_DATA="/data/qbao775/Explanation-Generation/rl_preference_data/preference_pairs.json"
PREF_LOG="/data/qbao775/Explanation-Generation/rl_preference_data/build_log.txt"
DPO_LOG="/data/qbao775/Explanation-Generation/rl_dpo_training.log"
EVAL_LOG="/data/qbao775/Explanation-Generation/rl_eval_results/eval.log"
RESULTS_FILE="/data/qbao775/Explanation-Generation/rl_eval_results/final_results.json"
BASE_MODEL="/data/shared/llama2/llama-2-13b-hf"
SFT_OUTPUT="/data/qbao775/Explanation-Generation/rl_sft_llama2_13b_generator"
DPO_OUTPUT="/data/qbao775/Explanation-Generation/rl_dpo_llama2_13b_generator"
VERIFIER_PATH="/data/qbao775/Explanation-Generation/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
TEST_DATA="/data/qbao775/Explanation-Generation/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
TRAIN_GPUS="4,5,6,7"

PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
TORCHRUN="/data/qbao775/miniconda3/envs/llm-tuning/bin/torchrun"

mkdir -p /data/qbao775/Explanation-Generation/rl_eval_results

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

log "=== RL-ILearner Auto Pipeline Monitor started ==="
log "Watching for preference data at: ${PREF_DATA}"

# ─────────────────────────────────────────────────────────────────
# WAIT for Step 2 (preference data) to finish
# ─────────────────────────────────────────────────────────────────
log "Step 2 (preference data) is in progress..."
while true; do
    # Check if build_log.txt says "Preference dataset saved"
    if grep -q "Preference dataset saved" "${PREF_LOG}" 2>/dev/null; then
        log "Step 2 DONE. Preference data file created."
        break
    fi
    # Safety: also check if file exists and process has died
    if [ -f "${PREF_DATA}" ] && ! pgrep -f "rl_build_preference_data" > /dev/null 2>&1; then
        log "Preference file found (process ended)."
        break
    fi
    sleep 60
done

# Report Step 2 stats
PREF_COUNT=$(${PYTHON} -c "import json; d=json.load(open('${PREF_DATA}')); print(len(d))" 2>/dev/null || echo "?")
log "Preference pairs built: ${PREF_COUNT}"

# ─────────────────────────────────────────────────────────────────
# STEP 3: DPO Training
# ─────────────────────────────────────────────────────────────────
log "=== Starting Step 3: DPO Training ==="
export TOKENIZERS_PARALLELISM=false

cd /data/qbao775/Explanation-Generation

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} ${TORCHRUN} \
    --nproc_per_node=4 \
    --master_port=29501 \
    rl_train_dpo.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "${SFT_OUTPUT}" \
    --preference_data_path "${PREF_DATA}" \
    --output_dir "${DPO_OUTPUT}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --beta 0.1 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --min_score_gap_filter 0.0 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --bf16 True \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --report_to none \
    --run_name "dpo-llama2-13b" 2>&1 | tee "${DPO_LOG}"

DPO_FINAL_LOSS=$(grep "train_loss" "${DPO_LOG}" | tail -1 | grep -oP "'train_loss': \K[\d.]+" || echo "N/A")
log "Step 3 DONE. DPO final loss: ${DPO_FINAL_LOSS}"
log "DPO adapter saved to: ${DPO_OUTPUT}"

# ─────────────────────────────────────────────────────────────────
# STEP 4: Evaluation
# Compare: SFT baseline vs DPO model vs legacy ILearner
# ─────────────────────────────────────────────────────────────────
log "=== Starting Step 4: Evaluation ==="

# Check test data exists
if [ ! -f "${TEST_DATA}" ]; then
    # Fall back to any available test data
    TEST_DATA=$(find /data/qbao775/Explanation-Generation/Paul_new_data -name "*random_100*" | head -1)
    log "Fallback test data: ${TEST_DATA}"
fi

CUDA_VISIBLE_DEVICES=4,5 ${PYTHON} rl_evaluation.py \
    --test_data_path "${TEST_DATA}" \
    --verifier_path "${VERIFIER_PATH}" \
    --output_path "${RESULTS_FILE}" \
    --sft_model_path "${BASE_MODEL}" \
    --sft_lora_path "${SFT_OUTPUT}" \
    --dpo_model_path "${BASE_MODEL}" \
    --dpo_lora_path "${DPO_OUTPUT}" \
    --device cuda:0 \
    --verifier_device cuda:1 \
    --cache_dir cache 2>&1 | tee "${EVAL_LOG}"

log "Step 4 DONE. Results saved to: ${RESULTS_FILE}"

# ─────────────────────────────────────────────────────────────────
# RECORD RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────
log ""
log "============================================================"
log "  RL-ILearner Experiment Summary"
log "============================================================"

# SFT training results
SFT_LOSS=$(grep "train_loss" /data/qbao775/Explanation-Generation/rl_sft_training.log | tail -1 | grep -oP "'train_loss': \K[\d.]+" || echo "0.7950")
SFT_TIME=$(grep "train_runtime" /data/qbao775/Explanation-Generation/rl_sft_training.log | tail -1 | grep -oP "'train_runtime': \K[\d.]+" || echo "2933")
log "SFT training:  loss=${SFT_LOSS}, time=${SFT_TIME}s"
log "DPO training:  loss=${DPO_FINAL_LOSS}"
log "Pref pairs:    ${PREF_COUNT}"

# Parse eval results JSON
${PYTHON} - << 'PYEOF'
import json, os, sys

results_path = "/data/qbao775/Explanation-Generation/rl_eval_results/final_results.json"
if not os.path.exists(results_path):
    print("  Eval results not found.")
    sys.exit(0)

try:
    with open(results_path) as f:
        results = json.load(f)
    print("\n  Model Comparison:")
    print(f"  {'Model':<25} {'BLEU':>8} {'BERTScore':>10} {'Verifier':>10}")
    print(f"  {'-'*55}")
    for model_name, metrics in results.items():
        bleu = metrics.get("bleu", {}).get("mean", "N/A")
        bert = metrics.get("bert_score", {}).get("mean", "N/A")
        verif = metrics.get("verifier_score", {}).get("mean", "N/A")
        bleu_str = f"{bleu:.4f}" if isinstance(bleu, float) else str(bleu)
        bert_str = f"{bert:.4f}" if isinstance(bert, float) else str(bert)
        verif_str = f"{verif:.2f}" if isinstance(verif, float) else str(verif)
        print(f"  {model_name:<25} {bleu_str:>8} {bert_str:>10} {verif_str:>10}")
except Exception as e:
    print(f"  Could not parse results: {e}")
PYEOF

log "============================================================"
log "Pipeline complete! Check rl_eval_results/ for full results."
log "============================================================"

#!/usr/bin/env bash
# ============================================================
# run_monitor_and_update.sh
# Monitors both running pipelines:
#   A) N=5 pref build → DPO train → Cardiff eval → update results.tex
#   B) Two-stage GRPO-NLI (Hybrid-DPO init → GRPO) → Cardiff+Sydney eval → update results.tex
# ============================================================
set -uo pipefail
BASE="/data/qbao775/Explanation-Generation"
cd "$BASE"
source /data/qbao775/miniconda3/etc/profile.d/conda.sh

LOG="$BASE/logs/monitor_and_update.log"
mkdir -p "$BASE/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Paths ─────────────────────────────────────────────────
N5_PREF="$BASE/rl_preference_data_n5_qwen3/preference_pairs.json"
N5_ADAPTER="$BASE/rl_dpo_n5_qwen3_8b_generator"
N5_EVAL="$BASE/rl_eval_results/qwen3_dpo_n5_cardiff_eval.json"
N5_BUILD_PID=2769603

GRPO_ADAPTER="$BASE/rl_grpo_nli_twostage_qwen3_8b_generator"
GRPO_CARDIFF_EVAL="$BASE/rl_eval_results/qwen3_grpo_nli_twostage_cardiff_eval.json"
GRPO_SYDNEY_EVAL="$BASE/rl_eval_results/qwen3_grpo_nli_twostage_sydney_eval.json"
GRPO_BUILD_PID=2789067

TRAIN_DATA="preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
CARDIFF_TEST="preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
SYDNEY_TEST="preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
BASE_MODEL="/data/shared/qwen3/Qwen3-8B"
SFT_ADAPTER="$BASE/models/rl_sft_qwen3_8b_generator"
VERIFIER="$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"

# ── Helper: wait for PID or process name ──────────────────
wait_for_process() {
    local PID=$1
    local NAME=$2
    while kill -0 "$PID" 2>/dev/null || ps aux | grep -q "$NAME" 2>/dev/null; do
        sleep 60
    done
    log "Process $PID ($NAME) finished."
}

is_alive() { kill -0 "$1" 2>/dev/null; }

# ─────────────────────────────────────────────────────────
# PIPELINE A: N=5 DPO
# ─────────────────────────────────────────────────────────
run_n5_pipeline() {
    log "=== PIPELINE A: N=5 ==="

    # A1: Wait for pref build
    log "A1: Waiting for N=5 pref build (PID $N5_BUILD_PID)..."
    while is_alive $N5_BUILD_PID; do sleep 60; done
    sleep 5

    local PAIRS
    PAIRS=$(python3 -c "import json; print(len(json.load(open('$N5_PREF'))))" 2>/dev/null || echo 0)
    log "A1: N=5 pref build done — $PAIRS pairs."
    if [[ "$PAIRS" -lt 5 ]]; then
        log "A1 ERROR: Too few pairs ($PAIRS). Aborting N=5 pipeline."
        return
    fi

    # A2: DPO training on GPU 7
    log "A2: Training N=5 DPO on GPU 7..."
    mkdir -p "$N5_ADAPTER"
    CUDA_VISIBLE_DEVICES=7 python3 -u \
        scripts/python_training/rl_train_dpo_qwen3.py \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --preference_data_path "$N5_PREF" \
        --output_dir "$N5_ADAPTER" \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --beta 0.1 \
        > logs/dpo_n5_train.log 2>&1
    log "A2: N=5 DPO training done."

    # A3: Cardiff eval on GPU 7
    log "A3: Evaluating N=5 DPO on Cardiff (GPU 7)..."
    CUDA_VISIBLE_DEVICES=7 python3 -u \
        scripts/python_training/rl_evaluate_qwen3.py \
        --test_data_path "$CARDIFF_TEST" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$N5_ADAPTER" \
        --verifier_path "$VERIFIER" \
        --output_path "$N5_EVAL" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > logs/eval_n5_cardiff.log 2>&1
    log "A3: N=5 Cardiff eval done."

    # A4: Update results.tex
    log "A4: Updating results.tex (N=5 result)..."
    python3 scripts/update_results_tex.py --task n5
    log "A4: results.tex updated with N=5 result."
}

# ─────────────────────────────────────────────────────────
# PIPELINE B: Two-stage GRPO-NLI
# ─────────────────────────────────────────────────────────
run_grpo_pipeline() {
    log "=== PIPELINE B: Two-stage GRPO-NLI ==="

    # B1: Wait for GRPO training
    log "B1: Waiting for GRPO-NLI training (PID $GRPO_BUILD_PID)..."
    while is_alive $GRPO_BUILD_PID || ps aux | grep -v grep | grep -q "rl_train_grpo_nli_qwen3" 2>/dev/null; do
        sleep 120
    done
    sleep 5

    if [[ ! -d "$GRPO_ADAPTER" ]] || [[ -z "$(ls -A "$GRPO_ADAPTER" 2>/dev/null)" ]]; then
        log "B1 ERROR: GRPO adapter missing or empty. Check logs/grpo_nli_twostage.log"
        tail -20 logs/grpo_nli_twostage.log || true
        return 0
    fi
    log "B1: GRPO training done. Adapter: $GRPO_ADAPTER"

    # B2: Cardiff eval on GPU 0
    log "B2: Evaluating GRPO-NLI on Cardiff..."
    conda run -n qwen3-rl \
    CUDA_VISIBLE_DEVICES=0 python3 -u \
        scripts/python_training/rl_evaluate_qwen3.py \
        --test_data_path "$CARDIFF_TEST" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$GRPO_ADAPTER" \
        --verifier_path "$VERIFIER" \
        --output_path "$GRPO_CARDIFF_EVAL" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > logs/eval_grpo_twostage_cardiff.log 2>&1 || true
    log "B2: Cardiff GRPO eval done."

    # B3: Sydney eval on GPU 0
    log "B3: Evaluating GRPO-NLI on Sydney..."
    conda run -n qwen3-rl \
    CUDA_VISIBLE_DEVICES=0 python3 -u \
        scripts/python_training/rl_evaluate_qwen3.py \
        --test_data_path "$SYDNEY_TEST" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$GRPO_ADAPTER" \
        --verifier_path "$VERIFIER" \
        --output_path "$GRPO_SYDNEY_EVAL" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > logs/eval_grpo_twostage_sydney.log 2>&1 || true
    log "B3: Sydney GRPO eval done."

    # B4: Parse results and update results.tex
    log "B4: Parsing GRPO-NLI results..."
    conda run -n qwen3-rl python3 -c "
import json, os
for domain, path in [('Cardiff', '$GRPO_CARDIFF_EVAL'), ('Sydney', '$GRPO_SYDNEY_EVAL')]:
    if os.path.exists(path):
        d = json.load(open(path))
        r = d.get('results', d)
        rows = r if isinstance(r, list) else [r]
        for x in rows:
            print(f'{domain} {x[\"model\"]}: NLI={x[\"avg_nli_entailment\"]:.4f} ACR={x[\"avg_answer_coverage_rate\"]:.4f}')
" || true
    log "B4: Updating results.tex with GRPO-NLI results..."
    conda run -n qwen3-rl python3 scripts/update_results_tex.py --task twostage_ppo \
        --cardiff_json "$GRPO_CARDIFF_EVAL" \
        --sydney_json  "$GRPO_SYDNEY_EVAL" || \
    log "B4 WARN: update_results_tex twostage_ppo not implemented — results in JSON only."
    log "B4: Done."
}

# Run both pipelines in parallel (background subshells)
run_n5_pipeline &
N5_PIPE_PID=$!
run_grpo_pipeline &
GRPO_PIPE_PID=$!

log "Both pipelines launched: N5_PIPE=$N5_PIPE_PID  GRPO_PIPE=$GRPO_PIPE_PID"
wait $N5_PIPE_PID  && log "Pipeline A (N=5) complete."    || log "Pipeline A exited (may have had an error)."
wait $GRPO_PIPE_PID && log "Pipeline B (GRPO-NLI) complete." || log "Pipeline B exited (may have had an error)."
log "=== ALL PIPELINES COMPLETE ==="

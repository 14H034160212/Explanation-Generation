#!/usr/bin/env bash
# ============================================================
# run_sequential_experiments.sh
# Monitors running DPO+Thinking evals, then sequentially runs:
#   1. DPO+Thinking eval (Cardiff + Sydney) — already launched, just wait
#   2. w_nli=0.9 DPO training + eval
#   3. N=5 preference pipeline + eval
# After each step, updates paper_draft/results.tex
# ============================================================
set -euo pipefail

BASE="/data/qbao775/Explanation-Generation"
cd "$BASE"
source /data/qbao775/miniconda3/etc/profile.d/conda.sh

LOG="$BASE/logs/sequential_experiments.log"
mkdir -p "$BASE/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

CARDIFF_THINK_JSON="$BASE/rl_eval_results/qwen3_dpo_thinking_multiplicative_cardiff_eval.json"
SYDNEY_THINK_JSON="$BASE/rl_eval_results/qwen3_dpo_thinking_multiplicative_sydney_eval.json"
WNLI09_CARDIFF_JSON="$BASE/rl_eval_results/qwen3_dpo_wnli09_cardiff_eval.json"
N5_CARDIFF_JSON="$BASE/rl_eval_results/qwen3_dpo_n5_cardiff_eval.json"

THINK_PIDS="848818 848819"

# ============================================================
# STEP 1: Wait for DPO+Thinking eval to complete
# ============================================================
log "=== STEP 1: Waiting for DPO+Thinking eval (Cardiff + Sydney) ==="
while true; do
    if [[ -f "$CARDIFF_THINK_JSON" && -f "$SYDNEY_THINK_JSON" ]]; then
        log "Both DPO+Thinking eval JSONs found. Proceeding."
        break
    fi
    # Check if processes died
    ALL_DEAD=true
    for PID in $THINK_PIDS; do
        if kill -0 "$PID" 2>/dev/null; then
            ALL_DEAD=false
            break
        fi
    done
    # Also check by name
    if ps aux | grep -q "rl_evaluate_qwen3_dpo_thinking" 2>/dev/null; then
        ALL_DEAD=false
    fi
    if $ALL_DEAD; then
        # Processes died — check if JSONs appeared (maybe just took a moment)
        sleep 10
        if [[ -f "$CARDIFF_THINK_JSON" && -f "$SYDNEY_THINK_JSON" ]]; then
            log "Both DPO+Thinking eval JSONs found after process exit. Proceeding."
            break
        fi
        log "ERROR: DPO+Thinking eval processes died without producing output JSONs."
        log "Cardiff log tail:"
        tail -20 "$BASE/logs/dpo_thinking_eval/cardiff.log" || true
        log "Relaunching..."
        conda activate qwen3-rl
        CUDA_VISIBLE_DEVICES=6 python3 -u \
            scripts/python_training/rl_evaluate_qwen3_dpo_thinking.py \
            --test_data_path preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
            --base_model_path /data/shared/qwen3/Qwen3-8B \
            --sft_lora_path models/rl_sft_qwen3_8b_generator \
            --dpo_lora_path rl_dpo_multiplicative_qwen3_8b_generator \
            --verifier_path models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
            --output_path "$CARDIFF_THINK_JSON" \
            --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
            > logs/dpo_thinking_eval/cardiff.log 2>&1 &
        CARDIFF_PID=$!
        CUDA_VISIBLE_DEVICES=7 python3 -u \
            scripts/python_training/rl_evaluate_qwen3_dpo_thinking.py \
            --test_data_path preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
            --base_model_path /data/shared/qwen3/Qwen3-8B \
            --sft_lora_path models/rl_sft_qwen3_8b_generator \
            --dpo_lora_path rl_dpo_multiplicative_qwen3_8b_generator \
            --verifier_path models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
            --output_path "$SYDNEY_THINK_JSON" \
            --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
            > logs/dpo_thinking_eval/sydney.log 2>&1 &
        SYDNEY_PID=$!
        THINK_PIDS="$CARDIFF_PID $SYDNEY_PID"
        log "Relaunched: Cardiff=$CARDIFF_PID Sydney=$SYDNEY_PID"
    fi
    sleep 120
done

log "=== STEP 1 DONE: Updating results.tex (thinking table) ==="
conda activate qwen3-rl 2>/dev/null || true
python3 scripts/update_results_tex.py \
    --task thinking \
    --cardiff_json "$CARDIFF_THINK_JSON" \
    --sydney_json  "$SYDNEY_THINK_JSON"
log "results.tex updated with DPO+Thinking results."


# ============================================================
# STEP 2: w_nli=0.9 DPO training + Cardiff eval
# ============================================================
log "=== STEP 2: w_nli=0.9 DPO training + eval ==="
WNLI09_ADAPTER="$BASE/rl_dpo_wnli09_qwen3_8b_generator"
WNLI09_PREF="$BASE/rl_preference_data_wnli09_qwen3"

if [[ ! -f "$WNLI09_CARDIFF_JSON" ]]; then
    # Train adapter if missing
    if [[ ! -d "$WNLI09_ADAPTER" ]]; then
        log "Training w_nli=0.9 DPO adapter on GPU 7..."
        conda activate qwen3-rl 2>/dev/null || true
        CUDA_VISIBLE_DEVICES=7 python3 -u rl_train_dpo_qwen3.py \
            --base_model_path /data/shared/qwen3/Qwen3-8B \
            --sft_lora_path models/rl_sft_qwen3_8b_generator \
            --preference_data_path "$WNLI09_PREF/preference_pairs.json" \
            --output_dir "$WNLI09_ADAPTER" \
            --num_train_epochs 5 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate 5e-5 \
            --beta 0.1 \
            > logs/dpo_wnli09_train.log 2>&1
        log "w_nli=0.9 training done."
    else
        log "w_nli=0.9 adapter already exists, skipping training."
    fi

    log "Evaluating w_nli=0.9 on Cardiff..."
    CUDA_VISIBLE_DEVICES=7 python3 -u rl_evaluate_qwen3.py \
        --test_data_path preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path models/rl_sft_qwen3_8b_generator \
        --dpo_lora_path "$WNLI09_ADAPTER" \
        --verifier_path models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path "$WNLI09_CARDIFF_JSON" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > logs/eval_wnli09_cardiff.log 2>&1
    log "w_nli=0.9 Cardiff eval done."
else
    log "w_nli=0.9 Cardiff eval JSON already exists, skipping."
fi

log "=== STEP 2 DONE: Updating results.tex (w_nli sweep table) ==="
python3 scripts/update_results_tex.py --task wnli_sweep
log "results.tex updated with w_nli sweep results."


# ============================================================
# STEP 3: N=5 preference pipeline + Cardiff eval
# ============================================================
log "=== STEP 3: N=5 preference pipeline ==="
N5_PREF="$BASE/rl_preference_data_n5_qwen3"
N5_ADAPTER="$BASE/rl_dpo_n5_qwen3_8b_generator"

if [[ ! -f "$N5_CARDIFF_JSON" ]]; then
    # Build N=5 preference data if missing
    if [[ ! -f "$N5_PREF/preference_pairs.json" ]]; then
        log "Building N=5 preference data on GPU 7..."
        mkdir -p "$N5_PREF"
        CUDA_VISIBLE_DEVICES=7 python3 -u rl_build_preference_data_nli.py \
            --base_model_path /data/shared/qwen3/Qwen3-8B \
            --sft_lora_path models/rl_sft_qwen3_8b_generator \
            --train_data_path "preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
            --verifier_path models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
            --output_path "$N5_PREF/preference_pairs.json" \
            --score_method multiplicative_acr \
            --num_samples 5 \
            --min_score_gap 0.2 \
            --max_examples 500 \
            --w_nli 0.7 \
            --w_ver 0.3 \
            --device cuda:0 \
            > logs/build_n5_pref.log 2>&1
        log "N=5 preference data built."
    else
        log "N=5 preference data already exists."
    fi

    # Train N=5 adapter
    if [[ ! -d "$N5_ADAPTER" ]]; then
        log "Training N=5 DPO adapter on GPU 7..."
        CUDA_VISIBLE_DEVICES=7 python3 -u rl_train_dpo_qwen3.py \
            --base_model_path /data/shared/qwen3/Qwen3-8B \
            --sft_lora_path models/rl_sft_qwen3_8b_generator \
            --preference_data_path "$N5_PREF/preference_pairs.json" \
            --output_dir "$N5_ADAPTER" \
            --num_train_epochs 5 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate 5e-5 \
            --beta 0.1 \
            > logs/dpo_n5_train.log 2>&1
        log "N=5 DPO training done."
    else
        log "N=5 adapter already exists, skipping training."
    fi

    log "Evaluating N=5 model on Cardiff..."
    CUDA_VISIBLE_DEVICES=7 python3 -u rl_evaluate_qwen3.py \
        --test_data_path preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path models/rl_sft_qwen3_8b_generator \
        --dpo_lora_path "$N5_ADAPTER" \
        --verifier_path models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path "$N5_CARDIFF_JSON" \
        --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
        > logs/eval_n5_cardiff.log 2>&1
    log "N=5 Cardiff eval done."
else
    log "N=5 Cardiff eval JSON already exists, skipping."
fi

log "=== STEP 3 DONE: Updating results.tex (N=5 result) ==="
python3 scripts/update_results_tex.py --task n5
log "results.tex updated with N=5 result."

log "=== ALL STEPS COMPLETE ==="
log "Final results.tex updated at $BASE/paper_draft/results.tex"

#!/bin/bash
# ============================================================
# Auto GPU Scheduler for Qwen3-8B RL Experiments
# Monitors GPUs 4-7, launches next experiment when GPU is free
# Usage: nohup ./gpu_scheduler.sh > scheduler.log 2>&1 &
# ============================================================

LOG="scheduler.log"
POLL_INTERVAL=60  # seconds between GPU checks
GPUS=(4 5 6 7)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

# ============================================================
# EXPERIMENT QUEUE — ordered by priority
# Format: "GPU_ID|OUTPUT_DIR|LOG_FILE|COMMAND..."
# GPU_ID: preferred GPU (will try others if busy)
# ============================================================
declare -a EXPERIMENT_QUEUE=(
    # --- DPO Ablations ---
    "any|rl_dpo_qwen3_v1_beta05_generator|rl_qwen3_dpo_v1_beta05.log|python3 rl_train_dpo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_sft_qwen3_8b_generator --preference_data_path ./rl_preference_data/preference_pairs.json --output_dir ./rl_dpo_qwen3_v1_beta05_generator --num_train_epochs 5 --per_device_train_batch_size 1 --max_steps 2000 --learning_rate 5e-6 --beta 0.5 --report_to none"
    "any|rl_dpo_qwen3_v1_beta01_lr1e5_generator|rl_qwen3_dpo_v1_lr1e5.log|python3 rl_train_dpo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_sft_qwen3_8b_generator --preference_data_path ./rl_preference_data/preference_pairs.json --output_dir ./rl_dpo_qwen3_v1_lr1e5_generator --num_train_epochs 5 --per_device_train_batch_size 1 --max_steps 2000 --learning_rate 1e-5 --beta 0.1 --report_to none"
    "any|rl_dpo_qwen3_v2_beta05_generator|rl_qwen3_dpo_v2_beta05.log|python3 rl_train_dpo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_sft_qwen3_8b_generator --preference_data_path ./rl_preference_data_v2/preference_pairs.json --output_dir ./rl_dpo_qwen3_v2_beta05_generator --num_train_epochs 5 --per_device_train_batch_size 1 --max_steps 2000 --learning_rate 5e-6 --beta 0.5 --report_to none"
    # --- PPO Variants (depends on DPO outputs above) ---
    "any|rl_ppo_qwen3_dpo_v1_beta05_generator|rl_qwen3_ppo_dpo_v1_beta05.log|python3 scripts/python_training/rl_train_ppo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_dpo_qwen3_v1_beta05_generator --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json --output_dir ./rl_ppo_qwen3_dpo_v1_beta05_generator --batch_size 4 --mini_batch_size 1 --learning_rate 1e-5 --max_questions 2000 --verifier_device cuda:0"
    "any|rl_ppo_qwen3_dpo_v2_beta05_generator|rl_qwen3_ppo_dpo_v2_beta05.log|python3 scripts/python_training/rl_train_ppo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_dpo_qwen3_v2_beta05_generator --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json --output_dir ./rl_ppo_qwen3_dpo_v2_beta05_generator --batch_size 4 --mini_batch_size 1 --learning_rate 1e-5 --max_questions 2000 --verifier_device cuda:0"
    "any|rl_ppo_qwen3_sft_lr5e5_generator|rl_qwen3_ppo_sft_lr5e5.log|python3 scripts/python_training/rl_train_ppo_qwen3.py --model_name_or_path /data/shared/qwen3/Qwen3-8B --sft_adapter_path ./rl_sft_qwen3_8b_generator --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json --output_dir ./rl_ppo_qwen3_sft_lr5e5_generator --batch_size 4 --mini_batch_size 1 --learning_rate 5e-5 --max_questions 2000 --verifier_device cuda:0"
)

QUEUE_INDEX=0

# Check if GPU is free (< 1000 MiB used)
is_gpu_free() {
    local gpu=$1
    local mem_used
    mem_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$mem_used" ] && [ "$mem_used" -lt 1000 ]
}

# Get output dir from experiment queue entry
get_output_dir() {
    echo "$1" | cut -d'|' -f2
}

# Get log file from experiment queue entry
get_log_file() {
    echo "$1" | cut -d'|' -f3
}

# Get command from experiment queue entry
get_command() {
    echo "$1" | cut -d'|' -f4-
}

# Check if experiment already has a completed output directory
is_done() {
    local output_dir="$1"
    [ -f "./${output_dir}/adapter_model.safetensors" ]
}

log "============================================"
log "Auto GPU Scheduler started"
log "Monitoring GPUs: ${GPUS[*]}"
log "Total experiments in queue: ${#EXPERIMENT_QUEUE[@]}"
log "============================================"

while true; do
    if [ "$QUEUE_INDEX" -ge "${#EXPERIMENT_QUEUE[@]}" ]; then
        log "All experiments completed or submitted. Scheduler exiting."
        break
    fi

    # Check each monitored GPU for availability
    for GPU in "${GPUS[@]}"; do
        if [ "$QUEUE_INDEX" -ge "${#EXPERIMENT_QUEUE[@]}" ]; then
            break
        fi

        if is_gpu_free "$GPU"; then
            ENTRY="${EXPERIMENT_QUEUE[$QUEUE_INDEX]}"
            OUTPUT_DIR=$(get_output_dir "$ENTRY")
            LOG_FILE=$(get_log_file "$ENTRY")
            CMD=$(get_command "$ENTRY")

            # Check if already done (previous run saved adapter)
            if is_done "$OUTPUT_DIR"; then
                log "SKIP: $OUTPUT_DIR already has adapter_model.safetensors, skipping."
                QUEUE_INDEX=$((QUEUE_INDEX + 1))
                continue
            fi

            log "GPU $GPU is FREE → Launching experiment $((QUEUE_INDEX + 1))/${#EXPERIMENT_QUEUE[@]}"
            log "  Output: $OUTPUT_DIR"
            log "  Log:    $LOG_FILE"

            env CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n qwen3-rl \
                $CMD > "$LOG_FILE" 2>&1 &

            log "  PID: $!"
            QUEUE_INDEX=$((QUEUE_INDEX + 1))

            # Small delay to let the process spin up before checking next GPU
            sleep 5
        fi
    done

    log "Polling in ${POLL_INTERVAL}s... (queue position: $QUEUE_INDEX/${#EXPERIMENT_QUEUE[@]})"
    sleep "$POLL_INTERVAL"
done

log "Scheduler finished."

#!/bin/bash
# Auto-launch DPO v3 training once preference_pairs.json exists
PREF_DATA="./rl_preference_data_v3/preference_pairs.json"
LOG="rl_dpo_v3_training.log"
PREF_LOG="rl_pref_v3.log"

echo "[$(date)] DPO v3 watcher started (PID $$)"
echo "[$(date)] Waiting for: $PREF_DATA"

while true; do
    if [ -f "$PREF_DATA" ] && [ -s "$PREF_DATA" ]; then
        SIZE=$(wc -c < "$PREF_DATA")
        echo "[$(date)] $PREF_DATA found ($SIZE bytes). Checking if it's valid JSON..."
        if python3 -c "import json; d=json.load(open('$PREF_DATA')); print(f'[OK] {len(d)} pairs')" 2>&1; then
            echo "[$(date)] Preference data ready! Launching DPO v3 training..."
            break
        else
            echo "[$(date)] File exists but invalid JSON yet, waiting..."
        fi
    fi
    # Check if preference gen process is still alive
    if ! ps -p 4018081 > /dev/null 2>&1; then
        echo "[$(date)] WARNING: PID 4018081 (pref gen) is no longer running!"
        if [ -f "$PREF_DATA" ] && [ -s "$PREF_DATA" ]; then
            echo "[$(date)] But output file exists — likely completed. Proceeding."
            break
        else
            echo "[$(date)] Output file not found. Pref gen may have failed. Exiting watcher."
            exit 1
        fi
    fi
    sleep 60
done

# Count pairs
N_PAIRS=$(python3 -c "import json; print(len(json.load(open('$PREF_DATA'))))")
echo "[$(date)] Found $N_PAIRS preference pairs. Starting DPO v3 training..."

CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n llm-tuning python3 rl_train_dpo.py \
    --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
    --sft_adapter_path ./rl_sft_llama2_13b_generator \
    --preference_data_path ./rl_preference_data_v3/preference_pairs.json \
    --output_dir ./rl_dpo_v3_llama2_13b_generator \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lora_r 16 --lora_alpha 32 \
    --bf16 True \
    > "$LOG" 2>&1

echo "[$(date)] DPO v3 training finished. Exit code: $?"
echo "[$(date)] Log: $LOG"

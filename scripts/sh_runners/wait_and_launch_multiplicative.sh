#!/bin/bash
# Wait for PIDs 2244824 (law rerank) and 2253459 (med2 rerank) to finish,
# then launch the multiplicative ACR pipeline on GPU4+6.

BASE=/data/qbao775/Explanation-Generation
LOG_DIR=$BASE/logs/multiplicative_acr
mkdir -p "$LOG_DIR"

echo "$(date) — Waiting for rerank jobs (PIDs 2244824, 2253459) to finish..."

while kill -0 2244824 2>/dev/null || kill -0 2253459 2>/dev/null; do
    LAW=$(python3 -c "import json; d=json.load(open('$BASE/rl_eval_results/pure_logic_queue/law_pure_logic_100.json')); print(len(d))" 2>/dev/null || echo "?")
    MED2=$(python3 -c "import json; d=json.load(open('$BASE/rl_eval_results/pure_logic_queue/med2_pure_logic_100.json')); print(len(d))" 2>/dev/null || echo "?")
    echo "$(date) — law=${LAW}/100  med2=${MED2}/100  (still waiting...)"
    sleep 300  # check every 5 minutes
done

echo "$(date) — Both rerank jobs done. Launching multiplicative ACR pipeline (both models)..."
cd "$BASE"
nohup bash scripts/sh_runners/run_multiplicative_acr_pipeline.sh both \
    > "$LOG_DIR/pipeline_both.log" 2>&1 &
PIPELINE_PID=$!
echo "$(date) — Pipeline launched (PID=$PIPELINE_PID). Logs: $LOG_DIR/pipeline_both.log"
echo $PIPELINE_PID > "$LOG_DIR/pipeline.pid"

#!/bin/bash
# ============================================================
# experiment_watcher.sh  — v2 (fixed)
# Monitors active experiments; auto-restarts on crash.
# Usage: nohup bash experiment_watcher.sh > watcher.log 2>&1 &
# ============================================================

PYTHON_LLM="/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
PYTHON_QWEN="/data/qbao775/miniconda3/envs/qwen3-rl/bin/python3"
WORKDIR="/data/qbao775/Explanation-Generation"
CHECK_INTERVAL=120   # seconds between checks

cd "$WORKDIR"

# Load API key if available
[ -f .env_openai ] && source .env_openai

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── PID state file ───────────────────────────────────────────
PID_TRACK="$WORKDIR/.watcher_state"
touch "$PID_TRACK"

get_pid() {
    grep "^$1=" "$PID_TRACK" 2>/dev/null | tail -1 | cut -d= -f2
}

set_pid() {
    # Remove old entry then append new
    grep -v "^$1=" "$PID_TRACK" > "${PID_TRACK}.tmp" 2>/dev/null && mv "${PID_TRACK}.tmp" "$PID_TRACK"
    echo "$1=$2" >> "$PID_TRACK"
}

is_alive() {
    local pid=$1
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

is_done() {
    local done_file=$1
    [[ -n "$done_file" && -f "$done_file" ]]
}

# ── Experiment configs ───────────────────────────────────────

# Cardiff K5 Baseline
start_cardiff_k5() {
    log "⟳  Starting Cardiff K5 Baseline"
    nohup $PYTHON_LLM rl_evaluation.py \
        --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/baselines_cardiff_k5_eval.json \
        --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back \
        --ilearner_k 5 --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:1 --cache_dir cache \
        >> rl_cardiff_k5_baseline.log 2>&1 &
    set_pid cardiff_k5 $!
    log "   Cardiff K5 → PID $!"
}

# Sydney K5 Baseline
# GPU 6 is occupied by PPO v1 verifier (~70GB). Use GPU 7 alone (65GB free).
# Both main model and verifier fit on GPU 7 together (~27GB + 14GB = 41GB).
start_sydney_k5() {
    log "⟳  Starting Sydney K5 Baseline (GPU 7, both models on cuda:0)"
    CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_LLM rl_evaluation.py \
        --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/baselines_sydney_k5_eval.json \
        --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back \
        --ilearner_k 5 --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:0 --cache_dir cache \
        >> rl_sydney_k5_baseline.log 2>&1 &
    set_pid sydney_k5 $!
    log "   Sydney K5 → PID $!"
}

# Law K5 Baseline — GPU 7, sequential after Sydney K5
start_law_k5() {
    log "⟳  Starting Law K5 Baseline (GPU 7, both models on cuda:0)"
    CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_LLM rl_evaluation.py \
        --test_data_path ./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/baselines_law_k5_eval.json \
        --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back \
        --ilearner_k 5 --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:0 --cache_dir cache \
        >> rl_law_k5_baseline.log 2>&1 &
    set_pid law_k5 $!
    log "   Law K5 → PID $!"
}

# Med Y1 K5 Baseline — GPU 7, sequential after Law K5
start_med_y1_k5() {
    log "⟳  Starting Med Y1 K5 Baseline (GPU 7, both models on cuda:0)"
    CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_LLM rl_evaluation.py \
        --test_data_path ./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/baselines_med_y1_k5_eval.json \
        --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back \
        --ilearner_k 5 --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:0 --cache_dir cache \
        >> rl_med_y1_k5_baseline.log 2>&1 &
    set_pid med_y1_k5 $!
    log "   Med Y1 K5 → PID $!"
}

# Med Y2 K5 Baseline — GPU 7, sequential after Med Y1 K5
start_med_y2_k5() {
    log "⟳  Starting Med Y2 K5 Baseline (GPU 7, both models on cuda:0)"
    CUDA_VISIBLE_DEVICES=7 nohup $PYTHON_LLM rl_evaluation.py \
        --test_data_path ./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/baselines_med_y2_k5_eval.json \
        --ilearner_model_path ./vicuna_13B_merged_all_generator_avg_3_lenexp_10_back \
        --ilearner_k 5 --ilearner_is_legacy \
        --device cuda:0 --verifier_device cuda:0 --cache_dir cache \
        >> rl_med_y2_k5_baseline.log 2>&1 &
    set_pid med_y2_k5 $!
    log "   Med Y2 K5 → PID $!"
}

# Qwen3 PPO v1 — GPU 4 single card (Ollama blocks GPU 6)
start_ppo_v1() {
    log "⟳  Starting Qwen3 PPO v1 (GPU 4, max_global_steps=1000)"
    CUDA_VISIBLE_DEVICES=4 nohup $PYTHON_QWEN scripts/python_training/rl_train_ppo_qwen3.py \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_dpo_qwen3_v1_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_dir ./rl_ppo_qwen3_hybrid_v1_2000steps \
        --num_epochs 16 --batch_size 4 --mini_batch_size 1 \
        --learning_rate 1e-5 --verifier_device cuda:0 \
        --max_global_steps 1000 --save_every_steps 200 \
        >> rl_ppo_hybrid_v1_2000steps.log 2>&1 &
    set_pid ppo_v1 $!
    log "   PPO v1 → PID $!"
}

# Qwen3 PPO v2 — GPU 5+7
start_ppo_v2() {
    log "⟳  Starting Qwen3 PPO v2 (GPU 5+7, max_global_steps=1000)"
    CUDA_VISIBLE_DEVICES=5,7 nohup $PYTHON_QWEN scripts/python_training/rl_train_ppo_qwen3.py \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_dpo_qwen3_v2_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_dir ./rl_ppo_qwen3_hybrid_v2_2000steps \
        --num_epochs 16 --batch_size 4 --mini_batch_size 1 \
        --learning_rate 1e-5 --verifier_device cuda:1 \
        --max_global_steps 1000 --save_every_steps 200 \
        >> rl_ppo_hybrid_v2_2000steps.log 2>&1 &
    set_pid ppo_v2 $!
    log "   PPO v2 → PID $!"
}

# ── Post-completion triggers ─────────────────────────────────

trigger_downstream() {
    # After Cardiff K5 done → start Sydney K5 if not already running/done
    if is_done rl_eval_results/baselines_cardiff_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_sydney_k5_eval.json; then
            local spid=$(get_pid sydney_k5)
            if ! is_alive "$spid"; then
                log "Cardiff K5 complete → triggering Sydney K5"
                start_sydney_k5
            fi
        fi
    fi

    # After Sydney K5 done → start Law K5 (sequential on GPU 7)
    if is_done rl_eval_results/baselines_sydney_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_law_k5_eval.json; then
            local lpid=$(get_pid law_k5)
            if ! is_alive "$lpid"; then
                log "Sydney K5 complete → triggering Law K5"
                start_law_k5
            fi
        fi
    fi

    # After Law K5 done → start Med Y1 K5
    if is_done rl_eval_results/baselines_law_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
            local m1pid=$(get_pid med_y1_k5)
            if ! is_alive "$m1pid"; then
                log "Law K5 complete → triggering Med Y1 K5"
                start_med_y1_k5
            fi
        fi
    fi

    # After Med Y1 K5 done → start Med Y2 K5
    if is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_med_y2_k5_eval.json; then
            local m2pid=$(get_pid med_y2_k5)
            if ! is_alive "$m2pid"; then
                log "Med Y1 K5 complete → triggering Med Y2 K5"
                start_med_y2_k5
            fi
        fi
    fi

    # After Med Y2 K5 done → run GPT-3.5 Sydney eval (needs GPU for verifier)
    if is_done rl_eval_results/baselines_med_y2_k5_eval.json; then
        if ! is_done rl_eval_results/gpt35_sydney_eval.json; then
            local g35pid=$(get_pid gpt35_sydney)
            if ! is_alive "$g35pid"; then
                log "Med Y2 K5 complete → triggering GPT-3.5 Sydney eval on GPU 7"
                CUDA_VISIBLE_DEVICES=7 nohup \
                    /data/qbao775/miniconda3/envs/llm-tuning/bin/python3 \
                    scripts/eval_pretrained_explanations.py \
                    --input_path ./Paul_new_data/Sydney/Sydney_gpt-35_random_100_correct.json \
                    --model_name "GPT-3.5" \
                    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
                    --output_path ./rl_eval_results/gpt35_sydney_eval.json \
                    --verifier_device cuda:0 \
                    >> rl_gpt35_sydney_eval.log 2>&1 &
                set_pid gpt35_sydney $!
                log "   GPT-3.5 Sydney eval → PID $!"
            fi
        fi
    fi

    # After Law K5 done → run GPT-4/GPT-4o-mini eval on GPU 7 (verifier needs GPU)
    if is_done rl_eval_results/baselines_law_k5_eval.json; then
        if ! is_done rl_eval_results/gpt4o_mini_cardiff_eval.json; then
            local gpid=$(get_pid gpt_eval)
            if ! is_alive "$gpid"; then
                log "Law K5 complete → triggering GPT-4/GPT-4o-mini eval on GPU 7"
                OPENAI_API_KEY="${OPENAI_API_KEY:-}" nohup bash run_gpt_eval_pretrained.sh \
                    >> rl_gpt_eval_pretrained.log 2>&1 &
                set_pid gpt_eval $!
                log "   GPT eval → PID $!"
            fi
        fi
    fi

    # After PPO v1 done → run Cardiff+Sydney eval for v1
    if is_done rl_ppo_qwen3_hybrid_v1_2000steps/adapter_model.safetensors; then
        if ! is_done rl_eval_results/qwen3_ppo_hybrid_v1_cardiff_eval.json; then
            local epid=$(get_pid ppo_v1_eval)
            if ! is_alive "$epid"; then
                log "PPO v1 complete → launching Cardiff+Sydney eval"
                nohup bash -c "
                    CUDA_VISIBLE_DEVICES=4,6 $PYTHON_QWEN rl_evaluate_qwen3.py \
                        --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
                        --dpo_lora_path ./rl_ppo_qwen3_hybrid_v1_2000steps \
                        --output_path ./rl_eval_results/qwen3_ppo_hybrid_v1_cardiff_eval.json \
                        --device cuda:0 --verifier_device cuda:1 --nli_device cpu \
                        > rl_qwen3_ppo_v1_cardiff_eval.log 2>&1
                    CUDA_VISIBLE_DEVICES=4,6 $PYTHON_QWEN rl_evaluate_qwen3.py \
                        --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
                        --dpo_lora_path ./rl_ppo_qwen3_hybrid_v1_2000steps \
                        --output_path ./rl_eval_results/qwen3_ppo_hybrid_v1_sydney_eval.json \
                        --device cuda:0 --verifier_device cuda:1 --nli_device cpu \
                        >> rl_qwen3_ppo_v1_cardiff_eval.log 2>&1
                " >> rl_qwen3_ppo_v1_cardiff_eval.log 2>&1 &
                set_pid ppo_v1_eval $!
            fi
        fi
    fi

    # After PPO v2 done → run Cardiff+Sydney eval for v2
    if is_done rl_ppo_qwen3_hybrid_v2_2000steps/adapter_model.safetensors; then
        if ! is_done rl_eval_results/qwen3_ppo_hybrid_v2_cardiff_eval.json; then
            local epid=$(get_pid ppo_v2_eval)
            if ! is_alive "$epid"; then
                log "PPO v2 complete → launching Cardiff+Sydney eval"
                nohup bash -c "
                    CUDA_VISIBLE_DEVICES=5,7 $PYTHON_QWEN rl_evaluate_qwen3.py \
                        --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
                        --dpo_lora_path ./rl_ppo_qwen3_hybrid_v2_2000steps \
                        --output_path ./rl_eval_results/qwen3_ppo_hybrid_v2_cardiff_eval.json \
                        --device cuda:0 --verifier_device cuda:1 --nli_device cpu \
                        > rl_qwen3_ppo_v2_cardiff_eval.log 2>&1
                    CUDA_VISIBLE_DEVICES=5,7 $PYTHON_QWEN rl_evaluate_qwen3.py \
                        --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
                        --dpo_lora_path ./rl_ppo_qwen3_hybrid_v2_2000steps \
                        --output_path ./rl_eval_results/qwen3_ppo_hybrid_v2_sydney_eval.json \
                        --device cuda:0 --verifier_device cuda:1 --nli_device cpu \
                        >> rl_qwen3_ppo_v2_cardiff_eval.log 2>&1
                " >> rl_qwen3_ppo_v2_cardiff_eval.log 2>&1 &
                set_pid ppo_v2_eval $!
            fi
        fi
    fi
}

# ── Status summary ───────────────────────────────────────────

print_status() {
    log "─────────────────────────────────────────────"
    log "STATUS SUMMARY"

    # Cardiff K5
    if is_done rl_eval_results/baselines_cardiff_k5_eval.json; then
        log "  ✅ Cardiff K5 Baseline — COMPLETED"
    else
        pid=$(get_pid cardiff_k5)
        if is_alive "$pid"; then
            last=$(tail -1 rl_cardiff_k5_baseline.log 2>/dev/null | grep -oP '\d+%.*' | head -c60)
            log "  🔄 Cardiff K5 Baseline — PID=$pid | $last"
        else
            log "  ❌ Cardiff K5 Baseline — DEAD (PID=$pid)"
        fi
    fi

    # Sydney K5
    if is_done rl_eval_results/baselines_sydney_k5_eval.json; then
        log "  ✅ Sydney K5 Baseline — COMPLETED"
    else
        pid=$(get_pid sydney_k5)
        if is_alive "$pid"; then
            last=$(tail -1 rl_sydney_k5_baseline.log 2>/dev/null | grep -oP '\d+%.*' | head -c60)
            log "  🔄 Sydney K5 Baseline — PID=$pid | $last"
        else
            log "  ❌ Sydney K5 Baseline — DEAD (PID=$pid)"
        fi
    fi

    # Law K5
    if is_done rl_eval_results/baselines_law_k5_eval.json; then
        log "  ✅ Law K5 Baseline — COMPLETED"
    elif ! is_done rl_eval_results/baselines_sydney_k5_eval.json; then
        log "  ⏳ Law K5 Baseline — waiting for Sydney K5"
    else
        pid=$(get_pid law_k5)
        if is_alive "$pid"; then
            last=$(tail -1 rl_law_k5_baseline.log 2>/dev/null | grep -oP '\d+%.*' | head -c60)
            log "  🔄 Law K5 Baseline — PID=$pid | $last"
        else
            log "  ❌ Law K5 Baseline — DEAD (PID=$pid)"
        fi
    fi

    # Med Y1 K5
    if is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
        log "  ✅ Med Y1 K5 Baseline — COMPLETED"
    elif ! is_done rl_eval_results/baselines_law_k5_eval.json; then
        log "  ⏳ Med Y1 K5 Baseline — waiting for Law K5"
    else
        pid=$(get_pid med_y1_k5)
        if is_alive "$pid"; then
            last=$(tail -1 rl_med_y1_k5_baseline.log 2>/dev/null | grep -oP '\d+%.*' | head -c60)
            log "  🔄 Med Y1 K5 Baseline — PID=$pid | $last"
        else
            log "  ❌ Med Y1 K5 Baseline — DEAD (PID=$pid)"
        fi
    fi

    # Med Y2 K5
    if is_done rl_eval_results/baselines_med_y2_k5_eval.json; then
        log "  ✅ Med Y2 K5 Baseline — COMPLETED"
    elif ! is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
        log "  ⏳ Med Y2 K5 Baseline — waiting for Med Y1 K5"
    else
        pid=$(get_pid med_y2_k5)
        if is_alive "$pid"; then
            last=$(tail -1 rl_med_y2_k5_baseline.log 2>/dev/null | grep -oP '\d+%.*' | head -c60)
            log "  🔄 Med Y2 K5 Baseline — PID=$pid | $last"
        else
            log "  ❌ Med Y2 K5 Baseline — DEAD (PID=$pid)"
        fi
    fi

    # PPO v1
    if is_done rl_ppo_qwen3_hybrid_v1_2000steps/adapter_model.safetensors; then
        log "  ✅ Qwen3 PPO v1 — COMPLETED"
    else
        pid=$(get_pid ppo_v1)
        if is_alive "$pid"; then
            last=$(tail -1 rl_ppo_hybrid_v1_2000steps.log 2>/dev/null | grep -oP 'Step \d+.*reward: [0-9.]+' | head -c80)
            log "  🔄 Qwen3 PPO v1 — PID=$pid | $last"
        else
            log "  ❌ Qwen3 PPO v1 — DEAD (PID=$pid)"
        fi
    fi

    # PPO v2
    if is_done rl_ppo_qwen3_hybrid_v2_2000steps/adapter_model.safetensors; then
        log "  ✅ Qwen3 PPO v2 — COMPLETED"
    else
        pid=$(get_pid ppo_v2)
        if is_alive "$pid"; then
            last=$(tail -1 rl_ppo_hybrid_v2_2000steps.log 2>/dev/null | grep -oP 'Step \d+.*reward: [0-9.]+' | head -c80)
            log "  🔄 Qwen3 PPO v2 — PID=$pid | $last"
        else
            log "  ❌ Qwen3 PPO v2 — DEAD (PID=$pid)"
        fi
    fi

    log "─────────────────────────────────────────────"
}

# ── Check and restart dead experiments ──────────────────────

check_and_restart() {
    # Cardiff K5
    if ! is_done rl_eval_results/baselines_cardiff_k5_eval.json; then
        pid=$(get_pid cardiff_k5)
        if ! is_alive "$pid"; then
            log "⚠️  Cardiff K5 died (PID=$pid) — restarting in 15s"
            sleep 15
            start_cardiff_k5
        fi
    fi

    # Sydney K5 — must wait for Cardiff K5 to finish (both use GPU 6+7)
    if ! is_done rl_eval_results/baselines_sydney_k5_eval.json; then
        if is_done rl_eval_results/baselines_cardiff_k5_eval.json; then
            pid=$(get_pid sydney_k5)
            if ! is_alive "$pid"; then
                log "⚠️  Sydney K5 died (PID=$pid) — restarting in 15s"
                sleep 15
                start_sydney_k5
            fi
        else
            log "  (Sydney K5 waiting for Cardiff K5 to free GPU 6+7)"
        fi
    fi

    # PPO v1
    if ! is_done rl_ppo_qwen3_hybrid_v1_2000steps/adapter_model.safetensors; then
        pid=$(get_pid ppo_v1)
        if ! is_alive "$pid"; then
            log "⚠️  PPO v1 died (PID=$pid) — restarting in 30s"
            sleep 30
            start_ppo_v1
        fi
    fi

    # PPO v2
    if ! is_done rl_ppo_qwen3_hybrid_v2_2000steps/adapter_model.safetensors; then
        pid=$(get_pid ppo_v2)
        if ! is_alive "$pid"; then
            log "⚠️  PPO v2 died (PID=$pid) — restarting in 30s"
            sleep 30
            start_ppo_v2
        fi
    fi

    # Law K5 — only restart if Sydney K5 already done (otherwise trigger_downstream handles start)
    if is_done rl_eval_results/baselines_sydney_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_law_k5_eval.json; then
            pid=$(get_pid law_k5)
            if ! is_alive "$pid"; then
                log "⚠️  Law K5 died (PID=$pid) — restarting in 15s"
                sleep 15
                start_law_k5
            fi
        fi
    fi

    # Med Y1 K5
    if is_done rl_eval_results/baselines_law_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
            pid=$(get_pid med_y1_k5)
            if ! is_alive "$pid"; then
                log "⚠️  Med Y1 K5 died (PID=$pid) — restarting in 15s"
                sleep 15
                start_med_y1_k5
            fi
        fi
    fi

    # Med Y2 K5
    if is_done rl_eval_results/baselines_med_y1_k5_eval.json; then
        if ! is_done rl_eval_results/baselines_med_y2_k5_eval.json; then
            pid=$(get_pid med_y2_k5)
            if ! is_alive "$pid"; then
                log "⚠️  Med Y2 K5 died (PID=$pid) — restarting in 15s"
                sleep 15
                start_med_y2_k5
            fi
        fi
    fi
}

# ── Main ─────────────────────────────────────────────────────

log "====== Experiment Watcher v2 Started ======"
log "Check interval: ${CHECK_INTERVAL}s"

# PIDs are read from .watcher_state (already written before launching watcher)
# Do NOT hardcode PIDs here — they change every restart
print_status

while true; do
    sleep "$CHECK_INTERVAL"
    check_and_restart
    trigger_downstream
    print_status
done

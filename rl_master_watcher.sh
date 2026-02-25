#!/bin/bash
# RL-ILearner Master Auto-Pipeline Watcher
# 全自动监控并依次执行所有训练/评估任务
# 运行于后台，不需要人工干预
#
# 负责：
#   [A] LLaMA-2-13B DPO v3:  pref_data_v3(已跑) → DPO v3训练 → 评估
#   [B] Qwen3-8B:             SFT(已跑) → pref_data_qwen3 → DPO qwen3 → 评估

LOG="/data/qbao775/Explanation-Generation/rl_master_watcher.log"
WORK_DIR="/data/qbao775/Explanation-Generation"
cd "$WORK_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "Master watcher started (PID $$)"
log "Monitoring: Qwen3-SFT + DPO-v3 pref data"
log "=========================================="

# ──────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────
wait_for_file() {
    # wait_for_file <path> <pid_to_watch> <label>
    local path="$1" pid="$2" label="$3"
    log "[$label] Waiting for: $path (watching PID $pid)"
    while true; do
        if [ -f "$path" ] && [ -s "$path" ]; then
            if python3 -c "import json; d=json.load(open('$path')); print(len(d),' items')" 2>/dev/null; then
                log "[$label] File ready: $path"
                return 0
            fi
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            log "[$label] WARNING: PID $pid died. Checking file one more time..."
            if [ -f "$path" ] && [ -s "$path" ]; then
                log "[$label] File exists after process exit — proceeding."
                return 0
            fi
            log "[$label] ERROR: Process died and file not found. Aborting this branch."
            return 1
        fi
        sleep 60
    done
}

wait_for_dir() {
    # wait_for_dir <dir> <pid_to_watch> <label>
    # 等待最终 adapter 存在 AND 进程已退出（避免 epoch checkpoint 误触发）
    local dir="$1" pid="$2" label="$3"
    log "[$label] Waiting for adapter dir: $dir (watching PID $pid)"
    while true; do
        # 先检查进程是否已结束
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            sleep 60; continue  # 进程还在跑，继续等
        fi
        # 进程结束后检查 adapter 文件
        if [ -d "$dir" ] && ls "$dir"/adapter_config.json 2>/dev/null; then
            log "[$label] Adapter dir ready: $dir"
            return 0
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            log "[$label] WARNING: PID $pid died. Checking dir..."
            if [ -d "$dir" ] && ls "$dir"/*.json 2>/dev/null | head -1 | grep -q .; then
                log "[$label] Dir exists after process exit — proceeding."
                return 0
            fi
            log "[$label] ERROR: Process died and adapter dir not found."
            return 1
        fi
        sleep 60
    done
}

# ──────────────────────────────────────────────────────────
# Branch A: LLaMA-2-13B DPO v3
# ──────────────────────────────────────────────────────────
run_branch_a() {
    log "[A] === LLaMA-2-13B DPO v3 branch started ==="

    # A1: 等待 preference data v3
    wait_for_file "./rl_preference_data_v3/preference_pairs.json" "4018081" "A-pref" || return 1
    N_PAIRS=$(python3 -c "import json; print(len(json.load(open('./rl_preference_data_v3/preference_pairs.json'))))")
    log "[A] Preference pairs v3: $N_PAIRS pairs"

    # A2: DPO v3 训练 (GPU 4,5,6,7 — 等 Qwen3 SFT 用完 GPU 7 后再开始，此时用 4,5)
    # 先用 GPU 4,5 (DPO v3 pref gen 完成后 GPU 4,5 会释放)
    log "[A] Starting DPO v3 training on GPU 4,5,6..."
    CUDA_VISIBLE_DEVICES=4,5,6 conda run -n llm-tuning python3 rl_train_dpo.py \
        --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
        --sft_adapter_path ./rl_sft_llama2_13b_generator \
        --preference_data_path ./rl_preference_data_v3/preference_pairs.json \
        --output_dir ./rl_dpo_v3_llama2_13b_generator \
        --num_train_epochs 5 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing True \
        --learning_rate 5e-5 \
        --lora_r 16 --lora_alpha 32 \
        --bf16 True \
        > rl_dpo_v3_training.log 2>&1
    DPO_V3_EXIT=$?
    log "[A] DPO v3 training finished (exit=$DPO_V3_EXIT)"

    if [ $DPO_V3_EXIT -ne 0 ]; then
        log "[A] ERROR: DPO v3 training failed. Check rl_dpo_v3_training.log"
        return 1
    fi

    # A3: 评估 DPO v3 (Cardiff + Sydney)
    log "[A] Starting DPO v3 evaluation — Cardiff..."
    CUDA_VISIBLE_DEVICES=4,5 conda run -n llm-tuning python3 rl_evaluation.py \
        --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/dpo_v3_cardiff_eval.json \
        --base_model_path /data/shared/llama2/llama-2-13b-hf \
        --sft_lora_path ./rl_sft_llama2_13b_generator \
        --dpo_lora_path ./rl_dpo_v3_llama2_13b_generator \
        --device cuda:0 --verifier_device cuda:1 \
        --nli_device cpu \
        > rl_dpo_v3_cardiff_eval.log 2>&1
    log "[A] Cardiff eval done (exit=$?)"

    log "[A] Starting DPO v3 evaluation — Sydney..."
    CUDA_VISIBLE_DEVICES=4,5 conda run -n llm-tuning python3 rl_evaluation.py \
        --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/dpo_v3_sydney_eval.json \
        --base_model_path /data/shared/llama2/llama-2-13b-hf \
        --sft_lora_path ./rl_sft_llama2_13b_generator \
        --dpo_lora_path ./rl_dpo_v3_llama2_13b_generator \
        --device cuda:0 --verifier_device cuda:1 \
        --nli_device cpu \
        > rl_dpo_v3_sydney_eval.log 2>&1
    log "[A] Sydney eval done (exit=$?)"

    log "[A] === LLaMA-2-13B DPO v3 branch COMPLETE ==="
    # 打印结果
    python3 -c "
import json
for fname, label in [('./rl_eval_results/dpo_v3_cardiff_eval.json','Cardiff'),
                     ('./rl_eval_results/dpo_v3_sydney_eval.json','Sydney')]:
    try:
        d = json.load(open(fname))
        for r in d.get('results',[]):
            print(f\"[A] {label} | {r['model']}: BLEU={r.get('avg_bleu',0):.4f} NLI={r.get('avg_nli_entailment',0):.4f} Verifier={r.get('avg_verifier_score',0):.4f}\")
    except: pass
" 2>/dev/null | tee -a "$LOG"
}

# ──────────────────────────────────────────────────────────
# Branch B: Qwen3-8B SFT → pref → DPO → eval
# ──────────────────────────────────────────────────────────
run_branch_b() {
    log "[B] === Qwen3-8B branch started ==="

    # B1: 等待 SFT adapter
    wait_for_dir "./rl_sft_qwen3_8b_generator" "4113985" "B-sft" || return 1
    log "[B] SFT complete. Checking SFT log..."
    tail -5 rl_sft_qwen3_training.log 2>/dev/null | tee -a "$LOG"

    # B2: 生成偏好数据 (GPU 6+7: generator=cuda:0=GPU6, verifier=cuda:1=GPU7)
    # Qwen3-8B(~16GB) + Alpaca-7B verifier(~13GB) — GPU 7 单卡(81GB)足够放两个
    # 但用 CUDA_VISIBLE_DEVICES=7 时两个模型都映射到 cuda:0，串行不冲突
    log "[B] Starting Qwen3 preference data generation on GPU 7..."
    mkdir -p ./rl_preference_data_qwen3
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_build_preference_data_qwen3.py \
        --generator_path /data/shared/qwen3/Qwen3-8B \
        --lora_adapter_path ./rl_sft_qwen3_8b_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
        --output_path ./rl_preference_data_qwen3/preference_pairs.json \
        --num_samples 3 --min_score_gap 0.1 --max_questions 500 \
        --generator_device cuda:0 --verifier_device cuda:0 \
        > rl_pref_qwen3.log 2>&1
    PREF_EXIT=$?
    log "[B] Preference data gen done (exit=$PREF_EXIT)"

    if [ $PREF_EXIT -ne 0 ] || [ ! -f "./rl_preference_data_qwen3/preference_pairs.json" ]; then
        log "[B] ERROR: Preference data generation failed. Check rl_pref_qwen3.log"
        return 1
    fi
    N_PAIRS=$(python3 -c "import json; print(len(json.load(open('./rl_preference_data_qwen3/preference_pairs.json'))))")
    log "[B] Qwen3 preference pairs: $N_PAIRS"

    # B3: DPO 训练 (GPU 7)
    log "[B] Starting Qwen3-8B DPO training on GPU 7..."
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_train_dpo_qwen3.py \
        --model_name_or_path /data/shared/qwen3/Qwen3-8B \
        --sft_adapter_path ./rl_sft_qwen3_8b_generator \
        --preference_data_path ./rl_preference_data_qwen3/preference_pairs.json \
        --output_dir ./rl_dpo_qwen3_8b_generator \
        --num_train_epochs 5 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --beta 0.1 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --bf16 True \
        --save_strategy epoch \
        --logging_steps 10 \
        --report_to none \
        > rl_dpo_qwen3_training.log 2>&1
    DPO_EXIT=$?
    log "[B] Qwen3 DPO training done (exit=$DPO_EXIT)"

    if [ $DPO_EXIT -ne 0 ]; then
        log "[B] ERROR: DPO training failed. Check rl_dpo_qwen3_training.log"
        return 1
    fi

    # B4: 评估 Qwen3 SFT + DPO (需要一个适配 Qwen3 的评估脚本)
    log "[B] Qwen3 evaluation — running inference script..."
    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_evaluate_qwen3.py \
        --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path ./rl_sft_qwen3_8b_generator \
        --dpo_lora_path ./rl_dpo_qwen3_8b_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/qwen3_cardiff_eval.json \
        --device cuda:0 --verifier_device cuda:0 \
        --nli_device cpu \
        > rl_qwen3_cardiff_eval.log 2>&1
    log "[B] Qwen3 Cardiff eval done (exit=$?)"

    CUDA_VISIBLE_DEVICES=7 conda run -n qwen3-rl python3 rl_evaluate_qwen3.py \
        --test_data_path ./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path ./rl_sft_qwen3_8b_generator \
        --dpo_lora_path ./rl_dpo_qwen3_8b_generator \
        --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path ./rl_eval_results/qwen3_sydney_eval.json \
        --device cuda:0 --verifier_device cuda:0 \
        --nli_device cpu \
        > rl_qwen3_sydney_eval.log 2>&1
    log "[B] Qwen3 Sydney eval done (exit=$?)"

    log "[B] === Qwen3-8B branch COMPLETE ==="
}

# ──────────────────────────────────────────────────────────
# 并行运行两个 branch
# ──────────────────────────────────────────────────────────
run_branch_a &
PID_A=$!
run_branch_b &
PID_B=$!

log "Branch A PID: $PID_A | Branch B PID: $PID_B"

wait $PID_A
log "Branch A finished (exit=$?)"
wait $PID_B
log "Branch B finished (exit=$?)"

log "=========================================="
log "All branches complete. Final results:"
log "=========================================="
python3 -c "
import json, os
results_dir = './rl_eval_results'
for fname in sorted(os.listdir(results_dir)):
    if fname.endswith('.json') and 'eval' in fname:
        try:
            d = json.load(open(os.path.join(results_dir, fname)))
            print(f'--- {fname} ---')
            for r in d.get('results', []):
                print(f\"  {r['model']}: BLEU={r.get('avg_bleu',0):.4f} NLI={r.get('avg_nli_entailment',0):.4f} Verifier={r.get('avg_verifier_score',0):.4f}\")
        except Exception as e:
            print(f'  {fname}: error reading ({e})')
" 2>/dev/null | tee -a "$LOG"

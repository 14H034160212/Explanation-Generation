#!/bin/bash
# =============================================================================
# Auto-watcher: DPO v2 training + Intermediate evaluation
# Polls for rl_preference_data_v2/preference_pairs.json every 60s.
# When found: runs DPO v2 on GPUs 4,5 then evals SFT vs DPO vs DPO-v2.
#
# Usage: nohup bash rl_auto_dpo_v2.sh > rl_dpo_v2_pipeline.log 2>&1 &
# =============================================================================

# Use llm-tuning conda env executables directly (no conda activate needed)
PYTHON="/data/qbao775/miniconda3/envs/llm-tuning/bin/python"
TORCHRUN="/data/qbao775/miniconda3/envs/llm-tuning/bin/torchrun"
WORKDIR="/data/qbao775/Explanation-Generation"

BASE_MODEL="/data/shared/llama2/llama-2-13b-hf"
VERIFIER_PATH="${WORKDIR}/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
TEST_DATA="${WORKDIR}/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
SFT_OUTPUT="${WORKDIR}/rl_sft_llama2_13b_generator"
DPO_OUTPUT="${WORKDIR}/rl_dpo_llama2_13b_generator"
DPO_V2_OUTPUT="${WORKDIR}/rl_dpo_v2_llama2_13b_generator"
PREF_V2="${WORKDIR}/rl_preference_data_v2/preference_pairs.json"
EVAL_OUTPUT="${WORKDIR}/rl_eval_results/intermediate_results.json"
MASTER_PORT=29502

export TOKENIZERS_PARALLELISM=false
cd "${WORKDIR}"

echo "========================================================"
echo "  DPO v2 Auto-watcher started at $(date)"
echo "  Polling for: ${PREF_V2}"
echo "========================================================"

# ─── Poll until preference data v2 is ready ───
while [ ! -f "${PREF_V2}" ]; do
    echo "[$(date '+%H:%M:%S')] Waiting for preference_pairs.json ..."
    sleep 60
done

echo ""
echo "[$(date '+%H:%M:%S')] >>> Preference data v2 FOUND!"
PAIR_COUNT=$("${PYTHON}" -c "import json; d=json.load(open('${PREF_V2}')); print(len(d))" 2>/dev/null || echo "?")
echo "[$(date '+%H:%M:%S')] Total pairs: ${PAIR_COUNT}"

# ─── Step 3A-v2: DPO training on GPUs 4,5 ───
echo ""
echo "[$(date '+%H:%M:%S')] >>> STEP 3A-v2: DPO v2 training on GPUs 4,5"
echo "    Pairs: ${PAIR_COUNT}  Output: ${DPO_V2_OUTPUT}"

CUDA_VISIBLE_DEVICES=4,5 "${TORCHRUN}" \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    "${WORKDIR}/rl_train_dpo.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "${SFT_OUTPUT}" \
    --preference_data_path "${PREF_V2}" \
    --output_dir "${DPO_V2_OUTPUT}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --beta 0.1 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --min_score_gap_filter 0.05 \
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
    --run_name "dpo-v2-llama2-13b"

echo ""
echo "[$(date '+%H:%M:%S')] >>> DPO v2 DONE → ${DPO_V2_OUTPUT}"

# ─── Intermediate Eval: SFT vs DPO vs DPO-v2 (GPUs 4,5) ───
echo ""
echo "[$(date '+%H:%M:%S')] >>> Intermediate evaluation: SFT vs DPO vs DPO-v2"

CUDA_VISIBLE_DEVICES=4,5 "${PYTHON}" "${WORKDIR}/rl_evaluation.py" \
    --test_data_path "${TEST_DATA}" \
    --verifier_path "${VERIFIER_PATH}" \
    --output_path "${EVAL_OUTPUT}" \
    --sft_model_path "${BASE_MODEL}" \
    --sft_lora_path "${SFT_OUTPUT}" \
    --dpo_model_path "${BASE_MODEL}" \
    --dpo_lora_path "${DPO_OUTPUT}" \
    --dpo_v2_model_path "${BASE_MODEL}" \
    --dpo_v2_lora_path "${DPO_V2_OUTPUT}" \
    --ppo_model_path "${BASE_MODEL}" \
    --ppo_lora_path "${WORKDIR}/rl_ppo_llama2_13b_generator" \
    --device cuda:0 \
    --verifier_device cuda:1 \
    --cache_dir cache

echo ""
echo "========================================================"
echo "  Intermediate eval DONE at $(date)"
echo "  Results saved to: ${EVAL_OUTPUT}"
echo "========================================================"

# Print summary table
"${PYTHON}" - <<'PYEOF'
import json
try:
    with open("./rl_eval_results/intermediate_results.json") as f:
        data = json.load(f)
    print("\n=== INTERMEDIATE RESULTS (SFT vs DPO vs DPO-v2) ===")
    print(f"{'Model':<25} {'BLEU':>8} {'BERTScore':>10} {'Verifier':>10} {'Time(s)':>9}")
    print("-" * 65)
    for r in data:
        name = r.get("model_name", "?")
        bleu  = r.get("avg_bleu", 0)
        bert  = r.get("avg_bert_score", 0)
        ver   = r.get("avg_verifier_score", 0)
        t     = r.get("total_time", 0)
        print(f"{name:<25} {bleu:>8.4f} {bert:>10.4f} {ver:>10.4f} {t:>9.1f}")
except Exception as e:
    print(f"Could not parse results: {e}", file=__import__('sys').stderr)
PYEOF

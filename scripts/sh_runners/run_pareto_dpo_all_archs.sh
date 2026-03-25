#!/bin/bash
# =============================================================================
# Pareto-Dominant DPO Pipeline — All Architectures
# Chosen must be >= rejected on BOTH ACR AND NLI simultaneously.
# This ensures every DPO gradient pushes both metrics upward, eliminating
# the per-architecture trade-off seen with verifier-only pair selection.
#
# Usage:
#   bash scripts/sh_runners/run_pareto_dpo_all_archs.sh [qwen3|llama2|llama3|all]
# Default: all
# =============================================================================
set -euo pipefail

MODEL=${1:-all}
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/pareto_dpo
mkdir -p "$LOG_DIR"

PYTHON_QWEN3=/data/qbao775/miniconda3/envs/qwen3-rl/bin/python3
PYTHON_LLAMA=/data/qbao775/miniconda3/envs/llm-tuning/bin/python3

BASE_QWEN3=/data/shared/qwen3/Qwen3-8B
BASE_LLAMA2=/data/shared/llama2/llama-2-13b-hf
BASE_LLAMA3=/data/shared/llama3/Meta-Llama-3-8B-hf

SFT_QWEN3=$BASE/rl_sft_qwen3_8b_generator
SFT_LLAMA2=$BASE/rl_sft_llama2_13b_generator
SFT_LLAMA3=$BASE/rl_sft_llama3_8b_generator

VERIFIER=$BASE/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA=$BASE/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json

N_SAMPLES=5
MAX_Q=500
MIN_COMBINED_DELTA=0.05
DPO_EPOCHS=5
DPO_BS=1
DPO_GRAD_ACCUM=8

DOMAINS=(cardiff sydney law med_y1 med_y2)
declare -A TEST_FILES=(
  [cardiff]="$BASE/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
  [sydney]="$BASE/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"
  [law]="$BASE/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"
  [med_y1]="$BASE/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"
  [med_y2]="$BASE/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
)

echo "======================================================"
echo " Pareto-Dominant DPO Pipeline  $(date)"
echo " Architectures: $MODEL"
echo " N_samples=$N_SAMPLES  min_combined_delta=$MIN_COMBINED_DELTA"
echo "======================================================"

# =============================================================================
# QWEN3-8B
# =============================================================================
if [[ "$MODEL" == "qwen3" || "$MODEL" == "all" ]]; then

  PREF_OUT=$BASE/rl_preference_data_pareto_qwen3/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_pareto_qwen3_8b_generator
  mkdir -p "$(dirname "$PREF_OUT")" "$DPO_OUT"

  echo ""
  echo "--- [Qwen3-8B] Stage 1: Pareto-dominant preference data ---"
  CUDA_VISIBLE_DEVICES=6 $PYTHON_QWEN3 "$SCRIPTS/rl_build_preference_data_pareto_qwen3.py" \
      --generator_path "$BASE_QWEN3" \
      --lora_adapter_path "$SFT_QWEN3" \
      --verifier_path "$VERIFIER" \
      --data_path "$DATA" \
      --output_path "$PREF_OUT" \
      --num_samples $N_SAMPLES \
      --max_questions $MAX_Q \
      --min_combined_delta $MIN_COMBINED_DELTA \
      --generator_device cuda:0 \
      --verifier_device cuda:0 \
      --nli_device cuda:0 \
      2>&1 | tee "$LOG_DIR/qwen3_build_pref.log"

  echo ""
  echo "--- [Qwen3-8B] Stage 2: DPO training ---"
  CUDA_VISIBLE_DEVICES=6,7 $PYTHON_QWEN3 "$SCRIPTS/rl_train_dpo_qwen3.py" \
      --model_name_or_path "$BASE_QWEN3" \
      --sft_adapter_path "$SFT_QWEN3" \
      --preference_data_path "$PREF_OUT" \
      --output_dir "$DPO_OUT" \
      --num_train_epochs $DPO_EPOCHS \
      --per_device_train_batch_size $DPO_BS \
      --gradient_accumulation_steps $DPO_GRAD_ACCUM \
      --bf16 True \
      --report_to none \
      2>&1 | tee "$LOG_DIR/qwen3_dpo_train.log"

  echo ""
  echo "--- [Qwen3-8B] Stage 3: Evaluation on all 5 domains ---"
  for DOMAIN in "${DOMAINS[@]}"; do
    OUT_FILE="$BASE/rl_eval_results/qwen3_dpo_pareto_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=6 $PYTHON_QWEN3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path "$BASE_QWEN3" \
        --sft_lora_path "$SFT_QWEN3" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "${TEST_FILES[$DOMAIN]}" \
        --verifier_path "$VERIFIER" \
        --output_path "$OUT_FILE" \
        --device cuda:0 \
        --verifier_device cuda:0 \
        2>&1 | tee "$LOG_DIR/qwen3_eval_${DOMAIN}.log"
  done

fi

# =============================================================================
# LLAMA-2-13B
# =============================================================================
if [[ "$MODEL" == "llama2" || "$MODEL" == "all" ]]; then

  PREF_OUT=$BASE/rl_preference_data_pareto_llama2/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_pareto_llama2_13b_generator
  mkdir -p "$(dirname "$PREF_OUT")" "$DPO_OUT"

  echo ""
  echo "--- [LLaMA-2-13B] Stage 1: Pareto-dominant preference data ---"
  CUDA_VISIBLE_DEVICES=4 $PYTHON_LLAMA "$SCRIPTS/rl_build_preference_data_pareto.py" \
      --generator_path "$BASE_LLAMA2" \
      --lora_adapter_path "$SFT_LLAMA2" \
      --verifier_path "$VERIFIER" \
      --data_path "$DATA" \
      --output_path "$PREF_OUT" \
      --num_samples $N_SAMPLES \
      --max_questions $MAX_Q \
      --min_combined_delta $MIN_COMBINED_DELTA \
      --generator_device cuda:0 \
      --verifier_device cuda:0 \
      --nli_device cuda:0 \
      2>&1 | tee "$LOG_DIR/llama2_build_pref.log"

  echo ""
  echo "--- [LLaMA-2-13B] Stage 2: DPO training ---"
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port 29504 \
      "$SCRIPTS/rl_train_dpo.py" \
      --model_name_or_path "$BASE_LLAMA2" \
      --sft_adapter_path "$SFT_LLAMA2" \
      --preference_data_path "$PREF_OUT" \
      --output_dir "$DPO_OUT" \
      --num_train_epochs $DPO_EPOCHS \
      --per_device_train_batch_size $DPO_BS \
      --gradient_accumulation_steps $DPO_GRAD_ACCUM \
      --beta 0.1 \
      --bf16 True \
      --report_to none \
      2>&1 | tee "$LOG_DIR/llama2_dpo_train.log"

  echo ""
  echo "--- [LLaMA-2-13B] Stage 3: Evaluation on all 5 domains ---"
  for DOMAIN in "${DOMAINS[@]}"; do
    OUT_FILE="$BASE/rl_eval_results/llama2_dpo_pareto_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=4 $PYTHON_LLAMA "$SCRIPTS/rl_evaluation.py" \
        --sft_model_path "$BASE_LLAMA2" \
        --sft_lora_path "$SFT_LLAMA2" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "${TEST_FILES[$DOMAIN]}" \
        --verifier_path "$VERIFIER" \
        --output_path "$OUT_FILE" \
        --device cuda:0 \
        --verifier_device cuda:0 \
        2>&1 | tee "$LOG_DIR/llama2_eval_${DOMAIN}.log"
  done

fi

# =============================================================================
# LLAMA-3-8B
# =============================================================================
if [[ "$MODEL" == "llama3" || "$MODEL" == "all" ]]; then

  PREF_OUT=$BASE/rl_preference_data_pareto_llama3/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_pareto_llama3_8b_generator
  mkdir -p "$(dirname "$PREF_OUT")" "$DPO_OUT"

  echo ""
  echo "--- [LLaMA-3-8B] Stage 1: Pareto-dominant preference data ---"
  CUDA_VISIBLE_DEVICES=5 $PYTHON_LLAMA "$SCRIPTS/rl_build_preference_data_pareto.py" \
      --generator_path "$BASE_LLAMA3" \
      --lora_adapter_path "$SFT_LLAMA3" \
      --verifier_path "$VERIFIER" \
      --data_path "$DATA" \
      --output_path "$PREF_OUT" \
      --num_samples $N_SAMPLES \
      --max_questions $MAX_Q \
      --min_combined_delta $MIN_COMBINED_DELTA \
      --generator_device cuda:0 \
      --verifier_device cuda:0 \
      --nli_device cuda:0 \
      2>&1 | tee "$LOG_DIR/llama3_build_pref.log"

  echo ""
  echo "--- [LLaMA-3-8B] Stage 2: DPO training ---"
  CUDA_VISIBLE_DEVICES=5 $PYTHON_LLAMA "$SCRIPTS/rl_train_dpo.py" \
      --model_name_or_path "$BASE_LLAMA3" \
      --sft_adapter_path "$SFT_LLAMA3" \
      --preference_data_path "$PREF_OUT" \
      --output_dir "$DPO_OUT" \
      --num_train_epochs $DPO_EPOCHS \
      --per_device_train_batch_size $DPO_BS \
      --gradient_accumulation_steps $DPO_GRAD_ACCUM \
      --beta 0.1 \
      --bf16 True \
      --report_to none \
      2>&1 | tee "$LOG_DIR/llama3_dpo_train.log"

  echo ""
  echo "--- [LLaMA-3-8B] Stage 3: Evaluation on all 5 domains ---"
  for DOMAIN in "${DOMAINS[@]}"; do
    OUT_FILE="$BASE/rl_eval_results/llama3_dpo_pareto_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=5 $PYTHON_LLAMA "$SCRIPTS/rl_evaluation.py" \
        --sft_model_path "$BASE_LLAMA3" \
        --sft_lora_path "$SFT_LLAMA3" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "${TEST_FILES[$DOMAIN]}" \
        --verifier_path "$VERIFIER" \
        --output_path "$OUT_FILE" \
        --device cuda:0 \
        --verifier_device cuda:0 \
        2>&1 | tee "$LOG_DIR/llama3_eval_${DOMAIN}.log"
  done

fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================"
echo " All done!  $(date)"
echo " Results in: $BASE/rl_eval_results/  (*_pareto_*)"
echo "======================================================"

$PYTHON_QWEN3 - <<'PYEOF'
import json, os, glob

BASE = "/data/qbao775/Explanation-Generation/rl_eval_results"
files = sorted(glob.glob(f"{BASE}/*_pareto_*_eval.json"))
if not files:
    print("No pareto eval results found yet.")
else:
    print(f"{'File':<55} {'ACR':>6} {'NLI':>6} {'Ver':>5}")
    print("-" * 78)
    for path in files:
        try:
            d = json.load(open(path))
            for r in d.get("results", []):
                name = os.path.basename(path).replace("_eval.json","")
                acr = r.get("avg_answer_coverage_rate", 0)
                nli = r.get("avg_nli_entailment", 0)
                ver = r.get("avg_verifier_score", 0)
                print(f"{name:<55} {acr:>6.4f} {nli:>6.4f} {ver:>5.3f}")
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")
PYEOF

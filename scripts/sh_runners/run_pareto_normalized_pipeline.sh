#!/bin/bash
# ============================================================
# Pareto-Normalized DPO Pipeline
# R_i = w_nli*norm_q(NLI_i) + w_acr*norm_q(ACR_i) + w_ver*norm_q(Ver_i) - gamma*len
# + Pareto filter: chosen must beat rejected on >= 2 of 3 metrics
#
# Key difference from multiplicative_acr:
#   - Within-question min-max normalization → each metric contributes equally
#   - Soft ACR weight (not hard gate) → NLI signal preserved
#   - Pareto filter → chosen genuinely better on multiple axes simultaneously
# ============================================================
set -euo pipefail

MODEL=${1:-both}   # qwen3 | llama2 | both
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/pareto_normalized
mkdir -p "$LOG_DIR"

W_NLI=0.5    # NLI dominant but not overwhelming
W_ACR=0.3    # ACR soft weight (not gate)
W_LEN=0.002
MIN_GAP=0.05
N_SAMPLES=5
MAX_Q=500
PARETO_MIN_WINS=2

echo "======================================================"
echo " Pareto-Normalized DPO Pipeline  $(date)"
echo " w_nli=$W_NLI w_acr=$W_ACR gamma=$W_LEN pareto_min_wins=$PARETO_MIN_WINS"
echo "======================================================"

# ---- QWEN3 branch ----
if [[ "$MODEL" == "qwen3" || "$MODEL" == "both" ]]; then

  PREF_OUT=$BASE/rl_preference_data_pareto_qwen3/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_pareto_qwen3_8b_generator
  mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT"

  echo ""
  echo "--- [Qwen3] Stage 1: Building pareto-normalized preference data ---"
  CUDA_VISIBLE_DEVICES=4,6 conda run -n qwen3-rl python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
      --model_type qwen3 \
      --score_method pareto_normalized \
      --generator_path /data/shared/qwen3/Qwen3-8B \
      --lora_adapter_path "$BASE/models/rl_sft_qwen3_8b_generator" \
      --data_path "$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
      --output_path "$PREF_OUT" \
      --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
      --num_samples $N_SAMPLES \
      --min_score_gap $MIN_GAP \
      --max_questions $MAX_Q \
      --w_nli $W_NLI \
      --w_acr $W_ACR \
      --w_len_penalty $W_LEN \
      --pareto_min_wins $PARETO_MIN_WINS \
      --generator_device cuda:0 \
      --verifier_device cuda:1 \
      --nli_device cpu \
      2>&1 | tee "$LOG_DIR/qwen3_build_pref.log"

  echo ""
  echo "--- [Qwen3] Stage 2: DPO training ---"
  CUDA_VISIBLE_DEVICES=4 conda run -n qwen3-rl python3 "$SCRIPTS/rl_train_dpo_qwen3.py" \
      --model_name_or_path /data/shared/qwen3/Qwen3-8B \
      --sft_adapter_path "$BASE/models/rl_sft_qwen3_8b_generator" \
      --preference_data_path "$PREF_OUT" \
      --output_dir "$DPO_OUT" \
      --num_train_epochs 5 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --bf16 True \
      --report_to none \
      2>&1 | tee "$LOG_DIR/qwen3_dpo_train.log"

  echo ""
  echo "--- [Qwen3] Stage 3: Evaluating on all 5 domains ---"
  for DOMAIN in cardiff sydney law med_y1 med_y2; do
    case $DOMAIN in
      cardiff) TEST_FILE="$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" ;;
      sydney)  TEST_FILE="$BASE/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" ;;
      law)     TEST_FILE="$BASE/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" ;;
      med_y1)  TEST_FILE="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" ;;
      med_y2)  TEST_FILE="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" ;;
    esac
    OUT_FILE="$BASE/rl_eval_results/qwen3_dpo_pareto_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=4 conda run -n qwen3-rl python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --dpo_lora_path "$DPO_OUT" \
        --sft_lora_path "$BASE/models/rl_sft_qwen3_8b_generator" \
        --test_data_path "$TEST_FILE" \
        --output_path "$OUT_FILE" \
        --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/qwen3_eval_${DOMAIN}.log"
  done
fi

# ---- LLAMA-2 branch ----
if [[ "$MODEL" == "llama2" || "$MODEL" == "both" ]]; then

  PREF_OUT2=$BASE/rl_preference_data_pareto_llama2/preference_pairs.json
  DPO_OUT2=$BASE/rl_dpo_pareto_llama2_13b_generator
  mkdir -p "$(dirname $PREF_OUT2)" "$DPO_OUT2"

  echo ""
  echo "--- [LLaMA-2] Stage 1: Building pareto-normalized preference data ---"
  CUDA_VISIBLE_DEVICES=4,6 conda run -n llm-tuning python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
      --model_type llama2 \
      --score_method pareto_normalized \
      --generator_path /data/shared/llama2/llama-2-13b-hf \
      --lora_adapter_path "$BASE/models/rl_sft_llama2_13b_generator" \
      --data_path "$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
      --output_path "$PREF_OUT2" \
      --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
      --num_samples $N_SAMPLES \
      --min_score_gap $MIN_GAP \
      --max_questions $MAX_Q \
      --w_nli $W_NLI \
      --w_acr $W_ACR \
      --w_len_penalty $W_LEN \
      --pareto_min_wins $PARETO_MIN_WINS \
      --generator_device cuda:0 \
      --verifier_device cuda:1 \
      --nli_device cpu \
      2>&1 | tee "$LOG_DIR/llama2_build_pref.log"

  echo ""
  echo "--- [LLaMA-2] Stage 2: DPO training ---"
  CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port 29504 \
      "$SCRIPTS/rl_train_dpo.py" \
      --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
      --sft_adapter_path "$BASE/models/rl_sft_llama2_13b_generator" \
      --preference_data_path "$PREF_OUT2" \
      --output_dir "$DPO_OUT2" \
      --num_train_epochs 5 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --beta 0.1 \
      --bf16 True \
      --report_to none \
      2>&1 | tee "$LOG_DIR/llama2_dpo_train.log"

  echo ""
  echo "--- [LLaMA-2] Stage 3: Evaluating on all 5 domains ---"
  for DOMAIN in cardiff sydney law med_y1 med_y2; do
    case $DOMAIN in
      cardiff) TEST_FILE="$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json" ;;
      sydney)  TEST_FILE="$BASE/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json" ;;
      law)     TEST_FILE="$BASE/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" ;;
      med_y1)  TEST_FILE="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" ;;
      med_y2)  TEST_FILE="$BASE/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" ;;
    esac
    OUT_FILE="$BASE/rl_eval_results/llama2_dpo_pareto_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=4,6 conda run -n llm-tuning python3 "$SCRIPTS/rl_evaluation.py" \
        --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
        --lora_adapter_path "$DPO_OUT2" \
        --test_data_path "$TEST_FILE" \
        --output_path "$OUT_FILE" \
        --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
        --verifier_device cuda:1 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/llama2_eval_${DOMAIN}.log"
  done
fi

echo ""
echo "======================================================"
echo " All done!  $(date)"
echo " Results in: $BASE/rl_eval_results/  (*_pareto_*)"
echo "======================================================"

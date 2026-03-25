#!/bin/bash
# ============================================================
# Multiplicative-ACR Pipeline (matches methodology figure)
# R_i = (w_nli * S_nli * w_ver * S_ver_norm - gamma*len) * I[acr>=0.5]
#
# Stage 1: Build preference data (GPU4=generator, GPU6=verifier)
# Stage 2: DPO training  (Qwen3 single-GPU on GPU4; LLaMA-2 dual-GPU on GPU4+6)
# Stage 3: Eval all 5 domains (both models)
# Note: GPU5 (ollama) and GPU7 (uvicorn) are occupied by other services.
#       Both models run sequentially on CUDA_VISIBLE_DEVICES=4,6.
#
# Usage:
#   bash run_multiplicative_acr_pipeline.sh [qwen3|llama2|both]
# ============================================================
set -euo pipefail

MODEL=${1:-both}   # qwen3 | llama2 | both
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/multiplicative_acr
mkdir -p "$LOG_DIR"

# Hyperparameters (matching methodology figure defaults)
W_NLI=0.7
W_VER=0.3
W_LEN=0.002
ACR_THRESH=0.5
MIN_GAP=0.05
N_SAMPLES=5
MAX_Q=500

echo "======================================================"
echo " Multiplicative-ACR Pipeline  $(date)"
echo " MODEL=$MODEL  w_nli=$W_NLI w_ver=$W_VER gamma=$W_LEN acr_thresh=$ACR_THRESH"
echo "======================================================"

# -------------------------------------------------------
# QWEN3-8B branch
# -------------------------------------------------------
if [[ "$MODEL" == "qwen3" || "$MODEL" == "both" ]]; then

  PREF_OUT=$BASE/rl_preference_data_multiplicative_qwen3/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_multiplicative_qwen3_8b_generator
  mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT"

  echo ""
  echo "--- [Qwen3] Stage 1: Building multiplicative-ACR preference data ---"
  CUDA_VISIBLE_DEVICES=4,6 conda run -n qwen3-rl python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
      --model_type qwen3 \
      --score_method multiplicative_acr \
      --generator_path /data/shared/qwen3/Qwen3-8B \
      --lora_adapter_path "$BASE/models/rl_sft_qwen3_8b_generator" \
      --data_path "$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
      --output_path "$PREF_OUT" \
      --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
      --num_samples $N_SAMPLES \
      --min_score_gap $MIN_GAP \
      --max_questions $MAX_Q \
      --w_nli $W_NLI \
      --w_ver $W_VER \
      --w_len_penalty $W_LEN \
      --acr_threshold $ACR_THRESH \
      --generator_device cuda:0 \
      --verifier_device cuda:1 \
      --nli_device cpu \
      2>&1 | tee "$LOG_DIR/qwen3_build_pref.log"

  echo ""
  echo "--- [Qwen3] Stage 2: DPO training on multiplicative-ACR pairs ---"
  CUDA_VISIBLE_DEVICES=4,6 conda run -n qwen3-rl python3 "$SCRIPTS/rl_train_dpo_qwen3.py" \
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
    OUT_FILE="$BASE/rl_eval_results/qwen3_dpo_multiplicative_acr_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=4,6 conda run -n qwen3-rl python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --model_path /data/shared/qwen3/Qwen3-8B \
        --lora_path "$DPO_OUT" \
        --sft_lora_path "$BASE/models/rl_sft_qwen3_8b_generator" \
        --test_data_path "$TEST_FILE" \
        --output_path "$OUT_FILE" \
        --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
        --verifier_device cuda:1 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/qwen3_eval_${DOMAIN}.log"
  done

fi

# -------------------------------------------------------
# LLAMA-2-13B branch
# -------------------------------------------------------
if [[ "$MODEL" == "llama2" || "$MODEL" == "both" ]]; then

  PREF_OUT=$BASE/rl_preference_data_multiplicative_llama2/preference_pairs.json
  DPO_OUT=$BASE/rl_dpo_multiplicative_llama2_13b_generator
  mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT"

  echo ""
  echo "--- [LLaMA-2] Stage 1: Building multiplicative-ACR preference data ---"
  CUDA_VISIBLE_DEVICES=4,6 conda run -n llm-tuning python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
      --model_type llama2 \
      --score_method multiplicative_acr \
      --generator_path /data/shared/llama2/llama-2-13b-hf \
      --lora_adapter_path "$BASE/models/rl_sft_llama2_13b_generator" \
      --data_path "$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
      --output_path "$PREF_OUT" \
      --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
      --num_samples $N_SAMPLES \
      --min_score_gap $MIN_GAP \
      --max_questions $MAX_Q \
      --w_nli $W_NLI \
      --w_ver $W_VER \
      --w_len_penalty $W_LEN \
      --acr_threshold $ACR_THRESH \
      --generator_device cuda:0 \
      --verifier_device cuda:1 \
      --nli_device cpu \
      2>&1 | tee "$LOG_DIR/llama2_build_pref.log"

  echo ""
  echo "--- [LLaMA-2] Stage 2: DPO training on multiplicative-ACR pairs ---"
  CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port 29503 \
      "$SCRIPTS/rl_train_dpo.py" \
      --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
      --sft_adapter_path "$BASE/models/rl_sft_llama2_13b_generator" \
      --preference_data_path "$PREF_OUT" \
      --output_dir "$DPO_OUT" \
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
    OUT_FILE="$BASE/rl_eval_results/llama2_dpo_multiplicative_acr_${DOMAIN}_eval.json"
    echo "  Evaluating $DOMAIN → $OUT_FILE"
    CUDA_VISIBLE_DEVICES=4,6 conda run -n llm-tuning python3 "$SCRIPTS/rl_evaluation.py" \
        --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
        --lora_adapter_path "$DPO_OUT" \
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
echo " Results in: $BASE/rl_eval_results/  (multiplicative_acr_*)"
echo " Logs in:    $LOG_DIR"
echo "======================================================"

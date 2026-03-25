#!/bin/bash
# Resume multiplicative ACR pipeline from DPO training stage
# (pref data already built: rl_preference_data_multiplicative_qwen3/preference_pairs.json)
set -euo pipefail

BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/multiplicative_acr

echo "======================================================"
echo " Resuming Multiplicative-ACR Pipeline  $(date)"
echo " Starting from Qwen3 DPO training"
echo "======================================================"

# ---- Qwen3 DPO training ----
PREF_OUT=$BASE/rl_preference_data_multiplicative_qwen3/preference_pairs.json
DPO_OUT=$BASE/rl_dpo_multiplicative_qwen3_8b_generator
mkdir -p "$DPO_OUT"

echo ""
echo "--- [Qwen3] Stage 2: DPO training (147 pairs, 5 epochs) ---"
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
  OUT_FILE="$BASE/rl_eval_results/qwen3_dpo_multiplicative_acr_${DOMAIN}_eval.json"
  echo "  Evaluating $DOMAIN → $OUT_FILE"
  CUDA_VISIBLE_DEVICES=4 conda run -n qwen3-rl python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
      --model_path /data/shared/qwen3/Qwen3-8B \
      --lora_path "$DPO_OUT" \
      --sft_lora_path "$BASE/models/rl_sft_qwen3_8b_generator" \
      --test_data_path "$TEST_FILE" \
      --output_path "$OUT_FILE" \
      --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
      --verifier_device cuda:0 \
      --nli_device cpu \
      2>&1 | tee "$LOG_DIR/qwen3_eval_${DOMAIN}.log"
done

# ---- LLaMA-2 full pipeline ----
PREF_OUT2=$BASE/rl_preference_data_multiplicative_llama2/preference_pairs.json
DPO_OUT2=$BASE/rl_dpo_multiplicative_llama2_13b_generator
mkdir -p "$(dirname $PREF_OUT2)" "$DPO_OUT2"

echo ""
echo "--- [LLaMA-2] Stage 1: Building multiplicative-ACR preference data ---"
CUDA_VISIBLE_DEVICES=4,6 conda run -n llm-tuning python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
    --model_type llama2 \
    --score_method multiplicative_acr \
    --generator_path /data/shared/llama2/llama-2-13b-hf \
    --lora_adapter_path "$BASE/models/rl_sft_llama2_13b_generator" \
    --data_path "$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json" \
    --output_path "$PREF_OUT2" \
    --verifier_path "$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2" \
    --num_samples 5 \
    --min_score_gap 0.05 \
    --max_questions 500 \
    --w_nli 0.7 \
    --w_ver 0.3 \
    --w_len_penalty 0.002 \
    --acr_threshold 0.5 \
    --generator_device cuda:0 \
    --verifier_device cuda:1 \
    --nli_device cpu \
    2>&1 | tee "$LOG_DIR/llama2_build_pref.log"

echo ""
echo "--- [LLaMA-2] Stage 2: DPO training ---"
CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port 29503 \
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
  OUT_FILE="$BASE/rl_eval_results/llama2_dpo_multiplicative_acr_${DOMAIN}_eval.json"
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

echo ""
echo "======================================================"
echo " All done!  $(date)"
echo "======================================================"

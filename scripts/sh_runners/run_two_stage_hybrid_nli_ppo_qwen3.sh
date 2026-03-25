#!/bin/bash
# ============================================================
# Two-Stage Pipeline: Hybrid-DPO → NLI-PPO (Qwen3-8B)
#
# Stage 1: Build multiplicative_acr preference data
# Stage 2: Train Hybrid-DPO (offline, preference pairs)
# Stage 3: Continue with NLI-PPO (online, DeBERTa reward)
# Stage 4: Evaluate on all 5 domains
#
# Motivation: DPO provides rapid ACR alignment (offline).
# NLI-PPO then fine-tunes entailment quality online using a
# pure NLI reward signal, directly targeting the alignment tax.
# This two-stage approach is motivated by the Med Y2 result:
# PPO+NLI-SFT achieves NLI=0.333 vs PPO+NLI-DPO=0.293,
# suggesting online NLI-RL is highly effective as a second stage.
#
# Usage:
#   bash scripts/sh_runners/run_two_stage_hybrid_nli_ppo_qwen3.sh [GPU_ID]
#
# Outputs:
#   models/rl_dpo_hybrid_twostage_qwen3_8b_generator/   (after DPO)
#   models/rl_ppo_nli_twostage_qwen3_8b_generator/       (after PPO)
#   rl_eval_results/qwen3_twostage_{domain}_eval.json    (all 5 domains)
# ============================================================
set -euo pipefail

GPU=${1:-6}
BASE=/data/qbao775/Explanation-Generation
SCRIPTS=$BASE/scripts/python_training
LOG_DIR=$BASE/logs/two_stage_hybrid_nli_ppo
mkdir -p "$LOG_DIR"

BASE_MODEL=/data/shared/qwen3/Qwen3-8B
SFT_ADAPTER=$BASE/models/rl_sft_qwen3_8b_generator
VERIFIER_PATH=$BASE/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA_PATH=$BASE/preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json

PREF_OUT=$BASE/rl_preference_data_twostage_qwen3/preference_pairs.json
DPO_OUT=$BASE/rl_dpo_hybrid_twostage_qwen3_8b_generator
PPO_OUT=$BASE/rl_ppo_nli_twostage_qwen3_8b_generator
mkdir -p "$(dirname $PREF_OUT)" "$DPO_OUT" "$PPO_OUT"

echo "======================================================"
echo " Two-Stage Hybrid-DPO → NLI-PPO Pipeline  $(date)"
echo " GPU=$GPU"
echo " Stage 1: multiplicative_acr preference data (N=3)"
echo " Stage 2: Hybrid-DPO training"
echo " Stage 3: NLI-PPO online RL (init from DPO)"
echo "======================================================"

# ---- Stage 1: Build Hybrid preference data ----
echo ""
echo "--- Stage 1: Building Hybrid preference data (multiplicative_acr) ---"
CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
    python3 "$SCRIPTS/rl_build_preference_data_nli.py" \
    --model_type qwen3 \
    --score_method multiplicative_acr \
    --generator_path "$BASE_MODEL" \
    --lora_adapter_path "$SFT_ADAPTER" \
    --data_path "$DATA_PATH" \
    --output_path "$PREF_OUT" \
    --verifier_path "$VERIFIER_PATH" \
    --num_samples 3 \
    --min_score_gap 0.1 \
    --max_questions 500 \
    --w_nli 0.7 \
    --w_ver 0.3 \
    --w_len_penalty 0.002 \
    --acr_threshold 0.5 \
    --generator_device cuda:0 \
    --verifier_device cuda:0 \
    --nli_device cpu \
    2>&1 | tee "$LOG_DIR/stage1_build_pref.log"

N_PAIRS=$(python3 -c "import json; d=json.load(open('$PREF_OUT')); print(len(d))")
echo "  -> $N_PAIRS preference pairs"

# ---- Stage 2: Hybrid-DPO training ----
echo ""
echo "--- Stage 2: Hybrid-DPO training ---"
CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
    python3 "$SCRIPTS/rl_train_dpo_qwen3.py" \
    --model_name_or_path "$BASE_MODEL" \
    --sft_adapter_path "$SFT_ADAPTER" \
    --preference_data_path "$PREF_OUT" \
    --output_dir "$DPO_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --report_to none \
    2>&1 | tee "$LOG_DIR/stage2_dpo_train.log"

echo "  -> Hybrid-DPO adapter saved to $DPO_OUT"

# ---- Stage 3: NLI-PPO (initialised from DPO adapter) ----
echo ""
echo "--- Stage 3: NLI-PPO online RL (init from Hybrid-DPO) ---"
# Key design: pass Hybrid-DPO adapter as sft_adapter_path so PPO
# starts from the DPO-aligned checkpoint, not raw SFT.
CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
    python3 "$SCRIPTS/rl_train_ppo_nli_qwen3.py" \
    --model_name_or_path "$BASE_MODEL" \
    --sft_adapter_path "$DPO_OUT" \
    --output_dir "$PPO_OUT" \
    --data_path "$DATA_PATH" \
    --batch_size 4 \
    --mini_batch_size 1 \
    --ppo_epochs 4 \
    --learning_rate 1e-5 \
    --max_questions 500 \
    --num_epochs 1 \
    --max_global_steps 500 \
    --save_every_steps 100 \
    --nli_device cpu \
    2>&1 | tee "$LOG_DIR/stage3_ppo_nli.log"

echo "  -> NLI-PPO adapter saved to $PPO_OUT"

# ---- Stage 4: Evaluate both DPO and PPO checkpoints ----
echo ""
echo "--- Stage 4: Evaluating DPO and PPO checkpoints on all 5 domains ---"

CARDIFF_TEST=$BASE/preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
SYDNEY_TEST=$BASE/preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
LAW_TEST=$BASE/preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json
MEDY1_TEST=$BASE/preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json
MEDY2_TEST=$BASE/preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json

for DOMAIN in cardiff sydney law med_y1 med_y2; do
    case $DOMAIN in
        cardiff) TEST_FILE="$CARDIFF_TEST" ;;
        sydney)  TEST_FILE="$SYDNEY_TEST" ;;
        law)     TEST_FILE="$LAW_TEST" ;;
        med_y1)  TEST_FILE="$MEDY1_TEST" ;;
        med_y2)  TEST_FILE="$MEDY2_TEST" ;;
    esac

    # Evaluate DPO checkpoint
    DPO_EVAL=$BASE/rl_eval_results/qwen3_twostage_dpo_${DOMAIN}_eval.json
    echo "  [DPO] $DOMAIN..."
    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$SFT_ADAPTER" \
        --dpo_lora_path "$DPO_OUT" \
        --test_data_path "$TEST_FILE" \
        --output_path "$DPO_EVAL" \
        --verifier_path "$VERIFIER_PATH" \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/eval_dpo_${DOMAIN}.log"

    # Evaluate PPO checkpoint
    PPO_EVAL=$BASE/rl_eval_results/qwen3_twostage_ppo_${DOMAIN}_eval.json
    echo "  [PPO] $DOMAIN..."
    CUDA_VISIBLE_DEVICES=$GPU conda run -n qwen3-rl \
        python3 "$SCRIPTS/rl_evaluate_qwen3.py" \
        --base_model_path "$BASE_MODEL" \
        --sft_lora_path "$PPO_OUT" \
        --test_data_path "$TEST_FILE" \
        --output_path "$PPO_EVAL" \
        --verifier_path "$VERIFIER_PATH" \
        --verifier_device cuda:0 \
        --nli_device cpu \
        2>&1 | tee "$LOG_DIR/eval_ppo_${DOMAIN}.log"
done

# ---- Summary ----
echo ""
echo "======================================================"
echo " Two-Stage Pipeline Summary"
echo "======================================================"
printf "%-15s %-35s %-8s %-8s %-8s\n" "Domain" "Model" "NLI" "ACR" "Ver"
printf "%-15s %-35s %-8s %-8s %-8s\n" "------" "-----" "---" "---" "---"
for DOMAIN in cardiff sydney law med_y1 med_y2; do
    for STAGE in dpo ppo; do
        EVAL_OUT=$BASE/rl_eval_results/qwen3_twostage_${STAGE}_${DOMAIN}_eval.json
        if [ -f "$EVAL_OUT" ]; then
            python3 -c "
import json
d = json.load(open('$EVAL_OUT'))
for r in d['results']:
    if r['model'] != '':
        print(f\"$DOMAIN            {r['model']:<35} {r['avg_nli_entailment']:.4f}   {r['avg_answer_coverage_rate']:.4f}   {r['avg_verifier_score']:.2f}\")
        break
"
        fi
    done
done

echo ""
echo "Logs in: $LOG_DIR"
echo "Done!  $(date)"

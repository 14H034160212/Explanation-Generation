#!/bin/bash
# =============================================================================
# RLearner-LLM: Complete Training Pipeline
# =============================================================================
# Hardware: 4x A100 (80GB) — GPUs 4,5,6,7 (GPUs 0-3 reserved for other jobs)
# Environment: conda activate llm-tuning  (TRL 0.7.1, Transformers 4.31.0)
# Model: LLaMA-2-13B-HF (at /data/shared/llama2/llama-2-13b-hf)
#
# Pipeline:
#   Step 1: SFT fine-tune LLaMA-2-13B on PeerWise data (LoRA, ~6-8h)
#   Step 2: Build DPO preference pairs (multi-sample + verifier scoring)
#   Step 2.5: Synthetic data augmentation (optional — needs API key)
#   Step 3A: DPO training (RECOMMENDED — stable, efficient, ~3-4h)
#   Step 3B: PPO training (ALTERNATIVE — uncomment to use instead)
#   Step 4: Evaluate and compare all models
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Base model (LLaMA-2-13B HuggingFace format, available on server)
BASE_MODEL="/data/shared/llama2/llama-2-13b-hf"

# Trained verifier (Alpaca-7B way-2 evaluator trained on Cardiff+Sydney merged data)
VERIFIER_PATH="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"

# Data paths
MERGED_TRAIN_DATA="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
TEST_DATA="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"

# Output directories
SFT_OUTPUT="./rl_sft_llama2_13b_generator"
PREFERENCE_DATA="./rl_preference_data/preference_pairs.json"
# Use v2 preference data (min_score_gap=0.1, 3 samples → more pairs) when available
PREFERENCE_DATA_V2="./rl_preference_data_v2/preference_pairs.json"
SYNTHETIC_DATA="./rl_preference_data/synthetic_pairs.json"
AUGMENTED_DATA="./rl_preference_data/preference_pairs_augmented.json"
DPO_OUTPUT="./rl_dpo_llama2_13b_generator"
DPO_V2_OUTPUT="./rl_dpo_v2_llama2_13b_generator"
PPO_OUTPUT="./rl_ppo_llama2_13b_generator"
EVAL_OUTPUT="./rl_eval_results/comparison.json"

# Synthetic data augmentation (Step 2.5)
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Training hyperparameters
SFT_EPOCHS=3
DPO_EPOCHS=2      # Original run: 2 epochs on 165 pairs (10 steps)
DPO_V2_EPOCHS=5   # v2 run: 5 epochs on ~450+ pairs (much more training signal)
DPO_BETA=0.1
LORA_R=16
LORA_ALPHA=32
BATCH_SIZE=1
GRAD_ACCUM=8          # Effective batch = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS = 32

# GPUs 4-7 (leave 0-3 for other jobs on the server)
NUM_GPUS=4
TRAIN_GPUS="4,5,6,7"
MASTER_PORT=29500

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
# Activate the conda env that has TRL 0.7.1 + PEFT 0.4.0 + accelerate 0.21.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-tuning

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="RLearner-LLM"

mkdir -p ./rl_preference_data ./rl_eval_results

echo "========================================================"
echo "  RLearner-LLM Pipeline Starting"
echo "  Base Model: ${BASE_MODEL}"
echo "  GPUs: ${TRAIN_GPUS} (${NUM_GPUS}x A100 80GB)"
echo "  Environment: llm-tuning (TRL 0.7.1)"
echo "========================================================"

# =============================================================================
# STEP 1: SFT Fine-tuning with LoRA
# Fine-tunes LLaMA-2-13B on the merged PeerWise dataset.
# Output: LoRA adapter saved to $SFT_OUTPUT
# Memory: ~26GB/GPU with bf16 + gradient checkpointing on A100 80GB
# =============================================================================
echo ""
echo ">>> STEP 1: SFT Fine-tuning with LoRA"
echo "    Model:  ${BASE_MODEL}"
echo "    Data:   ${MERGED_TRAIN_DATA}"
echo "    Output: ${SFT_OUTPUT}"
echo ""

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    rl_train_sft.py \
    --model_name_or_path "${BASE_MODEL}" \
    --data_path "${MERGED_TRAIN_DATA}" \
    --output_dir "${SFT_OUTPUT}" \
    --num_train_epochs ${SFT_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --max_seq_length 1024 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --bf16 True \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --report_to none \
    --run_name "sft-llama2-13b"

echo ">>> STEP 1 DONE. SFT adapter saved to ${SFT_OUTPUT}"

# =============================================================================
# STEP 2: Build DPO Preference Dataset
# Generates N explanations per question, scores with Verifier, creates pairs.
# Generator on GPU 4, Verifier on GPU 5 (separated to avoid OOM).
# =============================================================================
echo ""
echo ">>> STEP 2: Building DPO Preference Dataset"
echo "    Generator: ${SFT_OUTPUT} (on GPU 4)"
echo "    Verifier:  ${VERIFIER_PATH} (on GPU 5)"
echo "    Output:    ${PREFERENCE_DATA}"
echo ""

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} python rl_build_preference_data.py \
    --generator_path "${BASE_MODEL}" \
    --lora_adapter_path "${SFT_OUTPUT}" \
    --verifier_path "${VERIFIER_PATH}" \
    --data_path "${MERGED_TRAIN_DATA}" \
    --output_path "${PREFERENCE_DATA}" \
    --num_samples 6 \
    --max_new_tokens 512 \
    --min_score_gap 0.3 \
    --add_hard_negatives \
    --hard_negative_ratio 0.2 \
    --generator_device cuda:0 \
    --verifier_device cuda:1 \
    --cache_dir cache

echo ">>> STEP 2 DONE. Preference data saved to ${PREFERENCE_DATA}"

# =============================================================================
# STEP 2.5: Synthetic Data Augmentation (optional)
# Generates expert-level positives and hard negatives via GPT-4o or Claude.
# Skip if no API key is set — organic preference pairs are sufficient.
# =============================================================================
if [ -n "${OPENAI_API_KEY}" ]; then
    echo ""
    echo ">>> STEP 2.5: Generating Synthetic Data (GPT-4o CoT + Hard Negatives)"
    python rl_generate_synthetic_data.py \
        --data_path "${MERGED_TRAIN_DATA}" \
        --output_path "${SYNTHETIC_DATA}" \
        --merge_with "${PREFERENCE_DATA}" \
        --api_provider openai \
        --api_key "${OPENAI_API_KEY}" \
        --model gpt-4o \
        --num_questions 1000 \
        --negatives_per_question 2 \
        --api_delay 0.3
    AUGMENTED_DATA="${SYNTHETIC_DATA}"
    echo ">>> STEP 2.5 DONE. Augmented dataset: ${AUGMENTED_DATA}"
elif [ -n "${ANTHROPIC_API_KEY}" ]; then
    echo ""
    echo ">>> STEP 2.5: Generating Synthetic Data (Claude-3.5-Sonnet CoT + Hard Negatives)"
    python rl_generate_synthetic_data.py \
        --data_path "${MERGED_TRAIN_DATA}" \
        --output_path "${SYNTHETIC_DATA}" \
        --merge_with "${PREFERENCE_DATA}" \
        --api_provider anthropic \
        --api_key "${ANTHROPIC_API_KEY}" \
        --model claude-3-5-sonnet-20241022 \
        --num_questions 1000 \
        --negatives_per_question 2 \
        --api_delay 0.5
    AUGMENTED_DATA="${SYNTHETIC_DATA}"
    echo ">>> STEP 2.5 DONE. Augmented dataset: ${AUGMENTED_DATA}"
else
    echo ""
    echo ">>> STEP 2.5 SKIPPED (no OPENAI_API_KEY or ANTHROPIC_API_KEY set)"
    echo "    Using organic preference data: ${PREFERENCE_DATA}"
    AUGMENTED_DATA="${PREFERENCE_DATA}"
fi

# =============================================================================
# STEP 3A: DPO Training (RECOMMENDED)
# Trains the SFT model to prefer high-scoring explanations.
# Output: DPO LoRA adapter saved to $DPO_OUTPUT
# =============================================================================
echo ""
echo ">>> STEP 3A: DPO Training (Recommended Path)"
echo "    Base model:   ${BASE_MODEL}"
echo "    SFT adapter:  ${SFT_OUTPUT}"
echo "    Pref data:    ${AUGMENTED_DATA}"
echo "    Output:       ${DPO_OUTPUT}"
echo "    Beta:         ${DPO_BETA}"
echo ""

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=$((MASTER_PORT + 1)) \
    rl_train_dpo.py \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_adapter_path "${SFT_OUTPUT}" \
    --preference_data_path "${AUGMENTED_DATA}" \
    --output_dir "${DPO_OUTPUT}" \
    --num_train_epochs ${DPO_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --beta ${DPO_BETA} \
    --max_length 1024 \
    --max_prompt_length 512 \
    --min_score_gap_filter 0.3 \
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
    --run_name "dpo-llama2-13b"

echo ">>> STEP 3A DONE. DPO adapter saved to ${DPO_OUTPUT}"

# =============================================================================
# STEP 3A-v2: DPO Re-training with improved preference data (RECOMMENDED)
# Uses v2 preference pairs (min_score_gap=0.1, 3 samples) and more epochs.
# Run this after rl_preference_data_v2/ is complete for better verifier scores.
# Uncomment when ./rl_preference_data_v2/preference_pairs.json is ready.
# =============================================================================
# if [ -f "${PREFERENCE_DATA_V2}" ]; then
#     echo ""
#     echo ">>> STEP 3A-v2: DPO Re-training with v2 preference data"
#     echo "    Pref data: ${PREFERENCE_DATA_V2}"
#     echo "    Output:    ${DPO_V2_OUTPUT}"
#     echo ""
#     CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun \
#         --nproc_per_node=${NUM_GPUS} \
#         --master_port=$((MASTER_PORT + 2)) \
#         rl_train_dpo.py \
#         --model_name_or_path "${BASE_MODEL}" \
#         --sft_adapter_path "${SFT_OUTPUT}" \
#         --preference_data_path "${PREFERENCE_DATA_V2}" \
#         --output_dir "${DPO_V2_OUTPUT}" \
#         --num_train_epochs ${DPO_V2_EPOCHS} \
#         --per_device_train_batch_size ${BATCH_SIZE} \
#         --gradient_accumulation_steps ${GRAD_ACCUM} \
#         --lora_r ${LORA_R} \
#         --lora_alpha ${LORA_ALPHA} \
#         --lora_dropout 0.05 \
#         --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#         --beta ${DPO_BETA} \
#         --max_length 1024 \
#         --max_prompt_length 512 \
#         --min_score_gap_filter 0.05 \
#         --learning_rate 5e-5 \
#         --lr_scheduler_type cosine \
#         --warmup_ratio 0.1 \
#         --weight_decay 0.0 \
#         --bf16 True \
#         --logging_steps 5 \
#         --save_strategy steps \
#         --save_steps 200 \
#         --save_total_limit 2 \
#         --report_to none \
#         --run_name "dpo-v2-llama2-13b"
#     echo ">>> STEP 3A-v2 DONE. DPO v2 adapter saved to ${DPO_V2_OUTPUT}"
# fi

# =============================================================================
# STEP 3B: PPO Training (ALTERNATIVE — uncomment to run instead of DPO)
# Online RL using the Verifier as a live reward model.
# Actor on GPU 4 (cuda:0), Verifier on GPU 7 (cuda:1) via CUDA_VISIBLE_DEVICES=4,7
# =============================================================================
# echo ""
# echo ">>> STEP 3B: PPO Training (Alternative Path)"
#
# CUDA_VISIBLE_DEVICES=4,7 python rl_train_ppo.py \
#     --model_name_or_path "${BASE_MODEL}" \
#     --sft_adapter_path "${SFT_OUTPUT}" \
#     --verifier_path "${VERIFIER_PATH}" \
#     --verifier_device cuda:1 \
#     --data_path "${MERGED_TRAIN_DATA}" \
#     --output_dir "${PPO_OUTPUT}" \
#     --lora_r ${LORA_R} \
#     --lora_alpha ${LORA_ALPHA} \
#     --learning_rate 1e-5 \
#     --batch_size 4 \
#     --mini_batch_size 1 \
#     --ppo_epochs 4 \
#     --max_questions 500
#
# echo ">>> STEP 3B DONE. PPO adapter saved to ${PPO_OUTPUT}"

# =============================================================================
# STEP 4: Evaluation — Compare all models
# Runs inference with SFT, DPO, and legacy ILearner models on the test set.
# =============================================================================
echo ""
echo ">>> STEP 4: Evaluation"
echo "    Test set: ${TEST_DATA}"
echo "    Output:   ${EVAL_OUTPUT}"
echo ""

CUDA_VISIBLE_DEVICES=4,5 python rl_evaluation.py \
    --test_data_path "${TEST_DATA}" \
    --verifier_path "${VERIFIER_PATH}" \
    --output_path "${EVAL_OUTPUT}" \
    --sft_model_path "${BASE_MODEL}" \
    --sft_lora_path "${SFT_OUTPUT}" \
    --dpo_model_path "${BASE_MODEL}" \
    --dpo_lora_path "${DPO_OUTPUT}" \
    --ilearner_model_path "./llama_2_13B_merged_all_generator_avg_3_lenexp_10" \
    --ilearner_k 5 \
    --ilearner_is_legacy \
    --device cuda:0 \
    --verifier_device cuda:1 \
    --cache_dir cache

echo ""
echo "========================================================"
echo "  RLearner-LLM Pipeline COMPLETE!"
echo "  Results: ${EVAL_OUTPUT}"
echo "========================================================"


# =============================================================================
# ADDITIONAL: Per-domain evaluation
# Uncomment to evaluate separately on each dataset domain.
# =============================================================================

# for DOMAIN in Cardiff Sydney Auckland_law uk_medical_year1 uk_medical_year2; do
#     TEST_FILE="./Paul_new_data/${DOMAIN}/${DOMAIN}_vicuna_13b_finetuned_random_100.json"
#     if [ -f "${TEST_FILE}" ]; then
#         echo "  Evaluating domain: ${DOMAIN}"
#         CUDA_VISIBLE_DEVICES=4,5 python rl_evaluation.py \
#             --test_data_path "${TEST_FILE}" \
#             --verifier_path "${VERIFIER_PATH}" \
#             --output_path "./rl_eval_results/${DOMAIN}_comparison.json" \
#             --sft_model_path "${BASE_MODEL}" \
#             --sft_lora_path "${SFT_OUTPUT}" \
#             --dpo_model_path "${BASE_MODEL}" \
#             --dpo_lora_path "${DPO_OUTPUT}" \
#             --device cuda:0 \
#             --verifier_device cuda:1
#     fi
# done

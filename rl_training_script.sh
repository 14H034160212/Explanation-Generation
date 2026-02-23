#!/bin/bash
# =============================================================================
# RL-ILearner: Complete Training Pipeline
# =============================================================================
# Hardware: 8x A100 (80GB)
# Models: Qwen3.5-14B-Instruct (preferred) / Qwen3-14B-Instruct (fallback) /
#         Llama-3.3-70B-Instruct (large scale)
#
# Pipeline:
#   Step 1: SFT fine-tune the base model on PeerWise data (LoRA)
#   Step 2: Build DPO preference pairs (multi-sample + verifier scoring)
#   Step 3A: DPO training (RECOMMENDED - stable, efficient)
#   Step 3B: PPO training (ALTERNATIVE - online RL, higher ceiling)
#   Step 4: Evaluate and compare all models
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────
# CONFIG: Adjust these paths before running
# ─────────────────────────────────────────────

# Base model: use Qwen3.5 if available, else Qwen3-14B
# If running offline, set to a local path.
BASE_MODEL="Qwen/Qwen3.5-14B-Instruct"
# BASE_MODEL="Qwen/Qwen3-14B-Instruct"          # Fallback if Qwen3.5 not available
# BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct" # Large-scale experiment

# Existing verifier (trained LLaMA-2 evaluator)
VERIFIER_PATH="./llama_2_13B_merged_all_evaluator"

# Data paths
MERGED_TRAIN_DATA="./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
TEST_DATA="./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"

# Output directories
SFT_OUTPUT="./rl_sft_qwen3_generator"
PREFERENCE_DATA="./rl_preference_data/preference_pairs.json"
SYNTHETIC_DATA="./rl_preference_data/synthetic_pairs.json"
AUGMENTED_DATA="./rl_preference_data/preference_pairs_augmented.json"
DPO_OUTPUT="./rl_dpo_qwen3_generator"
PPO_OUTPUT="./rl_ppo_qwen3_generator"
EVAL_OUTPUT="./rl_eval_results/comparison.json"

# Synthetic data generation (亮点三)
# Set your API key here or via environment variable
OPENAI_API_KEY="${OPENAI_API_KEY:-}"   # export OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"  # export ANTHROPIC_API_KEY=sk-ant-...

# Training hyperparameters
SFT_EPOCHS=3
DPO_EPOCHS=2
DPO_BETA=0.1          # KL divergence penalty (lower = more aggressive)
LORA_R=16
LORA_ALPHA=32
BATCH_SIZE=1           # Per-device; increase if VRAM allows
GRAD_ACCUM=8           # Effective batch = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS

NUM_GPUS=8
MASTER_PORT=29500

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="RL-ILearner"

mkdir -p ./rl_preference_data ./rl_eval_results

echo "========================================================"
echo "  RL-ILearner Pipeline Starting"
echo "  Base Model: ${BASE_MODEL}"
echo "  GPUs: ${NUM_GPUS}x A100 (80GB)"
echo "========================================================"

# =============================================================================
# STEP 1: SFT Fine-tuning with LoRA
# Fine-tunes Qwen3.5/Qwen3/Llama-3 on PeerWise explanation data.
# Output: LoRA adapter saved to $SFT_OUTPUT
# =============================================================================
echo ""
echo ">>> STEP 1: SFT Fine-tuning with LoRA"
echo "    Model: ${BASE_MODEL}"
echo "    Data:  ${MERGED_TRAIN_DATA}"
echo "    Output: ${SFT_OUTPUT}"
echo ""

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} rl_train_sft.py \
    --model_name_or_path "${BASE_MODEL}" \
    --data_path "${MERGED_TRAIN_DATA}" \
    --output_dir "${SFT_OUTPUT}" \
    --num_train_epochs ${SFT_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
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
    --tf32 True \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name "sft-${BASE_MODEL##*/}" \
    --deepspeed ./rl_configs/ds_zero3.json

echo ">>> STEP 1 DONE. SFT adapter saved to ${SFT_OUTPUT}"

# =============================================================================
# STEP 2: Build DPO Preference Dataset
# Generates N explanations per question, scores with Verifier, creates pairs.
# NOTE: Generator on cuda:0-5, Verifier on cuda:6 (separated to avoid OOM).
# =============================================================================
echo ""
echo ">>> STEP 2: Building DPO Preference Dataset"
echo "    Generator: ${SFT_OUTPUT}"
echo "    Verifier:  ${VERIFIER_PATH}"
echo "    Output:    ${PREFERENCE_DATA}"
echo ""

# Run on 2 GPUs (generator + verifier separated)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python rl_build_preference_data.py \
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
    --verifier_device cuda:5 \
    --cache_dir cache

echo ">>> STEP 2 DONE. Preference data saved to ${PREFERENCE_DATA}"

# =============================================================================
# STEP 2.5: Synthetic Data Augmentation (亮点三 — GPT-4o/Claude CoT)
# Generates expert-level positive demos + model-based hard negatives.
# Skip this step if you don't have an API key (preference_pairs.json still works).
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
    echo "    Using organic preference data only: ${PREFERENCE_DATA}"
    AUGMENTED_DATA="${PREFERENCE_DATA}"
fi

# =============================================================================
# STEP 3A: DPO Training (RECOMMENDED)
# Trains the SFT model to prefer high-scoring explanations via DPO.
# Output: DPO LoRA adapter saved to $DPO_OUTPUT
# =============================================================================
echo ""
echo ">>> STEP 3A: DPO Training (Recommended Path)"
echo "    Base:       ${BASE_MODEL}"
echo "    SFT Adapter: ${SFT_OUTPUT}"
echo "    Prefs:      ${PREFERENCE_DATA}"
echo "    Output:     ${DPO_OUTPUT}"
echo "    Beta:       ${DPO_BETA}"
echo ""

deepspeed --num_gpus ${NUM_GPUS} --master_port $((MASTER_PORT + 1)) rl_train_dpo.py \
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
    --max_length 1024 \
    --max_prompt_length 512 \
    --min_score_gap_filter 0.3 \
    --beta ${DPO_BETA} \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --bf16 True \
    --tf32 True \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name "dpo-${BASE_MODEL##*/}" \
    --deepspeed ./rl_configs/ds_zero3.json

echo ">>> STEP 3A DONE. DPO adapter saved to ${DPO_OUTPUT}"

# =============================================================================
# STEP 3B: PPO Training (ALTERNATIVE — comment out if running DPO only)
# Online RL using Verifier as the real-time reward model.
# =============================================================================
# echo ""
# echo ">>> STEP 3B: PPO Training (Alternative Path)"
#
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
#     --num_processes ${NUM_GPUS} \
#     --mixed_precision bf16 \
#     rl_train_ppo.py \
#     --model_name_or_path "${BASE_MODEL}" \
#     --sft_adapter_path "${SFT_OUTPUT}" \
#     --verifier_path "${VERIFIER_PATH}" \
#     --data_path "${MERGED_TRAIN_DATA}" \
#     --output_dir "${PPO_OUTPUT}" \
#     --lora_r ${LORA_R} \
#     --lora_alpha ${LORA_ALPHA} \
#     --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --learning_rate 1e-5 \
#     --batch_size 8 \
#     --mini_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --ppo_epochs 4 \
#     --target_kl 0.1 \
#     --kl_penalty kl \
#     --adap_kl_ctrl True \
#     --log_with wandb \
#     --tracker_project_name "RL-ILearner"
#
# echo ">>> STEP 3B DONE. PPO adapter saved to ${PPO_OUTPUT}"

# =============================================================================
# STEP 4: Evaluation — Compare all models
# =============================================================================
echo ""
echo ">>> STEP 4: Evaluation"
echo "    Test set: ${TEST_DATA}"
echo "    Output:   ${EVAL_OUTPUT}"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python rl_evaluation.py \
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
echo "  RL-ILearner Pipeline COMPLETE!"
echo "  Results: ${EVAL_OUTPUT}"
echo "========================================================"


# =============================================================================
# ADDITIONAL EXPERIMENTS: Per-domain evaluation
# Uncomment to run evaluation on each domain separately.
# =============================================================================

# echo ">>> Running per-domain evaluation..."
#
# for DOMAIN in Cardiff Sydney Auckland_law uk_medical_year1 uk_medical_year2; do
#     TEST_FILE="./Paul_new_data/${DOMAIN}/${DOMAIN}_vicuna_13b_finetuned_random_100.json"
#     if [ -f "${TEST_FILE}" ]; then
#         echo "  Domain: ${DOMAIN}"
#         CUDA_VISIBLE_DEVICES=0,1 python rl_evaluation.py \
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


# =============================================================================
# LARGE-SCALE EXPERIMENT: Llama-3.3-70B with QLoRA
# Requires --use_4bit flag; runs DPO on 8x A100 via ZeRO-3 + 4-bit quantization
# =============================================================================

# LLAMA70B_MODEL="meta-llama/Llama-3.3-70B-Instruct"
# LLAMA70B_SFT_OUTPUT="./rl_sft_llama3_70b_generator"
# LLAMA70B_DPO_OUTPUT="./rl_dpo_llama3_70b_generator"
#
# deepspeed --num_gpus 8 --master_port 29501 rl_train_sft.py \
#     --model_name_or_path "${LLAMA70B_MODEL}" \
#     --data_path "${MERGED_TRAIN_DATA}" \
#     --output_dir "${LLAMA70B_SFT_OUTPUT}" \
#     --use_4bit True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lora_r 64 \
#     --lora_alpha 128 \
#     --deepspeed ./rl_configs/ds_zero3.json

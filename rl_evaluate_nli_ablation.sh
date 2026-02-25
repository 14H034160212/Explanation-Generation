#!/bin/bash
# NLI Scale Ablation: DeBERTa Small vs Large
# Correcting previous failed run

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-tuning

MODELS=("cross-encoder/nli-deberta-v3-small" "cross-encoder/nli-deberta-v3-large")
TEST_FILES=("./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"
            "./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json")
LABELS=("cardiff" "sydney")

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    TEST_FILE="${TEST_FILES[$i]}"
    
    for NLI_MODEL in "${MODELS[@]}"; do
        MODEL_LABEL=$(echo $NLI_MODEL | rev | cut -d'-' -f1 | rev)
        echo ">>> Running NLI Ablation: ${LABEL} with ${MODEL_LABEL}"
        
        # Note: We omit --skip_nli to enable NLI evaluation. 
        # The default nli_model in rl_evaluation.py is small, so we explicitly set it.
        
        CUDA_VISIBLE_DEVICES=4,5 python3 rl_evaluation.py \
            --test_data_path "${TEST_FILE}" \
            --nli_model "${NLI_MODEL}" \
            --output_path "./rl_eval_results/nli_ablation_${MODEL_LABEL}_${LABEL}.json" \
            --sft_model_path /data/shared/llama2/llama-2-13b-hf \
            --sft_lora_path ./rl_sft_llama2_13b_generator \
            --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
            --verifier_device cuda:1 \
            --device cuda:0 \
            > "rl_nli_ablation_${MODEL_LABEL}_${LABEL}.log" 2>&1
    done
done

source "$(conda info --base)/etc/profile.d/conda.sh"
DOMAINS=("Auckland_law" "Medicine_year1" "Medicine_year2")
TEST_FILES=("./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" 
            "./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" 
            "./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json")
LABELS=("law" "med_y1" "med_y2")

for i in "${!DOMAINS[@]}"; do
    echo ">>> Qwen3 Evaluation: ${LABELS[$i]}"
    CUDA_VISIBLE_DEVICES=6,7 conda run -n qwen3-rl python3 rl_evaluate_qwen3.py \
        --test_data_path "${TEST_FILES[$i]}" \
        --output_path "./rl_eval_results/qwen3_${LABELS[$i]}_eval.json" \
        > "rl_qwen3_${LABELS[$i]}_eval.log" 2>&1
done

#!/bin/bash
# Run Qwen3 cross-domain evaluations in parallel on idle GPUs 4, 5, 7
# Law -> GPU 4, Medicine Y1 -> GPU 5, Medicine Y2 -> GPU 7

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p ./rl_eval_results

echo ">>> [$(date)] Starting 3 Qwen3 domain evals in parallel on GPUs 4, 5, 7"

# GPU 4: Auckland Law
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n qwen3-rl python3 rl_evaluate_qwen3.py \
    --test_data_path "./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json" \
    --output_path "./rl_eval_results/qwen3_law_eval.json" \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
    > rl_qwen3_law_eval.log 2>&1 &
PID_LAW=$!
echo ">>> Law eval started (PID $PID_LAW) on GPU 4"

# GPU 5: Medicine Year 1
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n qwen3-rl python3 rl_evaluate_qwen3.py \
    --test_data_path "./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json" \
    --output_path "./rl_eval_results/qwen3_med_y1_eval.json" \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
    > rl_qwen3_med_y1_eval.log 2>&1 &
PID_Y1=$!
echo ">>> Medicine Y1 eval started (PID $PID_Y1) on GPU 5"

# GPU 7: Medicine Year 2
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n qwen3-rl python3 rl_evaluate_qwen3.py \
    --test_data_path "./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json" \
    --output_path "./rl_eval_results/qwen3_med_y2_eval.json" \
    --device cuda:0 --verifier_device cuda:0 --nli_device cpu \
    > rl_qwen3_med_y2_eval.log 2>&1 &
PID_Y2=$!
echo ">>> Medicine Y2 eval started (PID $PID_Y2) on GPU 7"

echo ">>> All 3 evals launched. Waiting for completion..."
wait $PID_LAW && echo ">>> [$(date)] Law eval DONE" || echo ">>> [$(date)] Law eval FAILED"
wait $PID_Y1  && echo ">>> [$(date)] Med Y1 eval DONE" || echo ">>> [$(date)] Med Y1 eval FAILED"
wait $PID_Y2  && echo ">>> [$(date)] Med Y2 eval DONE" || echo ">>> [$(date)] Med Y2 eval FAILED"

echo ">>> [$(date)] All domain evaluations completed."

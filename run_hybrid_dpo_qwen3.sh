#!/bin/bash
# =============================================================================
# Phase 1b: Hybrid DPO for Qwen3-8B  (0.5*NLI + 0.5*Verifier reward)
# Run AFTER run_nli_dpo_qwen3.sh completes (or on separate GPUs).
# =============================================================================

set -e

PYTHON=/data/qbao775/miniconda3/envs/qwen3-rl/bin/python3
WORKDIR=/data/qbao775/Explanation-Generation
cd "$WORKDIR"

BASE_MODEL=/data/shared/qwen3/Qwen3-8B
SFT_ADAPTER=./rl_sft_qwen3_8b_generator
VERIFIER=./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA=./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json
PREF_OUT=./rl_preference_data_hybrid_qwen3/preference_pairs.json
DPO_OUT=./rl_dpo_qwen3_hybrid_generator

CARDIFF_TEST=./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
SYDNEY_TEST=./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
EVAL_CARDIFF=./rl_eval_results/qwen3_dpo_hybrid_cardiff_eval.json
EVAL_SYDNEY=./rl_eval_results/qwen3_dpo_hybrid_sydney_eval.json

echo "============================================================"
echo "Phase 1b-A: Build Hybrid preference pairs (Qwen3-8B)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=6 $PYTHON rl_build_preference_data_nli.py \
    --model_type qwen3 \
    --score_method hybrid \
    --generator_path "$BASE_MODEL" \
    --lora_adapter_path "$SFT_ADAPTER" \
    --verifier_path "$VERIFIER" \
    --data_path "$DATA" \
    --output_path "$PREF_OUT" \
    --num_samples 3 \
    --min_score_gap 0.03 \
    --max_questions 500 \
    --generator_device cuda:0 \
    --nli_device cpu \
    --verifier_device cuda:0 \
    --cache_dir ./cache

echo ""
echo "============================================================"
echo "Phase 1b-B: DPO training with Hybrid pairs (Qwen3-8B)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=6,7 $PYTHON rl_train_dpo_qwen3.py \
    --model_name_or_path "$BASE_MODEL" \
    --sft_adapter_path "$SFT_ADAPTER" \
    --preference_data_path "$PREF_OUT" \
    --output_dir "$DPO_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --beta 0.1

echo ""
echo "============================================================"
echo "Phase 1b-C: Evaluate Qwen3-8B DPO-Hybrid (Cardiff + Sydney)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=6,7 $PYTHON rl_evaluate_qwen3.py \
    --base_model_path "$BASE_MODEL" \
    --sft_lora_path "$SFT_ADAPTER" \
    --dpo_lora_path "$DPO_OUT" \
    --test_data_path "$CARDIFF_TEST" \
    --verifier_path "$VERIFIER" \
    --output_path "$EVAL_CARDIFF" \
    --device cuda:0 --verifier_device cuda:1

CUDA_VISIBLE_DEVICES=6,7 $PYTHON rl_evaluate_qwen3.py \
    --base_model_path "$BASE_MODEL" \
    --sft_lora_path "$SFT_ADAPTER" \
    --dpo_lora_path "$DPO_OUT" \
    --test_data_path "$SYDNEY_TEST" \
    --verifier_path "$VERIFIER" \
    --output_path "$EVAL_SYDNEY" \
    --device cuda:0 --verifier_device cuda:1

echo ""
echo "============================================================"
echo "FINAL COMPARISON: NLI vs Hybrid vs Old-DPO (Verifier)"
echo "============================================================"
$PYTHON - <<'PYEOF'
import json, os

def show(label, path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return
    d = json.load(open(path))
    for r in d.get("results", []):
        nli = r.get("avg_nli_entailment", "N/A")
        nli_s = f"{nli:.4f}" if isinstance(nli, float) else nli
        print(f"  [{label}] {r['model']:30s}  NLI={nli_s}  "
              f"BERTans={r.get('avg_bert_score_f1_answer_anchored',0):.4f}  "
              f"ACR={r.get('avg_answer_coverage_rate',0):.4f}  "
              f"Ver={r.get('avg_verifier_score',0):.4f}")

print("Cardiff results:")
show("Old-DPO(Ver)", "./rl_eval_results/qwen3_ppo_cardiff_eval.json")
show("DPO-NLI",      "./rl_eval_results/qwen3_dpo_nli_cardiff_eval.json")
show("DPO-Hybrid",   "./rl_eval_results/qwen3_dpo_hybrid_cardiff_eval.json")

print("\nSydney results:")
show("Old-DPO(Ver)", "./rl_eval_results/qwen3_ppo_sydney_eval.json")
show("DPO-NLI",      "./rl_eval_results/qwen3_dpo_nli_sydney_eval.json")
show("DPO-Hybrid",   "./rl_eval_results/qwen3_dpo_hybrid_sydney_eval.json")
PYEOF

echo "Phase 1b complete."

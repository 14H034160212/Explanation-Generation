#!/bin/bash
# =============================================================================
# Phase 2b: Hybrid DPO for LLaMA-2-13B  (0.5*NLI + 0.5*Verifier reward)
# Run AFTER run_nli_dpo_llama2.sh completes.
# =============================================================================

set -e

PYTHON=/data/qbao775/miniconda3/envs/llm-tuning/bin/python3
WORKDIR=/data/qbao775/Explanation-Generation
cd "$WORKDIR"

BASE_MODEL=/data/shared/llama2/llama-2-13b-hf
SFT_ADAPTER=./rl_sft_llama2_13b_generator
VERIFIER=./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2
DATA=./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json
PREF_OUT=./rl_preference_data_hybrid_llama2/preference_pairs.json
DPO_OUT=./rl_dpo_hybrid_llama2_generator

CARDIFF_TEST=./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json
SYDNEY_TEST=./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json
EVAL_CARDIFF=./rl_eval_results/llama2_dpo_hybrid_cardiff_eval.json
EVAL_SYDNEY=./rl_eval_results/llama2_dpo_hybrid_sydney_eval.json

echo "============================================================"
echo "Phase 2b-A: Build Hybrid preference pairs (LLaMA-2-13B)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=4 $PYTHON rl_build_preference_data_nli.py \
    --model_type llama2 \
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
echo "Phase 2b-B: DPO training with Hybrid pairs (LLaMA-2)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON rl_train_dpo.py \
    --model_name_or_path "$BASE_MODEL" \
    --sft_adapter_path "$SFT_ADAPTER" \
    --preference_data_path "$PREF_OUT" \
    --output_dir "$DPO_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --lora_r 16 --lora_alpha 32 \
    --bf16 True

echo ""
echo "============================================================"
echo "Phase 2b-C: Evaluate LLaMA-2 DPO-Hybrid (Cardiff + Sydney)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=4,5 $PYTHON rl_evaluation.py \
    --test_data_path "$CARDIFF_TEST" \
    --base_model_path "$BASE_MODEL" \
    --sft_lora_path "$SFT_ADAPTER" \
    --dpo_lora_path "$DPO_OUT" \
    --verifier_path "$VERIFIER" \
    --output_path "$EVAL_CARDIFF" \
    --device cuda:0 --verifier_device cuda:1

CUDA_VISIBLE_DEVICES=4,5 $PYTHON rl_evaluation.py \
    --test_data_path "$SYDNEY_TEST" \
    --base_model_path "$BASE_MODEL" \
    --sft_lora_path "$SFT_ADAPTER" \
    --dpo_lora_path "$DPO_OUT" \
    --verifier_path "$VERIFIER" \
    --output_path "$EVAL_SYDNEY" \
    --device cuda:0 --verifier_device cuda:1

echo ""
echo "============================================================"
echo "FINAL COMPARISON: NLI vs Hybrid vs DPO v2 (Verifier)"
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
show("DPO-v2(Ver)", "./rl_eval_results/cardiff_eval_nli.json")
show("DPO-NLI",     "./rl_eval_results/llama2_dpo_nli_cardiff_eval.json")
show("DPO-Hybrid",  "./rl_eval_results/llama2_dpo_hybrid_cardiff_eval.json")

print("\nSydney results:")
show("DPO-v2(Ver)", "./rl_eval_results/sydney_eval_nli.json")
show("DPO-NLI",     "./rl_eval_results/llama2_dpo_nli_sydney_eval.json")
show("DPO-Hybrid",  "./rl_eval_results/llama2_dpo_hybrid_sydney_eval.json")
PYEOF

echo "Phase 2b complete."

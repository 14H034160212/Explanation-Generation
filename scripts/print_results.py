import json
import glob
files = [
    "rl_eval_results/qwen3_dpo_med_y1_eval.json",
    "rl_eval_results/qwen3_dpo_med_y2_eval.json",
    "rl_eval_results/dpo_v3_med_y2_final_eval.json",
    "rl_eval_results/baselines_law_remaining_eval.json",
    "rl_eval_results/baselines_med_y1_remaining_eval.json"
]
for f in files:
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
            print(f"--- {f} ---")
            for model, metrics in data.items():
                print(f"{model}: BLEU={metrics.get('bleu', 0):.4f}, BERT(S)={metrics.get('bertscore_f1', 0):.4f}, BERT(A)={metrics.get('answer_bertscore_f1', 0):.4f}, ACR={metrics.get('acr', 0):.4f}, NLI={metrics.get('nli_entailment', 0):.4f}, Verifier={metrics.get('verifier_score', 0):.4f}")
    except Exception as e:
        print(f"Failed to load {f}: {e}")

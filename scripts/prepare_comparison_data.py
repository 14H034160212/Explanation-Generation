import json
import os
import random

def prepare_comparison(sft_path, ppo_path, output_path, n_samples=50):
    if not os.path.exists(sft_path) or not os.path.exists(ppo_path):
        print(f"Error: Missing files. SFT: {os.path.exists(sft_path)}, PPO: {os.path.exists(ppo_path)}")
        return

    with open(sft_path, 'r') as f:
        sft_data = json.load(f)
    with open(ppo_path, 'r') as f:
        ppo_data = json.load(f)
    
    # SFT structure: detailed_results['Qwen3-8B-SFT']['generated_explanations']
    # If using qwen3_cardiff_eval.json
    sft_exps = sft_data["detailed_results"]["Qwen3-8B-SFT"]["generated_explanations"]
    
    # PPO structure: detailed_results['Qwen3-8B-DPO']['generated_explanations'] 
    # (Assuming it's named Qwen3-8B-DPO if dpo_lora_path was used as adapter)
    # Let me check the exact key in the new ppo output later.
    ppo_key = "Qwen3-8B-DPO" if "Qwen3-8B-DPO" in ppo_data["detailed_results"] else "Qwen3-8B-SFT"
    ppo_exps = ppo_data["detailed_results"][ppo_key]["generated_explanations"]

    # Load original test data to get questions
    test_data_path = sft_data.get("test_data_path", "./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    if isinstance(test_data, dict) and "results" in test_data:
        test_data = test_data["results"] # Handle different formats if needed

    comparison_data = []
    indices = list(range(min(len(sft_exps), len(ppo_exps), len(test_data))))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    
    for idx in indices:
        item = test_data[idx]
        comparison_data.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "target": item.get("output", item.get("Explanation", "")),
            "model_a_output": sft_exps[idx],
            "model_b_output": ppo_exps[idx],
            "model_a_name": "SFT",
            "model_b_name": "PPO-Hybrid"
        })

    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=4)
    print(f"Comparison data saved to {output_path} with {len(comparison_data)} samples.")

if __name__ == "__main__":
    prepare_comparison(
        "rl_eval_results/qwen3_cardiff_eval.json",
        "rl_eval_results/qwen3_ppo_cardiff_eval.json",
        "rl_eval_results/cardiff_sft_vs_ppo_comparison.json"
    )

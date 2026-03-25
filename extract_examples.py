import json
import glob
import os

files = glob.glob('rl_eval_results/llama2_dpo_hybrid_cross_domain_*_eval.json')
files += glob.glob('rl_eval_results/llama2_dpo_hybrid_cardiff_eval.json')
files += glob.glob('rl_eval_results/llama2_dpo_hybrid_sydney_eval.json')

for f in files:
    try:
        data = json.load(open(f))
    except Exception as e:
        continue
    
    test_path = data.get('test_data_path')
    basename = os.path.basename(test_path)
    cmd = os.popen(f"find /data/qbao775/Explanation-Generation -name '{basename}' | head -n 1").read().strip()
    if cmd:
        test_path = cmd
    
    if not test_path or not os.path.exists(test_path):
        print(f"Could not find test data for {f}: {test_path} (basename {basename})")
        continue
        
    test_data = json.load(open(test_path))
    
    models = [res['model'] for res in data['detailed_results']]
    
    sft_idx = 0
    hybrid_idx = -1
    for idx, m in enumerate(models):
        if 'SFT' in m.upper():
            sft_idx = idx
        if 'HYBRID' in m.upper() or 'RLEARNER' in m.upper() or 'DPO (K=1)' in m:
            hybrid_idx = idx
            
    sft_nli = data['detailed_results'][sft_idx]['nli_entailment_scores']
    hybrid_nli = data['detailed_results'][hybrid_idx]['nli_entailment_scores']
    
    best_idx = -1
    max_diff = -1
    for i in range(min(len(sft_nli), len(hybrid_nli))):
        diff = hybrid_nli[i] - sft_nli[i]
        sft_text = data['detailed_results'][sft_idx]['generated_explanations'][i].strip()
        hybrid_text = data['detailed_results'][hybrid_idx]['generated_explanations'][i].strip()
        
        if hybrid_nli[i] > 0.9 and sft_nli[i] < 0.2 and len(sft_text) > 30 and len(hybrid_text) > 30:
            if diff > max_diff:
                max_diff = diff
                best_idx = i
                
    if best_idx != -1:
        print(f"=== Dataset: {os.path.basename(f)} ===")
    else:
        for i in range(min(len(sft_nli), len(hybrid_nli))):
            diff = hybrid_nli[i] - sft_nli[i]
            if diff > max_diff:
                max_diff = diff
                best_idx = i
        print(f"=== Dataset (Fallback): {os.path.basename(f)} ===")
        
    print(f"Index: {best_idx}")
    print(f"Question: {test_data[best_idx].get('question', '').strip()}")
    options = test_data[best_idx].get('options', {})
    if not options:
        options = test_data[best_idx].get('choices', {})
    print(f"Options: {options}")
    answer_key = str(test_data[best_idx].get('answer', ''))
    if not answer_key:
        answer_key = str(test_data[best_idx].get('correct_answer', ''))
    print(f"Correct Answer: {answer_key}")
    print("-" * 20)
    
    for m_idx, m in enumerate(models):
        print(f"[{m}] (NLI: {data['detailed_results'][m_idx]['nli_entailment_scores'][best_idx]:.3f})")
        print(data['detailed_results'][m_idx]['generated_explanations'][best_idx].strip()[:800])
        print("\n")
    print("\n\n")

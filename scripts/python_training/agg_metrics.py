import json
import os
import numpy as np

results_dir = "/data/qbao775/Explanation-Generation/rl_eval_results/"
files = [
    "law_high_bleu.json",
    "law_hyper_bleu.json",
    "law_hyper_bleu_v2.json",
    "law_absolute_dominance_v3.json",
    "med_y1_high_bleu.json",
    "med_y2_dominance.json",
    "cardiff_dominance.json",
    "cardiff_hyper_bleu.json",
    "cardiff_hyper_bleu_v2.json",
    "cardiff_absolute_dominance_v3.json",
    "sydney_dominance.json",
    "sydney_hybrid_best_of_32.json",
    "sydney_hyper_bleu.json",
    "sydney_hyper_bleu_v2.json",
    "sydney_hyper_bleu_v3.json",
    "sydney_hyper_bleu_v4.json",
    "sydney_absolute_dominance_v5.json",
    "sydney_absolute_dominance_v6.json"
]

for f in files:
    path = os.path.join(results_dir, f)
    if not os.path.exists(path):
        continue
    
    with open(path, "r") as fh:
        try:
            data = json.load(fh)
        except:
            continue
    
    if not data:
        continue
    
    nli_list = []
    ver_list = []
    bleu_list = []
    acr_list = []
    
    for item in data:
        m = item.get("metrics", {})
        if not m: continue
        nli_list.append(m.get("nli", 0))
        ver_list.append(m.get("verifier", 0))
        bleu_list.append(m.get("bleu", 0))
        acr_list.append(m.get("acr", 0))
    
    if not nli_list: continue
    
    print(f"--- {f} (Count: {len(data)}) ---")
    print(f"NLI: {np.mean(nli_list):.4f}")
    print(f"Verifier: {np.mean(ver_list):.4f}")
    print(f"BLEU: {np.mean(bleu_list):.4f}")
    print(f"ACR: {np.mean(acr_list):.4f}")

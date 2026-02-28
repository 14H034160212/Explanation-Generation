import json
import torch
import os
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import re

def extract_correct_option_text(input_text):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m: return None
    letter = m.group(1)
    # Be more robust with regex
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    return opt_m.group(1).strip() if opt_m else None

@torch.no_grad()
def rescore_json(json_path, nli_model_name, device="cuda", fallback_test_data=None):
    print(f"Processing {json_path} with {nli_model_name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get hypotheses from test data if possible
    test_data_path = data.get("test_data_path") or fallback_test_data
    hypotheses = []
    if test_data_path and os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        hypotheses = [item.get("explanation", "") for item in test_data]
    else:
        print(f"  Warning: No valid test data found at {test_data_path}. Skipping file.")
        return

    # Load NLI
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    model.eval()
    
    entailment_idx = next((k for k, v in model.config.id2label.items() if "entail" in v.lower()), 1)

    # Prepare hypotheses
    hypotheses = []
    for item in test_data:
        _, opt_text = (None, None)
        # We need the same logic as rl_evaluation.py
        m = re.search(r"The correct answer is Option ([A-Z])", item["input"])
        if m:
            letter = m.group(1)
            opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
            opt_m = re.search(opt_pat, item["input"], re.DOTALL)
            opt_text = opt_m.group(1).strip() if opt_m else ""
        hypotheses.append(opt_text or "")

    # Process each model result
    for current_res in data.get("results", []):
        model_name = current_res.get("model", "Unknown")
        model_exps = []
        
        # Try finding exps in current_res or detailed_results
        if "generated_explanations" in current_res:
            model_exps = current_res["generated_explanations"]
        elif "detailed_results" in data:
            det = data["detailed_results"]
            if isinstance(det, dict) and model_name in det:
                model_exps = det[model_name].get("generated_explanations", [])
            elif isinstance(det, list):
                for d in det:
                    if d.get("model") == model_name:
                        model_exps = d.get("generated_explanations", [])
                        break
        
        if not model_exps:
            print(f"  Warning: No explanations found for {model_name}. Skipping.")
            continue
            
        print(f"  Rescoring {model_name} ({len(model_exps)} exps)...")
        
        nli_scores = []
        batch_size = 4
        for idx in range(0, len(model_exps), batch_size):
            batch_exp = model_exps[idx:idx+batch_size]
            batch_hyp = hypotheses[idx:idx+batch_size]
            
            # Use zip to handle potential length mismatches
            valid_pairs = [(e, h) for e, h in zip(batch_exp, batch_hyp) if h and e]
            
            if not valid_pairs:
                nli_scores.extend([0.0] * len(batch_exp))
                continue
            
            valid_exp = [e for e, h in valid_pairs]
            valid_hyp = [h for e, h in valid_pairs]
            
            enc = tokenizer(valid_exp, valid_hyp, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            batch_scores = probs[:, entailment_idx].cpu().tolist()
            
            nli_scores.extend(batch_scores)
            if len(batch_scores) < len(batch_exp):
                nli_scores.extend([0.0] * (len(batch_exp) - len(batch_scores)))
        
        current_res["nli_entailment_scores_large"] = nli_scores
        avg_large = sum(nli_scores)/len(nli_scores) if nli_scores else 0
        current_res["avg_nli_entailment_large"] = round(avg_large, 4)
        print(f"    Avg NLI (Large): {avg_large:.4f}")

    out_path = json_path.replace(".json", "_nli_large.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_paths", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-large")
    args = parser.parse_args()
    
    # Hardcoded fallback mapping for domains
    FALLBACK_MAPPING = {
        "cardiff": "./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",
        "sydney": "./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json",
        "law": "./PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json",
        "med_y1": "./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json",
        "med_y2": "./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"
    }
    
    for jp in args.json_paths:
        # Try to infer domain if test_data_path might be missing
        fallback_path = None
        for domain, path in FALLBACK_MAPPING.items():
            if domain in jp.lower():
                fallback_path = path
                break
        
        rescore_json(jp, args.nli_model, device=args.device, fallback_test_data=fallback_path)

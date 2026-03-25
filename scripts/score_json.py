import json
import argparse
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

VERIFIER_INSTRUCTION = (
    "As an explanation evaluation expert, can you evaluate the quality of the following "
    "explanation for the given exam question and provide a score from 1 to 5?"
)
VERIFIER_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

def extract_correct_option_text(input_text):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    opt_text = opt_m.group(1).strip() if opt_m else None
    return letter, opt_text

def get_verifier_score(ver_model, ver_tok, question_input, explanation):
    prompt = VERIFIER_TEMPLATE.format(
        instruction=VERIFIER_INSTRUCTION,
        input=f"{question_input}\n\nExplanation: {explanation}",
    )
    inputs = ver_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(next(ver_model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = ver_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=ver_tok.eos_token_id)
    resp = ver_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    nums = re.findall(r"\d+(?:\.\d+)?", resp)
    return min(max(float(nums[0]), 1.0), 5.0) if nums else 3.0

def compute_nli_batch(nli_model, nli_tok, premises, hypotheses, ent_idx, device, batch_size=16):
    scores = []
    for i in range(0, len(premises), batch_size):
        enc = nli_tok(premises[i:i+batch_size], hypotheses[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, ent_idx].cpu().tolist())
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_json", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--verifier_path", default="/data/qbao775/Explanation-Generation/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.gen_json, "r") as f:
        gen_data = json.load(f)
    if isinstance(gen_data, list):
        # Assume it's a list of explanations
        explanations = gen_data
    elif "results" in gen_data:
        # Standard evaluate output
        models = list(gen_data["detailed_results"].keys())
        explanations = gen_data["detailed_results"][models[0]]["generated_explanations"]
    else:
        raise ValueError("Unknown gen_json format")

    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    # Load models
    print("Loading verifier...")
    ver_tok = AutoTokenizer.from_pretrained(args.verifier_path, use_fast=False)
    ver_model = AutoModelForCausalLM.from_pretrained(args.verifier_path, torch_dtype=torch.bfloat16).to(args.device).eval()

    print("Loading NLI...")
    nli_tok = AutoTokenizer.from_pretrained(args.nli_model)
    nli_mdl = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(args.device).eval()
    ent_idx = next((k for k, v in nli_mdl.config.id2label.items() if "entail" in v.lower()), 1)

    nli_scores = []
    ver_scores = []
    
    count = min(len(explanations), len(test_data))
    premises = []
    hypotheses = []
    
    print(f"Scoring {count} samples...")
    for i in tqdm(range(count)):
        item = test_data[i]
        expl = explanations[i]
        input_text = item.get("input", "").strip()
        correct_opt = item.get("correct_option_text", item.get("correct_option", "")).strip()
        if not correct_opt:
            _, correct_opt = extract_correct_option_text(input_text)
        
        vs = get_verifier_score(ver_model, ver_tok, input_text, expl)
        ver_scores.append(vs)
        
        premises.append(expl)
        hypotheses.append(correct_opt if correct_opt else "")

    nli_scores = compute_nli_batch(nli_mdl, nli_tok, premises, hypotheses, ent_idx, args.device)
    
    avg_nli = sum(nli_scores) / count
    avg_ver = sum(ver_scores) / count
    
    print(f"AVG NLI: {avg_nli:.4f}")
    print(f"AVG Verifier: {avg_ver:.4f}")
    
    res = {
        "avg_nli": avg_nli,
        "avg_verifier": avg_ver,
        "nli_scores": nli_scores,
        "verifier_scores": ver_scores
    }
    with open(args.output_path, "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()

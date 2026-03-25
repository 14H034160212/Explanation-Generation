"""
Best-of-N Reranking for ARC-Challenge (Qwen3-8B)
Usage:
    CUDA_VISIBLE_DEVICES=4 conda run -n qwen3-rl python3 rl_rerank_arc.py \
        --test_data_path /data/qbao775/Explanation-Generation/preference_data/Generalization/arc_challenge_random_100.json \
        --base_model_path /data/shared/qwen3/Qwen3-8B \
        --sft_lora_path /data/qbao775/Explanation-Generation/models/rl_sft_qwen3_8b_generator \
        --dpo_lora_path /data/qbao775/Explanation-Generation/models/phase6/sciq_hybrid_dpo_qwen3 \
        --verifier_path /data/qbao775/Explanation-Generation/models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
        --output_path /data/qbao775/Explanation-Generation/rl_eval_results/arc_rerank_n16_hybrid_dpo.json \
        --n_samples 16 --weight_nli 0.7 --weight_ver 0.3
"""

import re
import argparse
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import torch
from bert_score import score as bert_score_func
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As an explanation evaluation expert, can you evaluate the quality of the following "
    "explanation for the given exam question and provide a score from 1 to 5?"
)
VERIFIER_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

def load_generator(base_path: str, lora_path: Optional[str], device: str):
    from peft import PeftModel
    logger.info(f"Loading Qwen3-8B from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, tokenizer

def load_verifier(verifier_path: str, device: str):
    logger.info(f"Loading verifier from {verifier_path}...")
    tokenizer = AutoTokenizer.from_pretrained(verifier_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(verifier_path, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tokenizer

def load_nli_model(model_name: str, device: str, cache_dir: str):
    logger.info(f"Loading NLI model {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
    id2label = mdl.config.id2label or {}
    ent_idx = next((k for k, v in id2label.items() if "entail" in v.lower()), 1)
    return mdl, tok, ent_idx

def extract_correct_option_text(input_text: str):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m: return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    opt_text = opt_m.group(1).strip() if opt_m else None
    return letter, opt_text

def generate_candidates(model, tokenizer, instruction: str, input_text: str, device: str, n: int, max_tokens: int) -> List[str]:
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
    candidates = []
    for i in range(n):
        gen = out[i][inputs["input_ids"].shape[1]:]
        candidates.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return candidates

@torch.no_grad()
def get_verifier_scores_batch(ver_model, ver_tok, question_input: str, explanations: List[str]) -> List[float]:
    scores = []
    for expl in explanations:
        prompt = VERIFIER_TEMPLATE.format(
            instruction=VERIFIER_INSTRUCTION,
            input=f"{question_input}\n\nExplanation: {expl}",
        )
        inputs = ver_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(next(ver_model.parameters()).device)
        out = ver_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=ver_tok.eos_token_id)
        resp = ver_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        nums = re.findall(r"\d+(?:\.\d+)?", resp)
        scores.append(min(max(float(nums[0]), 1.0), 5.0) if nums else 3.0)
    return scores

@torch.no_grad()
def compute_nli_scores_batch(nli_model, nli_tok, premises, hypothesis: str, ent_idx, device):
    enc = nli_tok(premises, [hypothesis]*len(premises), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, ent_idx].cpu().tolist()

def compute_acr(explanation: str, correct_option: str) -> float:
    if not correct_option: return 0.0
    keywords = [w for w in correct_option.lower().split() if len(w) >= 4]
    if not keywords: return 0.0
    exp_lower = explanation.lower()
    return sum(1 for kw in keywords if kw in exp_lower) / len(keywords)

@torch.no_grad()
def compute_bert_score_batch(candidates, reference, device):
    if not reference: return [0.0] * len(candidates)
    P, R, F1 = bert_score_func(candidates, [reference]*len(candidates), lang="en", device=device, verbose=False)
    return F1.tolist()

def compute_acr_batch(candidates, correct_option) -> List[float]:
    if not correct_option: return [0.0] * len(candidates)
    keywords = [w for w in correct_option.lower().split() if len(w) >= 4]
    if not keywords: return [0.0] * len(candidates)
    
    scores = []
    for expl in candidates:
        exp_lower = expl.lower()
        match_count = sum(1 for kw in keywords if kw in exp_lower)
        scores.append(match_count / len(keywords))
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--sft_lora_path", required=True)
    parser.add_argument("--dpo_lora_path", default=None)
    parser.add_argument("--verifier_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--weight_nli", type=float, default=0.25)
    parser.add_argument("--weight_ver", type=float, default=0.25)
    parser.add_argument("--weight_acr", type=float, default=0.25)
    parser.add_argument("--weight_bert", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    with open(args.test_data_path, "r") as f: data = json.load(f)
    gen_model, gen_tok = load_generator(args.base_model_path, args.dpo_lora_path, args.device)
    ver_model, ver_tok = load_verifier(args.verifier_path, args.device)
    nli_model, nli_tok, nli_ent_idx = load_nli_model(args.nli_model, args.device, args.cache_dir)

    final_results = []
    total_nli, total_ver, total_acr, total_bert = 0, 0, 0, 0

    for i, item in enumerate(data):
        instruction, input_text = item.get("instruction", ""), item.get("input", "")
        ref_text = item.get("output", item.get("Explanation", ""))
        _, correct_opt = extract_correct_option_text(input_text)
        if not correct_opt: correct_opt = item.get("correct_option_text", "")

        candidates = generate_candidates(gen_model, gen_tok, instruction, input_text, args.device, args.n_samples, 300)
        nli_scores = compute_nli_scores_batch(nli_model, nli_tok, candidates, correct_opt, nli_ent_idx, args.device)
        ver_scores = get_verifier_scores_batch(ver_model, ver_tok, input_text, candidates)
        acr_scores = compute_acr_batch(candidates, correct_opt)
        
        # Use correct_opt as reference for BERTScore(Ans) if Golden explanation is missing
        ref_for_bert = ref_text if ref_text and len(ref_text.strip()) > 0 else correct_opt
        bert_scores = compute_bert_score_batch(candidates, ref_for_bert, args.device)
        
        hybrid_scores = [
            args.weight_nli * n + 
            args.weight_ver * (v/5.0) + 
            args.weight_acr * a +
            args.weight_bert * b
            for n, v, a, b in zip(nli_scores, ver_scores, acr_scores, bert_scores)
        ]
        best_idx = hybrid_scores.index(max(hybrid_scores))
        
        best_expl = candidates[best_idx]
        best_nli = nli_scores[best_idx]
        best_ver = ver_scores[best_idx]
        best_acr = acr_scores[best_idx]
        best_bert = bert_scores[best_idx]
        
        total_nli += best_nli
        total_ver += best_ver
        total_acr += best_acr
        total_bert += best_bert
        
        final_results.append({
            "question": input_text,
            "best_explanation": best_expl,
            "nli": best_nli,
            "ver": best_ver,
            "acr": best_acr,
            "bert": best_bert
        })
        if (i+1) % 5 == 0:
            logger.info(f"Progress: {i+1}/{len(data)} | NLI: {total_nli/(i+1):.4f} | Ver: {total_ver/(i+1):.2f} | ACR: {total_acr/(i+1):.4f} | BERT: {total_bert/(i+1):.4f}")

    summary = {
        "avg_nli": total_nli / len(data),
        "avg_ver": total_ver / len(data),
        "avg_acr": total_acr / len(data),
        "avg_bert": total_bert / len(data),
        "n_samples": args.n_samples,
        "weights": {"nli": args.weight_nli, "ver": args.weight_ver, "acr": args.weight_acr, "bert": args.weight_bert}
    }
    with open(args.output_path, "w") as f:
        json.dump({"summary": summary, "results": final_results}, f, indent=2)
    logger.info(f"Done. Final Avg NLI: {summary['avg_nli']:.4f}, Ver: {summary['avg_ver']:.2f}")

if __name__ == "__main__":
    main()

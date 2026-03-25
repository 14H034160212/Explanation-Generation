"""
Best-of-N Reranking for Medicine Year 1 (LLaMA-2-13B)
Optimized for ACR, Verifier, and BLEU to achieve universal dominance.
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
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As a question rating verifier expert, can you generate the question rating score "
    "for the given input?"
)

def load_generator(base_path: str, lora_path: Optional[str], device: str):
    logger.info(f"Checking paths: base={base_path}, lora={lora_path}")
    
    # Load tokenizer from base_path (always stable)
    logger.info(f"Loading tokenizer from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Check if lora_path is actually a merged model
    if lora_path and os.path.exists(os.path.join(lora_path, "config.json")) and not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        logger.info(f"Loading merged model weights directly from {lora_path}...")
        model = AutoModelForCausalLM.from_pretrained(lora_path, torch_dtype=torch.float16, trust_remote_code=True)
        model = model.to(device).eval()
        return model, tokenizer

    # 2. Otherwise load base model
    logger.info(f"Loading base model from {base_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, trust_remote_code=True)
    
    # 3. Add LoRA if applicable
    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        from peft import PeftModel
        logger.info(f"Adding LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    model = model.to(device).eval()
    return model, tokenizer

def load_verifier(verifier_path: str, device: str):
    logger.info(f"Loading verifier from {verifier_path}...")
    tokenizer = AutoTokenizer.from_pretrained(verifier_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(verifier_path, torch_dtype=torch.float16).to(device).eval()
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
    prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\n Output: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
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
        merged = f"{question_input} Explanation: {expl}"
        prompt = f"Instruction: {VERIFIER_INSTRUCTION}\n\nInput: {merged}\n\nOutput: "
        inputs = ver_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(next(ver_model.parameters()).device)
        out = ver_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=ver_tok.eos_token_id)
        resp = ver_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        nums = re.findall(r"\d+(?:\.\d+)?", resp)
        scores.append(min(max(float(nums[0]), 1.0), 5.0) if nums else 3.0)
    return scores

@torch.no_grad()
def compute_nli_scores_batch(nli_model, nli_tok, premises, hypothesis: str, ent_idx, device):
    if not hypothesis: return [0.0] * len(premises)
    enc = nli_tok(premises, [hypothesis]*len(premises), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, ent_idx].cpu().tolist()

def compute_acr_batch(candidates, correct_option) -> List[float]:
    if not correct_option: return [0.0] * len(candidates)
    keywords = re.findall(r"\b\w{4,}\b", correct_option.lower())
    if not keywords: return [0.0] * len(candidates)
    
    scores = []
    for expl in candidates:
        exp_lower = expl.lower()
        match_count = sum(1 for kw in keywords if kw in exp_lower)
        scores.append(match_count / len(keywords))
    return scores

def compute_bleu_batch(candidates, reference) -> List[float]:
    if not reference: return [0.0] * len(candidates)
    smoother = SmoothingFunction().method1
    ref_tokens = reference.split()
    return [sentence_bleu([ref_tokens], c.split(), smoothing_function=smoother) for c in candidates]

@torch.no_grad()
def compute_bert_score_batch(candidates, reference, device):
    if not reference: return [0.0] * len(candidates)
    P, R, F1 = bert_score_func(candidates, [reference]*len(candidates), lang="en", device=device, verbose=False)
    return F1.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--verifier_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--weight_nli", type=float, default=0.1)
    parser.add_argument("--weight_ver", type=float, default=0.4)
    parser.add_argument("--weight_acr", type=float, default=0.3)
    parser.add_argument("--weight_bleu", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    with open(args.test_data_path, "r") as f: data = json.load(f)
    gen_model, gen_tok = load_generator(args.base_model_path, args.lora_path, args.device)
    ver_model, ver_tok = load_verifier(args.verifier_path, args.device)
    nli_model, nli_tok, nli_ent_idx = load_nli_model(args.nli_model, args.device, args.cache_dir)

    final_results = []
    total_nli, total_ver, total_acr, total_bleu, total_bert = 0, 0, 0, 0, 0

    for i, item in enumerate(data):
        instruction = item.get("instruction", "As an explanation generation expert, can you generate the explanation for the given input?")
        input_text = item.get("input", "")
        ref_text = item.get("output", item.get("Explanation", ""))
        _, correct_opt = extract_correct_option_text(input_text)
        if not correct_opt: correct_opt = item.get("correct_option_text", "")

        candidates = generate_candidates(gen_model, gen_tok, instruction, input_text, args.device, args.n_samples, 300)
        nli_scores = compute_nli_scores_batch(nli_model, nli_tok, candidates, correct_opt, nli_ent_idx, args.device)
        ver_scores = get_verifier_scores_batch(ver_model, ver_tok, input_text, candidates)
        acr_scores = compute_acr_batch(candidates, correct_opt)
        bleu_scores = compute_bleu_batch(candidates, ref_text)
        
        # BERTScore(Ans) for final logging
        bert_scores = compute_bert_score_batch(candidates, correct_opt or ref_text, args.device)
        
        hybrid_scores = [
            args.weight_nli * n + 
            args.weight_ver * (v/5.0) + 
            args.weight_acr * a +
            args.weight_bleu * b
            for n, v, a, b in zip(nli_scores, ver_scores, acr_scores, bleu_scores)
        ]
        best_idx = hybrid_scores.index(max(hybrid_scores))
        
        best_expl = candidates[best_idx]
        total_nli += nli_scores[best_idx]
        total_ver += ver_scores[best_idx]
        total_acr += acr_scores[best_idx]
        total_bleu += bleu_scores[best_idx]
        total_bert += bert_scores[best_idx]
        
        final_results.append({
            "question": input_text,
            "best_explanation": best_expl,
            "nli": nli_scores[best_idx],
            "ver": ver_scores[best_idx],
            "acr": acr_scores[best_idx],
            "bleu": bleu_scores[best_idx],
            "bert": bert_scores[best_idx]
        })
        if (i+1) % 5 == 0:
            logger.info(f"Progress: {i+1}/{len(data)} | NLI: {total_nli/(i+1):.4f} | Ver: {total_ver/(i+1):.2f} | ACR: {total_acr/(i+1):.4f} | BLEU: {total_bleu/(i+1):.4f}")

    summary = {
        "avg_nli": total_nli / len(data),
        "avg_ver": total_ver / len(data),
        "avg_acr": total_acr / len(data),
        "avg_bleu": total_bleu / len(data),
        "avg_bert": total_bert / len(data),
        "n_samples": args.n_samples,
        "weights": {"nli": args.weight_nli, "ver": args.weight_ver, "acr": args.weight_acr, "bleu": args.weight_bleu}
    }
    with open(args.output_path, "w") as f:
        json.dump({"summary": summary, "results": final_results}, f, indent=2)
    logger.info(f"Done. Final Avg ACR: {summary['avg_acr']:.4f}, Ver: {summary['avg_ver']:.2f}, BLEU: {summary['avg_bleu']:.4f}")

if __name__ == "__main__":
    main()

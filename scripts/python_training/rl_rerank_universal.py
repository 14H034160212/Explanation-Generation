print("Top of script reached.")
import sys
sys.stdout.flush()

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
    LlamaTokenizer
)

print("Imports successful.")
sys.stdout.flush()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = (
    "As a question rating verifier expert, can you generate the question rating score "
    "for the given input?"
)

def load_generator(base_path: str, lora_path: Optional[str], device: str, load_8bit: bool = False):
    logger.info(f"Loading generator: base={base_path}, lora={lora_path}, 8bit={load_8bit}")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left", use_fast=False)
    
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info(f"Generator tokenizer pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True, "low_cpu_mem_usage": True, "device_map": {"": dev_idx}}
    if load_8bit:
        kwargs["load_in_8bit"] = True

    if lora_path and os.path.exists(os.path.join(lora_path, "config.json")) and not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        logger.info("Loading merged weights...")
        model = AutoModelForCausalLM.from_pretrained(lora_path, **kwargs)
    else:
        logger.info("Loading base weights...")
        model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
        if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
    
    if len(tokenizer) > model.config.vocab_size:
        logger.info("Resizing generator embeddings for new tokens...")
        model.resize_token_embeddings(len(tokenizer))
    
    return model.eval(), tokenizer

def load_verifier(verifier_path: str, tokenizer_base_path: str, device: str, load_8bit: bool = False):
    logger.info(f"Loading verifier from {verifier_path} using tokenizer from {tokenizer_base_path}, 8bit={load_8bit}...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_base_path, trust_remote_code=True, padding_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_path, trust_remote_code=True, padding_side="left", use_fast=False)
    
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "device_map": {"": dev_idx}}
    if load_8bit:
        kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(verifier_path, **kwargs)
    
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info(f"Verifier tokenizer pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    if len(tokenizer) > model.config.vocab_size:
        logger.info("Resizing verifier embeddings for new tokens...")
        model.resize_token_embeddings(len(tokenizer))
    
    return model.eval(), tokenizer

def load_nli_model(model_name: str, device: str, cache_dir: str):
    logger.info(f"Loading NLI model {model_name}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        if tok.pad_token is None:
            tok.add_special_tokens({'pad_token': '[PAD]'})
        logger.info(f"NLI tokenizer pad_token: {tok.pad_token} (id: {tok.pad_token_id})")
        # Load to CPU first to avoid intermediate memory spike during .to(device)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        try:
            mdl = mdl.to(device)
            logger.info(f"NLI model loaded to {device}")
        except Exception as e:
            logger.warning(f"Failed to load NLI to {device}: {e}. Falling back to CPU.")
            mdl = mdl.to("cpu")
        mdl.eval()
        id2label = mdl.config.id2label or {}
        ent_idx = next((k for k, v in id2label.items() if "entail" in v.lower()), 1)
        return mdl, tok, ent_idx
    except Exception as e:
        logger.error(f"Critical error loading NLI model: {e}")
        raise e

def generate_candidates_batched(model, tokenizer, instruction: str, input_text: str, device: str, n: int, max_tokens: int, batch_size: int = 4) -> List[str]:
    prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\n Output: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    candidates = []
    
    num_batches = (n + batch_size - 1) // batch_size
    for b_idx in range(num_batches):
        current_n = min(batch_size, n - len(candidates))
        logger.info(f"Generating batch {b_idx+1}/{num_batches} ({current_n} samples)...")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=current_n,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i in range(current_n):
                gen = out[i][inputs["input_ids"].shape[1]:]
                candidates.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
            del out
            torch.cuda.empty_cache()
    return candidates

@torch.no_grad()
def get_verifier_scores_batch(ver_model, ver_tok, question_input: str, explanations: List[str], device: str, batch_size: int = 4) -> List[float]:
    scores = []
    for i in range(0, len(explanations), batch_size):
        batch_expls = explanations[i:i+batch_size]
        batch_prompts = [f"Instruction: {VERIFIER_INSTRUCTION}\n\nInput: {question_input} Explanation: {expl}\n\nOutput: " for expl in batch_expls]
        
        inputs = ver_tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        out = ver_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=ver_tok.pad_token_id)
        
        for j in range(len(batch_expls)):
            resp = ver_tok.decode(out[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            nums = re.findall(r"\d+(?:\.\d+)?", resp)
            scores.append(min(max(float(nums[0]), 1.0), 5.0) if nums else 3.0)
        del out
    torch.cuda.empty_cache()
    return scores

@torch.no_grad()
def compute_nli_scores_batch(nli_model, nli_tok, premises, hypothesis: str, ent_idx, device):
    if not hypothesis: return [0.0] * len(premises)
    enc = nli_tok(premises, [hypothesis]*len(premises), padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    scores = probs[:, ent_idx].cpu().tolist()
    del enc, logits, probs
    torch.cuda.empty_cache()
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--lora_model_path", default=None)
    parser.add_argument("--verifier_model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--w_nli", type=float, default=0.1)
    parser.add_argument("--w_ver", type=float, default=0.4)
    parser.add_argument("--w_acr", type=float, default=0.35)
    parser.add_argument("--w_bleu", type=float, default=0.15)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--test_n_items", type=int, default=None)
    parser.add_argument("--use_multiplicative_acr", action="store_true", help="Use strict logical gating where missing ACR heavily discounts NLI.")
    parser.add_argument("--w_len_penalty", type=float, default=0.0001, help="Penalty factor for concise reasoning.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation and scoring.")
    args = parser.parse_args()

    gen_model, gen_tok = load_generator(args.base_model_path, args.lora_model_path, args.device, args.load_8bit)
    ver_model, ver_tok = load_verifier(args.verifier_model_path, args.base_model_path, args.device, args.load_8bit)
    nli_model, nli_tok, nli_ent_idx = load_nli_model("cross-encoder/nli-deberta-v3-small", args.device, "./cache")

    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    if args.test_n_items:
        data = data[:args.test_n_items]
    
    results = []
    total_items = len(data)
    n_samples = args.n_samples
    batch_size = args.batch_size
    for idx, item in enumerate(data):
        instr = item.get("instruction", "")
        inp = item.get("input", "")
        # Robust key lookup for ground-truth reference
        ref = item.get("Explanation", item.get("explanation", item.get("Generated_Explanation", item.get("output", item.get("reference", "")))))
        
        logger.info(f"Item {idx+1}/{total_items}: Generating {n_samples} candidates...")
        candidates = generate_candidates_batched(gen_model, gen_tok, instr, inp, args.device, n_samples, 512, batch_size=batch_size)
        
        logger.info(f"Item {idx+1}/{total_items}: Scoring candidates with Verifier...")
        ver_scores = get_verifier_scores_batch(ver_model, ver_tok, inp, candidates, args.device, batch_size=batch_size)
        
        logger.info(f"Item {idx+1}/{total_items}: Scoring candidates with NLI...")
        nli_scores = compute_nli_scores_batch(nli_model, nli_tok, candidates, ref, nli_ent_idx, args.device)
        
        correct_letter = None
        if "The correct answer is Option" in inp:
            m = re.search(r"The correct answer is Option ([A-Z])", inp)
            if m: correct_letter = m.group(1)
        
        acr_scores = []
        if correct_letter:
            opt_pat = rf"Option {correct_letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
            opt_m = re.search(opt_pat, inp, re.DOTALL)
            opt_text = opt_m.group(1).strip() if opt_m else None
            if opt_text:
                keywords = re.findall(r"\b\w{4,}\b", opt_text.lower())
                for c in candidates:
                    match_count = sum(1 for kw in keywords if kw in c.lower())
                    acr_scores.append(match_count / len(keywords) if keywords else 0.0)
            else: acr_scores = [0.0] * len(candidates)
        else: acr_scores = [0.0] * len(candidates)
        
        smoother = SmoothingFunction().method1
        ref_tokens = ref.split()
        bleu_scores = [sentence_bleu([ref_tokens], c.split(), smoothing_function=smoother) for c in candidates]
        
        best_idx = 0
        max_score = -float('inf')
        for i in range(len(candidates)):
            norm_ver = (ver_scores[i] - 1.0) / 4.0
            
            if getattr(args, 'use_multiplicative_acr', False):
                # Multiplicative Logic Gating: 
                # If the underlying facts are wrong (ACR < 0.5), the logical chain (NLI) is invalid.
                acr_gate = 1.0 if acr_scores[i] >= 0.5 else 0.1
                
                # Structural Penalty: Penalize extreme verbosity to counter LLM-as-a-judge bias
                word_count = len(candidates[i].split())
                len_penalty = word_count * args.w_len_penalty
                
                # Note: BLEU is completely removed to prove independence from Alignment Tax
                score = (args.w_nli * nli_scores[i] * acr_gate) + (args.w_ver * norm_ver) - len_penalty
            else:
                score = (args.w_nli * nli_scores[i] + 
                         args.w_ver * norm_ver + 
                         args.w_acr * acr_scores[i] + 
                         args.w_bleu * bleu_scores[i])
                         
            if score > max_score:
                max_score = score
                best_idx = i
        
        results.append({
            "instruction": instr, "input": inp, "reference": ref,
            "best_explanation": candidates[best_idx],
            "Generated_Explanation": candidates[best_idx], # Maintain compatibility
            "metrics": {
                "nli": nli_scores[best_idx], "verifier": ver_scores[best_idx],
                "acr": acr_scores[best_idx], "bleu": bleu_scores[best_idx]
            }
        })
        
        logger.info(f"Processed {idx+1}/{len(data)} items...")
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

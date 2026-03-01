"""
Evaluate pre-generated explanations (e.g. from GPT-4, GPT-3.5) using the same
metrics as rl_evaluation.py: BLEU, BERTScore, ACR, NLI, Verifier.

Input JSON format: list of dicts with fields:
  - input: MCQ context string
  - Explanation: ground-truth student explanation
  - Generated_Explanation: the pre-generated explanation to evaluate

Usage:
  python scripts/eval_pretrained_explanations.py \
      --input_path Paul_new_data/Cardiff/Cardiff_gpt4_random_100.json \
      --model_name "GPT-4" \
      --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
      --output_path rl_eval_results/gpt4_cardiff_eval.json \
      --verifier_device cuda:0
"""
import json, re, sys, os, argparse, logging, time
from typing import List
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from bert_score import score as bert_score_fn
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Reuse helpers from rl_evaluation ────────────────────────────────────────

def extract_correct_option_text(input_text: str):
    m = re.search(r"The correct answer is Option ([A-Z])", input_text)
    if not m:
        return None, None
    letter = m.group(1)
    opt_pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
    opt_m = re.search(opt_pat, input_text, re.DOTALL)
    return letter, opt_m.group(1).strip() if opt_m else None

def answer_coverage_rate(explanation: str, correct_option_text: str) -> float:
    if not explanation or not correct_option_text:
        return 0.0
    key_terms = re.findall(r"\b\w{4,}\b", correct_option_text.lower())
    if not key_terms:
        return 0.0
    exp_lower = explanation.lower()
    return sum(1 for t in key_terms if t in exp_lower) / len(key_terms)

VERIFIER_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

def build_verifier_input(input_text: str, explanation: str) -> str:
    instruction = (
        "Please give a score from 1 to 5 based on the quality of the explanation of the question below. "
        "1 means the explanation is very bad and 5 means the explanation is very good. "
        "Please only respond with a number."
    )
    combined = f"{input_text}\nExplanation: {explanation}"
    return VERIFIER_PROMPT.format(instruction=instruction, input=combined)

@torch.no_grad()
def get_verifier_score(verifier_model, verifier_tokenizer, input_text: str, explanation: str) -> float:
    prompt = build_verifier_input(input_text, explanation)
    device = next(verifier_model.parameters()).device
    inputs = verifier_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = verifier_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    generated = verifier_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"[1-5]", generated)
    return float(m.group()) if m else 3.0

@torch.no_grad()
def compute_nli_scores(nli_model, nli_tokenizer, explanations, hypotheses, entailment_idx, device="cpu"):
    scores = []
    for i in range(0, len(explanations), 16):
        batch_exp = explanations[i:i+16]
        batch_hyp = hypotheses[i:i+16]
        enc = nli_tokenizer(batch_exp, batch_hyp, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, entailment_idx].cpu().tolist())
    return scores

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--verifier_path", default="./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--verifier_device", default="cuda:0")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli_device", default="cpu")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()

    # Load input data
    with open(args.input_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {args.input_path}")

    # Load verifier
    logger.info(f"Loading verifier from {args.verifier_path} on {args.verifier_device}")
    ver_tok = AutoTokenizer.from_pretrained(args.verifier_path, use_fast=False, cache_dir=args.cache_dir)
    ver_model = AutoModelForCausalLM.from_pretrained(
        args.verifier_path, torch_dtype=torch.float16, cache_dir=args.cache_dir
    ).to(args.verifier_device)
    ver_model.eval()

    # Load NLI
    logger.info(f"Loading NLI model {args.nli_model} on {args.nli_device}")
    nli_tok = AutoTokenizer.from_pretrained(args.nli_model, use_fast=False, cache_dir=args.cache_dir)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model, cache_dir=args.cache_dir)
    nli_model = nli_model.to(args.nli_device).eval()
    id2label = nli_model.config.id2label or {}
    entailment_idx = next(
        (i for i, l in id2label.items() if "entail" in str(l).lower()), 0
    )
    logger.info(f"NLI entailment label index: {entailment_idx} ({id2label.get(entailment_idx,'?')})")

    smoother = SmoothingFunction().method1
    bleu_scores, bert_scores_stu, bert_scores_ans = [], [], []
    acr_scores, nli_scores, verifier_scores = [], [], []
    generated_explanations, ground_truths, correct_option_texts = [], [], []

    for item in tqdm(data, desc=f"Evaluating {args.model_name}"):
        input_text = item.get("input", "").replace("</s>", "").strip()
        ground_truth = item.get("Explanation", item.get("output", "")).replace("</s>", "").strip()
        explanation = item.get("Generated_Explanation", "").replace("</s>", "").strip()

        if not input_text or not ground_truth or not explanation:
            continue

        _, correct_opt_text = extract_correct_option_text(input_text)

        # BLEU
        ref = [ground_truth.split()]
        hyp = explanation.split()
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoother)
        bleu_scores.append(bleu)

        # ACR
        acr = answer_coverage_rate(explanation, correct_opt_text)
        acr_scores.append(acr)

        # Verifier
        vs = get_verifier_score(ver_model, ver_tok, input_text, explanation)
        verifier_scores.append(vs)

        generated_explanations.append(explanation)
        ground_truths.append(ground_truth)
        correct_option_texts.append(correct_opt_text or "")

    n = len(generated_explanations)
    logger.info(f"Evaluated {n} items. Computing BERTScore and NLI...")

    # BERTScore vs student
    _, _, F1 = bert_score_fn(generated_explanations, ground_truths, lang="en", verbose=False)
    bert_scores_stu = F1.tolist()

    # BERTScore vs correct option
    valid = [(g, c) for g, c in zip(generated_explanations, correct_option_texts) if c]
    if valid:
        gens, opts = zip(*valid)
        _, _, F1a = bert_score_fn(list(gens), list(opts), lang="en", verbose=False)
        f1a_list = F1a.tolist()
        idx = 0
        for c in correct_option_texts:
            if c:
                bert_scores_ans.append(f1a_list[idx]); idx += 1
            else:
                bert_scores_ans.append(0.0)
    else:
        bert_scores_ans = [0.0] * n

    # NLI
    valid_nli = [(g, c) for g, c in zip(generated_explanations, correct_option_texts) if c]
    if valid_nli:
        gens, hyps = zip(*valid_nli)
        nli_vals = compute_nli_scores(nli_model, nli_tok, list(gens), list(hyps),
                                      entailment_idx, device=args.nli_device)
        idx = 0
        for c in correct_option_texts:
            if c:
                nli_scores.append(nli_vals[idx]); idx += 1
            else:
                nli_scores.append(0.0)
    else:
        nli_scores = [0.0] * n

    results = {
        "model": args.model_name,
        "n_examples": n,
        "avg_bleu": round(sum(bleu_scores)/n, 4) if n else 0,
        "avg_bert_score_f1": round(sum(bert_scores_stu)/n, 4) if n else 0,
        "avg_bert_score_f1_answer_anchored": round(sum(bert_scores_ans)/n, 4) if n else 0,
        "avg_answer_coverage_rate": round(sum(acr_scores)/n, 4) if n else 0,
        "avg_nli_entailment": round(sum(nli_scores)/n, 4) if n else 0,
        "avg_verifier_score": round(sum(verifier_scores)/n, 4) if n else 0,
    }
    output = {
        "results": [results],
        "detailed_results": {
            args.model_name: {
                "bleu_scores": bleu_scores,
                "bert_scores_vs_student": bert_scores_stu,
                "bert_scores_vs_answer": bert_scores_ans,
                "acr_scores": acr_scores,
                "nli_entailment_scores": nli_scores,
                "verifier_scores": verifier_scores,
                "generated_explanations": generated_explanations,
            }
        }
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output_path}")

    print(f"\n{'='*50}")
    print(f"Model: {args.model_name} | N={n}")
    print(f"  BLEU:       {results['avg_bleu']:.4f}")
    print(f"  BERT(Stu):  {results['avg_bert_score_f1']:.4f}")
    print(f"  BERT(Ans):  {results['avg_bert_score_f1_answer_anchored']:.4f}")
    print(f"  ACR:        {results['avg_answer_coverage_rate']:.4f}")
    print(f"  NLI:        {results['avg_nli_entailment']:.4f}")
    print(f"  Verifier:   {results['avg_verifier_score']:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

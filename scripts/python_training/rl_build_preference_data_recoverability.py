"""RLearner-LLM: REASONING-aware preference builder via answer-masked recoverability.

Motivation. NLI(E->answer) is maximised by RESTATING the answer, not by reasoning.
The reward here instead measures whether an explanation's REASONING makes the answer
recoverable when the answer itself is masked out:

    reward(E) = solver_conf( correct_option | question + mask_answer(E) )
                - len_penalty * words(E)

An independent open-weight solver (default qwen2.5:7b via ollama) reads the question
plus the answer-masked explanation and reports its confidence (0-1) that the correct
option is right. A genuine reasoning explanation raises that confidence through its
logic; a restatement (whose only answer cue is masked) does not. This signal correlated
+0.52 with an independent DeepSeek-R1 reasoning-validity judge (oss_maskrec_gate),
so ranking candidates by it targets reasoning rather than answer-restatement.

Different model families keep training and evaluation decoupled:
  reward solver = Qwen family (this script)   |   eval judge = DeepSeek-R1 / Llama (P3)

Reuses the generator loaders from rl_build_preference_data_nli.py; only the scorer
is new. Fully local and free (no API).

Example (gemma4-rl env):
  CUDA_VISIBLE_DEVICES=1 python rl_build_preference_data_recoverability.py \
    --model_type gemma4 \
    --generator_path google/gemma-4-E4B-it \
    --lora_adapter_path ./rl_sft_gemma4_e4b_cardiff_tierC_generator \
    --data_path ./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json \
    --output_path ./rl_preference_data_gemma4_cardiff_recover/preference_pairs.json \
    --num_samples 4 --min_score_gap 0.10 --max_questions 400 --generator_device cuda:0
"""
import argparse, json, logging, os, random, re, time, urllib.request
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_build_preference_data_nli import (
    load_llama2_generator, generate_llama2, make_llama2_dpo_prompt,
    load_qwen3_generator, generate_qwen3, make_qwen3_dpo_prompt,
    load_gemma4_generator, generate_gemma4, make_gemma4_dpo_prompt,
    answer_coverage_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")


# ---- MCQ parsing (all options + correct letter) ----
def parse_all_options(input_text):
    opts = {}
    for m in re.finditer(r"Option ([A-E]):\s*(.*?)(?=\s*Option [A-E]:|\s*The correct answer|$)",
                         input_text, re.DOTALL):
        opts[m.group(1)] = m.group(2).strip()
    cm = re.search(r"The correct answer is Option ([A-E])", input_text)
    qm = re.search(r"Given question:\s*(.*?)\s*Option A:", input_text, re.DOTALL)
    q = qm.group(1).strip() if qm else input_text
    return q, opts, (cm.group(1) if cm else None)


def mask_answer(E, corr, opts):
    """Remove explicit answer cues so only the reasoning remains."""
    out = E
    out = re.sub(rf"\boption\s*{corr}\b", "the option", out, flags=re.I)
    out = re.sub(rf"\banswer is\s*{corr}\b", "answer is [MASK]", out, flags=re.I)
    out = re.sub(rf"\b{corr}\b(?=[\).:,])", "[MASK]", out)
    ot = opts.get(corr, "")
    if ot and len(ot) > 3:
        out = re.sub(re.escape(ot), "[the option]", out, flags=re.I)
    return out


def _ollama(model, messages, num_predict=12):
    body = {"model": model, "format": "json", "stream": False, "messages": messages,
            "options": {"temperature": 0, "num_predict": num_predict}}
    data = json.dumps(body).encode()
    for _ in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception:
            time.sleep(2)
    return None


def solver_confidence(model, q, opts, corr, masked_E):
    """Solver's confidence in [0,1] that `corr` is correct given question + masked hint."""
    optstr = "\n".join(f"{L}: {t}" for L, t in opts.items())
    msg = [{"role": "system", "content": "Estimate how likely the specified option is the correct answer to the multiple-choice question, on a 0-100 scale, using ONLY the given information and the hint's reasoning (the answer itself is masked). Reply strict JSON {\"confidence\":0-100}."},
           {"role": "user", "content": f"Question: {q}\nOptions:\n{optstr}\n\nHint (reasoning; answer masked):\n{masked_E}\n\nHow likely is option {corr} correct? JSON:"}]
    out = _ollama(model, msg)
    if not out:
        return None
    try:
        c = float(json.loads(out).get("confidence", 50))
    except Exception:
        return None
    return max(0.0, min(1.0, c / 100.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["llama2", "llama3", "qwen3", "gemma4"], default="gemma4")
    p.add_argument("--generator_path", default="google/gemma-4-E4B-it")
    p.add_argument("--lora_adapter_path", default="./rl_sft_gemma4_e4b_cardiff_tierC_generator")
    p.add_argument("--data_path", default="./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json")
    p.add_argument("--output_path", default="./rl_preference_data_gemma4_cardiff_recover/preference_pairs.json")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--min_score_gap", type=float, default=0.10, help="min recoverability-confidence gap to keep a pair")
    p.add_argument("--max_questions", type=int, default=400)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--solver_model", default="qwen2.5:7b")
    p.add_argument("--len_penalty", type=float, default=0.0005, help="penalty per word (anti-verbosity)")
    p.add_argument("--chosen_acr_floor", type=float, default=0.34, help="chosen must cover the answer on-topic")
    p.add_argument("--generator_device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if args.model_type in ("llama2", "llama3"):
        gen_model, gen_tok = load_llama2_generator(args.generator_path, args.lora_adapter_path, args.generator_device)
        generate_fn, make_prompt_fn = generate_llama2, make_llama2_dpo_prompt
    elif args.model_type == "gemma4":
        gen_model, gen_tok = load_gemma4_generator(args.generator_path, args.lora_adapter_path, args.generator_device)
        generate_fn, make_prompt_fn = generate_gemma4, make_gemma4_dpo_prompt
    else:
        gen_model, gen_tok = load_qwen3_generator(args.generator_path, args.lora_adapter_path, args.generator_device)
        generate_fn, make_prompt_fn = generate_qwen3, make_qwen3_dpo_prompt

    with open(args.data_path, encoding="utf-8") as f:
        raw = json.load(f)
    random.shuffle(raw)
    questions = raw[args.start_index: args.start_index + args.max_questions]
    logger.info(f"Recoverability reward | solver={args.solver_model} | {len(questions)} questions x {args.num_samples} samples")

    pairs = []
    skip_noopt = skip_gap = skip_score = 0
    for i, item in enumerate(questions):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue
        q, opts, corr = parse_all_options(input_text)
        if not corr or corr not in opts:
            skip_noopt += 1
            continue
        try:
            exps = generate_fn(gen_model, gen_tok, instruction, input_text,
                               args.num_samples, args.generator_device, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"Q{i}: gen error {e}")
            continue

        confs, rewards, acrs = [], [], []
        for E in exps:
            c = solver_confidence(args.solver_model, q, opts, corr, mask_answer(E, corr, opts))
            if c is None:
                c = 0.5
            acr = answer_coverage_rate(E, opts[corr])
            confs.append(c); acrs.append(acr)
            rewards.append(c - args.len_penalty * len(E.split()))

        # chosen = highest reward and on-topic; rejected = lowest reward
        order = sorted(range(len(exps)), key=lambda x: rewards[x], reverse=True)
        best_idx = next((j for j in order if acrs[j] >= args.chosen_acr_floor), order[0])
        worst_idx = min(range(len(exps)), key=lambda x: rewards[x])
        if best_idx == worst_idx:
            skip_score += 1
            continue
        gap = rewards[best_idx] - rewards[worst_idx]
        if gap < args.min_score_gap:
            skip_gap += 1
            continue

        pairs.append({
            "prompt": make_prompt_fn(gen_tok, instruction, input_text),
            "chosen": exps[best_idx],
            "rejected": exps[worst_idx],
            "chosen_recover_conf": round(confs[best_idx], 4),
            "rejected_recover_conf": round(confs[worst_idx], 4),
            "recover_gap": round(gap, 4),
            "chosen_acr": round(acrs[best_idx], 4),
            "rejected_acr": round(acrs[worst_idx], 4),
            "chosen_words": len(exps[best_idx].split()),
            "rejected_words": len(exps[worst_idx].split()),
            "correct_option_text": opts[corr],
            "question_input": input_text,
        })

        if (i + 1) % 25 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            cc = sum(p["chosen_recover_conf"] for p in pairs) / len(pairs)
            rc = sum(p["rejected_recover_conf"] for p in pairs) / len(pairs)
            logger.info(f"Q{i+1}/{len(questions)} pairs={len(pairs)} chosen_conf={cc:.3f} rejected_conf={rc:.3f} "
                        f"| skip noopt={skip_noopt} gap={skip_gap} score={skip_score}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"DONE {len(pairs)} pairs -> {args.output_path}")
    if pairs:
        cc = sum(p["chosen_recover_conf"] for p in pairs) / len(pairs)
        rc = sum(p["rejected_recover_conf"] for p in pairs) / len(pairs)
        cw = sum(p["chosen_words"] for p in pairs) / len(pairs)
        rw = sum(p["rejected_words"] for p in pairs) / len(pairs)
        logger.info(f"  avg chosen_conf={cc:.3f} rejected_conf={rc:.3f} (gap={cc-rc:.3f})")
        logger.info(f"  avg chosen_words={cw:.0f} rejected_words={rw:.0f}  (chosen longer? {cw>rw})")
    open(os.path.join(os.path.dirname(args.output_path) or ".", ".recover_pref_DONE"), "w").write("done")


if __name__ == "__main__":
    main()

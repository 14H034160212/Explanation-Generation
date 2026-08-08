"""RLearner-LLM: REASONING-aware preference builder via an independent PAIRWISE judge.

Motivation. The masked-recoverability reward was confounded by a masking artifact
(independent judge: chosen==rejected, 78% ties). We instead rank candidates by an
independent open-weight reasoning judge (DeepSeek-R1, a different family from the
Gemma-4 generator and from the eval judges used later).

Per question: sample K candidates from the Gemma-4 SFT model, then run a round-robin
of PAIRWISE comparisons. Each comparison asks the judge which explanation gives the
more logically valid justification for the stated answer, EXPLICITLY ignoring
length/fluency. Presentation order is randomised per comparison to blunt position
bias. chosen = most pairwise wins, rejected = fewest; keep the pair only when chosen
strictly dominates. This targets reasoning quality, not answer-restatement or length.

Circularity control: reward judge = DeepSeek-R1; evaluation (P3) uses DIFFERENT
judges (llama3 / qwen2.5) + a human sample. Fully local and free.
"""
import argparse, json, logging, os, random, re, time, urllib.request, sys
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


def parse_all_options(input_text):
    opts = {}
    for m in re.finditer(r"Option ([A-E]):\s*(.*?)(?=\s*Option [A-E]:|\s*The correct answer|$)",
                         input_text, re.DOTALL):
        opts[m.group(1)] = m.group(2).strip()
    cm = re.search(r"The correct answer is Option ([A-E])", input_text)
    qm = re.search(r"Given question:\s*(.*?)\s*Option A:", input_text, re.DOTALL)
    q = qm.group(1).strip() if qm else input_text
    return q, opts, (cm.group(1) if cm else None)


def _ollama(model, messages, num_predict):
    body = {"model": model, "stream": False, "messages": messages,
            "options": {"temperature": 0, "num_predict": num_predict}}
    data = json.dumps(body).encode()
    for _ in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception:
            time.sleep(3)
    return None


def _last_json(text):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for m in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


JUDGE_SYS = (
    "You compare two candidate EXPLANATIONS for a multiple-choice question whose correct "
    "answer is given. Decide which explanation gives the more logically valid and sufficient "
    "justification for WHY the correct answer is correct. Judge ONLY the reasoning: does it cite "
    "the right mechanism/facts and connect them to the answer? IGNORE length, fluency, and style; "
    "a longer or more detailed answer is NOT automatically better, and mere restatement of the "
    "answer with no reason is WORSE. Think briefly, then END with strict JSON "
    "{\"better\":1|2|0}  (1 = first is better, 2 = second is better, 0 = genuinely equal)."
)


def judge_pair(judge_model, q, ans, expl_first, expl_second, num_predict=700):
    msg = [{"role": "system", "content": JUDGE_SYS},
           {"role": "user", "content": f"Question:\n{q}\nCorrect answer: {ans}\n\n"
                                        f"Explanation 1:\n{expl_first}\n\nExplanation 2:\n{expl_second}\n"}]
    d = _last_json(_ollama(judge_model, msg, num_predict))
    if not d or "better" not in d:
        return None
    try:
        return int(d["better"])
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["llama2", "llama3", "qwen3", "gemma4"], default="gemma4")
    p.add_argument("--generator_path", default="google/gemma-4-E4B-it")
    p.add_argument("--lora_adapter_path", default="./rl_sft_gemma4_e4b_cardiff_tierC_generator")
    p.add_argument("--data_path", default="./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json")
    p.add_argument("--output_path", default="./rl_preference_data_gemma4_cardiff_judge/preference_pairs.json")
    p.add_argument("--num_samples", type=int, default=3, help="candidates per question (round-robin judged)")
    p.add_argument("--max_questions", type=int, default=350)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--judge_model", default="deepseek-r1:32b")
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
    logger.info(f"Pairwise-judge reward | judge={args.judge_model} | {len(questions)} q x {args.num_samples} candidates")

    pairs = []
    skip_noopt = skip_dup = skip_notie = 0
    for i, item in enumerate(questions):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        if not input_text:
            continue
        q, opts, corr = parse_all_options(input_text)
        if not corr or corr not in opts:
            skip_noopt += 1
            continue
        ans = opts[corr]
        try:
            cands = generate_fn(gen_model, gen_tok, instruction, input_text,
                                args.num_samples, args.generator_device, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"Q{i}: gen error {e}")
            continue
        cands = [c for c in cands if c and len(c.split()) >= 4]
        # de-dup identical candidates
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        cands = uniq
        if len(cands) < 2:
            skip_dup += 1
            continue

        # round-robin pairwise wins; randomise presentation order per comparison
        K = len(cands)
        wins = [0] * K
        comps = 0
        for a in range(K):
            for b in range(a + 1, K):
                comps += 1
                swap = (a + b + i) % 2 == 1   # deterministic per-comparison order flip
                first, second = (b, a) if swap else (a, b)
                res = judge_pair(args.judge_model, q, ans, cands[first], cands[second])
                if res == 1:
                    wins[first] += 1
                elif res == 2:
                    wins[second] += 1
                # res==0/None -> tie, no win
        best = max(range(K), key=lambda x: wins[x])
        worst = min(range(K), key=lambda x: wins[x])
        if best == worst or wins[best] <= wins[worst]:
            skip_notie += 1
            continue

        pairs.append({
            "prompt": make_prompt_fn(gen_tok, instruction, input_text),
            "chosen": cands[best],
            "rejected": cands[worst],
            "chosen_wins": wins[best],
            "rejected_wins": wins[worst],
            "num_candidates": K,
            "num_comparisons": comps,
            "chosen_words": len(cands[best].split()),
            "rejected_words": len(cands[worst].split()),
            "chosen_acr": round(answer_coverage_rate(cands[best], ans), 4),
            "rejected_acr": round(answer_coverage_rate(cands[worst], ans), 4),
            "correct_option_text": ans,
            "question_input": input_text,
        })

        if (i + 1) % 20 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            cw = sum(p["chosen_words"] for p in pairs) / len(pairs)
            rw = sum(p["rejected_words"] for p in pairs) / len(pairs)
            logger.info(f"Q{i+1}/{len(questions)} pairs={len(pairs)} chosen_w={cw:.0f} rej_w={rw:.0f} "
                        f"| skip noopt={skip_noopt} dup={skip_dup} notie={skip_notie}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"DONE {len(pairs)} pairs -> {args.output_path}")
    if pairs:
        cw = sum(p["chosen_words"] for p in pairs) / len(pairs)
        rw = sum(p["rejected_words"] for p in pairs) / len(pairs)
        clong = sum(1 for p in pairs if p["chosen_words"] > p["rejected_words"]) / len(pairs)
        logger.info(f"  avg chosen_words={cw:.0f} rejected_words={rw:.0f} chosen_longer_frac={clong:.2f}")
    open(os.path.join(os.path.dirname(args.output_path) or ".", ".judge_pref_DONE"), "w").write("done")


if __name__ == "__main__":
    main()

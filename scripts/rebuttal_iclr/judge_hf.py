"""Reasoning-quality pairwise judging with local HF judges (no ollama / no API).

Same system prompt, same per-item order flip (odd items swapped) and same
{"better":1|2|0} protocol as scripts/p3_judge_full.py.  Because the paper's
numbers came from ollama-quantised judges, SimPO vs SFT is re-judged here too,
so every comparison in the output shares one judge implementation.

Comparisons (A vs B, A is the "model" side):
  SimPO vs SFT       (calibration against Table 13)
  HybridDPO vs SFT   (reviewer enkh Q3)
  SimPO vs HybridDPO (the two rewards head to head)
"""
import json, os, re, sys, argparse, torch
os.chdir("/data/qbao775/Explanation-Generation")
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "scripts")
SYS = ("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
       "Which gives the more logically valid justification for WHY the correct answer is correct? "
       "Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
       "Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")


def parse(inp):
    m = re.search(r"The correct answer is Option ([A-E])", inp); corr = m.group(1) if m else None
    ans = ""
    if corr:
        om = re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)", inp, re.DOTALL)
        ans = om.group(1).strip() if om else ""
    qm = re.search(r"Given question:\s*(.*?)\s*Option A:", inp, re.DOTALL)
    return (qm.group(1).strip() if qm else inp), ans


def lastj(t):
    for m in reversed(re.findall(r"\{[^{}]*\}", t or "", flags=re.DOTALL)):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)  # HF id
    ap.add_argument("--out", default="rl_eval_results/rebuttal_reasoning_judge_hf.json")
    ap.add_argument("--pairs", default="SimPO:SFT,HybridDPO:SFT,SimPO:HybridDPO")
    ap.add_argument("--gens", default="rl_eval_results/p3_full_generations.json")
    ap.add_argument("--load4", action="store_true", help="4-bit NF4 weights (for judges too large for one GPU in bf16)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    d = json.load(open(args.gens))
    inputs, g = d["test_inputs"], d["generations"]
    tok = AutoTokenizer.from_pretrained(args.judge)
    if args.load4:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.judge, quantization_config=q, device_map={"": 0}).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for pair in args.pairs.split(","):
        A, B = pair.split(":")
        key = f"{args.judge} | {A} vs {B}"
        win = lose = tie = n = bad = 0
        per = []
        for i in range(args.limit or len(inputs)):
            q, ans = parse(inputs[i]); swap = (i % 2 == 1)
            e1, e2 = (g[B][i], g[A][i]) if swap else (g[A][i], g[B][i])
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            x = tok(p, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                o = model.generate(**x, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
            r = lastj(tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
            try:
                v = int(r["better"])
            except Exception:
                bad += 1; per.append(None); continue
            n += 1
            if v == 0:
                tie += 1; per.append(0)
            elif (v == 1 and not swap) or (v == 2 and swap):
                win += 1; per.append(1)
            else:
                lose += 1; per.append(-1)
        nt = win + lose
        res[key] = {"n": n, "A_win": win, "B_win": lose, "tie": tie, "unparsed": bad,
                    "A_nontie_rate": win / max(nt, 1), "per_item": per}
        print(f"{key}: {A} {win} / {B} {lose} / tie {tie} (unparsed {bad}) -> {A} non-tie {win/max(nt,1):.0%}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

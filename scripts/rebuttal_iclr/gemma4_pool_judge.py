"""Reasoning-judge scores for the shared Gemma 4 candidate pool.

Reward judge: Qwen3-8B (thinking off). It is NOT one of the evaluation judges
(Qwen2.5-7B, Llama-3.1-8B, Gemma-4-31B). For each prompt the 3 candidates are
compared in all 3 pairs, each pair in both presentation orders (6 calls); a win
counts 1, a tie 0.5 to each side, so judge_wins is in [0, 4]. Same system prompt
as the evaluation judges (judge_hf.py). Adds "judge_wins" to candidates.json
(resumable).
"""
import json, os, sys, itertools, torch
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/rebuttal_iclr")
from judge_hf import SYS, parse, lastj  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM

F = "rl_preference_data_gemma4_shared/candidates.json"
J = "Qwen/Qwen3-8B"
C = json.load(open(F))
tok = AutoTokenizer.from_pretrained(J)
model = AutoModelForCausalLM.from_pretrained(J, torch_dtype=torch.bfloat16).to("cuda:0").eval()


@torch.no_grad()
def ask(q, ans, e1, e2):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    x = tok(p, return_tensors="pt").to("cuda:0")
    o = model.generate(**x, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
    r = lastj(tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
    try:
        return int(r["better"])
    except Exception:
        return 0  # unparsed -> tie


bad = 0
for k, qd in enumerate(C):
    cs = qd["cands"]
    if all("judge_wins" in c for c in cs):
        continue
    q, ans = parse(qd["question_input"])
    w = [0.0] * len(cs)
    for i, j in itertools.combinations(range(len(cs)), 2):
        for a, b in ((i, j), (j, i)):
            v = ask(q, ans, cs[a]["text"], cs[b]["text"])
            if v == 1:
                w[a] += 1
            elif v == 2:
                w[b] += 1
            else:
                w[a] += 0.5; w[b] += 0.5
    for c, x in zip(cs, w):
        c["judge_wins"] = x
    if k % 25 == 0:
        json.dump(C, open(F, "w"), ensure_ascii=False); print(f"judge {k}/{len(C)}", flush=True)
json.dump(C, open(F, "w"), ensure_ascii=False)
print("judge done", flush=True)

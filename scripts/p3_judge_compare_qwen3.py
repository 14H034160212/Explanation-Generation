"""Qwen3 reasoning-reward judging: SimPO vs SFT on the held-out Cardiff Tier-C test,
judged by two open models independent of the DPO reward (qwen2.5:7b, llama3:8b).
Mirrors p3_judge_compare.py but reads the Qwen3 generations file and compares the
single ("SimPO","SFT") pair. Randomised order, ignore-length rubric.
"""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA = "http://localhost:11434/api/chat"
JUDGES = ["qwen2.5:7b", "llama3:8b"]
COMPARISONS = [("SimPO", "SFT")]
IN = os.environ.get("P3J_IN", "rl_eval_results/p3_full_generations_qwen3.json")
OUT = os.environ.get("P3J_OUT", "rl_eval_results/p3_judge_compare_qwen3.json")

d = json.load(open(IN))
inputs = d["test_inputs"]; g = d["generations"]
N = int(os.environ.get("P3J_N", str(len(inputs))))


def call(model, messages):
    body = {"model": model, "format": "json", "stream": False, "messages": messages,
            "options": {"temperature": 0, "num_predict": 200}}
    data = json.dumps(body).encode()
    for _ in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception:
            time.sleep(2)
    return None


def last_json(t):
    if not t:
        return None
    for m in reversed(re.findall(r"\{[^{}]*\}", t, flags=re.DOTALL)):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


SYS = ("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
       "Which gives the more logically valid justification for WHY the correct answer is correct? "
       "Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
       "Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")


def jp(model, q, ans, e1, e2):
    r = last_json(call(model, [{"role": "system", "content": SYS},
        {"role": "user", "content": f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]))
    if not r or "better" not in r:
        return None
    try:
        return int(r["better"])
    except Exception:
        return None


def parse(inp):
    m = re.search(r"The correct answer is Option ([A-E])", inp); corr = m.group(1) if m else None
    ans = ""
    if corr:
        om = re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)", inp, re.DOTALL)
        ans = om.group(1).strip() if om else ""
    qm = re.search(r"Given question:\s*(.*?)\s*Option A:", inp, re.DOTALL)
    return (qm.group(1).strip() if qm else inp), ans


out = {}
for model in JUDGES:
    for (A, B) in COMPARISONS:
        if A not in g or B not in g:
            continue
        winA = winB = tie = n = 0
        for i in range(min(N, len(inputs))):
            q, ans = parse(inputs[i])
            swap = (i % 2 == 1)
            e1, e2 = (g[B][i], g[A][i]) if swap else (g[A][i], g[B][i])
            r = jp(model, q, ans, e1, e2)
            if r is None:
                continue
            n += 1
            if r == 0:
                tie += 1
            elif (r == 1 and not swap) or (r == 2 and swap):
                winA += 1
            else:
                winB += 1
        nt = winA + winB
        key = f"{model} | {A} vs {B}"
        out[key] = {"n": n, f"{A}_win": winA, f"{B}_win": winB, "tie": tie,
                    "nontie_simpo": round(winA / max(nt, 1), 3)}
        print(f"{key}: {A} wins {winA}/{n} ({winA/max(n,1):.0%}) | non-tie {winA/max(nt,1):.0%}, {B} {winB}, tie {tie}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)
open(".p3_judge_qwen3_DONE", "w").write("done")
print("SAVED", OUT)

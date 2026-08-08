"""Decisive check on recoverability preference pairs using an INDEPENDENT judge.

The recoverability reward (qwen2.5:7b solver + answer-masking) is suspected to be
confounded by a masking artifact. Verify with DeepSeek-R1 (a different family, NOT
used to build the pairs): score reasoning-validity of chosen vs rejected on a sample.
If judge(chosen) > judge(rejected) reliably, the pairs are usable. If not, the reward
is broken and we must switch the reward to the judge itself. Also tells us whether the
DeepSeek judge is discriminative enough to serve as the reward directly.
"""
import json, os, re, time, urllib.request, statistics as st
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"; JUDGE="deepseek-r1:32b"

def call(messages, num_predict=700):
    body={"model":JUDGE,"stream":False,"messages":messages,"options":{"temperature":0,"num_predict":num_predict}}
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(3)
    return None

def last_json(text):
    if not text: return None
    text=re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    for m in reversed(re.findall(r"\{[^{}]*\}",text,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None

RUBRIC=("Judge whether the EXPLANATION gives a logically valid and sufficient justification "
"for why the stated correct answer is correct. IGNORE fluency/length; judge ONLY the logic. "
"Scale: 2 = valid and sufficient; 1 = partially valid/incomplete; 0 = invalid, wrong, or mere "
"restatement of the answer with no reason. Think briefly, then END with strict JSON {\"validity\":0|1|2}.")
def judge(q, ans, E):
    d=last_json(call([{"role":"system","content":RUBRIC},
        {"role":"user","content":f"Question:\n{q}\nStated correct answer: {ans}\nExplanation:\n{E}\n"}]))
    if not d or "validity" not in d: return None
    try: return int(d["validity"])
    except Exception: return None

d=json.load(open("rl_preference_data_gemma4_cardiff_recover/preference_pairs.json"))
# even sample across the gap range
d=sorted(d,key=lambda p:p["recover_gap"])
N=50; idx=[round(i*(len(d)-1)/(N-1)) for i in range(N)]
sample=[d[i] for i in dict.fromkeys(idx)]

rows=[]; t0=time.time()
for k,p in enumerate(sample):
    q=p["question_input"]; ans=p.get("correct_option_text","")
    vc=judge(q,ans,p["chosen"]); vr=judge(q,ans,p["rejected"])
    if vc is None or vr is None: continue
    rows.append({"vc":vc,"vr":vr,"gap":p["recover_gap"]})
    json.dump(rows,open("rl_eval_results/pref_pairs_judge_check.json","w"),indent=2)
    if (k+1)%10==0: print(f"{k+1}/{len(sample)} elapsed {int(time.time()-t0)}s",flush=True)

n=len(rows)
chosen_win=sum(1 for r in rows if r["vc"]>r["vr"])
rejected_win=sum(1 for r in rows if r["vr"]>r["vc"])
tie=sum(1 for r in rows if r["vc"]==r["vr"])
print(f"\nN={n} pairs judged by {JUDGE} (independent of reward)")
print(f"  mean validity  chosen={st.mean(r['vc'] for r in rows):.2f}  rejected={st.mean(r['vr'] for r in rows):.2f}")
print(f"  chosen better : {chosen_win}/{n} ({chosen_win/n:.0%})")
print(f"  rejected better: {rejected_win}/{n} ({rejected_win/n:.0%})")
print(f"  tie           : {tie}/{n} ({tie/n:.0%})")
print("VERDICT:", "PAIRS OK (chosen>rejected)" if chosen_win>rejected_win else "PAIRS BROKEN (reward confounded) -> switch reward to judge")
open(".pref_check_DONE","w").write("done")

"""Judge each SimPO seed vs SFT (qwen2.5, independent of reward) over 191; report mean+/-std."""
import json, os, re, time, urllib.request, statistics as st
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"; JUDGE="qwen2.5:7b"
d=json.load(open("rl_eval_results/p3_multiseed_generations.json")); inp=d["test_inputs"]; g=d["generations"]
def call(msgs):
    body={"model":JUDGE,"format":"json","stream":False,"messages":msgs,"options":{"temperature":0,"num_predict":200}}
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(2)
    return None
def lastj(t):
    if not t: return None
    for m in reversed(re.findall(r"\{[^{}]*\}",t,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None
SYS=("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
"Which gives the more logically valid justification for WHY the correct answer is correct? "
"Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
"Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")
def parse(x):
    m=re.search(r"The correct answer is Option ([A-E])",x); c=m.group(1) if m else None; a=""
    if c:
        om=re.search(rf"Option {c}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",x,re.DOTALL); a=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",x,re.DOTALL); return (qm.group(1).strip() if qm else x),a
def jp(q,a,e1,e2):
    r=lastj(call([{"role":"system","content":SYS},{"role":"user","content":f"Question:\n{q}\nCorrect answer: {a}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]))
    if not r or "better" not in r: return None
    try: return int(r["better"])
    except Exception: return None
rates=[]; res={}
for key in sorted(k for k in g if k.startswith("SimPO_s")):
    win=sft=tie=n=0
    for i in range(len(inp)):
        q,a=parse(inp[i]); swap=(i%2==1)
        e1,e2=(g["SFT"][i],g[key][i]) if swap else (g[key][i],g["SFT"][i])
        r=jp(q,a,e1,e2)
        if r is None: continue
        n+=1
        if r==0: tie+=1
        elif (r==1 and not swap) or (r==2 and swap): win+=1
        else: sft+=1
    nt=win+sft; rate=win/nt if nt else 0; rates.append(rate)
    res[key]={"n":n,"win":win,"sft":sft,"tie":tie,"nontie_winrate":round(rate,3)}
    print(f"{key}: win {win} / sft {sft} / tie {tie} -> non-tie {rate:.1%}",flush=True)
    json.dump(res,open("rl_eval_results/simpo_multiseed_judge.json","w"),indent=2)
if rates:
    print(f"\nSimPO vs SFT (qwen2.5, {len(rates)} seeds): non-tie win-rate mean={st.mean(rates):.1%} "
          f"std={ (st.pstdev(rates) if len(rates)>1 else 0):.1%}  range [{min(rates):.1%},{max(rates):.1%}]")
open(".simpo_multiseed_judge_DONE","w").write("done")

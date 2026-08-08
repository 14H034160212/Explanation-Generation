"""P3 independent-judge comparison: does preference tuning improve JUDGED reasoning
over SFT (even though it is ~3x longer)? Judges are independent of the DPO reward
(reward judge = DeepSeek-R1); here we use qwen2.5:7b and llama3:8b. Randomised order,
ignore-length instruction. Because preference-tuned outputs (~194w) are much longer
than SFT (~61w), any preference is length/format-CONFOUNDED — reported as such.
"""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"
JUDGES=["qwen2.5:7b","llama3:8b"]
COMPARISONS=[("reasoningSimPO","SFT"),("reasoningDPO_v2","SFT")]
N=int(os.environ.get("P3J_N","70"))

def call(model,messages):
    body={"model":model,"format":"json","stream":False,"messages":messages,"options":{"temperature":0,"num_predict":200}}
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(2)
    return None
def last_json(t):
    if not t: return None
    for m in reversed(re.findall(r"\{[^{}]*\}",t,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None
SYS=("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
"Which gives the more logically valid justification for WHY the correct answer is correct? "
"Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
"Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")
def jp(model,q,ans,e1,e2):
    d=last_json(call(model,[{"role":"system","content":SYS},
        {"role":"user","content":f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]))
    if not d or "better" not in d: return None
    try: return int(d["better"])
    except Exception: return None

def parse(inp):
    m=re.search(r"The correct answer is Option ([A-E])",inp); corr=m.group(1) if m else None
    ans=""
    if corr:
        om=re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",inp,re.DOTALL)
        ans=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",inp,re.DOTALL)
    return (qm.group(1).strip() if qm else inp), ans

d=json.load(open("rl_eval_results/p3_reasoning_generations.json"))
inputs=d["test_inputs"]; g=d["generations"]
out={}
for model in JUDGES:
    for (A,B) in COMPARISONS:
        if A not in g or B not in g: continue
        winA=winB=tie=0; n=0
        for i in range(min(N,len(inputs))):
            q,ans=parse(inputs[i])
            swap=(i%2==1)
            e1,e2=(g[B][i],g[A][i]) if swap else (g[A][i],g[B][i])
            r=jp(model,q,ans,e1,e2)
            if r is None: continue
            n+=1
            if r==0: tie+=1
            elif (r==1 and not swap) or (r==2 and swap): winA+=1
            else: winB+=1
        key=f"{model} | {A} vs {B}"
        out[key]={"n":n,f"{A}_win":winA,f"{B}_win":winB,"tie":tie}
        print(f"{key}: {A} wins {winA}/{n} ({winA/max(n,1):.0%}), {B} wins {winB}/{n} ({winB/max(n,1):.0%}), tie {tie}",flush=True)
        json.dump(out,open("rl_eval_results/p3_judge_compare.json","w"),indent=2)
open(".p3_judge_DONE","w").write("done")
print("SAVED rl_eval_results/p3_judge_compare.json")

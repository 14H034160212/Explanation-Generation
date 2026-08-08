"""AI-judge the exact 24 blinded human-eval pairs (SimPO vs SFT) with a strong judge
(deepseek-r1) + qwen2.5. Reconstruct the same 24 differing indices used in the sheet."""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"
d=json.load(open("rl_eval_results/p3_reasoning_generations.json")); g=d["generations"]
inputs=d["test_inputs"]; SFT=g["SFT"]; SIM=g["reasoningSimPO_fixed"]
def norm(s): return re.sub(r"\s+"," ",s.strip().lower())
diff=[i for i in range(len(inputs)) if norm(SFT[i])!=norm(SIM[i]) and len(SFT[i].split())>=5 and len(SIM[i].split())>=5]
k=min(24,len(diff)); idxs=[diff[round(j*(len(diff)-1)/(k-1))] for j in range(k)]; idxs=list(dict.fromkeys(idxs))
def call(model,msgs,np_):
    body={"model":model,"stream":False,"messages":msgs,"options":{"temperature":0,"num_predict":np_}}
    if model!="deepseek-r1:32b": body["format"]="json"
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(3)
    return None
def lastj(t):
    if not t: return None
    t=re.sub(r"<think>.*?</think>","",t,flags=re.DOTALL)
    for m in reversed(re.findall(r"\{[^{}]*\}",t,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None
SYS=("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
"Which gives the more logically valid justification for WHY the correct answer is correct? "
"Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
"Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")
def parse(inp):
    m=re.search(r"The correct answer is Option ([A-E])",inp); corr=m.group(1) if m else None; ans=""
    if corr:
        om=re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",inp,re.DOTALL); ans=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",inp,re.DOTALL); return (qm.group(1).strip() if qm else inp),ans
def jp(model,q,ans,e1,e2):
    r=lastj(call(model,[{"role":"system","content":SYS},{"role":"user","content":f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}],700 if "deepseek" in model else 200))
    if not r or "better" not in r: return None
    try: return int(r["better"])
    except Exception: return None
for model in ["deepseek-r1:32b","qwen2.5:7b"]:
    sim=sft=tie=n=0
    for i in idxs:
        q,ans=parse(inputs[i]); swap=(i%2==1)
        e1,e2=(SIM[i],SFT[i]) if swap else (SFT[i],SIM[i])   # match sheet's A/B swap
        r=jp(model,q,ans,e1,e2)
        if r is None: continue
        n+=1
        if r==0: tie+=1
        elif (r==2 and not swap) or (r==1 and swap): sim+=1
        else: sft+=1
    print(f"{model}: on {n}/{len(idxs)} sheet pairs -> SimPO {sim}, SFT {sft}, tie {tie}",flush=True)
open(".ai_sheet_DONE","w").write("done")

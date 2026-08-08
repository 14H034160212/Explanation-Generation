"""B: full 191-test independent-judge eval. qwen2.5 + llama3 on all tuned-vs-SFT;
deepseek-r1 as an extra reference on SimPO vs SFT. Randomised order, ignore length.
qwen2.5/llama3 are independent of the DPO/SimPO/OPD reward (DeepSeek). DeepSeek here
is the reward family, reported only as a cross-check reference."""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"
d=json.load(open("rl_eval_results/p3_full_generations.json"))
inputs=d["test_inputs"]; g=d["generations"]
TASKS=[("qwen2.5:7b","SimPO"),("qwen2.5:7b","OPD"),("qwen2.5:7b","DPO_v2"),
       ("llama3:8b","SimPO"),("llama3:8b","OPD"),("llama3:8b","DPO_v2"),
       ("deepseek-r1:32b","SimPO")]  # deepseek = reference (reward family)
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
    m=re.search(r"The correct answer is Option ([A-E])",inp); corr=m.group(1) if m else None
    ans=""
    if corr:
        om=re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",inp,re.DOTALL); ans=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",inp,re.DOTALL)
    return (qm.group(1).strip() if qm else inp), ans
def jp(model,q,ans,e1,e2):
    np_=700 if model=="deepseek-r1:32b" else 200
    r=lastj(call(model,[{"role":"system","content":SYS},
        {"role":"user","content":f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}],np_))
    if not r or "better" not in r: return None
    try: return int(r["better"])
    except Exception: return None
out={}
for model,A in TASKS:
    win=sft=tie=n=0
    for i in range(len(inputs)):
        q,ans=parse(inputs[i]); swap=(i%2==1)
        e1,e2=(g["SFT"][i],g[A][i]) if swap else (g[A][i],g["SFT"][i])
        r=jp(model,q,ans,e1,e2)
        if r is None: continue
        n+=1
        if r==0: tie+=1
        elif (r==1 and not swap) or (r==2 and swap): win+=1
        else: sft+=1
    out[f"{model} | {A} vs SFT"]={"n":n,"model_win":win,"SFT_win":sft,"tie":tie}
    nt=win+sft
    print(f"{model} | {A} vs SFT: model {win}/{n} ({win/max(n,1):.0%}) SFT {sft}/{n} ({sft/max(n,1):.0%}) tie {tie} | non-tie {win/max(nt,1):.0%}",flush=True)
    json.dump(out,open("rl_eval_results/p3_judge_full.json","w"),indent=2)
open(".p3judgefull_DONE","w").write("done")

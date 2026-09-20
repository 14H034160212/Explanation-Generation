"""Frontier-judge corroboration: gpt-5-mini judges SimPO vs SFT on the 191 held-out
Cardiff Tier-C test set (reuses p3_full_generations.json). Randomised order,
ignore-length rubric, same as the open judges. Economical: mini model, ~0.2M tokens."""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
KEY=open("/tmp/claude-1022/-data-qbao775-Explanation-Generation/21961b56-282e-4ae0-86d0-81befbe44722/scratchpad/.openai_key").read().strip()
URL="https://api.openai.com/v1/chat/completions"; MODEL="gpt-5-mini"
d=json.load(open("rl_eval_results/p3_full_generations.json")); inp=d["test_inputs"]; g=d["generations"]
SYS=("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
"Which gives the more logically valid justification for WHY the correct answer is correct? "
"Judge ONLY reasoning; IGNORE length/fluency/formatting; restatement with no reason is worse. "
"Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")
def parse(x):
    m=re.search(r"The correct answer is Option ([A-E])",x); c=m.group(1) if m else None; a=""
    if c:
        om=re.search(rf"Option {c}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",x,re.DOTALL); a=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",x,re.DOTALL); return (qm.group(1).strip() if qm else x),a
def call(msgs):
    body=json.dumps({"model":MODEL,"messages":msgs,"max_completion_tokens":2000}).encode()
    for _ in range(4):
        try:
            req=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r:
                j=json.loads(r.read()); return j["choices"][0]["message"]["content"], j.get("usage",{}).get("total_tokens",0)
        except Exception as e: time.sleep(4)
    return None,0
def jp(q,a,e1,e2):
    c,tok=call([{"role":"system","content":SYS},{"role":"user","content":f"Question:\n{q}\nCorrect answer: {a}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\nJSON:"}])
    if not c: return None,tok
    m=re.search(r'\{[^{}]*\}',c)
    try: return int(json.loads(m.group(0)).get("better")),tok
    except Exception: return None,tok
win=sft=tie=n=0; toks=0
A="SimPO"
for i in range(len(inp)):
    q,a=parse(inp[i]); swap=(i%2==1)
    e1,e2=(g["SFT"][i],g[A][i]) if swap else (g[A][i],g["SFT"][i])
    r,t=jp(q,a,e1,e2); toks+=t
    if r is None: continue
    n+=1
    if r==0: tie+=1
    elif (r==1 and not swap) or (r==2 and swap): win+=1
    else: sft+=1
    if (i+1)%40==0:
        print(f"{i+1}/{len(inp)}: SimPO {win} SFT {sft} tie {tie} | tokens {toks}",flush=True)
        json.dump({"model":MODEL,"A":A,"n":n,"SimPO_win":win,"SFT_win":sft,"tie":tie,"tokens":toks},open("rl_eval_results/p3_judge_gpt5mini.json","w"),indent=2)
nt=win+sft
print(f"\ngpt-5-mini | SimPO vs SFT on {n}: SimPO {win} / SFT {sft} / tie {tie} | non-tie SimPO {win/max(nt,1):.0%} | total_tokens {toks}")
json.dump({"model":MODEL,"A":A,"n":n,"SimPO_win":win,"SFT_win":sft,"tie":tie,"nontie_simpo":round(win/max(nt,1),3),"tokens":toks},open("rl_eval_results/p3_judge_gpt5mini.json","w"),indent=2)
open(".gpt5_judge_DONE","w").write("done")

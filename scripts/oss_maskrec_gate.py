"""OPEN-SOURCE masked-recoverability gate (free, fully local via ollama).

Replaces the GPT-4o probe with local open models to check whether the ONE signal
that worked (answer-masked recoverability, +0.47 vs GPT-4o validity) still holds
when both reward-solver and judge are open-weight -- i.e. whether we have a FREE,
local, in-loop reasoning reward we can retrain against.

Design (avoid circularity): different model families.
  SOLVER (recoverability reward) : qwen2.5:7b   (Qwen family, fast)
  JUDGE  (reasoning validity)    : deepseek-r1:32b (DeepSeek family, strong reasoning)

Signal per explanation E:
  mask the answer mention in E -> ask SOLVER to pick the MCQ option from
  (question + masked E) only. recovered=1 iff it picks the correct option.
  A true reasoning E lets the solver recover the answer WITHOUT being told it;
  a restatement (answer masked out) does not.
Correlate recovered with JUDGE reasoning-validity (0/1/2). Local-only.
"""
import json, re, os, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"
SOLVER="qwen2.5:7b"; JUDGE="deepseek-r1:32b"

def call(model, messages, json_format, num_predict):
    body={"model":model,"stream":False,"messages":messages,
          "options":{"temperature":0,"num_predict":num_predict}}
    if json_format: body["format"]="json"
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(3)
    return None

def last_json(text):
    if not text: return None
    # strip deepseek <think>..</think>, then grab the last {...} block
    text=re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    ms=re.findall(r"\{[^{}]*\}",text,flags=re.DOTALL)
    for m in reversed(ms):
        try: return json.loads(m)
        except Exception: continue
    return None

def parse(inp):
    opts={}
    for m in re.finditer(r"Option ([A-E]):\s*(.*?)(?=\s*Option [A-E]:|\s*The correct answer|$)", inp, re.DOTALL):
        opts[m.group(1)]=m.group(2).strip()
    cm=re.search(r"The correct answer is Option ([A-E])", inp)
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:", inp, re.DOTALL)
    return (qm.group(1).strip() if qm else inp), opts, (cm.group(1) if cm else None)

def mask_answer(E, corr, opts):
    out=E
    out=re.sub(rf"\boption\s*{corr}\b","the option",out,flags=re.I)
    out=re.sub(rf"\banswer is\s*{corr}\b","answer is [MASK]",out,flags=re.I)
    out=re.sub(rf"\b{corr}\b(?=[\).:,])","[MASK]",out)
    ot=opts.get(corr,"")
    if ot and len(ot)>3: out=re.sub(re.escape(ot),"[the option]",out,flags=re.I)
    return out

def solve(q, opts, masked_E):
    optstr="\n".join(f"{L}: {t}" for L,t in opts.items())
    msg=[{"role":"system","content":"Given a multiple-choice question and a hint (its reasoning; the answer itself is masked), choose the single best option using ONLY the hint's reasoning and the question. Reply strict JSON {\"choice\":\"A|B|C|D|E\"}."},
         {"role":"user","content":f"Question: {q}\nOptions:\n{optstr}\n\nHint:\n{masked_E}\n\nJSON:"}]
    d=last_json(call(SOLVER,msg,True,20))
    return (d or {}).get("choice","").strip().upper()[:1] if d else None

RUBRIC=("Judge whether the EXPLANATION gives a logically valid and sufficient justification "
"for why the stated correct answer is correct. IGNORE fluency/length; judge ONLY the logic. "
"Scale: 2 = valid and sufficient reasoning; 1 = partially valid / incomplete; 0 = invalid, "
"wrong, or mere restatement of the answer with no reason. "
"Think briefly, then END your reply with strict JSON {\"validity\":0|1|2}.")
def judge(q,opt,E):
    msg=[{"role":"system","content":RUBRIC},
         {"role":"user","content":f"Question:\n{q}\nStated correct answer: {opt}\nExplanation:\n{E}\n"}]
    d=last_json(call(JUDGE,msg,False,700))
    if not d or "validity" not in d: return None
    try: return int(d["validity"])
    except Exception: return None

def get_exps(data,kind):
    dr=data.get("detailed_results",{})
    items=dr.items() if isinstance(dr,dict) else [(x.get("model"),x) for x in dr] if isinstance(dr,list) else []
    for m,v in items:
        mm=m or ""
        if kind=="sft" and("SFT"in mm or "Baseline"in mm): return v.get("generated_explanations",[])
        if kind=="dpo" and("DPO"in mm or "RL-"in mm or "RLearner"in mm): return v.get("generated_explanations",[])
    return []

CELLS=[("Gemma-4 Cardiff","rl_eval_results/gemma4_cardiff_merged_eval.json"),
("LLaMA-2 Cardiff","rl_eval_results/llama2_dpo_hybrid_cardiff_eval.json"),
("Qwen3 Cardiff","rl_eval_results/qwen3_dpo_hybrid_cardiff_eval.json")]
test=json.load(open("preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",encoding="utf-8"))
N=int(os.environ.get("GATE_N","20")); rows=[]
t0=time.time()
for label,ej in CELLS:
    data=json.load(open(ej,encoding="utf-8"))
    for kind in ["sft","dpo"]:
        exps=get_exps(data,kind); done=0
        for i in range(min(N,len(exps),len(test))):
            E=exps[i]
            if not E: continue
            q,opts,corr=parse(test[i]["input"])
            if not corr or corr not in opts: continue
            choice=solve(q,opts,mask_answer(E,corr,opts)); v=judge(q,opts[corr],E)
            if choice is None or v is None: continue
            rows.append({"cell":label,"kind":kind,"recovered":1 if choice==corr else 0,"validity":v})
            done+=1
            json.dump(rows,open("rl_eval_results/oss_maskrec_gate.json","w"),indent=2)  # periodic save
        print(f"{label}/{kind}: {done}  (elapsed {int(time.time()-t0)}s)",flush=True)

def spear(xs,ys):
    n=len(xs)
    if n<3: return 0
    def rk(v):
        o=sorted(range(n),key=lambda i:v[i]); r=[0]*n
        for k,i in enumerate(o): r[i]=k
        return r
    rx,ry=rk(xs),rk(ys); mx=sum(rx)/n; my=sum(ry)/n
    cov=sum((rx[i]-mx)*(ry[i]-my) for i in range(n)); sx=(sum((r-mx)**2 for r in rx))**.5; sy=(sum((r-my)**2 for r in ry))**.5
    return cov/(sx*sy) if sx*sy else 0
rec=[x["recovered"] for x in rows]; val=[x["validity"] for x in rows]
import statistics as st
print(f"\nN={len(rows)}  (solver={SOLVER}, judge={JUDGE})")
print(f"masked-recoverability Spearman vs open-judge validity = {spear(rec,val):+.3f}   (GPT-4o gave +0.47)")
for kind in ["sft","dpo"]:
    sub=[x for x in rows if x["kind"]==kind]
    if sub: print(f"  {kind}: recover_rate={st.mean(x['recovered'] for x in sub):.3f} validity={st.mean(x['validity'] for x in sub):.2f} (n={len(sub)})")
for label,_ in CELLS:
    sub=[x for x in rows if x["cell"]==label]
    if sub: print(f"  {label}: recover_rate={st.mean(x['recovered'] for x in sub):.3f} validity={st.mean(x['validity'] for x in sub):.2f}")
json.dump(rows,open("rl_eval_results/oss_maskrec_gate.json","w"),indent=2)
open(".oss_gate_DONE","w").write("done")
print("SAVED rl_eval_results/oss_maskrec_gate.json")

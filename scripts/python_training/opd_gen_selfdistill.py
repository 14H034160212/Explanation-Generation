"""OPD prototype - Phase 1: build reward-filtered on-policy self-distillation data.

OPD-family, teacher-free (ReST/RAFT-style). The student (Gemma-4-SFT) samples K
on-policy trajectories per prompt; an independent reasoning judge (DeepSeek-R1)
scores each; we keep the highest-validity trajectory, breaking ties toward the
SHORTER one. Because targets are the student's OWN concise samples, later SFT on
them cannot inflate length (the failure mode of DPO/SimPO here). Output is a
standard SFT dataset {instruction, input, output} of self-generated concise,
judge-preferred explanations.
"""
import argparse, json, logging, os, re, sys, time, urllib.request, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/data/qbao775/Explanation-Generation")
from rl_build_preference_data_nli import load_gemma4_generator, generate_gemma4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OLLAMA="http://localhost:11434/api/chat"

def parse(inp):
    m=re.search(r"The correct answer is Option ([A-E])",inp); corr=m.group(1) if m else None
    ans=""
    if corr:
        om=re.search(rf"Option {corr}:\s*(.+?)(?:\s+Option [A-E]:|The correct answer|$)",inp,re.DOTALL)
        ans=om.group(1).strip() if om else ""
    qm=re.search(r"Given question:\s*(.*?)\s*Option A:",inp,re.DOTALL)
    return (qm.group(1).strip() if qm else inp), ans

def _ollama(model,messages,num_predict):
    body={"model":model,"stream":False,"messages":messages,"options":{"temperature":0,"num_predict":num_predict}}
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(3)
    return None
def _last_json(t):
    if not t: return None
    t=re.sub(r"<think>.*?</think>","",t,flags=re.DOTALL)
    for m in reversed(re.findall(r"\{[^{}]*\}",t,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None
RUBRIC=("Judge whether the EXPLANATION gives a logically valid and sufficient justification "
"for why the stated correct answer is correct. IGNORE fluency/length; judge ONLY the logic. "
"Scale: 2=valid and sufficient; 1=partial/incomplete; 0=invalid/wrong/mere restatement. "
"Think briefly, then END with strict JSON {\"validity\":0|1|2}.")
def judge(model,q,ans,E):
    d=_last_json(_ollama(model,[{"role":"system","content":RUBRIC},
        {"role":"user","content":f"Question:\n{q}\nStated correct answer: {ans}\nExplanation:\n{E}\n"}],700))
    if not d or "validity" not in d: return None
    try: return int(d["validity"])
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--generator_path",default="google/gemma-4-E4B-it")
    ap.add_argument("--sft_adapter",default="./rl_sft_gemma4_e4b_cardiff_tierC_generator")
    ap.add_argument("--data_path",default="./preference_data/Paul_new_data/Cardiff_tierC_generator_train.json")
    ap.add_argument("--out",default="./rl_preference_data_gemma4_cardiff_opd/selfdistill.json")
    ap.add_argument("--judge_model",default="deepseek-r1:32b")
    ap.add_argument("--num_samples",type=int,default=4)
    ap.add_argument("--max_questions",type=int,default=150)
    ap.add_argument("--max_new_tokens",type=int,default=160)   # cap to bias concise
    ap.add_argument("--max_words",type=int,default=100)         # discard verbose targets
    ap.add_argument("--min_validity",type=int,default=2)        # keep only fully-valid targets
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    import random; random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)

    model,tok=load_gemma4_generator(args.generator_path,args.sft_adapter,args.device)
    data=json.load(open(args.data_path,encoding="utf-8")); random.shuffle(data)
    data=data[:args.max_questions]
    logger.info(f"OPD self-distill | judge={args.judge_model} | {len(data)} q x {args.num_samples} | cap {args.max_new_tokens}tok/{args.max_words}w minV={args.min_validity}")

    targets=[]; kept=skip_lowV=skip_long=0
    for i,item in enumerate(data):
        instr=item.get("instruction","").strip(); inp=item.get("input","").strip()
        if not inp: continue
        q,ans=parse(inp)
        if not ans: continue
        try:
            cands=generate_gemma4(model,tok,instr,inp,args.num_samples,args.device,args.max_new_tokens)
        except Exception as e:
            logger.warning(f"Q{i} gen err {e}"); continue
        scored=[]
        for E in cands:
            if not E or len(E.split())<4: continue
            v=judge(args.judge_model,q,ans,E)
            if v is None: continue
            scored.append((v,len(E.split()),E))
        if not scored: continue
        # best validity, tie-break shorter
        scored.sort(key=lambda x:(-x[0],x[1]))
        bv,bw,bE=scored[0]
        if bv<args.min_validity: skip_lowV+=1; continue
        if bw>args.max_words: skip_long+=1; continue
        targets.append({"instruction":instr,"input":inp,"output":bE,"validity":bv,"words":bw})
        kept+=1
        if (i+1)%20==0:
            json.dump(targets,open(args.out,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
            import statistics as st
            logger.info(f"Q{i+1}/{len(data)} kept={kept} avg_words={st.mean(t['words'] for t in targets):.0f} "
                        f"| skip lowV={skip_lowV} long={skip_long}")
    json.dump(targets,open(args.out,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    import statistics as st
    logger.info(f"DONE kept={kept}/{len(data)} avg_words={st.mean(t['words'] for t in targets):.0f} -> {args.out}")
    open(".opd_gen_DONE","w").write("done")

if __name__=="__main__":
    main()

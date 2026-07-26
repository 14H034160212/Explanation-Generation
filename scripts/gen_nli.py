"""Generate explanations from a Gemma-4 (base+LoRA adapter) on a test set and
score NLI entailment with small + held-out large models. Verifier-free."""
import json, re, os, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
os.chdir("/data/qbao775/Explanation-Generation")

ap=argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="google/gemma-4-E4B-it")
a=ap.parse_args()

def opt_text(inp):
    m=re.search(r"The correct answer is Option ([A-Z])",inp)
    if not m: return ""
    L=m.group(1); om=re.search(rf"Option {L}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)",inp,re.DOTALL)
    return om.group(1).strip() if om else ""

test=json.load(open(a.test))
tok=AutoTokenizer.from_pretrained(a.base, cache_dir="cache")
base=AutoModelForCausalLM.from_pretrained(a.base, cache_dir="cache", torch_dtype=torch.bfloat16).to(a.dev).eval()
model=PeftModel.from_pretrained(base, a.adapter).to(a.dev).eval()

@torch.no_grad()
def gen(instr,inp):
    msgs=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(a.dev)
    o=model.generate(**x,max_new_tokens=300,do_sample=False,temperature=1.0,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()

exps=[gen(it["instruction"],it["input"]) for it in test]
hyps=[opt_text(it["input"]) for it in test]
del model, base; torch.cuda.empty_cache()

def nli(name):
    t=AutoTokenizer.from_pretrained(name,use_fast=False,cache_dir="cache")
    m=AutoModelForSequenceClassification.from_pretrained(name,cache_dir="cache").to(a.dev).eval()
    ei=next((k for k,v in m.config.id2label.items() if "entail" in v.lower()),1)
    sc=[]
    with torch.no_grad():
        for i in range(0,len(exps),8):
            e=exps[i:i+8]; h=hyps[i:i+8]; pr=[(x,y) for x,y in zip(e,h) if x and y]
            if not pr: sc+=[0.0]*len(e); continue
            enc=t([x for x,_ in pr],[y for _,y in pr],padding=True,truncation=True,max_length=512,return_tensors="pt").to(a.dev)
            sc+=torch.softmax(m(**enc).logits,dim=-1)[:,ei].cpu().tolist()
            if len(pr)<len(e): sc+=[0.0]*(len(e)-len(pr))
    del m; torch.cuda.empty_cache()
    return round(sum(sc)/len(sc),4)

res={"adapter":a.adapter,"nli_small":nli("cross-encoder/nli-deberta-v3-small"),
     "nli_large":nli("cross-encoder/nli-deberta-v3-large"),
     "avg_words":round(sum(len(e.split()) for e in exps)/len(exps),1),"n":len(exps)}
print(json.dumps(res))
json.dump(res, open(a.out,"w"), indent=2)

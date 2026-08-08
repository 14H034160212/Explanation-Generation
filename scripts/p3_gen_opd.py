import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"python_training"))
os.chdir("/data/qbao775/Explanation-Generation")
from rl_build_preference_data_nli import load_gemma4_generator
BASE="google/gemma-4-E4B-it"; ADAPTER="./rl_dpo_gemma4_e4b_cardiff_opd_generator"
TEST="./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json"; DEV="cuda:0"
N=int(os.environ.get("P3_N","100"))
test=json.load(open(TEST,encoding="utf-8"))[:N]
model,tok=load_gemma4_generator(BASE,ADAPTER,DEV)
@torch.no_grad()
def gen(instr,inp):
    m=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(DEV)
    o=model.generate(**x,max_new_tokens=300,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
gens=[gen(t.get("instruction","").strip(),t.get("input","").strip()) for t in test]
d=json.load(open("rl_eval_results/p3_reasoning_generations.json"))
g=d.get("generations",d); g["reasoningOPD"]=gens
json.dump({"test_inputs":[t["input"] for t in test],"generations":g},open("rl_eval_results/p3_reasoning_generations.json","w"),indent=2,ensure_ascii=False)
import statistics as st
print("v2 avg_words",st.mean(len(x.split()) for x in gens),"min",min(len(x.split()) for x in gens),"max",max(len(x.split()) for x in gens))
open(".p3_opdgen_DONE","w").write("done")

"""Re-generate DPO_v2 / SimPO / OPD with CORRECT composition base+SFT+adapter."""
import json, os, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
BASE="google/gemma-4-E4B-it"; SFT="./rl_sft_gemma4_e4b_cardiff_tierC_generator"; DEV="cuda:0"
N=int(os.environ.get("P3_N","100"))
TARGETS={"reasoningDPO_v2_fixed":"./rl_dpo_gemma4_e4b_cardiff_reasoning_v2_generator",
         "reasoningSimPO_fixed":"./rl_dpo_gemma4_e4b_cardiff_reasoning_simpo_generator",
         "reasoningOPD_fixed":"./rl_dpo_gemma4_e4b_cardiff_opd_generator"}
test=json.load(open("./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json",encoding="utf-8"))[:N]
tok=AutoTokenizer.from_pretrained(BASE,padding_side="left",trust_remote_code=True,cache_dir="cache")
if tok.pad_token is None: tok.pad_token=tok.eos_token
@torch.no_grad()
def gen(m,instr,inp):
    msg=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(DEV)
    o=m.generate(**x,max_new_tokens=300,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
import statistics as st, gc
for key,adapter in TARGETS.items():
    print("=== "+key+" ===",flush=True)
    m=Gemma4ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,trust_remote_code=True,cache_dir="cache")
    m=PeftModel.from_pretrained(m,SFT); m=m.merge_and_unload()
    m=PeftModel.from_pretrained(m,adapter); m=m.merge_and_unload(); m=m.to(DEV).eval()
    gens=[gen(m,t.get("instruction","").strip(),t.get("input","").strip()) for t in test]
    d=json.load(open("rl_eval_results/p3_reasoning_generations.json")); d["generations"][key]=gens
    json.dump(d,open("rl_eval_results/p3_reasoning_generations.json","w"),indent=2,ensure_ascii=False)
    print(f"{key}: avg_w={st.mean(len(x.split()) for x in gens):.0f} max={max(len(x.split()) for x in gens)}",flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()
open(".p3fixedall_DONE","w").write("done")

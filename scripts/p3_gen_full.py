"""B: full test-set (191) generation with CORRECT composition base(+SFT)(+adapter)."""
import json, os, gc, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
BASE="google/gemma-4-E4B-it"; SFT="./rl_sft_gemma4_e4b_cardiff_tierC_generator"; DEV="cuda:0"
test=json.load(open("./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json",encoding="utf-8"))
tok=AutoTokenizer.from_pretrained(BASE,padding_side="left",trust_remote_code=True,cache_dir="cache")
if tok.pad_token is None: tok.pad_token=tok.eos_token
CONFIG={"SFT":None,
        "SimPO":"./rl_dpo_gemma4_e4b_cardiff_reasoning_simpo_generator",
        "OPD":"./rl_dpo_gemma4_e4b_cardiff_opd_generator",
        "DPO_v2":"./rl_dpo_gemma4_e4b_cardiff_reasoning_v2_generator"}
@torch.no_grad()
def gen(m,instr,inp):
    msg=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(DEV)
    o=m.generate(**x,max_new_tokens=300,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
out={"test_inputs":[t["input"] for t in test],"generations":{}}
if os.path.exists("rl_eval_results/p3_full_generations.json"):
    out=json.load(open("rl_eval_results/p3_full_generations.json"))
import statistics as st
for key,adapter in CONFIG.items():
    if key in out["generations"] and len(out["generations"][key])==len(test): 
        print(key,"already done",flush=True); continue
    print("=== "+key+" ===",flush=True)
    m=Gemma4ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,trust_remote_code=True,cache_dir="cache")
    m=PeftModel.from_pretrained(m,SFT); m=m.merge_and_unload()
    if adapter: m=PeftModel.from_pretrained(m,adapter); m=m.merge_and_unload()
    m=m.to(DEV).eval()
    gens=[gen(m,t.get("instruction","").strip(),t.get("input","").strip()) for t in test]
    out["generations"][key]=gens
    json.dump(out,open("rl_eval_results/p3_full_generations.json","w"),indent=2,ensure_ascii=False)
    print(f"{key}: n={len(gens)} avg_w={st.mean(len(x.split()) for x in gens):.0f}",flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()
open(".p3full_gen_DONE","w").write("done")

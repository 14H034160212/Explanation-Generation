"""Corrected eval generation: compose base + SFT(merged) + target adapter.
The DPO/SimPO/OPD adapters were trained on the SFT-MERGED base, so eval must
re-merge SFT before applying them (earlier eval applied them to the bare base,
dropping SFT and letting the base markdown prior resurface)."""
import json, os, sys, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
BASE="google/gemma-4-E4B-it"; SFT="./rl_sft_gemma4_e4b_cardiff_tierC_generator"
TARGET=os.environ["TARGET_ADAPTER"]; KEY=os.environ["TARGET_KEY"]
N=int(os.environ.get("P3_N","40")); DEV="cuda:0"
test=json.load(open("./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json",encoding="utf-8"))[:N]
tok=AutoTokenizer.from_pretrained(BASE,padding_side="left",trust_remote_code=True,cache_dir="cache")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=Gemma4ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,trust_remote_code=True,cache_dir="cache")
m=PeftModel.from_pretrained(m,SFT); m=m.merge_and_unload()           # <-- SFT merged FIRST
m=PeftModel.from_pretrained(m,TARGET); m=m.merge_and_unload()        # <-- then target adapter
m=m.to(DEV).eval()
@torch.no_grad()
def gen(instr,inp):
    msg=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(DEV)
    o=m.generate(**x,max_new_tokens=300,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
gens=[gen(t.get("instruction","").strip(),t.get("input","").strip()) for t in test]
d=json.load(open("rl_eval_results/p3_reasoning_generations.json")); g=d["generations"]; g[KEY]=gens
json.dump(d,open("rl_eval_results/p3_reasoning_generations.json","w"),indent=2,ensure_ascii=False)
import statistics as st
print(f"{KEY}: n={len(gens)} avg_w={st.mean(len(x.split()) for x in gens):.0f} min={min(len(x.split()) for x in gens)} max={max(len(x.split()) for x in gens)}")
open(".p3fixed_DONE","w").write("done")

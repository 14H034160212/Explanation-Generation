import json, os, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
BASE="google/gemma-4-E4B-it"; SFT="./rl_sft_gemma4_e4b_cardiff_tierC_generator"; DEV="cuda:0"
SEED=os.environ["SEED"]; ADAPTER=f"./rl_dpo_gemma4_e4b_cardiff_simpo_seed{SEED}"
F="rl_eval_results/p3_multiseed_generations.json"
test=json.load(open("./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json",encoding="utf-8"))
tok=AutoTokenizer.from_pretrained(BASE,padding_side="left",trust_remote_code=True,cache_dir="cache")
if tok.pad_token is None: tok.pad_token=tok.eos_token
# init file: carry test_inputs + SFT gens from the full file
if os.path.exists(F): d=json.load(open(F))
else:
    full=json.load(open("rl_eval_results/p3_full_generations.json"))
    d={"test_inputs":full["test_inputs"],"generations":{"SFT":full["generations"]["SFT"]}}
m=Gemma4ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,trust_remote_code=True,cache_dir="cache")
m=PeftModel.from_pretrained(m,SFT); m=m.merge_and_unload()
m=PeftModel.from_pretrained(m,ADAPTER); m=m.merge_and_unload(); m=m.to(DEV).eval()
@torch.no_grad()
def gen(instr,inp):
    msg=[{"role":"user","content":f"{instr}\n\n{inp}"}]
    p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(p,return_tensors="pt").to(DEV)
    o=m.generate(**x,max_new_tokens=300,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:],skip_special_tokens=True).strip()
gens=[gen(t.get("instruction","").strip(),t.get("input","").strip()) for t in test]
d["generations"][f"SimPO_s{SEED}"]=gens
json.dump(d,open(F,"w"),indent=2,ensure_ascii=False)
import statistics as st
print(f"seed{SEED}: n={len(gens)} avg_w={st.mean(len(x.split()) for x in gens):.0f}",flush=True)
open(f".simpo_gen_s{SEED}_DONE","w").write("done")

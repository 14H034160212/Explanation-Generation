"""P3 generation: SFT vs reasoning-DPO on the held-out Cardiff test set.

Deterministic (greedy) generation so the comparison reflects each model's preferred
output. Loads base+adapter sequentially (freeing between) to fit one GPU. Judging is
done separately by INDEPENDENT judges (qwen2.5, llama3) not used as the DPO reward.
"""
import json, os, sys, torch, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_training"))
os.chdir("/data/qbao775/Explanation-Generation")
from rl_build_preference_data_nli import load_gemma4_generator

BASE="google/gemma-4-E4B-it"
ADAPTERS={
    "SFT": "./rl_sft_gemma4_e4b_cardiff_tierC_generator",
    "reasoningDPO": "./rl_dpo_gemma4_e4b_cardiff_reasoning_generator",
    "oldHybridDPO": "./rl_dpo_gemma4_e4b_cardiff_tierC_generator",
}
TEST="./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json"
N=int(os.environ.get("P3_N","100"))
DEV="cuda:0"

test=json.load(open(TEST,encoding="utf-8"))[:N]

@torch.no_grad()
def gen_greedy(model, tok, instruction, input_text, max_new_tokens=300):
    messages=[{"role":"user","content":f"{instruction}\n\n{input_text}"}]
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tok(prompt,return_tensors="pt").to(DEV)
    out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()

results={}
for name,adapter in ADAPTERS.items():
    if not os.path.exists(adapter):
        print(f"skip {name}: no adapter",flush=True); continue
    print(f"=== loading {name} ({adapter}) ===",flush=True)
    model,tok=load_gemma4_generator(BASE,adapter,DEV)
    gens=[]
    for i,item in enumerate(test):
        E=gen_greedy(model,tok,item.get("instruction","").strip(),item.get("input","").strip())
        gens.append(E)
        if (i+1)%25==0: print(f"  {name} {i+1}/{len(test)}",flush=True)
    results[name]=gens
    json.dump(results,open("rl_eval_results/p3_reasoning_generations.json","w"),indent=2,ensure_ascii=False)
    del model; gc.collect(); torch.cuda.empty_cache()

json.dump({"test_inputs":[t["input"] for t in test],"generations":results},
          open("rl_eval_results/p3_reasoning_generations.json","w"),indent=2,ensure_ascii=False)
print("SAVED rl_eval_results/p3_reasoning_generations.json",flush=True)
open(".p3_gen_DONE","w").write("done")

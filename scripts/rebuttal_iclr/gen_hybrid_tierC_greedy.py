"""Reviewer enkh Q3: put the original Hybrid-DPO policy into the Section 6.3
reasoning comparison.  Greedy generation on the 191 Cardiff Tier-C test
questions with exactly the decoding used for SFT / SimPO in
p3_full_generations.json (base -> +SFT merge -> +adapter merge, greedy,
max_new_tokens=300), stored under the key "HybridDPO"."""
import json, os, gc, statistics as st, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

BASE = "google/gemma-4-E4B-it"
SFT = "./rl_sft_gemma4_e4b_cardiff_tierC_generator"
ADAPTER = "./rl_dpo_gemma4_e4b_cardiff_tierC_generator"   # Hybrid-DPO, Tier-C (Table 7 row)
DEV = "cuda:0"
OUT = "rl_eval_results/p3_full_generations.json"

test = json.load(open("./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json", encoding="utf-8"))
out = json.load(open(OUT))
assert out["test_inputs"] == [t["input"] for t in test], "test set mismatch with p3_full_generations"
tok = AutoTokenizer.from_pretrained(BASE, padding_side="left", trust_remote_code=True, cache_dir="cache")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


@torch.no_grad()
def gen(m, instr, inp):
    msg = [{"role": "user", "content": f"{instr}\n\n{inp}"}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    x = tok(p, return_tensors="pt").to(DEV)
    o = m.generate(**x, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


m = Gemma4ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir="cache")
m = PeftModel.from_pretrained(m, SFT).merge_and_unload()
m = PeftModel.from_pretrained(m, ADAPTER).merge_and_unload().to(DEV).eval()
gens = []
for i, t in enumerate(test):
    gens.append(gen(m, t.get("instruction", "").strip(), t.get("input", "").strip()))
    if i % 20 == 0:
        print(i, len(gens[-1].split()), flush=True)
out = json.load(open(OUT))  # re-read: never clobber keys written meanwhile
out["generations"]["HybridDPO"] = gens
json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
print(f"HybridDPO: avg_w={st.mean(len(x.split()) for x in gens):.0f} "
      f"SFT avg_w={st.mean(len(x.split()) for x in out['generations']['SFT']):.0f}", flush=True)
open(".rebuttal_hybrid_gen_DONE", "w").write("done")

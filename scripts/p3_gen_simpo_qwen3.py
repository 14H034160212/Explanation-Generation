"""Qwen3 held-out generation for the reasoning-reward study (SFT vs SimPO).

Eval-composition rule (critical): the SimPO LoRA was trained on the SFT-merged base,
so it MUST be evaluated as  base -> merge SFT -> apply SimPO adapter.  We load the
base once, merge the SFT LoRA to get the SFT policy (generate "SFT"), then stack the
SimPO adapter on that same merged model (generate "SimPO").  Both are decoded greedily
at the same max_new_tokens so any win is not a length artefact.

Writes rl_eval_results/p3_full_generations_qwen3.json  with schema matching the judge
scripts:  {"test_inputs": [...], "generations": {"SFT": [...], "SimPO": [...]}}.
"""
import json, os, sys, statistics as st, torch
os.chdir("/data/qbao775/Explanation-Generation")
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = os.environ.get("P3_BASE", "/data/shared/qwen3/Qwen3-8B")
SFT = os.environ.get("P3_SFT", "./rl_sft_qwen3_8b_cardiff_tierC_generator")
SIMPO = os.environ.get("P3_SIMPO", "./rl_dpo_qwen3_8b_cardiff_reasoning_simpo_generator")
TEST = os.environ.get("P3_TEST", "./preference_data/Paul_new_data/Cardiff_tierC_generator_test.json")
OUT = os.environ.get("P3_OUT", "rl_eval_results/p3_full_generations_qwen3.json")
DEV = os.environ.get("P3_DEV", "cuda:0")

test = json.load(open(TEST, encoding="utf-8"))
N = int(os.environ.get("P3_N", str(len(test))))
test = test[:N]
print(f"[qwen3 gen] {len(test)} test questions | base={BASE}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE, padding_side="left", trust_remote_code=True, cache_dir="cache")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[qwen3 gen] loading base + merging SFT ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, cache_dir="cache")
model = PeftModel.from_pretrained(model, SFT)
model = model.merge_and_unload()   # -> SFT policy
model.config.use_cache = True
model.to(DEV).eval()


@torch.no_grad()
def gen(instr, inp):
    m = [{"role": "user", "content": f"{instr}\n\n{inp}"}]
    p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    x = tok(p, return_tensors="pt").to(DEV)
    o = model.generate(**x, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_all():
    return [gen(t.get("instruction", "").strip(), t.get("input", "").strip()) for t in test]


print("[qwen3 gen] generating SFT ...", flush=True)
sft_gen = run_all()
print("  SFT avg_words", round(st.mean(len(x.split()) for x in sft_gen), 1), flush=True)

print("[qwen3 gen] stacking SimPO adapter on the SFT-merged model ...", flush=True)
model = PeftModel.from_pretrained(model, SIMPO)   # SFT-merged base + SimPO LoRA
model.to(DEV).eval()
print("[qwen3 gen] generating SimPO ...", flush=True)
simpo_gen = run_all()
print("  SimPO avg_words", round(st.mean(len(x.split()) for x in simpo_gen), 1), flush=True)

os.makedirs("rl_eval_results", exist_ok=True)
json.dump({"test_inputs": [t["input"] for t in test],
           "generations": {"SFT": sft_gen, "SimPO": simpo_gen}},
          open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[qwen3 gen] wrote {OUT}", flush=True)
open(".p3_simpogen_qwen3_DONE", "w").write("done")

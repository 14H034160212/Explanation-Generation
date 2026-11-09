"""Pool the g4shared eval generations (5 test sets x 100 items) into the
{"test_inputs", "generations"} format judge_hf.py reads. Item order is identical
across systems (greedy decoding, full rows only), so judging pairs by position is exact."""
import json, os
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
TEST = {"cardiff": "preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",
        "sydney": "preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json",
        "law": "preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json",
        "medy1": "preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json",
        "medy2": "preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"}
SYSTEMS = {"SFT": ("sft", "Gemma4-E4B-SFT"), "NLI": ("nli", "Gemma4-E4B-DPO"),
           "MaskNLI": ("masknli", "Gemma4-E4B-DPO"), "NLI93": ("nli_n93", "Gemma4-E4B-DPO"),
           "HM": ("hm", "Gemma4-E4B-DPO"),
           "JudgeSimPO": ("judge_simpo", "Gemma4-E4B-DPO"), "JudgeNLISimPO": ("judgenli_simpo", "Gemma4-E4B-DPO"),
           "NLISimPO": ("nli_simpo", "Gemma4-E4B-DPO"), "JudgeLM": ("judge_lm", "Gemma4-E4B-DPO"), "JudgeLMs2": ("judge_lm_s2", "Gemma4-E4B-DPO"), "JudgeLMs3": ("judge_lm_s3", "Gemma4-E4B-DPO"), "JudgeNLILM": ("judgenli_lm", "Gemma4-E4B-DPO"), "NLIe1": ("nli_e1", "Gemma4-E4B-DPO"),
           "NLIe2": ("nli_e2", "Gemma4-E4B-DPO"), "NLIb05": ("nli_b05", "Gemma4-E4B-DPO")}
inputs, gens, doms = [], {k: [] for k in SYSTEMS}, []
for dom, tp in TEST.items():
    items = json.load(open(tp))
    rows = {}
    for k, (tag, name) in SYSTEMS.items():
        f = f"rl_eval_results/g4shared_{tag}_{dom}_eval.json"
        if not os.path.exists(f):
            continue  # system not run (yet): judge_hf only uses systems named in --pairs
        r = json.load(open(f))["detailed_results"][name]["generated_explanations"]
        assert len(r) == len(items), (f, len(r))
        rows[k] = r
    inputs += [it["input"].strip() for it in items]; doms += [dom] * len(items)
    for k in SYSTEMS:
        gens[k] += rows.get(k, [None] * len(items))
json.dump({"test_inputs": inputs, "domains": doms, "generations": gens},
          open("rl_eval_results/g4shared_judge_generations.json", "w"), ensure_ascii=False)
gens = {k: v for k, v in gens.items() if None not in v}
json.dump({"test_inputs": inputs, "domains": doms, "generations": gens},
          open("rl_eval_results/g4shared_judge_generations.json", "w"), ensure_ascii=False)
for k, v in gens.items():
    print(k, len(v), "mean words", round(sum(len(x.split()) for x in v) / len(v), 1))

"""Add reproduced-verifier scores ("verifier_repro_scores") to every g4shared eval file.
Evals ran with --verifier_path none, so this is the only verifier column for them.
Rows must cover the whole test file (same check as g4shared_audit.py)."""
import json, glob, os, re, sys
ROOT = "/data/qbao775/Explanation-Generation"; os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/python_training")
import rl_build_preference_data_nli as B  # noqa: E402
TEST = {"cardiff": "preference_data/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json",
        "sydney": "preference_data/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json",
        "law": "preference_data/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json",
        "medy1": "preference_data/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json",
        "medy2": "preference_data/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"}
vm, vt = B.load_verifier("models/verifier_way2_repro", "cuda:0")
for f in sorted(glob.glob("rl_eval_results/g4shared_*_eval.json")):
    dom = re.search(r"_(cardiff|sydney|law|medy1|medy2)_eval\.json$", f).group(1)
    qs = [it["input"].strip() for it in json.load(open(TEST[dom]))]
    d = json.load(open(f)); changed = False
    for name, r in d["detailed_results"].items():
        if "verifier_repro_scores" in r or len(r["generated_explanations"]) != len(qs):
            continue
        r["verifier_repro_scores"] = [B.get_verifier_score(vm, vt, q, e) for q, e in zip(qs, r["generated_explanations"])]
        changed = True
    if changed:
        json.dump(d, open(f, "w"), ensure_ascii=False, indent=1); print("rescored", f, flush=True)

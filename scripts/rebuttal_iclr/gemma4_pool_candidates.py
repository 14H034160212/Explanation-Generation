"""Shared candidate pool for the Gemma 4 single- vs dual-signal ablation.

Same 500 prompts as the merged pool behind Table 1 (seed-42 shuffle of the
13,211-example SFT corpus; verified to contain all 164 original pool
questions), same generator (Gemma 4 E4B + merged SFT), same sampling
(n=3, T=0.7, top-p 0.9).  Every candidate is scored ONCE with every signal,
so each reward variant is built from identical candidates:

  phase gen : explanations + NLI (deberta-v3-small) + answer-masked NLI + ACR + words
  phase ver : Alpaca-7B verifier score (separate pass; fits beside other jobs)

Output: rl_preference_data_gemma4_shared/candidates.json (resumable).
"""
import argparse, json, os, random, re, sys
ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts/python_training")
import torch  # noqa: E402
import rl_build_preference_data_nli as B  # noqa: E402

DATA = "preference_data/Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json"
OUT = "rl_preference_data_gemma4_shared/candidates.json"
SFT = "./rl_sft_gemma4_e4b_merged_generator"
# The original Alpaca-7B verifier is lost; default to the reproduction (train_verifier_repro.py).
VERIFIER = os.environ.get("VERIFIER", "./models/verifier_way2_repro")


def mask_terms(exp, opt):
    """Identical to scripts/rebuttal_iclr/main15_audit.mask_terms."""
    o = opt.strip().rstrip(".").strip()
    out = re.sub(re.escape(o), "[MASK]", exp, flags=re.IGNORECASE) if o else exp
    for t in sorted(set(re.findall(r"\b\w{4,}\b", opt.lower())), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(t)}\w*", "[MASK]", out, flags=re.IGNORECASE)
    return out


def phase_gen(args):
    raw = json.load(open(DATA, encoding="utf-8"))
    random.seed(42); random.shuffle(raw)
    qs = raw[:500]
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    have = {d["qid"] for d in done}
    torch.manual_seed(args.seed)
    nli_m, nli_t, ent = B.load_nli_model("cross-encoder/nli-deberta-v3-small", "cpu", "cache")
    gen_m, gen_t = B.load_gemma4_generator("google/gemma-4-E4B-it", SFT, "cuda:0")
    for qid, item in enumerate(qs):
        if qid in have:
            continue
        instr, inp = item.get("instruction", "").strip(), item.get("input", "").strip()
        _, opt = B.extract_correct_option_text(inp)
        if not inp or not opt:
            continue
        exps = B.generate_gemma4(gen_m, gen_t, instr, inp, 3, "cuda:0", 300)
        nli = B.score_by_nli(nli_m, nli_t, exps, opt, ent, "cpu")
        mnli = B.score_by_nli(nli_m, nli_t, [mask_terms(e, opt) for e in exps], opt, ent, "cpu")
        done.append({"qid": qid, "instruction": instr, "question_input": inp, "correct_option_text": opt,
                     "prompt": B.make_gemma4_dpo_prompt(gen_t, instr, inp),
                     "cands": [{"text": e, "nli": n, "masked_nli": m, "acr": B.answer_coverage_rate(e, opt),
                                "words": len(e.split())} for e, n, m in zip(exps, nli, mnli)]})
        if len(done) % 10 == 0:
            json.dump(done, open(OUT, "w"), ensure_ascii=False)
            print(f"gen {len(done)}/500", flush=True)
    json.dump(done, open(OUT, "w"), ensure_ascii=False)
    print("gen done", len(done), flush=True)


def phase_ver(args):
    data = json.load(open(OUT))
    vm, vt = B.load_verifier(VERIFIER, "cuda:0")
    for k, q in enumerate(data):
        for c in q["cands"]:
            if "verifier" not in c:
                c["verifier"] = B.get_verifier_score(vm, vt, q["question_input"], c["text"])
        if k % 25 == 0:
            json.dump(data, open(OUT, "w"), ensure_ascii=False); print(f"ver {k}/{len(data)}", flush=True)
    json.dump(data, open(OUT, "w"), ensure_ascii=False)
    print("ver done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "ver"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    phase_gen(a) if a.phase == "gen" else phase_ver(a)

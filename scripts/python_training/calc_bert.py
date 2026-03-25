import json
from bert_score import score as bert_score_fn
import numpy as np
import torch

def main():
    path = "/data/qbao775/Explanation-Generation/rl_eval_results/med_y1_high_bleu.json"
    with open(path, "r") as f:
        data = json.load(f)
    
    gen_exps = [item["Generated_Explanation"] for item in data]
    ref_stus = [item["Explanation"] for item in data]
    
    # BERT(Stu)
    print("Calculating BERT(Stu)...")
    P, R, F1_stu = bert_score_fn(gen_exps, ref_stus, lang="en", verbose=False, device="cuda:0")
    
    # BERT(Ans) - need to extract correct option text
    import re
    def extract_opt(inp):
        m = re.search(r"The correct answer is Option ([A-Z])", inp)
        if not m: return ""
        letter = m.group(1)
        pat = rf"Option {letter}:\s*(.+?)(?:\s+Option [A-Z]:|The correct answer|$)"
        mm = re.search(pat, inp, re.DOTALL)
        return mm.group(1).strip() if mm else ""

    ref_ans = [extract_opt(item["input"]) for item in data]
    print("Calculating BERT(Ans)...")
    P, R, F1_ans = bert_score_fn(gen_exps, ref_ans, lang="en", verbose=False, device="cuda:0")
    
    avg_stu = F1_stu.mean().item()
    avg_ans = F1_ans.mean().item()
    
    print(f"BERT(Stu): {avg_stu:.4f}")
    print(f"BERT(Ans): {avg_ans:.4f}")
    
    # Update JSON with BERTScores
    for i, item in enumerate(data):
        item["metrics"]["bert_score"] = F1_stu[i].item()
        item["metrics"]["bert_ans"] = F1_ans[i].item()
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()

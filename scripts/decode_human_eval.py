"""Decode annotator submissions (blinded A/B/Tie) into SimPO-vs-SFT counts using the key."""
import json, csv, os
SCRATCH="/tmp/claude-1022/-data-qbao775-Explanation-Generation/21961b56-282e-4ae0-86d0-81befbe44722/scratchpad"
import sys
if "--pages" in sys.argv:
    import json as _json
    pf=sys.argv[sys.argv.index("--pages")+1]
    _subs=_json.load(open(pf,encoding="utf-8"))
    open("/tmp/_pages_subs.jsonl","w",encoding="utf-8").write("\n".join(_json.dumps(x,ensure_ascii=False) for x in _subs))
SUB=os.path.join(SCRATCH,"eval_submissions.jsonl")
if "--pages" in sys.argv: SUB="/tmp/_pages_subs.jsonl"
KEY="rl_eval_results/human_eval_KEY.csv"
os.chdir("/data/qbao775/Explanation-Generation")
# key: q -> which letter is SimPO ('A' or 'B')
simpo_letter={}
for row in csv.DictReader(open(KEY,encoding="utf-8")):
    q=int(row["Q"]); simpo_letter[q]="A" if row["mapping"].startswith("A=SimPO") else "B"
if not os.path.exists(SUB) or os.path.getsize(SUB)==0:
    print("No submissions yet at",SUB); raise SystemExit
subs=[json.loads(l) for l in open(SUB,encoding="utf-8") if l.strip()]
print(f"{len(subs)} submission(s)\n")
agg_s=agg_f=agg_t=0
for s in subs:
    simpo=sft=tie=0
    for q,let in simpo_letter.items():
        a=s["answers"].get(f"q{q}")
        if a is None: continue
        if a=="Tie": tie+=1
        elif a==let: simpo+=1
        else: sft+=1
    n=simpo+sft+tie
    nt=simpo+sft
    print(f"[{s.get('who') or 'anon'}] SimPO {simpo} / SFT {sft} / Tie {tie}  (n={n}, non-tie SimPO {simpo/nt:.0%})" if nt else f"[{s.get('who')}] all ties")
    agg_s+=simpo; agg_f+=sft; agg_t+=tie
NT=agg_s+agg_f
print(f"\nAGGREGATE: SimPO {agg_s} / SFT {agg_f} / Tie {agg_t}  |  non-tie SimPO preference {agg_s/NT:.1%}" if NT else "all ties")

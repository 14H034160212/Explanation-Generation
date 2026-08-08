"""Independent cross-judge validation of the length-matched pairwise-judge pairs.

Reward judge was DeepSeek-R1; validating with it would be circular. Use llama3:8b
(a DIFFERENT family from both the Gemma-4 generator and the DeepSeek reward judge)
as an INDEPENDENT pairwise judge on the LENGTH-MATCHED subset. Randomised order to
blunt position bias. If llama3 still prefers `chosen` well above chance on
length-matched pairs, the preference captures reasoning quality, not length or one
judge's idiosyncrasy.
"""
import json, os, re, time, urllib.request
os.chdir("/data/qbao775/Explanation-Generation")
OLLAMA="http://localhost:11434/api/chat"; JUDGE="llama3:8b"

def call(messages, num_predict=200):
    body={"model":JUDGE,"format":"json","stream":False,"messages":messages,"options":{"temperature":0,"num_predict":num_predict}}
    data=json.dumps(body).encode()
    for _ in range(3):
        try:
            req=urllib.request.Request(OLLAMA,data=data,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())["message"]["content"]
        except Exception: time.sleep(2)
    return None

def last_json(t):
    if not t: return None
    for m in reversed(re.findall(r"\{[^{}]*\}",t,flags=re.DOTALL)):
        try: return json.loads(m)
        except Exception: continue
    return None

SYS=("Compare two EXPLANATIONS for a multiple-choice question whose correct answer is given. "
"Which gives the more logically valid justification for WHY the correct answer is correct? "
"Judge ONLY the reasoning; IGNORE length/fluency; restatement with no reason is worse. "
"Reply strict JSON {\"better\":1|2|0} (1=first,2=second,0=equal).")
def judge_pair(q,ans,e1,e2):
    d=last_json(call([{"role":"system","content":SYS},
        {"role":"user","content":f"Question:\n{q}\nCorrect answer: {ans}\n\nExplanation 1:\n{e1}\n\nExplanation 2:\n{e2}\n"}]))
    if not d or "better" not in d: return None
    try: return int(d["better"])
    except Exception: return None

d=json.load(open("rl_preference_data_gemma4_cardiff_judge/preference_pairs_lenmatched.json"))
q_first=q_second=tie=agree=disagree=0; n=0
for i,p in enumerate(d):
    q=p["question_input"]; ans=p.get("correct_option_text","")
    swap=(i%2==1)  # randomise which side chosen is on
    e1,e2=(p["rejected"],p["chosen"]) if swap else (p["chosen"],p["rejected"])
    res=judge_pair(q,ans,e1,e2)
    if res is None: continue
    n+=1
    # map back to chosen/rejected
    if res==0: tie+=1
    elif (res==1 and not swap) or (res==2 and swap): agree+=1   # chosen preferred
    else: disagree+=1
    if (i+1)%20==0: print(f"{i+1}/{len(d)} agree={agree} disagree={disagree} tie={tie}",flush=True)

print(f"\nIndependent judge {JUDGE} on {n} LENGTH-MATCHED pairs:")
print(f"  prefers CHOSEN (agrees w/ DeepSeek): {agree}/{n} ({agree/n:.0%})")
print(f"  prefers REJECTED (disagrees)       : {disagree}/{n} ({disagree/n:.0%})")
print(f"  tie                                : {tie}/{n} ({tie/n:.0%})")
non_tie=agree+disagree
if non_tie: print(f"  among non-ties, chosen-preference = {agree/non_tie:.0%} (>50% = real signal)")
open(".llama3_check_DONE","w").write("done")

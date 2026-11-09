"""Write the LaTeX tables / figure of the revised ICLR paper from the audit JSONs.

Every number in paper_draft_arxiv/tables/*.tex comes from:
  rl_eval_results/rebuttal_main15_audit.json        (Table 1, Fig. 1, App. tables)
  rl_eval_results/rebuttal_single_vs_dual_audit.json (single- vs dual-signal table)
  rl_eval_results/rebuttal_reasoning_judge_hf.json   (reasoning judges incl. Hybrid-DPO)
  rl_eval_results/rebuttal_tierC_policy_nli.json     (Tier-C policies on NLI metrics)
  the eval JSONs named in the audit (BLEU/BERTScore/ACR/verifier, same run)
Run:  python3 scripts/rebuttal_iclr/gen_tables.py
"""
import json, os
import numpy as np
from math import comb

ROOT = "/data/qbao775/Explanation-Generation"
OUT = f"{ROOT}/paper_draft_arxiv/tables"
os.makedirs(OUT, exist_ok=True)
R = f"{ROOT}/rl_eval_results"
main = json.load(open(f"{R}/rebuttal_main15_audit.json"))["cells"]
sd = json.load(open(f"{R}/rebuttal_single_vs_dual_audit.json"))["cells"]
judge = json.load(open(f"{R}/rebuttal_reasoning_judge_hf.json"))
tierc = json.load(open(f"{R}/rebuttal_tierC_policy_nli.json"))
DOM = {"Cardiff": "Cardiff Bio.", "Sydney": "Sydney Bio.", "Law": "Auckland Law", "MedY1": "UK Med Y1", "MedY2": "UK Med Y2"}
ARCH = {"LLaMA-2": "LLaMA-2-13B", "Qwen3": "Qwen3-8B", "Gemma4": "Gemma~4 E4B"}


def f3(x):
    return f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"$-${abs(x):.3f}".replace("0.", ".", 1)


def arrow(a, b, bold=False):
    s = f"{f3(a)}$\\to${f3(b)}"
    return f"\\textbf{{{s}}}" if bold else s


def summary_row(data, name):
    return next(r for r in data["results"] if r.get("model") == name)


# ---------------- Table 1 ----------------
rows = []
prev = None
for c in main:
    s, L, M = c["small"], c["large"], c["large_maskterms"]
    lo, hi = s["ci95"]
    if prev and prev != c["arch"]:
        rows.append("\\midrule")
    prev = c["arch"]
    rows.append(" & ".join([
        ARCH[c["arch"]], DOM[c["domain"]],
        arrow(s["sft"], s["rl"], lo > 0),
        f"[{lo:+.2f}, {hi:+.2f}]",
        arrow(L["sft"], L["rl"], L["rl"] > L["sft"]),
        arrow(M["sft"], M["rl"], M["rl"] > M["sft"]),
        f"{c['sft_words']:.0f}$\\to${c['rl_words']:.0f}",
        f"{100*c['sft_contains']:.0f}$\\to${100*c['rl_contains']:.0f}",
    ]) + " \\\\")
open(f"{OUT}/main_table.tex", "w").write(r"""\begin{table}[t]
\centering
\caption{SFT$\to$RLearner-LLM on the five test sets. Each pair of numbers comes from \emph{one} evaluation run on the same 100 questions (LLaMA-2 decodes by sampling at $T{=}0.5$, so its SFT score varies between runs; Qwen3 and Gemma~4 decode greedily). \emph{NLI}: training scorer (DeBERTa-v3-small), bold if the 95\% bootstrap CI of the change excludes zero. \emph{Held-out}: DeBERTa-v3-large, used nowhere in training. \emph{Masked}: held-out NLI after every key term of the correct option is replaced by \texttt{[MASK]} in the explanation (\S\ref{sec:masking}); bold marks an increase. \emph{Copies}: \% of explanations containing the correct-option text verbatim.}
\label{tab:main}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
Model & Test set & NLI & 95\% CI of $\Delta$ & Held-out NLI & Masked held-out NLI & Words & Copies (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- Figure 1: held-out delta vs masked held-out delta ----------------
mk = {"LLaMA-2": ("blue", "*"), "Qwen3": ("red", "square*"), "Gemma4": ("green!50!black", "triangle*")}
plots = []
for a, (col, m) in mk.items():
    pts = " ".join(f"({c['large']['diff']:.3f},{c['large_maskterms']['diff']:.3f})" for c in main if c["arch"] == a)
    plots.append(f"\\addplot[only marks,mark={m},mark size=2.4pt,color={col}] coordinates {{{pts}}};\n\\addlegendentry{{{ARCH[a]}}}")
open(f"{OUT}/fig_masking.tex", "w").write(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.62\linewidth,height=0.42\linewidth,
  xlabel={$\Delta$ held-out NLI (DPO $-$ SFT)}, ylabel={$\Delta$ held-out NLI, answer masked},
  xmin=-0.2,xmax=0.32,ymin=-0.18,ymax=0.18, grid=major, grid style={gray!20},
  legend pos=outer north east, legend style={font=\scriptsize}, tick label style={font=\scriptsize}, label style={font=\small}]
\addplot[black!50,dashed,forget plot] coordinates {(-0.2,0) (0.32,0)};
\addplot[black!50,dashed,forget plot] coordinates {(0,-0.18) (0,0.18)};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
\caption{Where the entailment gain comes from. Each point is one (model, test set) cell of Table~\ref{tab:main}. $x$: change in held-out NLI after DPO. $y$: the same change after the correct answer's key terms are masked in the explanation. LLaMA-2 cells (blue) gain on $x$ but fall below zero on $y$: the gain is restating the answer. Gemma~4 cells (green) stay above zero on $y$ while their outputs get shorter. Qwen3 (red) rises on $y$ but its outputs triple in length (Table~\ref{tab:main}).}
\label{fig:masking}
\end{figure}
""")

# ---------------- Appendix: full metrics, same run ----------------
rows = []
prev = None
for c in main:
    d = json.load(open(f"{ROOT}/{c['file']}"))
    names = [r["model"] for r in d["results"]]
    sft, rl = d["results"][0], d["results"][-1]
    if prev and prev != c["arch"]:
        rows.append("\\midrule")
    prev = c["arch"]
    for tag, r, n in (("SFT", sft, c["sft_n"]), ("RLearner", rl, c["rl_n"])):
        rows.append(" & ".join([ARCH[c["arch"]] if tag == "SFT" else "", DOM[c["domain"]] if tag == "SFT" else "", tag,
                                f"{r['avg_bleu']:.4f}", f"{r['avg_bert_score_f1']:.4f}", f"{r['avg_bert_score_f1_answer_anchored']:.4f}",
                                f"{r['avg_answer_coverage_rate']:.4f}", f"{r['avg_nli_entailment']:.4f}", f"{r['avg_verifier_score']:.2f}", str(n)]) + " \\\\")
open(f"{OUT}/full_metrics.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{All metrics for the cells of Table~\ref{tab:main}, SFT and RLearner-LLM from the same evaluation run (file names in Table~\ref{tab:provenance}). $n$ is the number of non-empty generations out of 100.}
\label{tab:full_metrics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllccccccc}
\toprule
Model & Test set & Policy & BLEU & BERT(Stu) & BERT(Ans) & ACR & NLI & Ver & $n$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- Appendix: single vs dual signal ----------------
LAB = {"L2-hybrid-xd": "Hybrid ($H_A$), cross-domain", "L2-nli-xd": "NLI only, cross-domain",
       "L2-verif-v1": "Verifier only (DPO v1)", "L2-verif-v2": "Verifier only (DPO v2)",
       "Q3-hybrid-xd": "Hybrid ($H_A$), cross-domain", "Q3-nli-xd": "NLI only, cross-domain",
       "Q3-hybrid-sd": "Hybrid ($H_A$), single-domain", "Q3-nli-sd": "NLI only, single-domain",
       "Q3-verif": "Verifier only"}
rows = []
prev = None
for c in sorted(sd, key=lambda c: (c["arch"] != "LLaMA-2", list(DOM).index(c["domain"]))):
    key = (c["arch"], c["domain"])
    if prev and prev != key:
        rows.append("\\midrule")
    prev = key
    d = json.load(open(f"{ROOT}/{c['file']}"))
    rl_name = "RL-DPO-v2 (K=1)" if "v2" in c["label"] else d["results"][1]["model"] if c["arch"] == "LLaMA-2" else "Qwen3-8B-DPO"
    if c["label"] == "L2-verif-v1":
        rl_name = "RL-DPO (K=1)"
    ver = summary_row(d, rl_name)["avg_verifier_score"]
    s, L, M = c["small"], c["large"], c["large_maskterms"]
    rows.append(" & ".join([ARCH[c["arch"]], DOM[c["domain"]], LAB[c["label"]], arrow(s["sft"], s["rl"]), f"{s['diff']:+.3f}",
                            arrow(L["sft"], L["rl"]), arrow(M["sft"], M["rl"]), f"{c['rl_words']:.0f}", f"{ver:.2f}"]) + " \\\\")
open(f"{OUT}/single_dual.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Single- vs dual-signal preference data, same pipeline and SFT checkpoints. Each row pairs a DPO policy with the SFT row of its own evaluation run (LLaMA-2 decodes by sampling, so its SFT score differs between runs; Qwen3 SFT outputs are shared). The hybrid reward does not separate from NLI-only or verifier-only training on any metric, and NLI-only policies show no degeneration in length or verifier score.}
\label{tab:single_dual}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllcccccc}
\toprule
Model & Test set & Preference signal & NLI & $\Delta$NLI & Held-out NLI & Masked held-out & Words & Ver \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


# ---------------- reasoning judges incl Hybrid ----------------
def sign_p(a, b):
    n = a + b; k = min(a, b)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def pstr(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = f"{p:.0e}"; m, x = e.split("e")
    return f"{m}{{\\times}}10^{{{int(x)}}}"


cells = {}
for k, v in judge.items():
    j, comp = [s.strip() for s in k.split("|")]
    cells[(j, comp)] = f"{v['A_win']}/{v['B_win']}/{v['tie']} ({100*v['A_nontie_rate']:.0f}\\%, $p{{=}}{pstr(sign_p(v['A_win'], v['B_win']))}$)"
JN = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct"}
rows = [" & ".join([JN[j]] + [cells[(j, comp)] for comp in ("SimPO vs SFT", "HybridDPO vs SFT", "SimPO vs HybridDPO")]) + " \\\\" for j in JN]
pol = [("SFT", "SFT"), ("HybridDPO", "Hybrid-DPO (NLI reward)"), ("SimPO", "SimPO (reasoning reward)"), ("DPO_v2", "DPO (reasoning reward)"), ("OPD", "Self-distillation (reasoning reward)")]
rows2 = [" & ".join([lab, f"{tierc[k]['words']:.0f}", f"{100*tierc[k]['contains']:.0f}", f"{tierc[k]['nli_small']:.3f}", f"{tierc[k]['nli_large']:.3f}", f"{tierc[k]['masked_large']:.3f}"]) + " \\\\" for k, lab in pol]
open(f"{OUT}/reasoning_vs_hybrid.tex", "w").write(r"""\begin{table}[t]
\centering
\caption{Reasoning reward vs.\ NLI reward on the 191 Cardiff Tier-C test questions (Gemma~4, same SFT checkpoint, greedy decoding). \textbf{Top:} pairwise reasoning judgements (first policy wins / second wins / ties; \% of non-tied pairs; two-sided sign test), same prompt and order flip as Table~\ref{tab:reasoning_judges}, judges run locally in bf16. \textbf{Bottom:} the same outputs on the entailment metrics.}
\label{tab:reasoning_vs_hybrid}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Judge & SimPO vs SFT & Hybrid-DPO vs SFT & SimPO vs Hybrid-DPO \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}

\vspace{4pt}
\resizebox{0.86\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Policy & Words & Copies (\%) & NLI (train) & NLI (held-out) & Masked held-out NLI \\
\midrule
""" + "\n".join(rows2) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------- provenance ----------------
SFTCK = {"LLaMA-2": "rl\\_sft\\_llama2\\_13b\\_generator", "Qwen3": "rl\\_sft\\_qwen3\\_8b\\_generator", "Gemma4": "rl\\_sft\\_gemma4\\_e4b\\_merged\\_generator"}


def pool(c):
    if c["arch"] == "LLaMA-2":
        return "hybrid\\_cross\\_domain\\_llama2", "$H_A$", 239, "rl\\_dpo\\_hybrid\\_cross\\_domain\\_llama2"
    if c["arch"] == "Gemma4":
        return "gemma4\\_merged", "$H_M$", 164, "rl\\_dpo\\_gemma4\\_e4b\\_merged"
    if c["domain"] in ("Cardiff", "Sydney"):
        return "n5\\_qwen3", "$H_M$", 143, "rl\\_dpo\\_n5\\_qwen3\\_8b"
    return "hybrid\\_cross\\_domain\\_qwen3", "$H_A$", 632, "rl\\_dpo\\_hybrid\\_cross\\_domain\\_qwen3"


rows = []
for c in main:
    p, v, n, ad = pool(c)
    rows.append(" & ".join([ARCH[c["arch"]], DOM[c["domain"]], SFTCK[c["arch"]], p, v, str(n), ad,
                            "\\texttt{" + os.path.basename(c["file"]).replace("_", "\\_") + "}"]) + " \\\\")
open(f"{OUT}/provenance.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Provenance of every cell in Tables~\ref{tab:main} and~\ref{tab:full_metrics}: SFT checkpoint, preference pool (\texttt{rl\_preference\_data\_*}), reward variant, number of pairs, DPO adapter (\texttt{*\_generator}), and the evaluation file in \texttt{rl\_eval\_results/} from which \emph{both} the SFT and the RLearner numbers are read. Test sets are the fixed 100-question splits of \S\ref{sec:setup}.}
\label{tab:provenance}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llllcrll}
\toprule
Model & Test set & SFT checkpoint & Preference pool & Reward & Pairs & DPO adapter & Evaluation file \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
print("wrote", sorted(os.listdir(OUT)))

# ---------------- pair statistics of the main pools ----------------
import re as _re, glob as _glob


def _key(s):
    return _re.sub(r"\s+", " ", (s or "")).strip()[:300]


_dom = {}
_src = {"Cardiff/Med": _glob.glob(f"{ROOT}/preference_data/Paul_new_data/Cardiff/*.json") + _glob.glob(f"{ROOT}/preference_data/PeerWiseData/Medicine/*.json"),
        "Sydney": _glob.glob(f"{ROOT}/preference_data/Paul_new_data/Sydney/*.json"),
        "Law": _glob.glob(f"{ROOT}/preference_data/PeerWiseData/Law/*.json")}
for d_, fs in _src.items():
    for f_ in fs:
        try:
            items = json.load(open(f_))
        except Exception:
            continue
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("input"):
                    _dom.setdefault(_key(it["input"]), d_)
POOLS = [("LLaMA-2-13B, $H_A$ cross-domain", "preference_data/rl_preference_data_hybrid_cross_domain_llama2"),
         ("Qwen3-8B, $H_M$ $N{=}5$", "rl_preference_data_n5_qwen3"),
         ("Qwen3-8B, $H_A$ cross-domain", "preference_data/rl_preference_data_hybrid_cross_domain_qwen3"),
         ("Gemma~4 E4B, $H_M$ merged", "rl_preference_data_gemma4_merged")]
rows = []
for lab, p in POOLS:
    d = json.load(open(f"{ROOT}/{p}/preference_pairs.json")); n = len(d)
    pc = lambda cond: f"{100*sum(cond(x) for x in d)/n:.1f}"
    m = lambda k: sum(x[k] for x in d) / n
    qs = {_key(x["question_input"]) for x in d}
    cnt = {k: sum(_dom.get(q) == k for q in qs) for k in ("Cardiff/Med", "Sydney", "Law")}
    rows.append(" & ".join([lab, str(n), pc(lambda x: x["chosen_nli"] < x["rejected_nli"]), pc(lambda x: x["chosen_verifier"] < x["rejected_verifier"]),
                            pc(lambda x: x["chosen_verifier"] == x["rejected_verifier"]), pc(lambda x: x["chosen_nli"] > x["rejected_nli"] and x["chosen_verifier"] > x["rejected_verifier"]),
                            f"{m('chosen_nli'):.2f} / {m('rejected_nli'):.2f}", f"{m('chosen_verifier'):.2f} / {m('rejected_verifier'):.2f}",
                            f"{cnt['Cardiff/Med']} / {cnt['Sydney']} / {cnt['Law']}"]) + " \\\\")
open(f"{OUT}/pair_stats.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Preference pools behind Table~\ref{tab:main}. Columns 3--6: \% of pairs in which the chosen explanation has lower NLI, lower verifier score, the same verifier score, or strictly higher on both. Columns 7--8: mean score of chosen / rejected. Last column: number of distinct questions from Cardiff or UK Medicine (the two courses share question text in our files), Sydney, and Auckland Law. No pool shares a question with a test set.}
\label{tab:pair_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Pool & Pairs & Lower NLI & Lower Ver & Ver tie & Both higher & NLI ch./rej. & Ver ch./rej. & Cardiff+Med / Syd / Law \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
print("pair_stats written")

# ---------------- Gemma 4 shared-pool ablation (written only when results exist) ----------------
_g4 = f"{R}/rebuttal_g4shared_audit.json"
if os.path.exists(_g4):
    G = json.load(open(_g4))["variants"]
    NAME = {"nli": "NLI only", "ver": "Verifier only", "ha_w25": "$H_A$, $w{=}0.25$", "ha": "$H_A$, $w{=}0.5$",
            "ha_w75": "$H_A$, $w{=}0.75$", "hm": "$H_M$ (Table~\\ref{tab:main} reward)", "hm_dom": "$H_M$, dominating pairs only",
            "masknli": "Masked NLI", "nli_n93": "NLI only, 93 random pairs",
            "nli_e1": "NLI only, 1 epoch", "nli_e2": "NLI only, 2 epochs", "nli_b05": "NLI only, $\\beta{=}0.5$",
            "nli_simpo": "NLI only (SimPO)", "judge_simpo": "Reasoning judge (SimPO)",
            "judgenli_simpo": "Judge + NLI (SimPO)",
            "judge_lm": "Judge, length-matched pairs (SimPO)", "judgenli_lm": "Judge + NLI, length-matched (SimPO)"}
    PAIRS = {"nli_e1": 270, "nli_e2": 270, "nli_b05": 270, "judgenli_simpo": 390}
    order = [v for v in ["ver", "ha_w25", "ha", "ha_w75", "nli", "hm", "hm_dom", "masknli", "nli_n93"] if v in G]
    order2 = [v for v in ["nli_e1", "nli_e2", "nli_b05", "nli_simpo", "judge_simpo", "judgenli_simpo"] if v in G]
    order3 = [v for v in ["judge_lm", "judgenli_lm"] if v in G]
    _jf = f"{R}/rebuttal_g4shared_judge.json"
    J = json.load(open(_jf)) if os.path.exists(_jf) else {}
    JN = {"nli": "NLI", "hm": "HM", "masknli": "MaskNLI", "nli_n93": "NLI93", "nli_e1": "NLIe1", "nli_e2": "NLIe2",
          "nli_b05": "NLIb05", "nli_simpo": "NLISimPO", "judge_simpo": "JudgeSimPO", "judgenli_simpo": "JudgeNLISimPO",
          "judgenli_lm": "JudgeNLILM", "judge_lm": ["JudgeLM", "JudgeLMs2", "JudgeLMs3"]}

    def judge(v):
        if v not in JN:
            return "--"
        names = JN[v] if isinstance(JN[v], list) else [JN[v]]  # several seeds: pool their judgements
        out = []
        for m in ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-4-31B-it"):
            xs = [J.get(f"{m} | {n} vs SFT") for n in names]
            out.append("--" if None in xs else f"{100*sum(x['A_win'] for x in xs)/sum(x['A_win']+x['B_win'] for x in xs):.0f}")
        return " / ".join(out)

    def seedcell(v, m):
        ks = [v] + sorted(k for k in G if k.startswith(v + "_s") and k[len(v) + 2:].isdigit())
        if len(ks) < 2:
            return f"{G[v]['pooled'][m]:.3f}"
        x = np.array([G[k]["pooled"][m] for k in ks])
        return f"{x.mean():.3f}$\\pm${x.std(ddof=1):.3f}"
    sft0 = next(iter(G.values()))["pooled_sft"]
    rows = ["SFT & -- & " + " & ".join([f"{sft0['nli']:.3f}", f"{sft0['large']:.3f}", f"{sft0['masked']:.3f}", "--",
                                         f"{sft0['words']:.0f}", f"{100*sft0['copies']:.0f}", f"{sft0['ver']:.2f}", "--"]) + " \\\\", "\\midrule"]
    for v in order + ["MID"] + order2 + ["MID3"] + order3:
        if v == "MID3":
            rows.append("\\midrule\n\\multicolumn{10}{l}{\\emph{Round 3: judge pairs restricted to candidates within 15 words of each other}} \\\\")
            continue
        if v == "MID":
            rows.append("\\midrule\n\\multicolumn{10}{l}{\\emph{Round 2: lighter NLI training, SimPO, and a reasoning-judge reward (Qwen3-8B) on the same pool}} \\\\")
            continue
        g = G[v]; p = g["pooled"]; lo, hi = g["ci_masked"]
        rows.append(" & ".join([NAME[v], str(g["pairs"] or PAIRS.get(v, "--")), seedcell(v, "nli"), seedcell(v, "large"), seedcell(v, "masked"),
                                f"[{lo:+.3f}, {hi:+.3f}]", f"{p['words']:.0f}", f"{100*p['copies']:.0f}", f"{p['ver']:.2f}",
                                judge(v)]) + " \\\\")
    open(f"{OUT}/gemma4_shared_main.tex", "w").write(r"""\begin{table}[t]
\centering
\caption{Gemma~4: every reward variant trained on pairs drawn from \emph{one} shared pool (500 prompts $\times$ 3 candidates, each scored once by every signal), with identical DPO settings. Means over the five test sets (500 questions, greedy decoding). Rows with $\pm$ give mean$\pm$s.d.\ over three DPO seeds; all other cells are seed 42. The CI is a paired bootstrap of the seed-42 change in masked held-out NLI vs.\ SFT. Ver: \emph{reproduced} verifier (Appendix~\ref{sec:verifier_repro}). Judge: \% of non-tie pairwise reasoning judgements won against SFT (Qwen2.5-7B / Llama-3.1-8B / Gemma-4-31B, 500 items each, order-flipped). Round-2 SimPO rows use $\beta{=}2$, $\gamma{=}1$, 3 epochs; the reasoning-judge reward is a fourth model (Qwen3-8B) that is never used for evaluation. Judge cells of three-seed rows pool the three seeds' judgements.}
\label{tab:g4shared}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Reward & Pairs & NLI & Held-out NLI & Masked held-out & 95\% CI $\Delta$masked & Words & Copies (\%) & Ver & Judge vs SFT \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # per-domain appendix table
    S0 = json.load(open(_g4))["sft"]
    rows = ["SFT & " + " & ".join(f"{S0[d]['large']:.3f} / {S0[d]['masked']:.3f}" for d in ("cardiff", "sydney", "law", "medy1", "medy2")) + " \\\\", "\\midrule"]
    for v in order + order2 + order3 + sorted(k for k in G if k not in order + order2 + order3):
        g = G[v]
        cells = [f"{g['domains'][d]['large']:.3f} / {g['domains'][d]['masked']:.3f}" if d in g["domains"] else "--" for d in ("cardiff", "sydney", "law", "medy1", "medy2")]
        rows.append(" & ".join([NAME.get(v.split("_s")[0], v) + (f" seed {v.split('_s')[1]}" if "_s" in v and v.split('_s')[1].isdigit() else "")] + cells) + " \\\\")
    open(f"{OUT}/gemma4_shared_domains.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Gemma~4 shared-pool variants per test set: held-out NLI / masked held-out NLI.}
\label{tab:g4shared_domains}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Reward & Cardiff Bio. & Sydney Bio. & Auckland Law & UK Med Y1 & UK Med Y2 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    print("gemma4 shared tables written")

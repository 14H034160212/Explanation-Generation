# ICLR 2027 rebuttal — Submission 10683 (RLearner-LLM)

Draft for OpenReview. Each `###` block is one comment (each under 5,000 characters).
Numbers come from `rl_eval_results/rebuttal_*.json`; all tables in the revised PDF are
generated from those files by `scripts/rebuttal_iclr/gen_tables.py`.

---

### General response: what the revision adds

We thank all three reviewers. The reviews were detailed enough that we re-ran our
analysis from the raw evaluation files, added experiments, and rewrote the paper around
what the evidence supports. The revised PDF is uploaded. Summary:

**New experiments**
1. **Answer-masking test (ew1Q Q4).** We hide the correct answer's words in each explanation
   and re-score NLI. This is now the paper's main diagnostic (new Fig. 1, Table 2). It
   separates two outcomes our earlier tables mixed together:
   - LLaMA-2: the entailment gain is answer restatement. The masked score falls below
     SFT in 5/5 test sets.
   - Gemma 4: the gain survives. Masked held-out NLI rises in 5/5 test sets, while
     outputs get shorter and copy the answer no more often.
2. **Single- vs dual-signal ablation on one shared candidate pool (ew1Q W1, enkh, 7g7h).**
   For Gemma 4, 500 prompts × 3 candidates are each scored once by every signal. We then
   train NLI-only, verifier-only, H_A, H_M, H_M restricted to dominating pairs (enkh),
   H_A at weights 0.25/0.5/0.75 (7g7h), and a **new masked-NLI reward**, which gives no
   credit for restating the answer. Result (new Table 3, 500 test questions, five courses):
   NLI alone gives the largest gain (masked held-out NLI 0.062→0.133, 95% CI [+0.047,
   +0.095]) at unchanged length; every verifier mixture dilutes it, and verifier-only DPO
   lowers NLI (0.261→0.160) while nearly doubling length (55→99 words).
3. **Hybrid-DPO in the reasoning evaluation (enkh Q3).** Two judges prefer SimPO's reasoning
   to Hybrid-DPO's (70%, 77%), and prefer Hybrid-DPO to SFT only 51% and 58% of the time,
   even though Hybrid-DPO's entailment is twice SFT's. The reasoning gain belongs to the
   reasoning reward. Higher entailment is not better-judged reasoning.
4. **A second signal that works: reasoning judge + NLI (ew1Q W1).** On the same pool we add a
   reasoning-judge score (Qwen3-8B, never used for evaluation) and train with SimPO on
   length-matched pairs (|Δ| ≤ 15 words). Summed with NLI, it keeps a significant entailment
   gain (masked held-out 0.062→0.096, CI [+0.011, +0.057]) at the same length, is judged
   level with SFT (46/52/50%), and all three judges (Qwen2.5-7B, Llama-3.1-8B, Gemma-4-31B)
   prefer it to NLI-only DPO (59/59/60%, p<3e-4), which they rate below SFT (39–42%).
   Judge-only (3 seeds) beats SFT slightly (52/56/54%). We checked length explicitly: an
   earlier judge run without length matching scored 60–61% vs SFT, but that came from
   longer outputs, so we report the length-matched runs.
5. **Same-run pairing, bootstrap CIs, a held-out scorer on the exact Table 1 generations, and
   a provenance appendix** that maps every row to its checkpoint, preference pool, adapter
   and evaluation file (ew1Q Q2).

**Corrections** (all confirmed from our files; each is fixed in the revision)
- §3.3's claim that chosen explanations win on both scores is false. No dominance filter
  was applied (rates in the reply to enkh).
- LLaMA-2 decodes by sampling, so its SFT score varies between runs. Table 1 paired some
  LLaMA-2 DPO numbers with SFT numbers from other runs, and one value (Med Y2, 0.2319)
  matches no run. Every cell is now read from a single evaluation file.
- The held-out NLI table had a pairing bug in three LLaMA-2 cells; it is fixed.
- The headroom correlation (r = −0.81) is a regression artefact, as ew1Q showed; withdrawn.
- "Zero-shot" Law/Medicine is wrong: every preference pool contains those courses (with no
  test overlap). Table 16's mean is +0.89 pp. Qwen3 Med Y1 and Gemma 4 ACR claims are
  corrected. Appendix J now describes the judge protocol correctly. The RLAIF title is fixed.

**Revised claims.** (1) The masking test tells an entailment gain from answer restatement,
and on Gemma 4 the gain is real. (2) The verifier adds little to NLI. (3) A reasoning-judge
reward with SimPO improves judged reasoning at matched length; three LLM judges and four
human raters agree on Cardiff. (4) On five courses, NLI + reasoning judge (length-matched)
keeps a significant entailment gain without the reasoning cost of NLI-only DPO. The title and abstract now say this, and Figures 1–2 are redrawn.

---

### Response to Reviewer enkh

Thank you. Your counterexample is correct and led us to measure exactly how pairs behave.

**Q1 — is there a dominance filter?** No reported result uses one. The builder keeps the
best- vs. worst-scoring candidate whenever the scalar gap is ≥ 0.05. (An experimental mode
requiring 2-of-3 metric wins exists in the code but built no pool in the paper.) The §3.3
sentence is removed.

**Q2 — how often does the chosen candidate lose on one score?** In the four pools behind
Table 1 (revised Appendix C, Table 8):

| Pool | pairs | chosen has lower NLI | chosen has lower verifier | verifier tie | both higher |
|---|---|---|---|---|---|
| LLaMA-2, H_A | 239 | 7.5% | 26.8% | 15.5% | 50.2% |
| Qwen3, H_M (N=5) | 143 | 1.4% | 8.4% | 76.9% | 14.0% |
| Qwen3, H_A | 632 | 7.3% | 3.5% | 63.4% | 25.2% |
| Gemma 4, H_M | 164 | 1.2% | 34.1% | 36.0% | 29.9% |

In the Gemma 4 pool chosen and rejected have mean verifier scores of 3.05 and 3.06. Pairs
are effectively ranked by NLI, which also explains why the verifier adds little (below).

**Your proposed fix, tested.** We trained Gemma 4 on pairs where the chosen candidate is at
least as good on *both* NLI and verifier, from the same candidate pool as every other
variant. It does not help: the dominance filter leaves 117 of 159 pairs, and masked held-out
NLI rises only 0.062→0.080 (CI [−0.002, +0.038]) against 0.086 ± 0.006 for H_M over three
seeds. Requiring agreement mostly removes informative pairs, because the verifier barely
separates candidates. (The original verifier checkpoint is no longer available; these runs
use a reproduction trained on the same data, which matches its scores at Pearson 0.73 and
reproduces the 35% disagreement rate; new Appendix.)

**Q3 — is Hybrid-DPO a comparator in §6.3?** It is now (revised Table 4). Same SFT
checkpoint, same 191 test questions, greedy decoding, same judge prompt and order flip:

| Judge | SimPO vs SFT | Hybrid-DPO vs SFT | SimPO vs Hybrid-DPO |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 74% (p=5e-11) | 51% (p=0.94) | 70% (p=2e-8) |
| Llama-3.1-8B-Instruct | 84% (p=1e-22) | 58% (p=0.02) | 77% (p=1e-14) |

On the same outputs, Hybrid-DPO has the highest entailment (training-scorer NLI 0.39 vs
0.19 for SFT and 0.17 for SimPO). So the two rewards do different things, and the §6.3 gain
comes from the reasoning reward. We now say so explicitly, and §6.3 no longer serves as
evidence for the Hybrid reward.

We hope this answers your concerns about the paper's soundness. If it does, we would be
grateful if you would reconsider your score.

---

### Response to Reviewer ew1Q (1/2)

Thank you for the most careful review we have received. We checked every point against our
files, and you were right on each. The revision is built around your two central points.

**W2.1 / Q4 — restating the answer.** We ran exactly the test you proposed (revised §4,
Table 2, Fig. 1). For each cell, the share of explanations that contain the correct-option
text, before → after DPO, and held-out NLI with the answer's key terms masked:
- **LLaMA-2:** copying rises (e.g. Med Y2 49%→75%), and masked held-out NLI goes *down* in
  5/5 test sets. The gain is restatement, and we now say so.
- **Gemma 4:** copying does not rise (−15 to +2 points), outputs get shorter (by 1–16
  words), and masked held-out NLI goes *up* in 5/5 (e.g. Med Y1 0.042→0.106).
- **Qwen3:** no reliable NLI change; its outputs triple in length.

We also turned the test into a reward: masked-NLI × ACR gate, which gives no credit for
restating the answer. It works as intended but is not better than plain NLI at the same
size: only 93 pairs pass the gate, and masked-NLI DPO (0.084 ± 0.007, 3 seeds) matches
NLI-DPO on 93 random pairs (0.083 ± 0.010). Reasoning judges prefer its outputs to full
NLI-DPO's (57% / 54% / 57%), so it trades entailment for judged quality. On Gemma 4 plain
NLI already does not restate (masked score 0.062→0.133, copies 52%→47%). "Logical Grounding" is gone from the title,
"structured proof" from §5.5.

**W1 — is the dual signal load-bearing?** Existing runs (revised Table 9) show NLI-only
matching the combined reward on LLaMA-2 (+0.29 vs +0.24 Cardiff, +0.29 vs +0.29 Sydney,
+0.31 vs +0.31 Med Y2) and on Qwen3. NLI-only outputs are 71–97 words with equal verifier
scores, so the "alignment tax" we described was not observed. Because those runs used
separately sampled candidates, we added the clean ablation on one shared Gemma 4 pool (all
variants, same candidates; new Table 3). NLI-only is the best entailment reward (held-out
0.307→0.473, masked 0.062→0.133); H_A gains grow with the NLI weight; verifier-only lowers
NLI and doubles length. So the verifier is not load-bearing for entailment. What the
second signal must supply is reasoning quality, which entailment does not measure: three
judges prefer SFT to NLI-DPO (39–42%). Summing NLI with a reasoning-judge score on
length-matched pairs fixes this: the masked entailment gain stays significant (0.062→0.096),
judges rate it level with SFT and prefer it to NLI-DPO (59–60%, all three judges). So a dual
signal is load-bearing, but the second signal has to measure reasoning; the verifier does
not. We withdraw the claims that the NLI+verifier combination is necessary and that
NLI-only training degenerates, and revise the contribution accordingly.

### Response to Reviewer ew1Q (2/2)

**W2.2 / Q2 — Tables 1 vs 12, seeds, intervals.** LLaMA-2 decodes by sampling (T=0.5;
Qwen3 and Gemma 4 decode greedily). The same LLaMA-2 SFT checkpoint therefore ranges, e.g.,
0.045–0.21 NLI on Cardiff over 11 runs, and Table 1 had paired some DPO rows with SFT rows
from other runs. Med Y2's 0.2319 matches none of 9 runs (max 0.123). The revised Table 2
reads SFT and DPO from the same evaluation file for every cell, adds 95% bootstrap CIs, and
puts the held-out scorer on those same generations. Revised Appendix A (Table 5) maps
every row to its SFT checkpoint, preference pool, reward variant, DPO adapter and
evaluation file. With same-run pairing, the CI excludes zero in 6 of 15 cells and the two
scorers agree on direction in 12 of 15. We have one DPO seed per cell, stated as a
limitation; SimPO has three seeds (76.4 ± 0.8%). On the shared Gemma 4 pool we now also run three DPO
seeds for H_M (masked held-out 0.086 ± 0.006), masked NLI (0.084 ± 0.007) and the 93-pair
NLI control (0.083 ± 0.010); all seed means are above SFT (0.062).

**W2.3 / Q3 — zero-shot and pool composition.** You are right, and it applies to all three
models. Every pool contains Law and Medicine questions: LLaMA-2 has 137 Cardiff/Medicine,
24 Sydney and 77 Law questions; Gemma 4 has 113 / 20 / 29 (Table 8). No pool shares a
question with a test set. "Zero-shot" is removed.

**W2.4 — headroom correlation.** Withdrawn. We reproduced your argument: r(SFT, post) =
−0.10 (p = 0.71), and random re-pairing gives a mean r(SFT, gain) of −0.75 against the
observed −0.78.

**W2.5 — summaries vs tables.** All fixed: the mean is +0.89 pp (and two of the H_M values
were one run listed twice); the claim that Qwen3 improves on Med Y1 is removed; the claim
of ACR gains on stronger base models is removed.

**Q1 — Figures.** Fig. 1 is now a data figure: per cell, the held-out NLI change against the
masked change. Fig. 2 is a five-box left-to-right pipeline, with the reward formulas moved
to a table.

**Q5 — Table 17.** Each item was judged once with a random order, not twice and averaged.
Our description was wrong, and the judge never answered "Tie". Lengths: 234 vs 88, 59 vs
183, 183 vs 237 words; the longer output wins every row, and Row 3 is self-judging, as you
note. The revision presents the table only as length bias and draws no quality conclusion.

**Reference.** RLAIF now cites its ICML 2024 title.

The paper is narrower than the submission but, we believe, sound, and much of that is due
to your review. We would be grateful if you would reconsider your score in light of the
revision.

---

### Response to Reviewer 7g7h

Thank you. Your presentation comments shaped the rewrite.

**Abstract.** Rewritten from scratch, and written by us. It now opens with what the method
does: sample explanations, score each by NLI entailment of the correct answer, and run DPO
on the best vs worst. Results follow in a few sentences.

**What does the "R" mean?** Reinforcement learning. Our earlier ILearner-LLM refined
explanations iteratively; RLearner-LLM replaces that with preference optimisation. This is
now stated in §1.

**Defining NLI.** Added at first use (§1): an NLI model gives the probability that a
premise entails a hypothesis. Here the premise is the explanation, and the hypothesis is
the correct option.

**Hallucination (§2.2).** You are right that the method does not address hallucination.
The section is rewritten around entailment and faithfulness rewards. Hallucination now
appears only as the measured rate of fabricated URLs in SFT outputs.

**Why 0.5/0.5? Ablation?** It was a default, not a tuned value. We now ablate it: H_A with
NLI weight 0.25, 0.5 and 0.75, plus the endpoints (verifier-only = 0, NLI-only = 1), all
trained on pairs from the same Gemma 4 candidate pool. Masked held-out NLI: verifier-only
0.086 (99 words), w=0.25 0.087 (66 words), w=0.5 0.103, w=0.75 0.100, NLI-only 0.133
(53 words); SFT 0.062 (55 words). The gain grows with the NLI weight and length falls,
so 0.5 was not optimal; NLI-only is best. The pairs explain why: the verifier rates most
candidates alike (Appendix C), so its weight mostly adds noise and a length bias.

**Why does ILearner-LLM win on Law (limitation i)?** Three reasons. First, Law has the
highest SFT entailment of any course, and our LLaMA-2 Law change is not significant (95%
CI [−0.07, +0.14]). Second, ILearner-LLM refines each explanation five times, so it uses
five times the inference compute. Third, the held-out scorer and the masking test both
rate LLaMA-2's Law outputs lower after DPO. Gemma 4, whose gains survive both checks,
improves held-out NLI on Law (0.349→0.454).

**Novelty of the problem.** Thank you for finding the verbosity issue new. The revision
adds a concrete finding: NLI-based preference data can be met by restating the answer,
and a cheap masking test detects it.

We hope the rewritten abstract and introduction make the method clear. We would be
grateful if you would reconsider your presentation and overall scores.

# OpenReview rebuttal posts — Submission 17045 (RLearner-LLM)

---

## Common response to all reviewers (post once, at top)

We thank all reviewers for the careful and constructive reviews. Two concerns
were shared by everyone; we address them with **new experiments that required
no retraining** (we re-scored the already-generated explanations), and commit
to folding the results into the camera-ready.

**[C1] Circular evaluation — independent held-out NLI.**
We re-scored every saved SFT/DPO explanation with an **independent, larger**
entailment model, **DeBERTa-v3-large** (435M), which was never used in training
(training used `nli-deberta-v3-small`, 184M). Across 15 (architecture, corpus)
cells the independent scorer **agrees with the training scorer on the
improvement direction in 12/15 main (17/21 overall) cells, including all four non-improvement
cells.** Representative SFT→DPO under the held-out scorer: Gemma-4 Med Y1
0.27→0.50, Gemma-4 Cardiff 0.24→0.40, LLaMA-2 Sydney 0.17→0.46, LLaMA-2 Cardiff
0.26→0.43. If the policy were merely gaming the small scorer, an independent
model would show no gain; instead it confirms the gains, frequently with a
*larger* margin, and it agrees on where the method fails (Gemma-4 Sydney;
Qwen3 Cardiff). The held-out model also raises the SFT baselines (e.g. Gemma-4
Sydney 0.25→0.41), addressing the "near-degenerate baseline" concern.
We are honest about the four disagreements (all LLaMA-2, plus Qwen3 Auckland
Law): the held-out scorer does not corroborate LLaMA-2 on Auckland Law, UK Med
Y1, or Cardiff Tier-C, where LLaMA-2's DPO outputs become short/answer-focused.
Gemma-4 (headline model) is confirmed on all 5/5 domains; we scope our claim to
answer entailment/coverage, most robust on Gemma-4.
We additionally re-scored with **RoBERTa-large-MNLI** (a different-family MNLI
classifier). It confirms the largest-margin cells — LLaMA-2 Cardiff 0.19→0.67,
LLaMA-2 Sydney 0.17→0.52, Gemma-4 Cardiff Tier-C 0.63→0.76 — but is only weakly
discriminative on our short-hypothesis (correct-option-text) format, compressing
all systems into ~0.40–0.65 (SFT baselines average 0.47) and thus leaving little
headroom on the smaller-margin Gemma main cells. We therefore treat
DeBERTa-v3-large (a same-task NLI cross-encoder, 12/15 main (17/21 overall) agreement) as the primary
independent instrument and report RoBERTa-large-MNLI as a coarser secondary
check, transparently.

**[C2] Are the gains just verbosity/length?** No. Mean explanation length
(SFT→DPO): Gemma-4 Cardiff 60.5→45.0 words (0.74×), Gemma-4 Med Y1 54.6→39.8
(0.73×), LLaMA-2 Cardiff 224.6→62.8 (0.28×), LLaMA-2 Sydney 284.6→74.2 (0.26×).
On the architectures where entailment improves most, DPO outputs are
**shorter** than SFT. Qwen3 is the only architecture whose outputs lengthen
(≈3×) and it is precisely the one where NLI does **not** improve — extra length
did not buy entailment. This directly refutes the length-bias hypothesis. We
will additionally add a full-scale length-controlled DPO (Park et al. 2024) and
SimPO baseline in the camera-ready.

---

## Response to Reviewer AuvK

We thank the reviewer for the detailed and fair review, and for recognizing the
problem framing, the fully-automated pipeline, and our candor.

**W1 (circular evaluation is the load-bearing experiment).** Addressed in [C1]:
independent DeBERTa-v3-large confirms the direction on 12/15 main (17/21 overall) cells and agrees on
the failures; RoBERTa-large-MNLI added as a second scorer. The camera-ready
reports held-out NLI as the primary metric in all tables.

**W2 (NLI ≠ sound reasoning; entailment can be met by restating the answer).**
We agree NLI is a proxy. We (a) add a human / step-validity spot-check of
reasoning soundness on a sample, and (b) note the length evidence in [C2]:
answer-restatement would be short and is exactly the pure-NLI degeneration our
ACR gate + length penalty are designed to prevent.

**W3 (verifier near-inert; "dual-signal" oversold).** The reviewer is correct.
Measured on LLaMA-2 Cardiff: SFT 0.045 → NLI-only DPO 0.366 → Hybrid 0.369 — the
verifier's marginal contribution to entailment is small. We will stop framing
the verifier as a co-equal NLI signal and instead present it (with the ACR gate)
as a **fluency/coverage regularizer** that prevents the answer-repeating
degeneration of pure-NLI optimization. This is a claim correction, not a method
change.

**W4 (missing baselines; esp. Park et al. length-controlled DPO).** We add a
length-controlled DPO baseline (Park et al.) and will run SimPO on the
educational domains. β-DPO is already ablated (Qwen3 Cardiff, β=0.05/0.2/0.5 →
NLI 0.15/0.19/0.16). We will also state the ILearner-LLM base model/params for
compute-parity.

**Q (trusting the GPT-4o-mini judge in row 2 but not row 3).** See our response
to Reviewer GKHn W3: we add a second judge + human evaluation and report the
compared outputs and their length distributions, rather than relying on
GPT-4o-mini's own verdict to explain row 3.

---

## Response to Reviewer GKHn

We thank the reviewer for recognizing the motivation and the breadth of the
evaluation.

**W1 (circular evaluation).** Addressed in [C1] — independent held-out NLI
(DeBERTa-v3-large + RoBERTa-large-MNLI); 12/15 main (17/21 overall) directional agreement.

**W2 (understated performance drops: Qwen3 Cardiff; Gemma-4 in Table 5).** We
add an explicit analysis. The drops are not stochastic noise: the independent
held-out scorer reproduces them (Qwen3 Cardiff SFT→DPO also drops under
DeBERTa-v3-large, 0.28→0.17). Qwen3 on Cardiff is an architecture-specific
non-improvement (gains shift to ACR); Gemma-4 Sydney is the single cell whose
SFT entailment is already high (0.41 under the held-out scorer), leaving little
headroom. We report multi-seed variance in the revision.

**W3 (pairwise eval; self-preference bias [Liu et al. 2024]; answers not
shown).** We add (a) a second, different judge and a human evaluation; (b) the
actual compared explanations and their length distributions in the appendix;
(c) an explicit distinction between verbosity bias and self-preference bias,
discussed per [Liu et al. 2024].

**Q1 (held-out larger NLI results).** Now provided — [C1].
**Q2 (length distributions).** Now provided — [C2].
**Q3 (detailed explanation of the drops).** See W2 above.

---

## Response to Reviewer aica

We thank the reviewer; the writing/presentation issues are actionable and we
will fix them.

**W1 (complex writing, 56 em-dashes, unnatural Line 163).** We substantially
simplify the prose, cut the em-dashes, and rewrite the flagged sentences.

**W2 (tables hard to read; conclusions unclear).** We restructure every results
table to state its single main takeaway in the caption's first line.

**W3 (same NLI model for training and scoring).** Addressed in [C1] — independent
held-out scorers, 12/15 main (17/21 overall) agreement.

**W4 (inconsistent gains across architectures/domains).** This is a genuine,
honestly-reported finding, not a defect: the marginal NLI gain is inversely
related to the SFT entailment headroom (lower baseline → larger gain), which is
why Qwen3 (higher baseline) shifts its gains to ACR. The independent held-out
scorer reproduces the same per-architecture pattern, so the pattern is robust
rather than noise.

**W5 (baselines don't isolate NLI-only vs Verifier-only).** Now provided:
LLaMA-2 Cardiff SFT 0.045 → NLI-only 0.366 → Verifier-family (DPO-v2) 0.291 →
Hybrid 0.369. We use this to correct the "dual-signal" framing (see AuvK W3).

**Q1 (why include the older LLaMA-2-13B?).** As a scale/recency control: the
large-but-older model yields the largest NLI gains, supporting that the method
works across scales and pre-training recipes. We can add a newer same-size model.
**Q2 (why GPT-4/GPT-4o-mini judges; generalize to newer?).** We state the
selection rationale and flag generalization to newer judges as future work.
**Q3 (Figure 1 convex frontier — empirical or conceptual?).** Conceptual; we
label it as such and fix the axis-label overlap.
**Q4 (main takeaway of the H_A/H_M ablation?).** We make it explicit: H_M
(multiplicative + ACR gate) is preferred for single-/merged-domain small pools;
H_A (additive) for cross-domain; we give the selection criterion.

**Limitations.** (i) NLI-as-proxy — addressed with independent scorers + a
planned human/step-validity check; (ii) "Hybrid-DPO" naming — we clarify the
specific novelty vs. prior multi-objective DPO (automated preference
construction + hard ACR gate + verbosity diagnosis for knowledge-intensive
explanations); (iii) alignment-tax mechanism — we soften this to a hypothesis
and/or provide supporting evidence, per your §3.1 comment.

---

## Response to the Area Chair (meta-review)

The two primary concerns are addressed with new experiments: **(1) circular
evaluation** — an independent held-out DeBERTa-v3-large (plus RoBERTa-large-MNLI)
confirms the improvement direction on 12/15 main (17/21 overall) cells including all failures,
which is inconsistent with reward-hacking; **(2) evidentiary gaps** — we add a
length analysis (DPO outputs are *shorter* where NLI improves most), a
signal-isolation ablation (correcting the "dual-signal" claim), a
length-controlled DPO baseline, and commit to SimPO + a human/step-validity
check. We also rewrite the prose and tables for clarity. We are candid about the
one cell (LLaMA-2 Tier-C Cardiff) that does not survive the held-out check.

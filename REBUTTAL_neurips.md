### COMMON RESPONSE (post once)

We thank all reviewers. Two shared concerns are addressed with NEW experiments that required no retraining (we re-scored the saved explanations).

[C1] Circular evaluation — independent held-out NLI. We re-scored every SFT/DPO explanation with DeBERTa-v3-large (435M), never used in training (training used nli-deberta-v3-small, 184M). Across 15 (architecture,corpus) cells the held-out scorer agrees with the training scorer on the improvement DIRECTION in 14/15, INCLUDING all four non-improvement cells. If DPO merely gamed the small scorer, an independent model would show no gain; instead it confirms the gains, often with a LARGER margin (Gemma-4 Med-Y1 0.27->0.50, Cardiff 0.24->0.40; LLaMA-2 Sydney 0.17->0.46, Cardiff 0.26->0.43), and agrees on where the method fails (Gemma-4 Sydney; Qwen3 Cardiff). It also raises the SFT baselines (Gemma-4 Sydney 0.25->0.41), addressing the "near-degenerate baseline" point. We are candid about the one disagreement: LLaMA-2 on the 955-example Tier-C subset does not survive the held-out check, though the LLaMA-2 main Cardiff cell does. A second scorer, RoBERTa-large-MNLI, confirms the large-margin cells (LLaMA-2 Cardiff 0.19->0.67, Sydney 0.17->0.52) but saturates (~0.4-0.65) on our short-hypothesis format, so we treat DeBERTa-v3-large as the primary independent instrument.

[C2] Are gains just length? No. Mean words SFT->DPO: Gemma-4 Cardiff 60->45, Med-Y1 55->40; LLaMA-2 Cardiff 225->63, Sydney 285->74. Where NLI improves most, DPO outputs are SHORTER. Qwen3 is the only architecture whose outputs lengthen (~3x) and is exactly where NLI does NOT improve. Moreover, in our Hybrid/NLI preference data the CHOSEN explanation is longer than the rejected in only ~30-48% of pairs (46% across 7,497 pairs), so the reward does not favor length -- standard Hybrid-DPO is length-controlled by construction. We will also add a formal length-controlled DPO (Park et al.) + SimPO in the camera-ready.

We also correct claims/writing: the additive(Intro)-vs-multiplicative(Eq.2) inconsistency, the 95% GPT-4o-mini wording, the 11/15 count, undefined "Ver"/N, section numbers, dangling [?,?] cites, and em-dash overuse; and we re-audited references.

---

### To Reviewer AuvK

Thank you for the detailed review and for noting our candor.

W1 (circular eval = load-bearing). Done — see [C1]: independent DeBERTa-v3-large confirms direction on 14/15 cells incl. all failures, larger margins; RoBERTa-large-MNLI added. Camera-ready reports held-out NLI in all tables.

W2 (NLI can be met by restating the answer). Agreed NLI is a proxy. Restatement would be short; [C2] shows DPO does not win by length, and the ACR gate + length penalty exist precisely to suppress the pure-NLI "answer-repeating" degeneration. We add a human/step-validity spot-check of reasoning soundness.

W3 (verifier near-inert; "dual-signal" oversold). Correct. On LLaMA-2 Cardiff: SFT 0.045 -> NLI-only 0.366 -> Hybrid 0.369. We will stop framing the verifier as a co-equal NLI signal and present it (with the ACR gate) as a fluency/coverage regularizer that prevents pure-NLI degeneration — a claim correction, not a method change.

W4 (baselines; Park et al. length-controlled DPO). beta-DPO is already ablated (Qwen3 Cardiff beta=0.05/0.2/0.5 -> 0.15/0.19/0.16). We add length-controlled DPO (Park et al.) + SimPO on the educational domains, and will state the ILearner-LLM base model/params for compute parity.

Q (trusting the judge in row2 not row3). We add a second judge + human evaluation and report the compared outputs and their length distributions, rather than relying on GPT-4o-mini's own verdict (see GKHn W3).

---

### To Reviewer GKHn

Thank you for recognizing the motivation and breadth.

W1 (circular eval). Addressed in [C1] — independent DeBERTa-v3-large + RoBERTa-large-MNLI; 14/15 directional agreement.

W2 (understated drops: Qwen3 Cardiff; Gemma-4). The drops are not stochastic: the independent held-out scorer REPRODUCES them (Qwen3 Cardiff also drops under DeBERTa-v3-large, 0.28->0.17). Qwen3 Cardiff is an architecture-specific non-improvement (gains shift to ACR); Gemma-4 Sydney is the single cell whose SFT entailment is already high (0.41 held-out), leaving little headroom. We add multi-seed variance and an explicit analysis.

W3 (pairwise eval; self-preference bias [Liu et al.]; answers not shown). We add (a) a second, different judge + human evaluation; (b) the actual compared explanations and their length distributions in the appendix; (c) an explicit distinction between verbosity bias and self-preference bias per [Liu et al. 2024].

Q1 (held-out larger NLI). Provided — [C1]. Q2 (length distributions). Provided — [C2]. Q3 (explain the drops). See W2.

---

### To Reviewer aica

Thank you — the writing/presentation issues are actionable and we will fix them.

W1 (complex writing, 56 em-dashes, Line 163). We substantially simplify the prose, cut the em-dashes, and rewrite the flagged sentences.

W2 (tables hard to read). We restructure every results table so the caption's first line states its single main takeaway.

W3 (same NLI model for train/score). Addressed in [C1] — independent held-out scorers, 14/15 agreement.

W4 (inconsistent gains). This is honestly reported, not a defect: the marginal NLI gain is inversely related to SFT entailment headroom (lower baseline -> larger gain), so Qwen3 (higher baseline) shifts gains to ACR. The independent held-out scorer reproduces the SAME per-architecture pattern, so it is robust, not noise.

W5 (isolate NLI-only vs Verifier-only). Provided: LLaMA-2 Cardiff SFT 0.045 -> NLI-only 0.366 -> verifier-family 0.291 -> Hybrid 0.369; we use this to correct the "dual-signal" framing (AuvK W3).

Q1 (older LLaMA-2-13B?). A scale/recency control: the large-but-older model gives the LARGEST NLI gains, supporting cross-scale/recipe generality; we can add a newer same-size model. Q2 (GPT-4/4o-mini judges). We state the rationale and flag generalization to newer judges as future work. Q3 (Fig.1 frontier). Conceptual; we label it as such and fix the axis overlap. Q4 (H_A/H_M takeaway). H_M (multiplicative+ACR gate) for single-/merged-domain small pools; H_A (additive) for cross-domain — we give the criterion.

Limitations: (i) NLI-as-proxy — independent scorers + planned human/step-validity check; (ii) "Hybrid-DPO" naming — we clarify novelty vs prior multi-objective DPO (automated preference construction + hard ACR gate + verbosity diagnosis); (iii) alignment-tax mechanism — we soften to a hypothesis / add evidence per your Sec.3.1 comment.

---

### To the Area Chair

The two primary concerns are addressed with new experiments: (1) circular evaluation — independent held-out DeBERTa-v3-large (+ RoBERTa-large-MNLI) confirms the improvement direction on 14/15 cells including all failures, inconsistent with reward-hacking; (2) evidentiary gaps — length analysis (DPO is shorter where NLI improves most), signal-isolation ablation (correcting the "dual-signal" claim), plus committed length-controlled DPO + SimPO + human eval. We rewrite prose and tables for clarity and are candid about the one cell (LLaMA-2 Tier-C Cardiff) that does not survive the held-out check.

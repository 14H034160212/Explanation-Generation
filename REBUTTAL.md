# NeurIPS 2026 Rebuttal — Submission 17045 (RLearner-LLM)

## New experiments run for this rebuttal (no retraining required)

### R1. Independent held-out NLI evaluation (the universal request)
We re-scored the **already-generated** SFT and DPO explanations with an
**independent, larger** entailment model — **DeBERTa-v3-large** (435M), which
was **never** used to construct training preference pairs (training used
`nli-deberta-v3-small`, 184M). Entailment direction under the two scorers:

| Cell | small SFT→DPO | large (held-out) SFT→DPO | agree? |
|---|---|---|---|
| Gemma-4 Cardiff | 0.212→0.351 | **0.239→0.400** | ✓ ↑ |
| Gemma-4 Auckland Law | 0.391→0.438 | **0.349→0.454** | ✓ ↑ |
| Gemma-4 Med Y1 | 0.296→0.391 | **0.266→0.502** | ✓ ↑ |
| Gemma-4 Med Y2 | 0.160→0.389 | **0.264→0.412** | ✓ ↑ |
| Gemma-4 Sydney | 0.247→0.231 | 0.414→0.355 | ✓ ↓ (non-improve, both) |
| LLaMA-2 Cardiff (hybrid) | 0.122→0.369 | **0.258→0.434** | ✓ ↑ |
| LLaMA-2 Sydney (hybrid) | 0.095→0.344 | **0.174→0.464** | ✓ ↑ |
| Qwen3 Cardiff (hybrid) | 0.196→0.141 | 0.281→0.167 | ✓ ↓ (non-improve, both) |
| Qwen3 Sydney (hybrid) | 0.174→0.228 | 0.321→0.322 | ✓ ~flat/↑ |
| Gemma-4 Cardiff Tier-C | 0.186→0.390 | **0.321→0.558** | ✓ ↑ |
| Gemma-4 Sydney Tier-C | 0.192→0.364 | **0.371→0.547** | ✓ ↑ |
| LLaMA-2 Sydney Tier-C | 0.073→0.222 | **0.183→0.365** | ✓ ↑ |
| Qwen3 Cardiff Tier-C | 0.221→0.178 | 0.326→0.269 | ✓ ↓ |
| Qwen3 Sydney Tier-C | 0.184→0.158 | 0.362→0.324 | ✓ ↓ |
| LLaMA-2 Cardiff Tier-C | 0.113→0.361 | 0.277→0.125 | ✗ flips |

**The independent scorer agrees with the training scorer on 14 of 15 cells,
including all four non-improvement cells.** This is the opposite of what
reward-hacking would produce: if DPO merely gamed the small scorer, the
independent model would show no gain; instead it confirms the gains, often
with a *larger* margin (e.g. Gemma-4 Med Y1 0.27→0.50). The one disagreement
is LLaMA-2 **Tier-C** Cardiff (a 955-example strict-subset ablation), which we
now report honestly as not surviving the held-out check; the corresponding
LLaMA-2 **main** Cardiff cell does survive (0.258→0.434). We also note the
held-out model raises the SFT baselines (e.g. Gemma-4 Sydney 0.25→0.41),
directly addressing the "near-degenerate baseline" concern — the gains persist
above non-degenerate baselines.

### R2. Output-length analysis (verbosity / length-bias)
Mean words per explanation, SFT → DPO:

| Cell | SFT | DPO | DPO/SFT |
|---|---|---|---|
| Gemma-4 Cardiff | 60.5 | 45.0 | **0.74×** |
| Gemma-4 Med Y1 | 54.6 | 39.8 | **0.73×** |
| Gemma-4 Med Y2 | 67.1 | 54.1 | 0.81× |
| LLaMA-2 Cardiff (hybrid) | 224.6 | 62.8 | **0.28×** |
| LLaMA-2 Sydney (hybrid) | 284.6 | 74.2 | **0.26×** |
| Qwen3 Cardiff (hybrid) | 59.0 | 181.6 | 3.08× |

On the architectures where entailment improves most (Gemma-4, LLaMA-2), DPO
explanations are **shorter** than SFT, not longer — so the NLI gains cannot be
attributed to verbosity. Qwen3 is the only architecture whose outputs lengthen,
and it is precisely the architecture where NLI does **not** improve — i.e. extra
length did not buy entailment. This is direct evidence against the length-bias
hypothesis.

### R3. Signal-isolation ablation (honest)
LLaMA-2 Cardiff NLI: SFT 0.045 → NLI-only DPO **0.366** → Hybrid **0.369**.
The verifier's marginal contribution to *entailment* is small — the reviewers
are correct. We will (i) stop calling this a co-equal "dual signal" for NLI and
(ii) reframe the verifier + ACR gate as a **fluency/coverage regularizer** whose
job is to prevent the answer-repeating degeneration that pure-NLI optimization
produces (consistent with our "verifier-blindness" finding and the length
analysis above), not to raise NLI. This is a framing/claim correction, not a
change to the method.

## Commitments in the revision
- Report the held-out DeBERTa-v3-large NLI for **all** cells in the main tables (add RoBERTa-large-MNLI as a second independent scorer).
- Add a length-controlled DPO baseline (Park et al. 2024) and run SimPO on the educational domains (β-DPO already ablated: Qwen3 Cardiff β=0.05/0.2/0.5 → 0.15/0.19/0.16).
- Add a small human / step-validity spot-check of reasoning soundness on a sample.
- Analyze the non-improvement cells (Gemma-4 Sydney, Qwen3 Cardiff) explicitly; report multi-seed variance.
- Rewrite for clarity: reconcile additive-vs-multiplicative (Intro vs Eq. 2), fix the 95%/GPT-4o-mini wording, define the "Ver" metric and candidate count N, fix section numbers and the dangling [?,?] citations, reduce em-dashes, and simplify the tables to lead with the main conclusion.

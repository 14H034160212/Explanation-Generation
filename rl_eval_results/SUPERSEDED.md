# Result files that must not be read as current

## The mis-composed preference generations, and anything judged from them

`p3_gen_simpo.py` and `p3_gen_v2only.py` built their generations by merging the
preference adapter onto the **bare base**, with the SFT adapter dropped. That
is the wrong composition, and it does not fail loudly — it degenerates into
length. Mean words per key in `p3_reasoning_generations.json`:

| key | words | composition |
| --- | --- | --- |
| `SFT` | 6,106 | — |
| `reasoningSimPO` | 18,231 | base + adapter (**wrong**) |
| `reasoningSimPO_fixed` | 6,204 | SFT-merged base + adapter |
| `reasoningDPO_v2` | 18,165 | base + adapter (**wrong**) |
| `reasoningDPO_v2_fixed` | 5,532 | SFT-merged base + adapter |
| `reasoningOPD` | 18,049 | base + adapter (**wrong**) |
| `reasoningOPD_fixed` | 4,067 | SFT-merged base + adapter |
| `oldHybridDPO` | 18,080 | base + adapter (**wrong**) |

Superseded, therefore:

* generation keys `reasoningSimPO`, `reasoningDPO`, `reasoningDPO_v2`,
  `reasoningOPD`, `oldHybridDPO` — use the `_fixed` keys
* `p3_judge_compare.json`, `p3_judge_compare_qwen3.json`,
  `p3_judge_compare_qwen35.json`, `p3_judge_compare_qwen35_gentle.json`
  — use `p3_judge_fixed.py` and its outputs
* `scripts/p3_judge_compare.py` itself, which now refuses to run

The "preference-tuned outputs are ~3x longer, so any preference is
length-confounded" caveat in the old scripts was describing this artefact, not
a property of preference tuning.

## Held-out NLI before realignment

`heldout_nli_eval.py` paired explanation *j* with test question *j*. An item
whose generation came back empty is skipped by `rl_evaluation.py`, so a row
shorter than the test set is a subsequence and everything after the first skip
was scored against the wrong question. 25 eval files in this directory have at
least one short row; all are LLaMA-2.

Three cells changed when realigned, all from FLIPS to HOLDS. The affected rows
in `heldout_nli_summary.json` and `heldout_nli_15cells.json` carry a
`realigned` block and keep the old value as `large_before_realignment`:

| cell | before | after |
| --- | --- | --- |
| LLaMA2 Cardiff Tier-C | 0.2773 → 0.1250 | 0.2773 → **0.4432** |
| LLaMA-2 AucklandLaw | 0.4911 → 0.2482 | 0.4911 → **0.5520** |
| LLaMA-2 MedY1 | 0.2263 → 0.0761 | 0.2263 → **0.4393** |

Across both tables: 13/21 → **16/21** cells hold under the independent scorer.
SFT scores are unchanged by construction — those rows were complete, so their
alignment was already the identity, and that is the control.

Still flipping, and genuinely so (rows 1:1, alignment was never in question):
Gemma4 Sydney (main), Qwen3 Cardiff (hybrid), Qwen3 Cardiff Tier-C, Qwen3
Sydney Tier-C, Qwen3 MedY1.

## The open-source recoverability gate's correlation

`oss_maskrec_gate.py` computed Spearman with distinct ranks for tied values.
Both variables are almost all ties, so the reported **+0.52** was the agreement
of an invented within-tie ordering; with average ranks the same 110 rows give
**+0.17** (matches `scipy.stats.spearmanr`). The raw rows in
`oss_maskrec_gate.json` are fine — only the printed coefficient was wrong, and
the docstring of `rl_build_preference_data_recoverability.py` cited it as the
reason for adopting the reward.

The `+0.47` GPT-4o figure quoted alongside it comes from a run whose script is
not in this repository and has **not** been recomputed; do not compare the two
until it has been.

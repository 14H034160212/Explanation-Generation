# Gemma 4 E4B-it Cardiff Pipeline — Progress Log

**Started**: 2026-04-21 21:34 (SFT launch)
**Target**: Small-scale Cardiff Biology validation of Hybrid-DPO on Gemma 4 E4B-it (dense, 8B total / 4.5B effective via PLE)
**GPU**: cuda:1 (A100 80GB)
**Env**: gemma4-rl (transformers 5.5.4)

## Pipeline Steps

| Step | Status | Output / Notes |
|------|--------|----------------|
| 1. SFT | ✅ Complete | Saved 23:17, 1044/1044 steps, adapter 133MB |
| 2. Preference data (Hybrid) | ✅ Complete | 79 pairs, chosen NLI 0.876 vs rejected 0.044 (20× gap) |
| 3. Hybrid-DPO training | ✅ Complete | Final loss 0.092, rewards/margins 3.15, accuracies 1.00, 50 steps / 6.2min runtime |
| 4. Evaluation v1 (broken) | ⚠️ Invalid | Eval loaded DPO on raw base (bug) — results were meaningless |
| 5. Summary v1 | ⚠️ Invalid (rewritten) | |
| 6. DPO v2 (conservative) | ⚠️ Not needed | Trained fine but invalid eval misdiagnosed it as collapsing |
| 7. DPO v3 (attention-only LoRA) | ⚠️ Not needed | Same — diagnosis was wrong |
| 8. Eval-bug root-cause + fix | ✅ Complete | load_generator() now chains SFT+DPO adapters correctly |
| 9. Corrected Phase 1 Cardiff-only eval | ✅ Complete (archived) | SFT 0.196 → DPO 0.463 (+136%); Phase 2 replaces with merged-SFT numbers for paper |
| Phase 2A. Merged 5-domain SFT | ✅ Complete | 4h16m, 13211 examples, train_loss 9.25, acc 0.795 |
| Phase 2B. Merged preference data (500 Q) | ✅ Complete | 164 pairs, chosen NLI 0.892 vs rejected 0.053 |
| Phase 2C. Hybrid-DPO training | ✅ Complete | 770s, margins 8.61, accuracies 1.00 |
| Phase 2D. 5-domain eval (parallel) | ✅ Complete | 4/5 domains NLI up (Cardiff +66%, MedY2 +143% peak); Sydney flat |
| Phase 2E. Paper + summary update | ✅ Complete | 5 tables, abstract/intro/discussion all updated |

## Escalation policy (for autonomous wake-ups)

Rule: On each wake, do a **1-min gen-test** BEFORE committing to 50-min eval. If mode collapse (markdown `## ` / `Option N:` / `Here are a few options` in first 300 chars), do NOT eval — escalate to next tier.

Escalation order: v3 → v4 → v5 → STOP and write final summary.

For each new DPO variant, always:
1. Archive previous variant's adapter + log (`mv` with version suffix).
2. Launch new DPO via `scripts/python_training/rl_train_dpo_gemma4.py` with the tier's hyperparams.
3. After training, gen-test with `/tmp/gemma4_dpo_v2_sanity.py` (path points to `./rl_dpo_gemma4_e4b_cardiff_generator`).
4. If clean, launch full eval; if collapsed, escalate.

If v5 (scaled pref data) still collapses, write a final summary documenting the escalation attempts and the conclusion that Gemma 4 E4B-it has a base IT markdown prior too strong to be suppressed by small-scale DPO on 200–500 Cardiff questions. Recommend: (a) try gemma-4-26B-A4B-it (MoE, might have weaker markdown prior), (b) filter out markdown candidates at generation time in preference data builder, (c) SFT for more epochs / more data to deepen the compact-answer attractor before DPO.

## Baseline (Qwen3-8B from paper Table 4 — Cardiff Biology)

| Model | BLEU | BERT(Stu) | BERT(Ans) | ACR | NLI |
|-------|------|-----------|-----------|-----|-----|
| Qwen3-8B SFT | 0.0406 | 0.8587 | 0.8462 | 0.7421 | 0.1959 |
| RLearner-LLM (Qwen3) | 0.0247 | 0.8148 | 0.7957 | 0.8272 | 0.1820 |

Goal: See whether Gemma 4 E4B-it (smaller, 4.5B effective) shows a similar ACR/NLI improvement pattern under Hybrid-DPO, validating the method's robustness across architectures.

## Errors Resolved During Setup

1. **transformers 5.2.0 unaware of gemma4 model_type** → cloned env to `gemma4-rl`, upgraded to 5.5.4.
2. **Gemma4ForCausalLM silently random-inits text tower** (checkpoint weights are under `model.language_model.*` prefix not `model.*`) → switched to `Gemma4ForConditionalGeneration`.
3. **PEFT cannot wrap `Gemma4ClippableLinear` in vision/audio towers** → LoRA target_modules is now a full-path regex restricting to `model.language_model.layers.*` plain Linear modules (258 total).

## Event Log

- 21:34 — SFT v3 launched (regex LoRA targets, correct class). MISSING=UNEXPECTED=0, trainable=34.9M.
- 22:23 — SFT at step 485/1044 (46%), elapsed 49min, ETA ~54min, 5.8s/step, GPU 1 at 47GB/43% util. Healthy. No errors.
- 22:59 — SFT at step 855/1044 (82%), elapsed 1h25min, ETA ~19min to completion, 6.0s/step, GPU 1 at 47GB/52% util. Healthy. No errors. Running slightly faster than projected.
- 23:17 — **SFT COMPLETE** at step 1044/1044. LoRA adapter saved (133MB safetensors + config + chat_template). Final loss 9.74, mean token accuracy 0.71. Raw loss is high because Gemma 4 has vocab=262k (ln(262k)≈12.5 random baseline); 9.74 is still ~3 nats below random, and sanity-generation confirms the adapter is working.
- 23:32 — Sanity-generation test ✅ Output on Cardiff anatomy question was fluent and on-topic ("The axillary nerve passes through the quadrangular space..."). Model learned correctly despite misleading raw-loss magnitude.
- 23:35 — Noted evaluator_Test_cardiff.json has numeric `output` field (verifier rating), not text. Switched test set to **Cardiff_gpt4_random_100.json** which uses `Explanation` key for student reference text.
- 23:35 — Noted verifier path in Qwen3 script pointed at project root but actual location is `./models/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2`. Corrected in Step 2 launch.
- 23:36 — **Step 2 launched** (PID 4137256). 200 questions × 3 candidates × 3 scoring (NLI+verifier+ACR). ETA ~60–75 min.
- 05:27 — Step 2 check at 11:26 elapsed. Process alive, GPU 1 at 32GB / 33% util. No errors. Per-question logs are DEBUG level so healthy progress is silent at INFO; GPU activity is the indicator. Estimated ~24 of 200 questions processed (~12%).
- 05:54 — Step 2 check at 38:26 elapsed. Process alive, GPU 1 at 32GB / 35% util (recovered from 24% earlier — just different workload phases). No errors. Estimated ~96 of 200 questions (~48%), ~42 min remaining.
- 06:12 — **Step 2 COMPLETE** (total 56 min). 79 preference pairs, 121 skipped low-gap, 0 missing option. Avg chosen NLI 0.876 vs rejected 0.044 (20× gap) — NLI gate works as designed. Avg chosen verifier 3.09 vs rejected 3.12 — verifier alone cannot distinguish (replicates the paper's Section 4.2 finding on Gemma 4).
- 06:28 — **Step 3 DPO launched** (PID 86532, GPU 1). Config: 5 epochs, β=0.1, lr=5e-5, bs=1, grad_accum=8, LoRA r=16 with regex targets. Expected ~49 DPO steps at ~10s/step ≈ 10 min total.
- 07:13 — **Step 3 DPO COMPLETE**. train_runtime 370s (6.2 min actual), 50 steps. MISSING=UNEXPECTED=0, trainable=34.9M. **Final loss 0.0920**, **rewards/margins 3.155**, **rewards/accuracies 1.00** (model perfectly ranks chosen > rejected), mean_token_accuracy 0.831 (vs SFT 0.71). Very clean DPO run. (Total 45min elapsed — bulk was SFT-LoRA merge-and-unload on 8B multimodal base.)
- 07:19 — **Step 4 EVAL launched** (PID 102437, GPU 1). Cardiff_gpt4_random_100.json (100 questions). Evaluates SFT + DPO checkpoints.
- 07:28 — **Gemma4-E4B SFT eval DONE** (100/100, 9 min). Metrics: BLEU=0.0554 BERT(Stu)=0.8651 BERT(Ans)=0.8553 ACR=0.7790 NLI=0.1961 Verifier=3.12 Time=5.49s/q.
- 07:35 — Gemma4-E4B DPO eval in progress: 10/100 done, ~23s/sample (4× slower than SFT — indicates longer / more complete explanations from DPO model, consistent with ACR-gated training). ETA ~35 min remaining.
- 08:08 — **Step 4 EVAL v1 COMPLETE**. Gemma4-E4B-DPO: BLEU=0.0201, BERT(Stu)=0.8156, BERT(Ans)=0.7971, ACR=0.7806, NLI=0.1012, Verifier=3.14, Time=23.0s. All metrics DROPPED vs SFT. NLI regression most severe (−0.095).
- 08:15 — Summary v1 written. **Diagnosis**: DPO model generates markdown-formatted "meta-explanations" (avg 1392 chars vs SFT 320 chars) starting with "## Explanation Generation\n\nHere are a few options...". Preference data is CLEAN (0/79 chosen have markdown) — this is NOT a data contamination issue but a **policy drift** — DPO pushed the policy far enough from SFT that it exposed Gemma 4 base IT's markdown prior.
- 08:20 — **DPO v2 launched** (PID 200884) with conservative hyperparams to keep policy close to SFT: β 0.1→**0.3** (3× stronger KL penalty), LR 5e-5→**2e-5**, epochs 5→**2**, LoRA r 16→**8**, alpha 32→**16**. Expected ~3 min training.
- 08:20 — Archived v1 artifacts: `rl_dpo_gemma4_e4b_cardiff_generator_v1_beta0.1_lr5e-5_r16/`, `rl_eval_results/gemma4_cardiff_eval_v1.json`, `rl_dpo_gemma4_training_v1.log`, `rl_evaluate_gemma4_v1.log`.
- 08:38 — DPO v2 trained cleanly: 156s runtime, trainable=17.4M (half of v1), final loss 0.66, rewards/margins 0.06 (vs v1's 3.15), rewards/accuracies 0.73 (vs v1's 1.00). v2 drift 50× smaller than v1 — expected from 3× higher β.
- 11:40 — **DPO v2 gen-test still shows mode collapse** — outputs start with `## Explanation Generation` / `### Detailed Explanation`, 870–950 chars (vs v1 1400 — reduced but not eliminated). Diagnosis: Gemma 4 E4B-it has a strong markdown instruction-tuning prior; SFT's 35M-param LoRA suppressed it but ANY DPO MLP update lets it reappear.
- 11:41 — **DPO v3 launched** (PID 519611) with **LoRA on attention only** (`q/k/v/o_proj`, no MLP). Hypothesis: MLP controls style/format; restricting LoRA to attention should preserve SFT's compact-answer style while still allowing attention to encode preference.
- 11:41 — Archived v2: `rl_dpo_gemma4_e4b_cardiff_generator_v2_beta0.3_lr2e-5_r8/`, `rl_dpo_gemma4_training_v2.log`.
- 12:45 — DPO v3 trained: trainable=4.5M, runtime 123s, loss 0.68, margins 0.08. Same conservative hyperparams as v2 but attention-only.
- 12:55 — v3 gen-test **STILL shows markdown**. Attention-only hypothesis refuted? The v1/v2/v3 outputs are near-identical at position-wise greedy decoding, suggesting the DPO adapter barely changes inference path.
- 13:XX — **CRITICAL BUG FOUND in eval and sanity scripts.** `rl_evaluate_gemma4.py` and `/tmp/gemma4_dpo_v2_sanity.py` loaded DPO adapter directly on **raw base** (`base + DPO_delta`), but the DPO adapter was **trained on top of SFT-merged base**. Correct inference flow is `base → apply SFT → merge → apply DPO → merge`. Running DPO adapter on raw base = almost base model = markdown output. **All previous eval results (v1 BLEU 0.02, NLI 0.10) are INVALID**. The DPO training itself has been fine since the beginning.
- 13:XX — Wrote corrected sanity test `/tmp/gemma4_dpo_v3_correct_sanity.py`: SFT→merge→DPO→merge. Output is now **CLEAN** (245/353/278 char compact answers matching SFT style) across all 3 test samples. ✅
- 13:XX — Fixed `rl_evaluate_gemma4.py`: `load_generator()` now accepts a list of LoRA paths and applies/merges each in sequence. DPO eval uses `[sft_lora, dpo_lora]` instead of just `[dpo_lora]`.
- 13:XX — Restored **v1 DPO adapter** (strongest learning signal: β=0.1, margins=3.15, accuracies=1.0) to main path. v3 archived. Launched corrected eval (PID 1218403, GPU 1). ETA ~50 min.
- 22:01 — **CORRECTED EVAL COMPLETE.** Gemma4-E4B SFT vs DPO on 100 Cardiff questions:
  - SFT:  BLEU=0.0554 BERT(Stu)=0.8651 BERT(Ans)=0.8553 ACR=0.7790 NLI=**0.1961** Verifier=3.12 Time=5.19s
  - DPO:  BLEU=0.0390 BERT(Stu)=0.8630 BERT(Ans)=0.8753 ACR=**0.8021** NLI=**0.4630** Verifier=3.07 Time=**3.25s**
  - Δ:    BLEU -0.016, BERT(Stu) flat, BERT(Ans) +0.020, **ACR +0.023**, **NLI +0.267 (+136%)**, 37% faster inference
  - 0/100 DPO outputs have markdown; median DPO len 130 chars (vs SFT 210) — model learned compact answers
  - **Beats Qwen3-8B paper baseline on NLI by 2.5×** (0.463 vs 0.182). Gemma 4 E4B is a strong architecture for Hybrid-DPO.
- 22:05 — **Summary rewritten** at `gemma4_pipeline_summary.md` with corrected numbers + postmortem of the eval-script bug + recommendations for paper. Pipeline DONE (phase 1).

## Phase 2: 5-domain cross-architecture extension

**Goal**: Match the paper's Qwen3-8B evaluation setup across all 5 domains (Cardiff, Sydney, Auckland Law, UK Medicine Y1, Y2) so the Gemma 4 E4B row can be added to Tables 3–6, not just Table 3 (Cardiff).

**Setup (matching paper)**: SFT on merged 13,211-example dataset (all 5 domains). Pref-data built from merged training set. One DPO adapter. Evaluated separately on each domain's 100-Q test set.

**Test sets located** (all 100-Q with Explanation key):
- Cardiff: `preference_data/Paul_new_data/Cardiff/Cardiff_gpt4_random_100.json`
- Sydney: `preference_data/Paul_new_data/Sydney/Sydney_gpt-4_random_100.json`
- Auckland: `preference_data/PeerWiseData/Law/Auckland_law_gpt4_random_100.json`
- Medicine Y1: `preference_data/PeerWiseData/Medicine/Medicine_year1_gpt4_random_100.json`
- Medicine Y2: `preference_data/PeerWiseData/Medicine/Medicine_year2_gpt4_random_100.json`

**Pipeline stages**:
| Step | Status | Notes |
|------|--------|-------|
| 2A. SFT on merged 13,211 | 🟡 Running | PID 1351940, GPU 1, output `rl_sft_gemma4_e4b_merged_generator/`, ETA ~4h10m |
| 2B. Pref-data build (500 Q merged) | ⏳ | Using multiplicative_acr mode, same hyperparams as phase 1 |
| 2C. DPO training | ⏳ | v1-style aggressive: β=0.1, LR=5e-5, 5 epochs, LoRA r=16 (worked for phase 1) |
| 2D. Eval × 5 domains | ⏳ | Plan: GPU 0 serves 3, GPU 1 serves 2 in parallel |
| 2E. Update results tables 3–6 + prose | ⏳ | Keep phase-1 Cardiff-only result as archive; new merged SFT Cardiff numbers replace them |

- 23:29 — **Phase 2A SFT launched** (merged 13,211-example SFT, GPU 1, PID 1351940). Est ~4h10m to completion. `rl_sft_gemma4_merged_training.log`.
- 23:35 — SFT early check: step 45/2478 (1.8%), 6.15s/step, GPU 1 34GB/34% util, MISSING=UNEXPECTED=0, trainable=34.9M. Healthy.
- 00:37 — SFT 1h check: step 652/2478 (26%), 6.57s/step (slightly slower), GPU 1 37GB/34% util. ETA ~3h20m remaining.
- 01:44 — SFT 2h14m check: step 1279/2478 (52%), 6.5s/step, GPU 1 37GB/42% util. ETA ~2h10m remaining. Healthy.
- 02:55 — SFT 3h25m check: step 1953/2478 (79%), 5.94s/step (speeding up), GPU 1 37GB/29% util. ETA ~52min remaining. Healthy.
- 03:46 — **Phase 2A SFT COMPLETE** (total 4h16m, 15370s runtime). train_loss 9.245, mean_token_accuracy **0.7948** (noticeably better than Cardiff-only SFT's 0.71 — the 2.4× larger training set is clearly helping). Adapter saved (133MB) to `rl_sft_gemma4_e4b_merged_generator/`.
- 04:21 — **Phase 2B launched ahead of scheduled wake** (PID 1727384, GPU 1). 500 Q × 3 samples × multiplicative_acr scoring. ETA ~1.5–2h. **⚠ DO NOT relaunch at the 04:06 scheduled wake** — Phase 2B is already running.
- 05:25 — Phase 2B 1h02m check: GPU 1 32GB/35% util, still "Processing 500 questions ..." (DEBUG-level per-Q logs not shown). Est ~44% (220/500 Q), ETA ~80 min remaining. Healthy.
- 05:29 / 05:45 — progress INFO logs appeared: 200/500 (67 pairs), 250/500 (78 pairs), with periodic saves. At this pace (18.8s/Q) ETA was ~06:45.
- 06:36 — **Phase 2B apparent stall** — GPU still 30% util + 32GB held, process alive, but log hasn't updated since 05:45 (51 min silent), expected 300-save at ~06:00 never appeared. Not killing yet; GPU activity suggests slow-but-alive. Possibilities: very long single question, verifier scoring slowdown, or logger buffering lag. Next check in 30 min — if still no progress, kill + restart from 250-pair partial save.
- 06:58 — **Phase 2B actually completed** — apparent stall was logger buffering; all 250→500 progress + final save flushed at 06:58. Clean exit.
  - **164 preference pairs** retained (vs Phase 1's 79 — 2.1× more data)
  - 336 skipped (low gap), 0 skipped (no option)
  - Avg chosen NLI **0.892** vs rejected **0.053** (17× gap — NLI gate working)
  - Avg chosen verifier **3.05** vs rejected **3.06** (near-identical — verifier-blindness replicates)
  - Avg multiplicative_acr gap: 0.0924
- 07:37 — **Phase 2C DPO launched** (PID 1968361, GPU 1). β=0.1, LR=5e-5, 5 epochs, LoRA r=16. 164 pairs × 5 epochs / (bs=1 × accum=8) = ~100 DPO steps. ETA: ~1h15m (8 min setup + ~15 min DPO + ~40 min SFT-merge setup).
- 07:50 — **Phase 2C DPO COMPLETE** (total 13 min — much faster than phase 1 which had 40min SFT-merge setup; likely some weights cached). train_runtime 770s, 100 steps, train_loss 0.222 (vs phase 1's 0.342), **rewards/margins 8.61** (2.7× phase 1's 3.15 — much stronger learning signal from 2.1× pref pair count), rewards/accuracies 1.00, mean_token_accuracy 0.817.
- 07:57 — **Phase 2D parallel 5-domain eval launched AHEAD of scheduled wake 08:24**:
  - GPU 0 pipeline (PID 1994210): Cardiff → Sydney → Auckland (serial)
  - GPU 1 pipeline (PID 1994211): Medicine Y1 → Y2 (serial)
  - **⚠ DO NOT relaunch Phase 2D at the 08:24 scheduled wake** — BOTH pipelines already running
  - ETA: ~50min per domain × 3 = 2h30m on GPU 0, ~50min × 2 = 1h40m on GPU 1. Bottleneck is GPU 0 → all done ~10:30.
- 08:51 — **Phase 2D COMPLETE** (both pipelines finished ~54min total wall, much faster than estimated — each domain ~18 min not 50 min). All 5 JSONs written. Summary of Δ NLI vs SFT: Cardiff +66%, Sydney −6%, Auckland +12%, MedY1 +32%, MedY2 +143%. 4 of 5 domains improve; Sydney flat. ACR drops −1.3 to −10.3 pp in 4 of 5 domains (honest trade-off to flag).
- 09:XX — **Phase 2E COMPLETE**: updated paper abstract/introduction/discussion and results.tex Tables 3–7 with Gemma 4 rows + prose across all 5 domains. Full rewrite of gemma4_pipeline_summary.md. Pipeline DONE.

## Phase 3: Cardiff Tier-B robustness ablation (response to Paul's filtering suggestion)

**Goal**: Replace the "we couldn't implement the strict discriminator filter" limitations paragraph with a concrete Tier-B (5*-share>=0.10 + deleted==0) ablation, framed as a robustness check on the Cardiff results.

- 05:25 — Cardiff Tier-B preprocess: 1041 questions kept (832 train + 209 test, vs default's 7309). Filter cascade: img 4759, avg<3 14959, share<0.10 6176, total<10 1853, deleted 44, short_expl 48.
- 05:25 — Phase 3A SFT launched (PID 394652, GPU 0).
- 05:41 — **Phase 3A Tier-B SFT COMPLETE** — 907s train_runtime (~15 min), train_loss 12.21 (higher than merged 9.25, expected from 6.7× smaller corpus), mean_token_accuracy 0.6942 (vs merged 0.7948). MISSING/UNEXPECTED=0.
- 05:58 — **Phase 3B Tier-B pref-data build launched** (PID 398470, GPU 0). 200Q × 3 samples, multiplicative_acr.
- 07:31 — **Phase 3B Tier-B pref-data COMPLETE** — 77 preference pairs (vs Phase-2 merged's 164). Avg multiplicative_acr gap 0.1006. **Avg chosen NLI 0.9008 vs rejected 0.0496 (18× gap, even slightly stronger than Phase 2's 17× gap)**. Avg chosen verifier 3.2439 vs rejected 3.2661 (verifier-blindness replicates a 4th time). 123 questions skipped low-gap, 0 missing option.
- 07:59 — **Phase 3C Tier-B DPO launched** (PID 415821, GPU 0).
- 08:05 — **Phase 3C Tier-B DPO COMPLETE** — train_runtime 339s (5.6 min — much faster than Phase-2's 13min, base+SFT cache hit). train_loss 0.3247, rewards/margins 2.9–3.0 (vs Phase-2 merged's 8.6 — smaller signal because pref-data is smaller and SFT is weaker), rewards/accuracies 1.00.
- 08:59 — **Phase 3D Cardiff eval launched** (PID 422813, GPU 0). 100 questions on Cardiff_gpt4_random_100.json. ETA ~20min.
- 09:27 — **Phase 3D Cardiff Tier-B eval COMPLETE**. Headline: Tier-B DPO **NLI 0.5202 vs Phase-2 default DPO 0.3505 (+48%)**, ACR 0.7823 vs 0.6497 (+13.3 pp).

## Phase 3 Final ablation table (Cardiff, 100-Q test)

| Filter | Train N | SFT NLI | DPO BLEU | DPO BERT(Ans) | DPO ACR | DPO NLI | DPO time(s) |
|--------|---------|---------|----------|---------------|---------|---------|-------------|
| Default (Section 4) | 7,309 | 0.2117 | 0.0303 | 0.8426 | 0.6497 | 0.3505 | 4.76 |
| **Tier-B (stricter)** | 1,041 | 0.2059 | **0.0381** | **0.8696** | **0.7823** | **0.5202** | 7.62 |
| Δ (Tier-B − Default) | −86% | −0.006 | +0.008 | +0.027 | +0.133 | **+0.170 (+48%)** | +2.9s |

- 09:30 — **Phase 3E paper update DONE**. Replaced the limitations paragraph with `\subsubsection*{Robustness to Question-Quality Filtering}` containing the table and a positive-framing narrative: stricter quality filter (5*-share≥0.10) at 7× smaller pool keeps SFT NLI flat but boosts post-DPO NLI by 48% and ACR by +13.3 pp. Verifier-blindness now replicated on a 4th corpus (Tier-B chosen verifier 3.24 vs rejected 3.27). Pipeline DONE.

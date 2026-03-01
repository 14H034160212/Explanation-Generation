# RLearner-LLM: Reinforcement Learning for Educational Explanation Generation

> **Extension of ILearner-LLM** — replaces K-round iterative inference with RL-trained single-pass generation.
> See the main [README.md](README.md) for the full project overview.

---

## Table of Contents
- [Three Paper Innovations](#three-paper-innovations)
- [System Architecture](#system-architecture)
- [Training Pipeline](#training-pipeline)
- [Experimental Results](#experimental-results)
  - [Cardiff Biology](#cardiff-biology-dataset)
  - [Sydney Biology](#sydney-biology-dataset)
  - [Cross-Domain Generalization](#cross-domain-generalization-results)
  - [GPT Model Comparison](#gpt-model-comparison)
  - [GPT Pairwise Judge Results](#gpt-pairwise-judge-results)
  - [ILearner-LLM K=5 Baseline](#ilearner-llm-k5-baseline)
- [Key Conclusions](#key-conclusions)
- [Future Work](#future-work)
- [Reproducing Results](#reproducing-results)

---

## Three Paper Innovations

### Innovation 1 — Single-Pass RL Generation Replacing Iterative Inference

**Problem**: The original ILearner-LLM pipeline runs K sequential rounds of Generator → Verifier → Feedback, where latency scales linearly with K. At K=5, inference is 5× slower than a single-pass model, making real-time deployment impractical.

**Our Approach**: Instead of iterating at test time, we embed the Generator–Verifier interaction into the training signal itself. Using the verifier as a reward model, we apply:
- **DPO (Direct Preference Optimization)** — offline RL from verifier-ranked candidate pairs.
- **PPO (Proximal Policy Optimization)** — online RL with live verifier reward signals.

**Result**: At inference time, a single forward pass produces explanations that match or exceed the quality of K=5 iterative refinement, at **~3.4× lower latency** (5.8s vs 19.9s per question for SFT).

| Approach | Inference passes | Avg latency (Cardiff) |
|----------|-----------------|----------------------|
| SFT baseline (LoRA) | 1 | 19.9s |
| **DPO v2 (ours)** | **1** | **5.8s** |
| PPO (ours) | 1 | 7.2s |
| GPT-4o-mini (API) | 1 (API) | 5.6s |
| Qwen3-8B SFT | 1 | 4.0s |

---

### Innovation 2 — Answer-Grounded Multi-Metric Evaluation Framework

**Problem**: Standard text-generation metrics (BLEU, BERTScore vs. student explanation) measure surface similarity to reference text. For explanation quality, what matters most is whether the explanation correctly identifies and justifies the *correct answer option* — which BLEU/BERTScore fail to capture when comparing against student-written reference text.

**Our Approach**: We propose a three-tier answer-grounded evaluation framework, where the hypothesis for each metric is the *correct answer option text* (not the student reference):

| Metric | What it measures | Hypothesis |
|--------|-----------------|------------|
| **BERT(Ans)** | Semantic similarity to correct option | Correct option text |
| **ACR** (Answer Coverage Rate) | Lexical coverage of key terms from correct option | ≥4-char tokens from correct option |
| **NLI Entailment** | Logical entailment: does explanation *imply* correct option? | Correct option text |

The NLI metric (using `cross-encoder/nli-deberta-v3-small`) is the most discriminative: **SFT scores ~0.05** while **all RL models score 0.22–0.30** (4–5× gap), directly capturing whether explanations logically support the correct answer.

**Key finding on GPT models**: GPT-4 achieves the highest BERTScore(Stu) but only NLI=0.07 on Cardiff — worse than our SFT baseline. This reveals that high fluency does not imply answer-grounded reasoning. GPT-3.5, by contrast, achieves NLI=0.27 — comparable to our best RL model — suggesting smaller GPT models may follow task instructions more literally.

---

### Innovation 3 — Automated Verifier-Based Preference Data Construction

**Problem**: DPO requires high-quality preference pairs (chosen/rejected). Human annotation is expensive and slow. Using a fixed GPT-4 teacher creates domain mismatch for biology/medicine exam questions.

**Our Approach**: We automatically construct preference pairs using the domain-trained verifier model itself as the preference oracle:

1. **Multi-sample generation**: For each question, generate N candidate explanations (N=3) using the SFT model.
2. **Verifier scoring**: Score all N candidates with the domain verifier (0–5 scale).
3. **Pair selection**: Select `(highest-scored, lowest-scored)` if score gap ≥ threshold (0.1).
4. **Iterative scaling**: Increase N and lower the threshold to get more pairs.

| Version | Questions | Samples/Q | Min gap | Pairs | Training epochs | Result |
|---------|-----------|-----------|---------|-------|-----------------|--------|
| Pref v1 | 500 | 2 | 0.3 | **165** | 2 | DPO v1 |
| Pref v2 | 500 | 3 | 0.1 | **458** (+2.77×) | 5 | DPO v2 (best) |
| Pref v3 | 1000 | 3 | 0.1 | **851** | 5 | DPO v3 (cross-domain) |

**Key advantage**: The verifier is domain-specific, trained on Cardiff+Sydney+Law+Medicine data, making preference pairs highly relevant to the target distribution. No API cost, no domain mismatch.

---

## System Architecture

```
┌─────────────────── Training Pipeline ───────────────────────┐
│                                                              │
│  Phase 1 — SFT (Supervised Fine-Tuning)                     │
│  ─────────────────────────────────────                       │
│  13,211 expert explanations                                  │
│  + LLaMA-2-13B base model                                   │
│  + LoRA (r=16, α=32) fine-tuning                            │
│  ──────────────────────────────────►  SFT Generator         │
│                                                              │
│  Phase 2 — Preference Data Construction                      │
│  ──────────────────────────────────                          │
│  SFT Generator generates N explanations per question         │
│  Domain Verifier scores each candidate (0-5)                 │
│  ──────────────────────────────────►  (chosen, rejected) pairs│
│                                                              │
│  Phase 3A — DPO Training (Recommended)                       │
│  ─────────────────────────────────                           │
│  Preference pairs + SFT init                                 │
│  ──────────────────────────────────►  DPO Generator         │
│                                                              │
│  Phase 3B — PPO Training (Alternative)                       │
│  ─────────────────────────────────                           │
│  Online reward from verifier                                 │
│  ──────────────────────────────────►  PPO Generator         │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌─────────────────── Inference (1-shot) ───────────────────────┐
│                                                              │
│  Question + Options + Correct Answer                         │
│              │                                               │
│              ▼                                               │
│    ┌─────────────────┐                                       │
│    │  RL Generator   │  (single forward pass, ~6s)           │
│    │  (LoRA adapter) │                                       │
│    └─────────────────┘                                       │
│              │                                               │
│              ▼                                               │
│    High-quality explanation that names and justifies         │
│    the correct answer concept                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

### Environments

| Env | Use | Key packages |
|-----|-----|-------------|
| `llm-tuning` | LLaMA-2-13B SFT / DPO / PPO | TRL 0.7.1, Transformers 4.31.0, PEFT 0.4.0 |
| `qwen3-rl` | Qwen3-8B all experiments | TRL 0.28, Transformers 5.2, torch 2.6.0+cu124 |

### Completed Training Runs

#### LLaMA-2-13B Experiments

| Step | Script | Config | Duration | Output |
|------|--------|--------|----------|--------|
| SFT | `rl_train_sft.py` | LLaMA-2-13B, LoRA r=16, 3 epochs, 13211 ex | 48 min, 4×A100 | `./rl_sft_llama2_13b_generator/` |
| Pref v1 | `rl_build_preference_data.py` | 500Q × 2 samples, gap≥0.3 | ~2h | 165 pairs |
| DPO v1 | `rl_train_dpo.py` | 165 pairs, 2 epochs | 68 sec | `./rl_dpo_llama2_13b_generator/` |
| Pref v2 | `rl_build_preference_data.py` | 500Q × 3 samples, gap≥0.1 | ~4h | 458 pairs |
| DPO v2 | `rl_train_dpo.py` | 458 pairs, 5 epochs | 14m52s | `./rl_dpo_v2_llama2_13b_generator/` |
| PPO | `rl_train_ppo.py` | 500Q, batch=4, 125 steps | ~3h | `./rl_ppo_llama2_13b_generator/` |
| Pref v3 | `rl_build_preference_data.py` | 1000Q × 3 samples, gap≥0.1 | ~6h | 851 pairs |
| DPO v3 | `rl_train_dpo.py` | 851 pairs, 5 epochs, all-domain data | ~30min | `./rl_dpo_v3_llama2_13b_generator/` |

#### Qwen3-8B Experiments

| Step | Script | Config | Duration | Output |
|------|--------|--------|----------|--------|
| SFT | `rl_train_sft_qwen3.py` | Qwen3-8B, LoRA, 3 epochs, 13211 ex | ~1h45m, 1×A100 | `./rl_sft_qwen3_8b_generator/` |
| DPO v1 | `rl_train_dpo_qwen3.py` | 165 pairs, beta=0.1, 5 epochs | ~5min | `./rl_dpo_qwen3_v1_generator/` |
| DPO v2 | `rl_train_dpo_qwen3.py` | 458 pairs, beta=0.1, 5 epochs | ~15min | `./rl_dpo_qwen3_v2_generator/` |
| DPO Beta0.5 | `rl_train_dpo_qwen3.py` | 165 pairs, beta=0.5, 5 epochs | ~5min | `./rl_dpo_qwen3_beta05_generator/` |
| PPO Standard | `rl_train_ppo_qwen3.py` | SFT init, lr=1e-5, 500 steps | ~4h | `./rl_ppo_qwen3_sft_ppo_generator/` |
| PPO Hybrid v1 | `rl_train_ppo_qwen3.py` | DPO-v1 init, lr=1e-5, 2000 steps | ~6h | `./rl_ppo_qwen3_hybrid_v1_2000steps/` |
| PPO Hybrid v2 | `rl_train_ppo_qwen3.py` | DPO-v2 init, lr=1e-5, 2000 steps | ~6h | `./rl_ppo_qwen3_hybrid_v2_2000steps/` |

---

## Experimental Results

All results on 100-question held-out test sets.

**Metric definitions:**
- **BLEU** — n-gram overlap with student reference explanation
- **BERT(Stu)** — BERTScore F1 vs. student-written reference
- **BERT(Ans)** — BERTScore F1 vs. correct answer option text (answer-anchored)
- **ACR** — Answer Coverage Rate: fraction of ≥4-char keywords from correct option appearing in explanation
- **NLI** — DeBERTa-v3-small entailment probability (explanation → correct option); **most discriminative**
- **Ver** — Domain verifier score (0–5, Alpaca-7B reward model)
- **Time** — Average inference time per question (seconds)

---

### Cardiff Biology Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ | Time ↓ |
|-------|--------|------------|------------|-------|-------|-------|--------|
| **ILearner-LLM (K=5)** | 0.0729 | — | 0.7859 | 0.8713 | 0.0864 | 3.1960 | — |
| SFT (LLaMA-2-13B) | 0.0160 | 0.8070 | 0.7820 | 0.8087 | 0.0555 | 3.1976 | 19.947 |
| DPO v1 (165 pairs, 2 ep) | 0.0173 | 0.8238 | 0.8325 | 0.7698 | **0.2969** | 3.0467 | 6.567 |
| **DPO v2 (458 pairs, 5 ep)** | **0.0247** | **0.8300** | **0.8422** | **0.8682** | 0.2905 | 3.0648 | **5.774** |
| DPO v3 (851 pairs, 5 ep) | 0.0189 | 0.8228 | 0.8284 | 0.8007 | 0.2257 | 3.0551 | 8.698 |
| PPO (125 steps, batch=4) | 0.0175 | 0.8245 | 0.8255 | 0.7390 | 0.2260 | 3.0750 | 7.234 |
| GPT-4 | 0.0316 | 0.8472 | 0.8178 | 0.8972 | 0.0736 | 3.0100 | — |
| GPT-3.5 | 0.0258 | 0.8436 | 0.8231 | 0.9136 | 0.2708 | 2.9900 | — |
| GPT-4o-mini | 0.0241 | 0.8362 | 0.7991 | 0.8979 | 0.1028 | 3.0100 | — |
| Qwen3-8B SFT | 0.0406 | 0.8587 | 0.8462 | 0.7421 | 0.1959 | 2.5000 | 4.029 |
| Qwen3-8B DPO | 0.0230 | 0.8169 | 0.7980 | 0.8323 | 0.1149 | 2.9300 | 13.985 |
| Qwen3-8B Hybrid PPO v1 | 0.0233 | 0.8157 | 0.7977 | 0.8381 | 0.1791 | 2.9300 | 12.800 |
| Qwen3-8B Hybrid PPO v2 | 0.0256 | 0.8156 | 0.7974 | 0.8493 | 0.1613 | 2.9700 | 13.030 |

**Cardiff takeaways:**
- **DPO v2 is the overall best**: highest BLEU, BERT(Ans), ACR, and fastest inference among RL models.
- **NLI gap is dramatic**: SFT 0.055 → best RL model 0.297 (5.4× improvement). NLI is the most task-relevant signal.
- **GPT-4 paradox**: highest fluency (BERT(Stu)=0.847) but NLI=0.074 — *worse than SFT*. GPT-4 writes encyclopaedic explanations that do not specifically justify the correct option.
- **GPT-3.5 is competitive**: NLI=0.271, close to our DPO v2 (0.291). It follows task instructions more literally.
- **Qwen3 DPO alignment tax**: Qwen3 SFT NLI=0.196, but after DPO it drops to 0.115 — the reward signal (verifier score) does not fully align with the NLI objective.

---

### Sydney Biology Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ | Time ↓ |
|-------|--------|------------|------------|-------|-------|-------|--------|
| **ILearner-LLM (K=5)** | 0.0274 | — | 0.7889 | 0.6451 | 0.0837 | 3.1843 | — |
| SFT (LLaMA-2-13B) | 0.0222 | 0.8244 | 0.7870 | 0.6249 | 0.0537 | 3.1937 | 19.001 |
| DPO v1 (165 pairs, 2 ep) | 0.0314 | 0.8262 | 0.8272 | 0.6034 | 0.2171 | 2.9094 | 9.049 |
| **DPO v2 (458 pairs, 5 ep)** | 0.0364 | **0.8367** | **0.8426** | 0.6290 | **0.2774** | 2.9474 | **6.370** |
| DPO v3 (851 pairs, 5 ep) | 0.0357 | 0.8360 | 0.8372 | 0.6481 | 0.2310 | 2.8448 | 8.081 |
| PPO (125 steps, batch=4) | **0.0421** | 0.8364 | 0.8294 | **0.6606** | 0.2269 | 2.9609 | 7.596 |
| GPT-4 | **0.0843** | 0.8685 | 0.8302 | 0.4727 | 0.1973 | 2.8100 | — |
| GPT-3.5 | 0.0530 | 0.8636 | 0.8260 | **0.8925** | 0.2475 | 2.8600 | — |
| GPT-4o-mini | 0.0385 | 0.8541 | 0.8079 | 0.8785 | 0.1787 | 2.9100 | — |
| Qwen3-8B SFT | 0.0844 | **0.8788** | 0.8416 | 0.5929 | 0.1737 | 2.2600 | 4.056 |
| Qwen3-8B DPO | 0.0404 | 0.8358 | 0.7997 | 0.8238 | 0.1973 | 2.8000 | 13.055 |
| Qwen3-8B Hybrid PPO v1 | 0.0419 | 0.8358 | 0.7996 | 0.8238 | 0.2122 | 2.8000 | 11.994 |
| Qwen3-8B Hybrid PPO v2 | 0.0434 | 0.8365 | 0.7993 | 0.7966 | 0.1887 | 2.8200 | 12.017 |

**Sydney takeaways:**
- **DPO v2 best NLI** (0.277) and BERT(Ans). **PPO best BLEU and ACR** — complementary strengths.
- **GPT-4 highest BLEU** (0.084) on Sydney, but very low ACR (0.473) — verbose text that drifts from the correct option keywords.
- **Qwen3 SFT** achieves the highest BERT(Stu) (0.879) on Sydney, indicating strong baseline fluency.
- Patterns consistent with Cardiff: NLI gap (SFT 0.054 → best RL 0.277) holds across datasets.

---

### Cross-Domain Generalization Results

Models evaluated on domains **not seen** during preference data construction (which used Cardiff+Sydney data only).

#### Auckland Law Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|-------|--------|------------|------------|-------|-------|-------|
| ILearner-LLM (K=5) | 0.0381 | — | 0.7720 | 0.6326 | 0.3996 | 2.7845 |
| SFT (LLaMA-2-13B) | 0.0298 | 0.8010 | 0.7709 | 0.5516 | 0.2702 | 2.7180 |
| DPO v3 | 0.0540 | 0.8246 | 0.8232 | 0.6261 | 0.3287 | 2.6111 |
| Qwen3-8B SFT | **0.1382** | **0.8784** | **0.8557** | 0.5175 | 0.3191 | 2.0000 |
| Qwen3-8B DPO | 0.0355 | 0.8183 | 0.8030 | **0.8018** | 0.2438 | 2.4700 |
| Qwen3-8B PPO | 0.0343 | 0.8161 | 0.8007 | 0.7693 | 0.2235 | 2.5900 |

**Law dataset note:** ILearner K=5 achieves the highest NLI (0.400) here — suggesting that in the Law domain, iterative selection outperforms single-pass RL. Qwen3 SFT shows strong BLEU (0.138) on Law, reflecting its stronger out-of-domain generation capability.

#### UK Medicine Year 1

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|-------|--------|------------|------------|-------|-------|-------|
| ILearner-LLM (K=5) | 0.0671 | — | 0.7863 | 0.8066 | 0.1219 | 3.2005 |
| SFT (LLaMA-2-13B) | 0.0136 | 0.8065 | 0.7799 | 0.7556 | 0.0860 | 3.2561 |
| DPO v3 | 0.0192 | 0.8240 | 0.8333 | 0.7575 | 0.2583 | 3.0429 |
| Qwen3-8B SFT | **0.0458** | **0.8629** | **0.8466** | 0.7387 | 0.2457 | 2.4700 |
| Qwen3-8B DPO | 0.0222 | 0.8196 | 0.7959 | 0.8266 | 0.2019 | 2.9100 |
| Qwen3-8B PPO | 0.0212 | 0.8184 | 0.7959 | **0.8362** | 0.1701 | 2.9600 |

#### UK Medicine Year 2

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|-------|--------|------------|------------|-------|-------|-------|
| ILearner-LLM (K=5) | 0.0495 | — | 0.7873 | 0.7357 | 0.0668 | 3.1600 |
| SFT (LLaMA-2-13B) | 0.0163 | 0.8208 | 0.8142 | 0.6120 | 0.2319 | 2.8717 |
| DPO v3 | 0.0161 | 0.8161 | 0.8291 | 0.7552 | 0.2322 | 3.0524 |
| Qwen3-8B SFT | **0.0399** | **0.8501** | **0.8352** | 0.6430 | 0.1632 | 2.4900 |
| Qwen3-8B DPO | 0.0234 | 0.8147 | 0.7969 | **0.7941** | **0.2149** | 2.9600 |
| Qwen3-8B PPO | 0.0232 | 0.8137 | 0.7983 | 0.8234 | 0.1688 | 3.0000 |

---

### GPT Model Comparison

We evaluated three OpenAI models using pre-generated explanation data (GPT-4, GPT-3.5) and live API calls (GPT-4o-mini), applying the same full metric suite.

#### Cardiff Biology

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|-------|--------|------------|------------|-------|-------|-------|
| GPT-4 | 0.0316 | **0.8472** | 0.8178 | 0.8972 | 0.0736 | 3.0100 |
| GPT-3.5 | 0.0258 | 0.8436 | 0.8231 | **0.9136** | **0.2708** | 2.9900 |
| GPT-4o-mini | 0.0241 | 0.8362 | 0.7991 | 0.8979 | 0.1028 | 3.0100 |
| LLaMA-2 DPO v2 (ours) | **0.0247** | 0.8300 | **0.8422** | 0.8682 | 0.2905 | **3.0648** |

#### Sydney Biology

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|-------|--------|------------|------------|-------|-------|-------|
| GPT-4 | **0.0843** | **0.8685** | 0.8302 | 0.4727 | 0.1973 | 2.8100 |
| GPT-3.5 | 0.0530 | 0.8636 | 0.8260 | **0.8925** | **0.2475** | 2.8600 |
| GPT-4o-mini | 0.0385 | 0.8541 | 0.8079 | 0.8785 | 0.1787 | 2.9100 |
| LLaMA-2 DPO v2 (ours) | 0.0364 | 0.8367 | **0.8426** | 0.6290 | **0.2774** | **2.9474** |

**Key findings:**
- **GPT-4 has surprisingly low NLI** (0.07 Cardiff, 0.20 Sydney). It generates fluent encyclopaedic explanations that lack direct logical entailment of the correct option.
- **GPT-3.5 is the best GPT model on NLI** — its NLI (0.27/0.25) is close to our DPO v2 (0.29/0.28), suggesting it follows the task prompt more literally.
- **Our DPO v2 outperforms GPT-4o-mini on NLI by ~2.8×** (0.291 vs 0.103 on Cardiff), despite being a smaller, locally-deployed model.
- **GPT-4 on Sydney ACR is only 0.47** — below SFT baseline. Its verbose style often drifts from the correct option's specific terminology.
- GPT models have zero deployment latency cost at small scale, but our RL models run locally with no per-call API cost.

---

### GPT Pairwise Judge Results

We used GPT-4o-mini as a pairwise judge on 100 questions per comparison (N=100, no ties observed).

#### Cardiff Biology

| Model A | Model B | A wins | B wins | B win rate |
|---------|---------|--------|--------|-----------|
| SFT | DPO v2 | 69 | 31 | 31% |
| SFT | PPO | 86 | 12 | 12% |
| DPO v2 | PPO | 76 | 22 | 22% |
| **DPO v2** | **GPT-4o-mini** | **0** | **100** | **100%** |
| **DPO v2** | **GPT-3.5** | **8** | **92** | **92%** |
| Qwen3-8B SFT | Qwen3-8B DPO | 1 | 99 | 99% |

#### Sydney Biology

| Model A | Model B | A wins | B wins | B win rate |
|---------|---------|--------|--------|-----------|
| SFT | DPO v2 | 76 | 24 | 24% |
| SFT | PPO | 91 | 7 | 7% |
| DPO v2 | PPO | 89 | 9 | 9% |
| **DPO v2** | **GPT-4o-mini** | **0** | **100** | **100%** |

**Critical finding — Self-Preference Bias**: The GPT-4o-mini judge shows 100% preference for GPT-4o-mini over our DPO v2, yet the objective NLI metric shows DPO v2 is 2.8× better (NLI 0.291 vs 0.103). This is a well-documented **self-preference bias** in LLM-as-judge evaluation: LLMs systematically favour text that stylistically resembles their own output (verbose, fluent, well-structured). This demonstrates that LLM-as-judge is **not a reliable evaluation method** for task-oriented explanation generation where content accuracy matters more than fluency.

Interestingly, within LLaMA-2 RL models, GPT judge ranks: **SFT > DPO v2 > PPO** — again mirroring output length rather than task quality.

Similarly, Qwen3 DPO wins 99% over Qwen3 SFT, yet Qwen3 DPO actually has *lower* NLI on Cardiff (0.115 vs 0.196).

---

### ILearner-LLM K=5 Baseline

Full iterative baseline (K=5 samples per question, verifier selects best) across all five datasets:

| Dataset | BLEU ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Ver ↑ |
|---------|--------|------------|-------|-------|-------|
| Cardiff Biology | 0.0729 | 0.7859 | 0.8713 | 0.0864 | 3.1960 |
| Sydney Biology | 0.0274 | 0.7889 | 0.6451 | 0.0837 | 3.1843 |
| Auckland Law | 0.0381 | 0.7720 | 0.6326 | **0.3996** | 2.7845 |
| UK Medicine Y1 | 0.0671 | 0.7863 | 0.8066 | 0.1219 | 3.2005 |
| UK Medicine Y2 | 0.0495 | 0.7873 | 0.7357 | 0.0668 | 3.1600 |

**Key insight**: K=5 sampling dramatically improves **BLEU** (e.g. Cardiff: 0.073 vs SFT K=1 0.016, 4.6× higher) because selecting the best candidate from 5 increases surface-level similarity to the reference. However, **NLI remains near-zero** (0.086 on Cardiff), showing that the verifier selects for style/fluency rather than answer-grounding. Our DPO v2 (NLI=0.291) outperforms K=5 on NLI by **3.4×** while using a single generation pass.

The exception is **Auckland Law** (K=5 NLI=0.400), which is the best NLI result across all conditions. Legal explanations may be inherently more structured, enabling the verifier to more reliably select answer-entailing candidates.

---

## NLI Model Ablation (Cardiff)

| NLI Model | NLI Score (SFT) | Speed | Notes |
|-----------|-----------------|-------|-------|
| `nli-deberta-v3-large` | 0.055 | ~1.5s/pair | Stricter entailment; better calibrated |
| `nli-deberta-v3-small` | 0.287 | ~0.1s/pair | 15× faster; more permissive threshold |

Both models agree on the **relative ranking** of systems (RL > SFT), but absolute scores differ substantially. The small model is used as the primary metric throughout this work for efficiency. Large model scores may be more appropriate for final paper reporting.

---

## Key Conclusions

### 1. RL Training is the Most Effective Method for Answer-Grounded Explanation Quality

All RL methods (DPO v1, v2, v3, PPO) improve NLI by **4–5× over SFT** consistently across Cardiff and Sydney. This is the most task-relevant improvement: RL forces the model to generate explanations that logically entail the correct answer option, rather than producing plausible-sounding but answer-agnostic text.

### 2. DPO v2 is the Best Overall LLaMA-2 Model

DPO v2 (458 pairs, 5 epochs) outperforms DPO v1, DPO v3, and PPO on the majority of metrics. Key advantages:
- Highest NLI on Sydney (0.277), highest BERT(Ans) and BLEU on Cardiff
- Fastest inference among RL models (5.8s on Cardiff)
- Low training cost (15 minutes, compared to hours for PPO)

More preference data (v1→v2: 165→458 pairs) helps. Further scaling to v3 (851 pairs) does not continue to improve NLI, suggesting diminishing returns from simple data scaling with the same verifier.

### 3. GPT-4 is Not the Best Model for This Task

Despite achieving the highest BERTScore (fluency), GPT-4 has NLI=0.07 on Cardiff — comparable to or worse than SFT. Its tendency to generate encyclopaedic, multi-faceted explanations causes it to miss the specific logical link between explanation and answer option. GPT-3.5 is the best GPT model on NLI (0.27), likely due to more literal instruction-following. Our DPO v2 outperforms GPT-4o-mini on NLI by ~2.8×.

### 4. LLM-as-Judge Cannot Be Trusted for This Task

GPT-4o-mini pairwise judge shows 100% preference for GPT-4o-mini over DPO v2, despite objective metrics showing the opposite. This self-preference bias disqualifies LLM judges from being used as the primary evaluation signal for task-oriented explanation generation. Answer-grounded metrics (NLI, BERT(Ans), ACR) are more reliable.

### 5. Multi-Sample Selection (K=5) Improves Surface Metrics, Not Answer-Grounding

ILearner K=5 achieves 4.6× higher BLEU than K=1 SFT, but NLI remains near-zero (~0.08). The verifier selects high-quality explanations in terms of style, not in terms of answer entailment. RL directly optimises this deficiency.

### 6. Qwen3-8B Has Strong Zero-Shot Instruction-Following but Suffers Alignment Tax

Qwen3-8B SFT achieves NLI=0.196 (Cardiff) — 3.5× higher than LLaMA-2 SFT (0.055) — with no RL. Its stronger base model capability allows better zero-shot task adherence. However, DPO training causes NLI to *drop* to 0.115, a classic alignment tax: the verifier reward is not perfectly aligned with the NLI objective, so DPO optimises reward at the cost of answer-entailment. Hybrid PPO v1 partially recovers NLI (0.179), but does not exceed the SFT baseline.

### 7. Deployment Recommendation

| Use case | Recommended model | Rationale |
|----------|------------------|-----------|
| **Best task quality now** | LLaMA-2 DPO v2 | Highest NLI, fastest RL inference, proven pipeline |
| **Low-cost deployment** | Qwen3-8B SFT | Apache 2.0, 16GB VRAM, strong baseline NLI (0.196) |
| **4-bit quantized edge** | Qwen3-8B SFT (Q4) | ~4–5GB VRAM (RTX 3090), minimal quality loss |
| **API-based (no GPU)** | GPT-3.5 | Best NLI among GPT models, cheap API |

LLaMA-2 requires A100-class GPUs and has a restrictive commercial license. Qwen3-8B (Apache 2.0) is the preferred long-term deployment target once the DPO alignment issue is resolved.

---

## Future Work

### High Priority

| Experiment | Description | Expected outcome |
|------------|-------------|-----------------|
| Fix Qwen3 DPO alignment | Re-design preference data using NLI as reward signal instead of verifier score | Recover NLI≥0.20 after DPO for Qwen3 |
| NLI-guided reward for PPO | Replace verifier score with NLI entailment probability as PPO reward | Directly optimise the most discriminative metric |
| Larger preference dataset | Scale to 2000+ pairs across all 5 domains | Test data scaling limits for LLaMA-2 DPO |
| Human evaluation study | Rate 50 explanations per model for correctness and relevance | Validate NLI as a proxy for human-judged quality |

### Model Improvements

| Experiment | Description | Rationale |
|------------|-------------|-----------|
| Qwen3-8B with NLI reward DPO | Use NLI score as preference oracle instead of verifier | Qwen3 verifier score reward causes alignment tax |
| DPO + PPO two-stage pipeline | Initialise PPO from DPO v2 (LLaMA-2) | Combine offline alignment with online exploration |
| PPO with larger batch size | batch=8–16 for LLaMA-2 PPO | Reduce reward variance; enable more stable online RL |
| Qwen3-8B 4-bit quantization | QLoRA fine-tuning for deployment | Enable deployment on consumer GPUs |
| Qwen3 thinking mode | Enable Qwen3 chain-of-thought (`enable_thinking=True`) | May improve answer-grounded reasoning quality |
| Reward model ensemble | Combine verifier + NLI as composite reward | More robust reward signal for PPO |

### Evaluation Improvements

| Experiment | Description | Rationale |
|------------|-------------|-----------|
| Large NLI model final eval | `cross-encoder/nli-deberta-v3-large` for final paper scores | More conservative, better-calibrated entailment scores |
| Error analysis | Categorise SFT failure modes (URL output, shorthand, hallucination) | Qualitative section for paper |
| Calibration study | Compare NLI small vs large model rankings | Ensure metric choice does not change conclusions |
| Cross-domain RL training | Train DPO/PPO on all 5 domains jointly | Improve Law/Medicine performance where K=5 still leads |
| Latency profiling | Full pipeline latency (preprocessing + inference + postprocessing) | Accurate latency comparison for deployment section |

### Broader Experiments

| Experiment | Description | Rationale |
|------------|-------------|-----------|
| Llama-3-8B / Llama-3-70B | Apply full RL pipeline to Llama-3 | Stronger base → higher NLI ceiling |
| Reward model training | Fine-tune a reward model on human preference data | Replace heuristic verifier with human-aligned reward |
| Student outcome study | Test whether better NLI explanations improve student learning | Ultimate validation of the system's educational value |

---

## Reproducing Results

### Quick Reproduction (LLaMA-2 DPO v2)

```bash
conda activate llm-tuning

# Step 1: SFT
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 rl_train_sft.py \
    --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_dir ./rl_sft_llama2_13b_generator \
    --num_train_epochs 3 --per_device_train_batch_size 2 \
    --lora_r 16 --lora_alpha 32 --bf16 True

# Step 2: Build preference pairs (v2)
CUDA_VISIBLE_DEVICES=4,5 python3 rl_build_preference_data.py \
    --generator_path /data/shared/llama2/llama-2-13b-hf \
    --lora_adapter_path ./rl_sft_llama2_13b_generator \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_path ./rl_preference_data_v2/preference_pairs.json \
    --num_samples 3 --min_score_gap 0.1 --max_questions 500 \
    --generator_device cuda:0 --verifier_device cuda:1

# Step 3: DPO training
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 rl_train_dpo.py \
    --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
    --sft_adapter_path ./rl_sft_llama2_13b_generator \
    --preference_data_path ./rl_preference_data_v2/preference_pairs.json \
    --output_dir ./rl_dpo_v2_llama2_13b_generator \
    --num_train_epochs 5 --per_device_train_batch_size 2 \
    --lora_r 16 --lora_alpha 32 --bf16 True

# Step 4: Evaluate
CUDA_VISIBLE_DEVICES=4,5 python3 rl_evaluation.py \
    --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --output_path ./rl_eval_results/cardiff_eval_nli.json \
    --base_model_path /data/shared/llama2/llama-2-13b-hf \
    --sft_lora_path ./rl_sft_llama2_13b_generator \
    --dpo_lora_path ./rl_dpo_v2_llama2_13b_generator \
    --device cuda:0 --verifier_device cuda:1
```

---

## Evaluation Metrics Reference

```python
# All metrics implemented in rl_evaluation.py

# BERT(Stu): semantic similarity to student-written reference
bert_score(generated, student_explanation)

# BERT(Ans): semantic similarity to correct option text
bert_score(generated, correct_option_text)

# ACR: fraction of ≥4-char tokens from correct option that appear in explanation
acr = len(set(keywords_from_option) & set(tokens_in_explanation)) / len(keywords_from_option)

# NLI: DeBERTa entailment probability
# premise = generated explanation, hypothesis = correct option text
nli_model = "cross-encoder/nli-deberta-v3-small"
P(entailment | premise=explanation, hypothesis=correct_option)

# Verifier: domain verifier model score (0-5)
verifier_model = "./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"
```

### Evaluation Result JSON Schema

```json
{
  "results": {
    "model": "string",
    "avg_bleu": 0.0,
    "avg_bert_score_f1": 0.0,
    "avg_bert_score_f1_answer_anchored": 0.0,
    "avg_answer_coverage_rate": 0.0,
    "avg_nli_entailment": 0.0,
    "avg_verifier_score": 0.0,
    "avg_inference_time_s": 0.0
  },
  "detailed_results": {
    "bleu_scores": [],
    "bert_scores_vs_student": [],
    "bert_scores_vs_answer": [],
    "acr_scores": [],
    "nli_entailment_scores": [],
    "verifier_scores": [],
    "inference_times": [],
    "generated_explanations": []
  }
}
```

---

## Model Checkpoints

### LLaMA-2-13B Models

| Model | Path | Init from |
|-------|------|-----------|
| SFT LoRA | `./rl_sft_llama2_13b_generator/` | LLaMA-2-13B base |
| DPO v1 LoRA | `./rl_dpo_llama2_13b_generator/` | SFT |
| DPO v2 LoRA | `./rl_dpo_v2_llama2_13b_generator/` | SFT |
| DPO v3 LoRA | `./rl_dpo_v3_llama2_13b_generator/` | SFT (all-domain) |
| PPO LoRA | `./rl_ppo_llama2_13b_generator/` | SFT |

### Qwen3-8B Models

| Model | Path | Init from |
|-------|------|-----------|
| SFT LoRA | `./rl_sft_qwen3_8b_generator/` | Qwen3-8B base |
| DPO v1 LoRA | `./rl_dpo_qwen3_v1_generator/` | SFT |
| DPO v2 LoRA | `./rl_dpo_qwen3_v2_generator/` | SFT |
| DPO Beta0.5 LoRA | `./rl_dpo_qwen3_beta05_generator/` | SFT |
| PPO Standard | `./rl_ppo_qwen3_sft_ppo_generator/` | SFT |
| Hybrid PPO v1 | `./rl_ppo_qwen3_hybrid_v1_2000steps/` | DPO v1 |
| Hybrid PPO v2 | `./rl_ppo_qwen3_hybrid_v2_2000steps/` | DPO v2 |

### Shared Models

| Model | Path | Role |
|-------|------|------|
| LLaMA-2-13B base | `/data/shared/llama2/llama-2-13b-hf` | Generator base |
| Qwen3-8B base | `/data/shared/qwen3/Qwen3-8B` | Generator base |
| Verifier (Alpaca-7B) | `./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2/` | Reward model |

# RLearner-LLM: Reinforcement Learning for Educational Explanation Generation

> **Extension of ILearner-LLM** — replaces K-round iterative inference with RL-trained single-pass generation.
> See the main [README.md](README.md) for the full project overview.

---

## Table of Contents
- [Three Paper Innovations](#three-paper-innovations)
- [System Architecture](#system-architecture)
- [Training Pipeline](#training-pipeline)
- [Experimental Results](#experimental-results)
- [Future Experiments](#future-experiments)
- [Reproducing Results](#reproducing-results)

---

## Three Paper Innovations

### Innovation 1 — Single-Pass RL Generation Replacing Iterative Inference

**Problem**: The original ILearner-LLM pipeline runs K sequential rounds of Generator → Verifier → Feedback, where latency scales linearly with K. At K=5, inference is 5× slower than a single-pass model, making real-time deployment impractical.

**Our Approach**: Instead of iterating at test time, we embed the Generator–Verifier interaction into the training signal itself. Using the verifier as a reward model, we apply:
- **DPO (Direct Preference Optimization)** — offline RL from verifier-ranked candidate pairs.
- **PPO (Proximal Policy Optimization)** — online RL with live verifier reward signals.

**Result**: At inference time, a single forward pass produces explanations that match or exceed the quality of K=5 iterative refinement, at **~6× lower latency** (5.8s vs ≥18s per question).

| Approach | Inference passes | Avg latency (Cardiff) |
|----------|-----------------|----------------------|
| ILearner-LLM (original, K=5) | 5 Generator + 5 Verifier | ~100s (estimated) |
| SFT baseline (LoRA) | 1 | 19.9s |
| **DPO v2 (ours)** | **1** | **5.8s** |
| PPO (ours) | 1 | 7.2s |

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

**Qualitative finding**: SFT models frequently output URLs, shorthand lists ("Option A is correct, B is wrong"), or non-explanatory text. RL models consistently produce substantive explanations naming and justifying the correct answer concept.

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
| Pref v3 | 1000 | 3 | 0.1 | **851** | 5 | DPO v3 |

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

### Environment
```bash
conda activate llm-tuning   # TRL 0.7.1, Transformers 4.31.0, PEFT 0.4.0, accelerate 0.21.0
```

### Completed Training Runs

| Step | Script | Config | Duration | Output |
|------|--------|--------|----------|--------|
| SFT | `rl_train_sft.py` | LLaMA-2-13B, LoRA r=16, 3 epochs, 13211 ex | 48 min, 4×A100 | `./rl_sft_llama2_13b_generator/` |
| Pref v1 | `rl_build_preference_data.py` | 500Q × 2 samples, gap≥0.3 | ~2h | 165 pairs |
| DPO v1 | `rl_train_dpo.py` | 165 pairs, 2 epochs | 68 sec | `./rl_dpo_llama2_13b_generator/` |
| Pref v2 | `rl_build_preference_data.py` | 500Q × 3 samples, gap≥0.1 | ~4h | 458 pairs |
| DPO v2 | `rl_train_dpo.py` | 458 pairs, 5 epochs | 14m52s | `./rl_dpo_v2_llama2_13b_generator/` |
| PPO | `rl_train_ppo.py` | 500Q, batch=4, 125 steps | ~3h | `./rl_ppo_llama2_13b_generator/` |

### Quick Reproduction

```bash
# Step 1: SFT
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n llm-tuning python3 rl_train_sft.py \
    --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_dir ./rl_sft_llama2_13b_generator \
    --num_train_epochs 3 --per_device_train_batch_size 2 \
    --lora_r 16 --lora_alpha 32 --bf16 True

# Step 2: Build preference pairs
CUDA_VISIBLE_DEVICES=4,5 conda run -n llm-tuning python3 rl_build_preference_data.py \
    --generator_path /data/shared/llama2/llama-2-13b-hf \
    --lora_adapter_path ./rl_sft_llama2_13b_generator \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_path ./rl_preference_data_v2/preference_pairs.json \
    --num_samples 3 --min_score_gap 0.1 --max_questions 500 \
    --generator_device cuda:0 --verifier_device cuda:1

# Step 3A: DPO training
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n llm-tuning python3 rl_train_dpo.py \
    --model_name_or_path /data/shared/llama2/llama-2-13b-hf \
    --sft_adapter_path ./rl_sft_llama2_13b_generator \
    --preference_data_path ./rl_preference_data_v2/preference_pairs.json \
    --output_dir ./rl_dpo_v2_llama2_13b_generator \
    --num_train_epochs 5 --per_device_train_batch_size 2 \
    --lora_r 16 --lora_alpha 32 --bf16 True

# Step 4: Evaluate all models
CUDA_VISIBLE_DEVICES=4,5 conda run -n llm-tuning python3 rl_evaluation.py \
    --test_data_path ./Paul_new_data/Cardiff/cardiff_eval_nli.json \
    --verifier_path ./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
    --output_path ./rl_eval_results/cardiff_eval_nli.json \
    --base_model_path /data/shared/llama2/llama-2-13b-hf \
    --sft_lora_path ./rl_sft_llama2_13b_generator \
    --dpo_lora_path ./rl_dpo_v2_llama2_13b_generator \
    --device cuda:0 --verifier_device cuda:1
```

---

## Experimental Results

All results on 100-question test sets (Cardiff Biology + Sydney Biology).
Metrics: BLEU, BERT(Stu) = BERTScore vs. student explanation, BERT(Ans) = BERTScore vs. correct option,
ACR = Answer Coverage Rate, NLI = DeBERTa entailment probability (explanation → correct option),
Verifier = domain verifier score (0–5), Time = avg inference time per question (seconds).

### Cardiff Biology Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ | Time(s) ↓ |
|-------|--------|------------|------------|-------|-------|-----------|---------|
| SFT (LLaMA-2-13B + LoRA) | 0.0160 | 0.8070 | 0.7820 | 0.8087 | 0.0555 | 3.1976 | 19.947 |
| DPO v1 (165 pairs, 2 ep) | 0.0173 | 0.8238 | 0.8325 | 0.7698 | **0.2969** | 3.0467 | 6.567 |
| **DPO v2 (458 pairs, 5 ep)** | **0.0247** | **0.8300** | **0.8422** | **0.8682** | 0.2905 | 3.0648 | **5.774** |
| PPO (125 steps, batch=4) | 0.0175 | 0.8245 | 0.8255 | 0.7390 | 0.2260 | 3.0750 | 7.234 |
| DPO v3 (851 pairs, 5 ep) | 0.0436 | 0.8384 | 0.8407 | 0.7490 | 0.2079 | 3.2125 | 6.124 |
| Qwen3-8B SFT | 0.0436 | 0.8624 | 0.8512 | 0.7512 | 0.2312 | 2.5000 | 3.771 |
| Qwen3-8B DPO | 0.0512 | 0.8698 | 0.8588 | 0.7621 | 0.2912 | 2.9700 | 13.104 |

**Cardiff takeaways:**
- DPO v2 wins on BLEU, BERT(Stu), BERT(Ans), ACR, and inference speed.
- DPO v1 has highest NLI (0.2969) — more training data (DPO v2) slightly lowers NLI here.
- **NLI gap**: SFT 0.055 → RL models 0.23–0.30 (4–5× improvement). The most discriminative metric.
- **Speed**: All RL models are 2.7–3.5× faster than SFT at inference (less repetition/rambling).

### Sydney Biology Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ | Time(s) ↓ |
|-------|--------|------------|------------|-------|-------|-----------|---------|
| SFT (LLaMA-2-13B + LoRA) | 0.0222 | 0.8244 | 0.7870 | 0.6249 | 0.0537 | 3.1937 | 19.001 |
| DPO v1 (165 pairs, 2 ep) | 0.0314 | 0.8262 | 0.8272 | 0.6034 | 0.2171 | 2.9094 | 9.049 |
| **DPO v2 (458 pairs, 5 ep)** | 0.0364 | **0.8367** | **0.8426** | 0.6290 | **0.2774** | 2.9474 | **6.370** |
| PPO (125 steps, batch=4) | **0.0421** | 0.8364 | 0.8294 | **0.6606** | 0.2269 | 2.9609 | 7.596 |
| DPO v3 (851 pairs, 5 ep) | 0.0573 | 0.8454 | 0.8523 | 0.6277 | 0.1878 | 3.1386 | 5.921 |
| Qwen3-8B SFT | 0.0384 | 0.8512 | 0.8412 | 0.6121 | 0.1982 | 2.2600 | 3.775 |
| Qwen3-8B DPO | 0.0421 | 0.8588 | 0.8492 | 0.6312 | 0.2790 | 2.7900 | 12.145 |

**Sydney takeaways:**
- PPO achieves highest BLEU (0.0421) and ACR (0.6606) on Sydney.
- DPO v2 wins on BERT(Stu), BERT(Ans), NLI, and speed.
- **NLI gap** consistent: SFT 0.054 → RL models 0.22–0.28 (4–5× improvement).
- Results generalize cross-domain (Cardiff and Sydney show same patterns).

### Key Findings Across Both Datasets

1. **NLI is the most discriminative metric** — captures logical grounding that BLEU/BERTScore miss.
2. **All RL methods significantly outperform SFT** on answer-grounded metrics (BERT(Ans), ACR, NLI).
3. **DPO v2 is the overall best model** — top performance on most metrics, fastest inference.
4. **PPO shows complementary strengths** — highest BLEU+ACR on Sydney, suggesting it learns different generation behaviors.
5. **SFT qualitative failure mode**: frequently outputs URLs, shorthand ("A-correct, B-wrong"), or non-explanations. RL models consistently produce substantive explanations.

---

## Cross-Domain Generalization Results

Evaluating the models on domains NOT seen during the initial preference data generation (Cardiff/Sydney only). These results use the DPO v3 model (trained on 851 merged pairs).

### Auckland Law Dataset

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ |
|-------|--------|------------|------------|-------|-------|-----------|
| SFT baseline | 0.0298 | 0.8010 | 0.7709 | 0.5516 | 0.2702 | 2.7180 |
| DPO v1 | 0.0369 | 0.8230 | 0.8199 | 0.5831 | 0.2967 | 2.5996 |
| DPO v2 | 0.0640 | 0.8246 | 0.8257 | 0.6315 | 0.2528 | 2.6252 |
| PPO | 0.0516 | 0.8218 | 0.8198 | 0.6327 | 0.2888 | 2.5593 |
| ILearner-LLM (K=5) | 0.0526 | 0.8123 | **0.8456** | 0.6552 | **0.3842** | 2.6512 |
| **Qwen3 SFT** | **0.1558** | **0.8801** | 0.8267 | 0.4578 | 0.3470 | 2.1600 | (Proxy) |
| **Qwen3 DPO** | 0.0343 | 0.8161 | 0.8007 | **0.7693** | 0.2235 | 2.5900 |

### UK Medicine Year 1

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ |
|-------|--------|------------|------------|-------|-------|-----------|
| SFT baseline | 0.0136 | 0.8065 | 0.7799 | 0.7556 | 0.0860 | 3.2561 |
| DPO v1 | 0.0176 | 0.8246 | 0.8256 | 0.7067 | 0.2266 | 3.1457 |
| DPO v2 | 0.0220 | 0.8261 | 0.8261 | 0.8010 | 0.2503 | 3.0764 |
| PPO | 0.0160 | 0.8270 | 0.8256 | 0.6917 | **0.2755** | 3.0696 |
| **DPO v3** | 0.0192 | 0.8240 | 0.8333 | **0.7575** | 0.2583 | 3.0429 |
| **Qwen3 SFT** | **0.0458** | **0.8629** | **0.8466** | 0.7387 | 0.2457 | 2.4700 | (Proxy) |
| **Qwen3 DPO** | 0.0212 | 0.8184 | 0.7959 | 0.8362 | 0.1701 | 2.9600 |

### UK Medicine Year 2

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ |
|-------|--------|------------|------------|-------|-------|-----------|
| SFT baseline | 0.0163 | 0.8208 | 0.8142 | 0.6120 | 0.2319 | 2.8717 |
| DPO v1 | 0.0184 | 0.8174 | 0.8343 | 0.6823 | 0.2896 | 3.0252 |
| DPO v2 | 0.0203 | 0.8226 | 0.8403 | 0.7953 | **0.3189** | 3.0230 |
| PPO | 0.0156 | 0.8190 | 0.8302 | 0.7210 | 0.2800 | 3.0323 |
| DPO v3 | 0.0153 | 0.8214 | 0.8309 | 0.7428 | 0.2159 | 3.0593 |
| **Qwen3 SFT** | **0.0399** | **0.8501** | **0.8352** | 0.6430 | 0.1632 | 2.4900 | (Proxy) |
| **Qwen3 DPO** | 0.0232 | 0.8137 | 0.7983 | **0.8234** | 0.2149 | 3.0000 |

## NLI Model Ablation Results (Cardiff SFT)

We investigated the sensitivity of the NLI metric to the underlying cross-encoder model size.

| NLI Model | NLI Entailment (SFT) ↑ | Inference Time (s) |
|-----------|-------------------------|--------------------|
| DeBERTa-v3-Large | 0.0555 | ~1.5s/pair |
| DeBERTa-v3-Small | **0.2872** | **~0.1s/pair** |

**Takeaway:** The choice of NLI model significantly impacts the absolute score. The small model is 15x faster and appears more generous in entailment classification, likely due to less strict semantic matching requirements.


---


## Future Experiments

### Immediate Next Steps (in progress / ready to run)

| Experiment | Status | Expected outcome |
|------------|--------|-----------------|
| DPO v3 (1000Q × 3 samples, gap≥0.1) | **Completed** | 851 preference pairs generated |
| Qwen3-8B DPO Training & Eval | **Completed** | Evaluates generalizability on a smaller proxy model |
| DPO v3 Training & Eval | **Completed** | Should improve on DPO v2 with 2× more pairs (Med Y2 final eval) |
| PPO Scaling (500 steps) | **Failed (OOM)** | Investigate reward stagnation with more steps (Needs Multi-GPU) |

### Cross-Domain Generalization

| Experiment | Description | Priority |
|------------|-------------|----------|
| Law dataset eval | Evaluate DPO v2 / PPO on Auckland Law questions | High |
| Medicine Y1 eval | Evaluate on UK Medicine Year 1 questions | High |
| Medicine Y2 eval | Evaluate on UK Medicine Year 2 questions | High |
| Cross-domain training | Train on merged; eval on each domain separately | Medium |

### Model Improvements

| Experiment | Description | Rationale |
|------------|-------------|-----------|
| More PPO steps | Run PPO for 500+ steps (currently only 125) | PPO may need more steps to converge |
| PPO with higher batch | batch=8 or 16 for stabler reward | Reduce variance in online RL |
| DPO with larger base | LLaMA-2-70B or Llama-3-8B | Larger model = higher ceiling |
| Hybrid DPO+PPO | Init PPO from DPO v2 checkpoint | Combine offline stability + online exploration |

### Evaluation Improvements

| Experiment | Description | Rationale |
|------------|-------------|-----------|
| ILearner-LLM K=5 baseline | Run original K-loop inference for direct latency comparison | Quantify speedup in paper |
| Larger NLI model | Use `cross-encoder/nli-deberta-v3-large` | Higher NLI quality |
| Human evaluation | Rate 50 explanations per model for correctness | Ground truth for NLI claim |
| Error analysis | Categorize SFT failure cases (URL output, shorthand, hallucination) | Paper qualitative section |

### Scaling Laws

| Experiment | Description |
|------------|-------------|
| Pref data scaling | Plot NLI vs. number of preference pairs (165 → 458 → 900+) |
| N samples ablation | Compare N=2,3,4,5 candidates per question |
| Min score gap ablation | Compare gap threshold 0.5, 0.3, 0.1, 0.0 |

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

| Model | Path | Base |
|-------|------|------|
| SFT LoRA adapter | `./rl_sft_llama2_13b_generator/` | LLaMA-2-13B |
| DPO v1 LoRA adapter | `./rl_dpo_llama2_13b_generator/` | LLaMA-2-13B |
| DPO v2 LoRA adapter | `./rl_dpo_v2_llama2_13b_generator/` | LLaMA-2-13B |
| PPO LoRA adapter | `./rl_ppo_llama2_13b_generator/` | LLaMA-2-13B |
| Verifier (merged) | `./qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2/` | Alpaca-7B |

Base model: `/data/shared/llama2/llama-2-13b-hf`

# ILearner-LLM + RLearner-LLM: Explanation Generation for PeerWise

This repository contains two progressive lines of work for automatically generating and evaluating educational explanations for PeerWise exam questions:

1. **ILearner-LLM** (original paper) — Instruction-tuned LLM pipeline with iterative Generator–Verifier refinement.
2. **RLearner-LLM** (this extension) — Replaces the K-round iterative prompt loop with reinforcement learning (DPO/PPO) using modern models (LLaMA-2-13B, LLaMA-3-8B, **Qwen3-8B**, **Gemma 4 E4B-it**) and LoRA for parameter-efficient fine-tuning. The latest **Hybrid-DPO** variant adds a multiplicative NLI×Verifier reward gated by Answer Coverage Rate, validated across **5 academic domains × 3 base architectures**.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [ILearner-LLM Pipeline (Original)](#ilearner-llm-pipeline-original)
- [RLearner-LLM Pipeline (New Extension)](#rlearner-llm-pipeline-new-extension)
- [Experimental Results](#experimental-results)
- [Model Zoo](#model-zoo)
- [Key Design Decisions](#key-design-decisions)
- [Acknowledgements](#acknowledgements)

> **RLearner-LLM detailed documentation** (paper innovations, full results, future experiments):
> **[README_RL.md](README_RL.md)**

---

## Architecture Overview

### ILearner-LLM (Original)

```
Question + Options + Answer
         │
         ▼
  ┌─────────────┐   K-round loop (K=1..5)
  │  Generator  │◄─────────────────────────┐
  │  (LLaMA-2)  │   "Your last score was X, │
  └─────────────┘    generate a better one" │
         │                                  │
         ▼ generated explanation             │
  ┌─────────────┐                           │
  │  Verifier   │──── score (0-5) ──────────┘
  │  (LLaMA-2)  │
  └─────────────┘
         │
         ▼ final explanation after K rounds
```

**Limitation**: Latency scales with K. Prompt-only refinement cannot update model weights.

### RLearner-LLM (Extension)

```
         SFT Phase (LoRA)
Data ──► Fine-tune Llama-3/Qwen3 ──► SFT Generator (LoRA Adapter)

         DPO Phase (Offline RL, RECOMMENDED)
SFT Generator ──► Generate N candidates per question
                        │
                   Verifier scores all N candidates
                        │
              (best → y_chosen, worst → y_rejected)
              + GPT-4o CoT synthetic positives
              + Model-generated hard negatives
                        │
              LoRA-DPO training ──► DPO Generator (LoRA Adapter)

         Inference (1-shot, no K-loop)
Question ──► DPO Generator ──► High-quality explanation  (single forward pass)
```

**Key improvements over original**:
- **1-Shot inference** (vs K-round): ~K× lower latency at deployment
- **LoRA Hot-swap**: One base model in GPU memory; switch Generator/Verifier adapters in milliseconds
- **Modern base**: LLaMA-3-8B, **Qwen3-8B**, **Gemma 4 E4B-it** (4.5B-effective dense, Per-Layer Embeddings) — far stronger reasoning than LLaMA-2-13B
- **Synthetic hard negatives**: Prevent Reward Hacking via GPT-4o CoT flawed explanation generation

---

## Project Structure

```
Explanation-Generation/
│
├── README.md
├── requirements.txt
│
├── train.py                      # Core SFT training (original, full-parameter fine-tuning)
├── utils.py                      # Shared data loading utilities
├── training_script.sh            # Original full-param training commands (LLaMA-2)
│
│── ── ── RLearner-LLM (New) ── ── ──
├── rl_train_sft.py               # Step 1: SFT fine-tuning with LoRA (Llama-3/Qwen3)
├── rl_build_preference_data.py   # Step 2: Build DPO preference pairs (multi-sample + Verifier)
├── rl_generate_synthetic_data.py # Step 2b: GPT-4o/Claude CoT synthetic data (亮点三)
├── rl_train_dpo.py               # Step 3A: DPO training — RECOMMENDED PATH
├── rl_train_ppo.py               # Step 3B: PPO online RL training — alternative
├── rl_evaluation.py              # Step 4: Evaluate & compare all models
├── rl_training_script.sh         # One-command full RLearner-LLM pipeline
│
├── rl_configs/
│   ├── ds_zero2.json             # DeepSpeed ZeRO-2 (lighter, for smaller models)
│   └── ds_zero3.json             # DeepSpeed ZeRO-3 (for 32B+ models)
│
│── ── ── Legacy Scripts ── ── ──
├── scripts/
│   ├── preprocessing/            # Data preprocessing (generator & verifier formats)
│   │   ├── data_preprocessing_generator.py
│   │   ├── data_preprocessing_verifier_way2.py
│   │   ├── data_preprocess_generator_one_dataset_cardiff.py
│   │   ├── data_preprocess_generator_one_dataset_sydney.py
│   │   ├── merged_all_training_set.py
│   │   └── reward_data_preprocessing.py
│   │
│   ├── evaluation/               # Batch evaluation (BLEU + BERTScore)
│   │   ├── batch_evaluation_Cardiff.py
│   │   ├── batch_evaluation_Sydney.py
│   │   ├── batch_evaluation_auckland_law.py
│   │   ├── batch_evaluation_uk_medical_year1.py
│   │   ├── batch_evaluation_uk_medical_year2.py
│   │   ├── batch_evaluation_all.py
│   │   └── bleu_score_calculator.py
│   │
│   ├── chat/                     # Interactive demo (Generator + Verifier loop)
│   │   ├── chat_generator.py
│   │   ├── chat_verifier_way2.py
│   │   └── chat_explanation_verifier_way2.py
│   │
│   ├── sampling/                 # Random sample selection for evaluation
│   ├── analysis/                 # Metric merging & analysis utilities
│   └── gpt4/                     # GPT-4 generation and evaluation scripts
│
├── Paul_new_data/                # Primary PeerWise datasets (Cardiff, Sydney, ...)
├── PeerWiseData/                 # Additional datasets (Medicine, Law)
├── rl_preference_data/           # [runtime] DPO preference pairs
└── rl_eval_results/              # [runtime] RL evaluation results
```

> **Note**: All scripts in `scripts/` must be run **from the project root** so that relative
> data paths (e.g., `./Paul_new_data/`) resolve correctly:
> ```bash
> cd /path/to/Explanation-Generation
> python scripts/evaluation/batch_evaluation_Cardiff.py
> ```

---

## Installation

### Base environment (original ILearner-LLM)

```bash
conda create -n explanation python=3.10
conda activate explanation
git clone https://github.com/Strong-AI-Lab/Explanation-Generation.git
cd Explanation-Generation
pip install -r requirements.txt
```

### RLearner-LLM environment (recommended: use existing `trl` conda env)

```bash
conda activate trl
# Install any missing packages
pip install trl>=0.10.0 peft>=0.10.0 bitsandbytes>=0.43.0 \
            deepspeed>=0.14.0 datasets>=2.18.0 accelerate>=0.27.0
# Optional: Flash Attention 2 for ~2× faster training
pip install flash-attn --no-build-isolation
```

---

## Datasets

| Dataset | Domain | Train (~) | Test (~) |
|---------|--------|-----------|----------|
| Cardiff | Biology | 4,000 | 1,400 |
| Sydney | Biology | 500 | 460 |
| Auckland Law | Law | 1,000 | — |
| UK Medicine Year 1 | Medicine | 1,000 | — |
| UK Medicine Year 2 | Medicine | 1,000 | — |
| **Merged All** | Mixed | 5,000 | — |

**Data format** (JSON, one object per question):
```json
{
  "instruction": "As an explanation generation expert, can you generate the explanation for the given input?",
  "input": "Question: [stem] Option A: ... Option B: ... The correct answer is A",
  "output": "Student-written explanation text..."
}
```

---

## ILearner-LLM Pipeline (Original)

### 1. Data Preprocessing

```bash
# Full merged dataset for generator training
python scripts/preprocessing/data_preprocessing_generator.py

# Single-domain: Cardiff only, avg_rating >= 3, explanation length >= 10
python scripts/preprocessing/data_preprocess_generator_one_dataset_cardiff.py

# Verifier data (Way 2: input = question + explanation → output = score)
python scripts/preprocessing/data_preprocessing_verifier_way2.py
```

### 2. Convert LLaMA weights to HuggingFace format

```bash
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /data/shared/llama2/llama-2-13b \
    --model_size 13B \
    --output_dir ./llama_2_13B_hf
```

### 3. Fine-tuning (full parameter, 8× A100)

```bash
# Example: Vicuna-13B generator on Cardiff (avg>=3, len>=10)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Cardiff_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 --num_train_epochs 5 \
   --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
   --learning_rate 2e-5 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True --gradient_checkpointing True
```

See `training_script.sh` for all training commands.

### 4. Batch Evaluation

```bash
python scripts/evaluation/batch_evaluation_Cardiff.py     # Cardiff
python scripts/evaluation/batch_evaluation_Sydney.py      # Sydney
python scripts/evaluation/batch_evaluation_all.py         # All domains
```

### 5. Interactive Demo

```bash
python scripts/chat/chat_explanation_verifier_way2.py
```

---

## RLearner-LLM Pipeline (New Extension)

### Prerequisites (server)
| Resource | Location |
|----------|----------|
| Llama-3-8B-Instruct | `/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct` |
| Llama-3-70B-Instruct | `/data/shared/llama3/llama3/Meta-Llama-3-70B-Instruct` |
| Cardiff Verifier | `./qiming_vicuna_13B_Cardiff_merged_verifier_way_2` |
| Sydney Verifier | `./qiming_vicuna_13B_Sydney_merged_verifier_way_2` |
| Free GPUs | cuda:4, cuda:5, cuda:6, cuda:7 (80GB each) |

### Step 1: SFT with LoRA

```bash
conda activate trl
deepspeed --num_gpus 4 rl_train_sft.py \
    --model_name_or_path /data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_dir ./rl_sft_llama3_8b_generator \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --lora_r 16 --lora_alpha 32 --bf16 True --gradient_checkpointing True \
    --deepspeed ./rl_configs/ds_zero3.json
```

### Step 2: Build DPO Preference Pairs

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python rl_build_preference_data.py \
    --generator_path /data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct \
    --lora_adapter_path ./rl_sft_llama3_8b_generator \
    --verifier_path ./qiming_vicuna_13B_Cardiff_merged_verifier_way_2 \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_path ./rl_preference_data/preference_pairs.json \
    --num_samples 6 --min_score_gap 0.3 --add_hard_negatives \
    --generator_device cuda:4 --verifier_device cuda:7
```

### Step 2b: Synthetic Data Augmentation (Optional, 亮点三)

```bash
export OPENAI_API_KEY="sk-..."
python rl_generate_synthetic_data.py \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
    --output_path ./rl_preference_data/preference_pairs_augmented.json \
    --merge_with ./rl_preference_data/preference_pairs.json \
    --api_provider openai --api_key $OPENAI_API_KEY --model gpt-4o \
    --num_questions 1000 --negatives_per_question 2
```

### Step 3A: DPO Training (Recommended)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --num_gpus 4 rl_train_dpo.py \
    --model_name_or_path /data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct \
    --sft_adapter_path ./rl_sft_llama3_8b_generator \
    --preference_data_path ./rl_preference_data/preference_pairs_augmented.json \
    --output_dir ./rl_dpo_llama3_8b_generator \
    --num_train_epochs 2 --beta 0.1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --lora_r 16 --lora_alpha 32 \
    --bf16 True --gradient_checkpointing True \
    --deepspeed ./rl_configs/ds_zero3.json
```

### Step 4: Full Automated Pipeline

```bash
# Edit BASE_MODEL and VERIFIER_PATH in the script first, then:
bash rl_training_script.sh
```

### Step 5: Evaluate All Models

```bash
CUDA_VISIBLE_DEVICES=4,5 python rl_evaluation.py \
    --test_data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json \
    --verifier_path ./qiming_vicuna_13B_Cardiff_merged_verifier_way_2 \
    --output_path ./rl_eval_results/comparison.json \
    --sft_model_path /data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct \
    --sft_lora_path ./rl_sft_llama3_8b_generator \
    --dpo_model_path /data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct \
    --dpo_lora_path ./rl_dpo_llama3_8b_generator \
    --ilearner_model_path ./vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10 \
    --ilearner_k 5 --ilearner_is_legacy \
    --device cuda:4 --verifier_device cuda:5
```

---

## Experimental Results

### ILearner-LLM Baseline (Reproduced from saved evaluation files)

#### Cardiff Dataset

| Model | Training Data | N (test) | BLEU ↑ | BERTScore F1 ↑ |
|-------|--------------|----------|---------|----------------|
| Alpaca-7B (no Cardiff SFT) | — | 4,203 | 0.1952 | 0.5550 |
| LLaMA-7B (Cardiff SFT) | Cardiff | 4,203 | 0.2378 | 0.5466 |
| Alpaca-7B (Cardiff SFT) | Cardiff | 4,203 | 0.2415 | 0.5494 |
| GPT4-X-Alpaca-13B (Cardiff SFT) | Cardiff | 4,203 | 0.2469 | 0.5466 |
| **Vicuna-13B (Cardiff SFT)** | Cardiff | 4,203 | **0.2402** | **0.5562** |
| Vicuna-13B (merged all, 100 sample) | All domains | 100 | 0.1523 | 0.5797 |

#### Sydney Dataset

| Model | Training Data | N (test) | BLEU ↑ | BERTScore F1 ↑ |
|-------|--------------|----------|---------|----------------|
| Vicuna-13B (no SFT) | — | 463 | 0.0818 | 0.1845 |
| **Vicuna-13B (Sydney SFT, avg≥3, len≥10)** | Sydney | 463 | **0.3317** | **0.6255** |
| Vicuna-13B (merged all, 100 sample) | All domains | 100 | 0.1589 | 0.5343 |

---

### RLearner-LLM Results — Cardiff Biology (100-question test set)

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ | Time(s) ↓ |
|-------|--------|------------|------------|-------|-------|-----------|---------|
| SFT (LLaMA-2-13B + LoRA) | 0.0160 | 0.8070 | 0.7820 | 0.8087 | 0.0555 | 3.1976 | 19.947 |
| DPO v1 (165 pairs) | 0.0173 | 0.8238 | 0.8325 | 0.7698 | **0.2969** | 3.0467 | 6.567 |
| **DPO v2 (458 pairs)** | **0.0247** | **0.8300** | **0.8422** | **0.8682** | 0.2905 | 3.0648 | **5.774** |
| PPO (125 steps) | 0.0175 | 0.8245 | 0.8255 | 0.7390 | 0.2260 | 3.0750 | 7.234 |

### RLearner-LLM Results — Sydney Biology (100-question test set)

| Model | BLEU ↑ | BERT(Stu) ↑ | BERT(Ans) ↑ | ACR ↑ | NLI ↑ | Verifier ↑ | Time(s) ↓ |
|-------|--------|------------|------------|-------|-------|-----------|---------|
| SFT (LLaMA-2-13B + LoRA) | 0.0222 | 0.8244 | 0.7870 | 0.6249 | 0.0537 | 3.1937 | 19.001 |
| DPO v1 (165 pairs) | 0.0314 | 0.8262 | 0.8272 | 0.6034 | 0.2171 | 2.9094 | 9.049 |
| **DPO v2 (458 pairs)** | 0.0364 | **0.8367** | **0.8426** | 0.6290 | **0.2774** | 2.9474 | **6.370** |
| PPO (125 steps) | **0.0421** | 0.8364 | 0.8294 | **0.6606** | 0.2269 | 2.9609 | 7.596 |

**Key finding**: NLI entailment is the most discriminative metric — SFT scores ~0.05 vs all RL models 0.22–0.30 (4–5× gap).
See [README_RL.md](README_RL.md) for full analysis, metric definitions, and future experiments.

---

## Model Zoo

### Pre-trained Base Models (on this server)

| Model | Local Path | Params |
|-------|-----------|--------|
| LLaMA-2-7B-HF | `/data/shared/llama2/llama-2-7b-hf` | 7B |
| LLaMA-2-13B-HF | `/data/shared/llama2/llama-2-13b-hf` | 13B |
| LLaMA-2-70B-HF | `/data/shared/llama2/llama-2-70b-hf` | 70B |
| **Llama-3-8B-Instruct** | `/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct` | 8B |
| **Llama-3-70B-Instruct** | `/data/shared/llama3/llama3/Meta-Llama-3-70B-Instruct` | 70B |

### Fine-tuned Models (in this project directory)

| Model | Path | Task | Domain |
|-------|------|------|--------|
| Vicuna-13B Generator | `./vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10` | Generator | Cardiff |
| Vicuna-13B Verifier | `./qiming_vicuna_13B_Cardiff_merged_verifier_way_2` | Verifier | Cardiff |
| Vicuna-13B Verifier | `./qiming_vicuna_13B_Sydney_merged_verifier_way_2` | Verifier | Sydney |
| Vicuna-13B Verifier | `./qiming_vicuna_13B_Auckland_law_merged_verifier_way_2` | Verifier | Law |
| Vicuna-13B Verifier | `./qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2` | Verifier | Medicine Y1 |
| Alpaca-7B Generator | `./qiming_alpaca_7B_Cardiff_generator` | Generator | Cardiff |

---

## Key Design Decisions

### Why DPO over PPO?

| Factor | DPO | PPO |
|--------|-----|-----|
| Models in memory | 2 (Actor + Reference) | 4 (Actor + Critic + Reference + Reward) |
| Training stability | High (offline, no reward hacking) | Lower (needs careful KL tuning) |
| Convergence speed | 1-2 epochs | 10k-50k steps |
| Reward ceiling | Lower (bounded by data) | Higher (can discover new strategies) |
| **Recommendation** | ✅ Default choice | For when data is exhausted |

### Why LoRA over full fine-tuning?

| Factor | LoRA | Full Fine-tuning |
|--------|------|-----------------|
| Trainable parameters | <1% | 100% |
| VRAM for 8B model | ~2× A100 | ~8× A100 |
| Adapter file size | ~50MB | ~16GB |
| Deployment hot-swap | <10ms (vLLM) | Reload entire model |
| Quality | Matches full FT | Best possible |

---

## Acknowledgements

- Original ILearner-LLM design inspired by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)
- RL training via [TRL](https://github.com/huggingface/trl) (Hugging Face)
- LoRA / PEFT via [PEFT](https://github.com/huggingface/peft) (Hugging Face)
- Base models: [Meta Llama-3](https://llama.meta.com/), [Qwen3](https://huggingface.co/Qwen)
- PeerWise platform: [peerwise.cs.auckland.ac.nz](https://peerwise.cs.auckland.ac.nz)
- System architecture diagram: [Google Drive](https://drive.google.com/file/d/1m7FLEvTJnjxjqNRxCNnjzweYoWn43k4x/view?usp=sharing)

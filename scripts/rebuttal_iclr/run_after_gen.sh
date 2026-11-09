#!/bin/bash
# Waits for the Hybrid-DPO greedy generation, then judges + Tier-C NLI on GPU 3 only.
cd /data/qbao775/Explanation-Generation
until [ -f .rebuttal_hybrid_gen_DONE ]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
PY=/data/qbao775/miniconda3/envs/gemma4-rl/bin/python
$PY scripts/rebuttal_iclr/judge_hf.py --judge Qwen/Qwen2.5-7B-Instruct
$PY scripts/rebuttal_iclr/judge_hf.py --judge meta-llama/Llama-3.1-8B-Instruct
/data/qbao775/miniconda3/envs/llm-tuning/bin/python scripts/rebuttal_iclr/tierC_policy_nli.py
touch .rebuttal_judge_DONE

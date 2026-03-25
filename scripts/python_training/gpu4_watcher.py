#!/usr/bin/env python3
"""
GPU 4 Experiment Watcher
Monitors GPU 4 free memory and auto-launches remaining experiments when space permits.
Uses lock files to prevent conflicts with other chains running on GPU 5/6/7.

Priority order:
  1. LLaMA-2 PPO-NLI-DPO evals (5 domains)   — needs PPO-NLI-DPO adapter
  2. Qwen3 NLI cross-domain evals (5 domains) — needs Qwen3 NLI XD DPO adapter
  3. Qwen3 hybrid cross-domain evals (5 domains) — needs Qwen3 hybrid XD DPO adapter
  4. LLaMA-2 NLI cross-domain evals (5 domains)  — needs LLaMA-2 NLI XD DPO adapter
  5. LLaMA-2 hybrid cross-domain evals (5 domains) — needs LLaMA-2 hybrid XD DPO adapter
"""

import subprocess
import os
import time
import logging
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = "/data/qbao775/Explanation-Generation"
EVAL_DIR    = f"{BASE}/rl_eval_results"
WATCHER_LOG = f"{BASE}/gpu4_watcher.log"
JOB_LOG     = f"{BASE}/gpu4_watcher_jobs.log"

PY_L = "/data/qbao775/miniconda3/envs/llm-tuning/bin/python3"
PY_Q = "/data/qbao775/miniconda3/envs/qwen3-rl/bin/python3"

LLAMA2  = "/data/shared/llama2/llama-2-13b-hf"
QWEN3   = "/data/shared/qwen3/Qwen3-8B"
SFT_L   = f"{BASE}/rl_sft_llama2_13b_generator"
SFT_Q   = f"{BASE}/rl_sft_qwen3_8b_generator"
VFR     = f"{BASE}/qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2"

GPU_ID  = 4   # Physical GPU index for nvidia-smi query
CV_DEV  = 4   # CUDA_VISIBLE_DEVICES value

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(WATCHER_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

# ── Test domains ───────────────────────────────────────────────────────────────
DOMAINS = [
    ("cardiff", f"{BASE}/Paul_new_data/Cardiff/Cardiff_vicuna_13b_finetuned_random_100.json"),
    ("sydney",  f"{BASE}/Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json"),
    ("law",     f"{BASE}/PeerWiseData/Law/Auckland_law_vicuna_13b_finetuned_random_100.json"),
    ("med_y1",  f"{BASE}/PeerWiseData/Medicine/Medicine_year1_vicuna_13b_finetuned_random_100.json"),
    ("med_y2",  f"{BASE}/PeerWiseData/Medicine/Medicine_year2_vicuna_13b_finetuned_random_100.json"),
]

# ── Helper: build experiment entries ──────────────────────────────────────────
def l2_eval(extra_flags, data, out):
    """LLaMA-2 eval command on GPU 4."""
    return (
        f"CUDA_VISIBLE_DEVICES={CV_DEV} {PY_L} {BASE}/rl_evaluation.py "
        f"--sft_model_path {LLAMA2} --sft_lora_path {SFT_L} "
        f"{extra_flags} "
        f"--verifier_path {VFR} "
        f"--test_data_path {data} --output_path {out} "
        f"--device cuda:0 --verifier_device cuda:0"
    )

def q3_eval(extra_flags, data, out):
    """Qwen3 eval command on GPU 4."""
    return (
        f"CUDA_VISIBLE_DEVICES={CV_DEV} {PY_Q} {BASE}/rl_evaluate_qwen3.py "
        f"--base_model_path {QWEN3} --sft_lora_path {SFT_Q} "
        f"{extra_flags} "
        f"--verifier_path {VFR} "
        f"--test_data_path {data} --output_path {out} "
        f"--device cuda:0 --verifier_device cuda:0 --nli_device cpu"
    )

# ── Experiment queue ───────────────────────────────────────────────────────────
# mem_mib: conservative GPU memory estimate (LLaMA-2 actor+verifier ~41 GB, Qwen3 ~31 GB)
EXPERIMENTS = []

# 1. LLaMA-2 PPO-NLI-DPO evals ─────────────────────────────────────────────────
_L2_PPO_NLI_DPO_ADAPTER = "rl_ppo_nli_llama2_dpo_generator/adapter_model.bin"
_L2_PPO_NLI_DPO_FLAGS   = f"--ppo_model_path {LLAMA2} --ppo_lora_path {BASE}/rl_ppo_nli_llama2_dpo_generator"
for dom, data in DOMAINS:
    out = f"{EVAL_DIR}/llama2_ppo_nli_dpo_{dom}_eval.json"
    EXPERIMENTS.append(dict(
        name     = f"LLaMA-2 PPO-NLI-DPO {dom}",
        deps     = [_L2_PPO_NLI_DPO_ADAPTER],
        output   = out,
        mem_mib  = 42000,
        cmd      = l2_eval(_L2_PPO_NLI_DPO_FLAGS, data, out),
    ))

# 2. Qwen3 NLI cross-domain evals ──────────────────────────────────────────────
_Q3_NLI_XD_ADAPTER = "rl_dpo_nli_cross_domain_qwen3_generator/adapter_model.safetensors"
for dom, data in DOMAINS:
    out = f"{EVAL_DIR}/qwen3_dpo_nli_cross_domain_{dom}_eval.json"
    EXPERIMENTS.append(dict(
        name     = f"Qwen3 NLI-XD {dom}",
        deps     = [_Q3_NLI_XD_ADAPTER],
        output   = out,
        mem_mib  = 32000,
        cmd      = q3_eval(
            f"--dpo_lora_path {BASE}/rl_dpo_nli_cross_domain_qwen3_generator",
            data, out
        ),
    ))

# 3. Qwen3 hybrid cross-domain evals ───────────────────────────────────────────
_Q3_HYB_XD_ADAPTER = "rl_dpo_hybrid_cross_domain_qwen3_generator/adapter_model.safetensors"
for dom, data in DOMAINS:
    out = f"{EVAL_DIR}/qwen3_dpo_hybrid_cross_domain_{dom}_eval.json"
    EXPERIMENTS.append(dict(
        name     = f"Qwen3 Hybrid-XD {dom}",
        deps     = [_Q3_HYB_XD_ADAPTER],
        output   = out,
        mem_mib  = 32000,
        cmd      = q3_eval(
            f"--dpo_lora_path {BASE}/rl_dpo_hybrid_cross_domain_qwen3_generator",
            data, out
        ),
    ))

# 4. LLaMA-2 NLI cross-domain evals ───────────────────────────────────────────
_L2_NLI_XD_ADAPTER = "rl_dpo_nli_cross_domain_generator/adapter_model.bin"
for dom, data in DOMAINS:
    out = f"{EVAL_DIR}/llama2_dpo_nli_cross_domain_{dom}_eval.json"
    EXPERIMENTS.append(dict(
        name     = f"LLaMA-2 NLI-XD {dom}",
        deps     = [_L2_NLI_XD_ADAPTER],
        output   = out,
        mem_mib  = 42000,
        cmd      = l2_eval(
            f"--dpo_model_path {LLAMA2} --dpo_lora_path {BASE}/rl_dpo_nli_cross_domain_generator",
            data, out
        ),
    ))

# 5. LLaMA-2 hybrid cross-domain evals ────────────────────────────────────────
_L2_HYB_XD_ADAPTER = "rl_dpo_hybrid_cross_domain_llama2_generator/adapter_model.bin"
for dom, data in DOMAINS:
    out = f"{EVAL_DIR}/llama2_dpo_hybrid_cross_domain_{dom}_eval.json"
    EXPERIMENTS.append(dict(
        name     = f"LLaMA-2 Hybrid-XD {dom}",
        deps     = [_L2_HYB_XD_ADAPTER],
        output   = out,
        mem_mib  = 42000,
        cmd      = l2_eval(
            f"--dpo_model_path {LLAMA2} --dpo_lora_path {BASE}/rl_dpo_hybrid_cross_domain_llama2_generator",
            data, out
        ),
    ))

log.info(f"Loaded {len(EXPERIMENTS)} experiments across 5 experiment groups.")

# ── Utilities ──────────────────────────────────────────────────────────────────
def get_gpu_free_mib(gpu_index: int) -> int:
    """Return free GPU memory in MiB for the given physical GPU index."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return int(r.stdout.strip())
    except Exception as e:
        log.warning(f"nvidia-smi failed: {e}")
        return 0


def deps_satisfied(exp: dict) -> bool:
    """Return True if all dependency files exist."""
    for dep in exp["deps"]:
        p = Path(BASE, dep)
        if not p.exists():
            return False
    return True


def is_already_running(output_path: str) -> bool:
    """
    Return True if another process is already working toward this output file.
    Checks both: (a) a .lock file, (b) pgrep for the output filename in cmdline.
    """
    # Lock file check
    if Path(output_path + ".lock").exists():
        return True
    # Process cmdline check (basename to match both relative and absolute paths)
    fname = Path(output_path).name
    try:
        r = subprocess.run(
            ["pgrep", "-f", fname],
            capture_output=True, text=True, timeout=5
        )
        pids = [p for p in r.stdout.strip().split("\n") if p and int(p) != os.getpid()]
        return len(pids) > 0
    except Exception:
        return False


def acquire_lock(output_path: str) -> bool:
    """
    Atomically create a lock file. Returns True if we got the lock, False if
    someone else already holds it.
    """
    lock = output_path + ".lock"
    try:
        # O_CREAT | O_EXCL is atomic: fails if file already exists
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(output_path: str):
    """Remove lock file."""
    try:
        Path(output_path + ".lock").unlink(missing_ok=True)
    except Exception:
        pass


def run_experiment(exp: dict) -> int:
    """
    Launch the experiment synchronously (blocks until done).
    Returns process exit code.
    """
    out = exp["output"]
    log.info(f"▶ START  {exp['name']}")
    log.info(f"  CMD: {exp['cmd']}")

    job_log = open(JOB_LOG, "a")
    job_log.write(f"\n\n{'='*70}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {exp['name']}\n{'='*70}\n")
    job_log.flush()

    proc = subprocess.Popen(
        exp["cmd"],
        shell=True,
        cwd=BASE,
        stdout=job_log,
        stderr=subprocess.STDOUT,
    )
    ret = proc.wait()
    job_log.close()

    if ret == 0 and Path(out).exists():
        log.info(f"✓ DONE   {exp['name']}")
    else:
        log.warning(f"✗ FAILED {exp['name']} (exit={ret}, output_exists={Path(out).exists()})")
    return ret


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("GPU 4 Watcher started.")
    log.info(f"Tracking {len(EXPERIMENTS)} experiments.")
    log.info("=" * 60)
    os.chdir(BASE)

    POLL_INTERVAL   = 60    # seconds between scans when idle
    MEM_BUFFER_MIB  = 6000  # keep 6 GB headroom

    consecutive_idle = 0

    while True:
        free_mib = get_gpu_free_mib(GPU_ID)
        log.info(f"[GPU{GPU_ID}] Free: {free_mib} MiB  —  scanning experiments ...")

        launched = False
        all_done_or_running = True  # optimistic; set False if any exp is pending

        for exp in EXPERIMENTS:
            out = exp["output"]

            # Already finished
            if Path(out).exists():
                continue

            all_done_or_running = False  # at least one pending

            # Dependencies not met yet
            if not deps_satisfied(exp):
                missing = [d for d in exp["deps"] if not Path(BASE, d).exists()]
                log.info(f"  [wait-deps]  {exp['name']}  missing: {missing}")
                continue

            # Already running elsewhere (or locked)
            if is_already_running(out):
                log.info(f"  [running-elsewhere]  {exp['name']}")
                continue

            # Check memory
            needed = exp["mem_mib"] + MEM_BUFFER_MIB
            if free_mib < needed:
                log.info(
                    f"  [low-mem]  {exp['name']}  need {needed} MiB, have {free_mib} MiB"
                )
                continue

            # Try to acquire lock (atomic)
            if not acquire_lock(out):
                log.info(f"  [lock-conflict]  {exp['name']}  — already locked")
                continue

            # ── Launch ──────────────────────────────────────────────────────
            try:
                ret = run_experiment(exp)
            finally:
                release_lock(out)

            launched = True
            consecutive_idle = 0
            # Re-scan immediately after finishing (no sleep)
            break

        else:
            # Loop completed without launching
            if all_done_or_running:
                log.info("All experiments are either done or running elsewhere. Exiting.")
                break

        if not launched:
            consecutive_idle += 1
            wait = min(POLL_INTERVAL * consecutive_idle, 300)  # back off up to 5 min
            log.info(f"  Nothing launchable right now. Waiting {wait}s ...")
            time.sleep(wait)
        else:
            consecutive_idle = 0
            # Brief pause to let GPU memory settle before re-check
            time.sleep(10)


if __name__ == "__main__":
    main()

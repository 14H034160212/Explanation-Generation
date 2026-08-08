"""Minimal SimPO trainer for Gemma-4 E4B-it (TRL 0.29 has no CPOTrainer/simpo).

SimPO objective (reference-free, LENGTH-NORMALIZED so there is no incentive to be
longer):
    L = -log sigmoid( beta * ( avg_logp(chosen) - avg_logp(rejected) ) - gamma )
where avg_logp is the mean per-token log-prob of the RESPONSE tokens only.

Trains a fresh LoRA on top of the SFT-merged model. Purpose: test whether a
length-normalized objective avoids the ~3x verbosity drift that plain DPO produced
on this task (v1/v2 both ~193 words vs SFT 61).
"""
import argparse, json, logging, os, math, torch
from torch.nn.utils import clip_grad_norm_
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resp_logprob_sum_and_len(model, tok, prompt, response, device):
    """Return (sum_logprob, n_response_tokens) for response tokens given prompt."""
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=768).input_ids
    full = tok(prompt + response + tok.eos_token, return_tensors="pt",
               truncation=True, max_length=1024).input_ids.to(device)
    p_len = p_ids.shape[1]
    if full.shape[1] <= p_len:
        return None, 0
    out = model(full).logits  # [1, T, V]
    logp = torch.log_softmax(out[:, :-1, :].float(), dim=-1)  # predict token t+1
    targets = full[:, 1:]  # [1, T-1]
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    resp_logp = tok_logp[p_len - 1:]  # response token log-probs
    return resp_logp.sum(), resp_logp.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--sft_adapter", default="./rl_sft_gemma4_e4b_cardiff_tierC_generator")
    ap.add_argument("--pref", default="./rl_preference_data_gemma4_cardiff_judge/preference_pairs_lenmatched.json")
    ap.add_argument("--out", default="./rl_dpo_gemma4_e4b_cardiff_reasoning_simpo_generator")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)   # gamma_beta_ratio = gamma/beta = 0.5
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.base, padding_side="left", trust_remote_code=True, cache_dir="cache")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("loading base + merging SFT LoRA ...")
    model = Gemma4ForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                           trust_remote_code=True, cache_dir="cache")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    model.config.use_cache = False

    target = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target,
                    bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lc)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.to(dev).train()

    pairs = json.load(open(args.pref, encoding="utf-8"))
    logger.info(f"{len(pairs)} pairs | beta={args.beta} gamma={args.gamma} lr={args.lr} epochs={args.epochs}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(pairs) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    step = 0
    for ep in range(args.epochs):
        order = list(range(len(pairs)))
        # deterministic shuffle by epoch (no Math.random dependency)
        order = order[ep % len(order):] + order[:ep % len(order)]
        opt.zero_grad()
        run_loss = run_acc = n = 0
        for bi, idx in enumerate(order):
            p = pairs[idx]
            lc_sum, lc_len = resp_logprob_sum_and_len(model, tok, p["prompt"], p["chosen"], dev)
            lr_sum, lr_len = resp_logprob_sum_and_len(model, tok, p["prompt"], p["rejected"], dev)
            if lc_len == 0 or lr_len == 0:
                continue
            avg_c = lc_sum / lc_len
            avg_r = lr_sum / lr_len
            logits = args.beta * (avg_c - avg_r) - args.gamma
            loss = -torch.nn.functional.logsigmoid(logits)
            (loss / args.accum).backward()
            run_loss += loss.item(); run_acc += 1.0 if (avg_c > avg_r) else 0.0; n += 1
            if (bi + 1) % args.accum == 0:
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    logger.info(f"ep{ep} step{step}/{total_steps} loss={run_loss/max(n,1):.4f} "
                                f"acc={run_acc/max(n,1):.3f} lr={sched.get_last_lr()[0]:.2e}")
                    run_loss = run_acc = n = 0
        # flush leftover grads
        opt.step(); sched.step(); opt.zero_grad()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    logger.info(f"DONE saved -> {args.out}")
    open(".reasoning_simpo_DONE", "w").write("done")


if __name__ == "__main__":
    main()

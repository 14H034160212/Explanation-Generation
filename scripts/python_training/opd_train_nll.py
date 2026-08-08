"""OPD prototype - Phase 2: NLL LoRA fine-tune on reward-filtered on-policy targets.

Standard next-token supervision on the self-generated, judge-preferred, concise
explanations from Phase 1. Because targets are the student's own concise samples,
this reinforces judge-preferred reasoning WITHOUT the verbosity drift that DPO/SimPO
showed. Fresh LoRA on the SFT-merged base (same target modules as the project).
"""
import argparse, json, logging, math, os, torch
from torch.nn.utils import clip_grad_norm_
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_prompt(tok, instruction, input_text):
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--sft_adapter", default="./rl_sft_gemma4_e4b_cardiff_tierC_generator")
    ap.add_argument("--data", default="./rl_preference_data_gemma4_cardiff_opd/selfdistill.json")
    ap.add_argument("--out", default="./rl_dpo_gemma4_e4b_cardiff_opd_generator")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.base, padding_side="right", trust_remote_code=True, cache_dir="cache")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info("loading base + merging SFT LoRA ...")
    model = Gemma4ForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                           trust_remote_code=True, cache_dir="cache")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    model.config.use_cache = False
    target = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target,
                                             bias="none", task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads(); model.print_trainable_parameters(); model.to(dev).train()

    data = json.load(open(args.data, encoding="utf-8"))
    logger.info(f"{len(data)} self-distill targets | epochs={args.epochs} lr={args.lr}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = math.ceil(len(data) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=steps)

    step = 0
    for ep in range(args.epochs):
        order = list(range(len(data))); order = order[ep % len(order):] + order[:ep % len(order)]
        opt.zero_grad(); run = n = 0
        for bi, idx in enumerate(order):
            it = data[idx]
            prompt = build_prompt(tok, it.get("instruction", "").strip(), it.get("input", "").strip())
            p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=768).input_ids
            full = tok(prompt + it["output"] + tok.eos_token, return_tensors="pt",
                       truncation=True, max_length=1024).input_ids.to(dev)
            p_len = p_ids.shape[1]
            if full.shape[1] <= p_len:
                continue
            labels = full.clone()
            labels[:, :p_len] = -100   # supervise response tokens only
            out = model(full, labels=labels)
            loss = out.loss
            (loss / args.accum).backward()
            run += loss.item(); n += 1
            if (bi + 1) % args.accum == 0:
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    logger.info(f"ep{ep} step{step}/{steps} nll={run/max(n,1):.4f} lr={sched.get_last_lr()[0]:.2e}")
                    run = n = 0
        opt.step(); sched.step(); opt.zero_grad()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    logger.info(f"DONE -> {args.out}")
    open(".opd_train_DONE", "w").write("done")


if __name__ == "__main__":
    main()

"""Fast reproduction of the lost Alpaca-7B verifier (way 2).

The original (training_script.sh, 'qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2')
was Alpaca-7B fully fine-tuned for 20 epochs on Cardiff_Sydney_merged_verifier_way_2.json.
That checkpoint is no longer on disk. This reproduction trains a LoRA (r=64, all linear
layers) on LLaMA-7B with the same data and Alpaca prompt, 2 epochs, loss on the response
only. 1,000 examples are held out (seed 0) to check that it predicts the ratings.

Output: ./models/verifier_way2_repro (merged weights, loadable by load_verifier()).
"""
import json, random, os, sys
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

ROOT = "/data/qbao775/Explanation-Generation"
os.chdir(ROOT)
BASE = "llama_7B_hf/llama-7b"
TOK = "llama_7B_hf/tokenizer"
DATA = "preference_data/Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json"
OUT = "models/verifier_way2_repro"
TMPL = ("Below is an instruction that describes a task, paired with an input that provides further "
        "context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
MAXLEN = 1024

d = json.load(open(DATA))
random.Random(0).shuffle(d)
val, train = d[:1000], d[1000:]
os.makedirs(OUT, exist_ok=True)
json.dump(val, open(f"{OUT}/heldout_1000.json", "w"), ensure_ascii=False)

tok = LlamaTokenizer.from_pretrained(TOK, bos_token="<s>", eos_token="</s>", unk_token="<unk>")  # old config has empty special tokens
tok.pad_token_id = 0


class DS(Dataset):
    def __init__(self, rows):
        self.ex = []
        for r in rows:
            p = tok(TMPL.format(**r), add_special_tokens=True)["input_ids"]
            a = tok(str(r["output"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            p = p[: MAXLEN - len(a)]  # truncate the prompt, never the label
            self.ex.append((p + a, [-100] * len(p) + a))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(b):
    n = max(len(x) for x, _ in b)
    ids = torch.tensor([x + [0] * (n - len(x)) for x, _ in b])
    lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in b])
    return {"input_ids": ids, "labels": lab, "attention_mask": (lab != -100) | (ids != 0)}


model = LlamaForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
args = TrainingArguments(output_dir=f"{OUT}_ckpt", num_train_epochs=2, per_device_train_batch_size=8,
                         gradient_accumulation_steps=2, learning_rate=2e-4, warmup_ratio=0.03,
                         lr_scheduler_type="cosine", bf16=True, logging_steps=20, save_strategy="no",
                         report_to="none", remove_unused_columns=False,
                         dataloader_num_workers=2)
Trainer(model=model, args=args, train_dataset=DS(train), data_collator=collate).train()
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("saved", OUT)

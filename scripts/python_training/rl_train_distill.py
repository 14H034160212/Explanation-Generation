import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Verify the file
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    print(f"Loading dataset from: {args.dataset_path}")

    # Load with stdlib json
    with open(args.dataset_path, "r") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} training samples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize the dataset
    def tokenize(example):
        inst = example.get('instruction', '') or ''
        inp = example.get('input', '') or ''
        out = example.get('output', '') or ''
        text = f"Instruction: {inst}\nInput: {inp}\nExplanation: {out}{tokenizer.eos_token}"
        encoded = tokenizer(text, truncation=True, max_length=768, padding="max_length")
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_list(raw_data)
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    print(f"Tokenized {len(tokenized)} samples")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        data_collator=data_collator,
    )

    print("Starting distillation training...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Distilled model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

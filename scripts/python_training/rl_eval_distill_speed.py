import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the distilled LoRA")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Data
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    # 2. Load Model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Loading distilled LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()
    model.eval()

    # 3. Best-of-1 Generation & Timing
    results = []
    total_time = 0.0

    print("Starting purely sequential inference benchmark...")
    for idx, item in enumerate(tqdm(test_data)):
        instruction = item.get("instruction", item.get("question", ""))
        input_text = item.get("input", item.get("choices", ""))
        
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nExplanation:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Time the generation exactly like SFT baseline
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Best-of-1 greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Decode
        input_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        
        results.append({
            "instruction": instruction,
            "input": input_text,
            "prediction": completion,
            "time_seconds": elapsed
        })

    # 4. Summary metrics
    avg_time = total_time / len(test_data)
    print(f"\n=== Benchmark Complete ===")
    print(f"Total time for {len(test_data)} samples: {total_time:.2f}s")
    print(f"Average time per query: {avg_time:.2f}s")

    # Save outputs for score evaluation
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()

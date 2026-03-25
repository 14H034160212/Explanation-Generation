import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/shared/qwen3/Qwen3-8B")
    parser.add_argument("--adapter_path", type=str, default="models/rl_sft_qwen3_8b_generator")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"Loaded adapter from {args.adapter_path}")

    with open(args.test_data, "r") as f:
        data = json.load(f)

    results = []
    # Iterative Explanatory Prompting (NeurIPS 2023 style)
    # We use a structured prompt that forces the model to think step-by-step
    # and then provide an explanation.
    system_prompt = "You are an expert tutor. Provide a logical, step-by-step explanation using the following format:\nThinking: <steps>\nExplanation: <final explanation>"

    for item in tqdm(data):
        context = item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        prompt = f"{system_prompt}\n\nContext: {context}\nQuestion: {question}\nCorrect Answer: {answer}\nLet's think step by step."
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        results.append({
            "question": question,
            "context": context,
            "answer": answer,
            "generated_explanation": response
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()

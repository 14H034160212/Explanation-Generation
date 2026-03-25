import json
import os
import random
from datasets import load_dataset

def format_sciq(split="test", num_samples=100):
    dataset = load_dataset("sciq", split=split)
    samples = list(dataset)
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    formatted_data = []
    for item in samples:
        question = item['question']
        correct_answer = item['correct_answer']
        distractors = [item['distractor1'], item['distractor2'], item['distractor3']]
        
        # Shuffle options
        options = distractors + [correct_answer]
        random.shuffle(options)
        
        correct_idx = options.index(correct_answer)
        correct_letter = chr(ord('A') + correct_idx)
        
        input_text = f"Given question: {question} "
        for i, opt in enumerate(options):
            input_text += f"Option {chr(ord('A') + i)}: {opt} "
        input_text += f"The correct answer is Option {correct_letter}."
        
        explanation = item.get('support', "")
        
        formatted_data.append({
            "instruction": "As an explanation generation expert, can you generate the explanation for the given input?",
            "input": input_text.strip(),
            "Explanation": explanation.strip()
        })
    return formatted_data

def format_arc(subset="ARC-Challenge", num_samples=100):
    dataset = load_dataset("ai2_arc", subset, split="test")
    samples = list(dataset)
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    formatted_data = []
    for item in samples:
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']  # e.g., 'A', 'B', '1', '2'
        
        # ARC sometimes uses '1', '2' instead of 'A', 'B'
        labels = choices['label']
        texts = choices['text']
        
        options_dict = dict(zip(labels, texts))
        
        # Standardize to A, B, C, D
        input_text = f"Given question: {question} "
        correct_letter = 'A'
        for i, (label, text) in enumerate(options_dict.items()):
            letter = chr(ord('A') + i)
            if label == answer_key:
                correct_letter = letter
            input_text += f"Option {letter}: {text} "
            
        input_text += f"The correct answer is Option {correct_letter}."
        
        formatted_data.append({
            "instruction": "As an explanation generation expert, can you generate the explanation for the given input?",
            "input": input_text.strip(),
            "Explanation": ""  # ARC doesn't have explanations
        })
    return formatted_data

if __name__ == "__main__":
    out_dir = "preference_data/Generalization"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Formatting SciQ Test (100)...")
    sciq_test = format_sciq("test", 100)
    with open(f"{out_dir}/sciq_random_100.json", "w", encoding="utf-8") as f:
        json.dump(sciq_test, f, indent=4)

    print("Formatting SciQ Train (1000)...")
    sciq_train = format_sciq("train", 1000)
    with open(f"{out_dir}/sciq_train_1000.json", "w", encoding="utf-8") as f:
        json.dump(sciq_train, f, indent=4)
        
    print("Formatting ARC-Challenge Test (100)...")
    arc_data = format_arc("ARC-Challenge", 100)
    with open(f"{out_dir}/arc_challenge_random_100.json", "w", encoding="utf-8") as f:
        json.dump(arc_data, f, indent=4)
        
    print("Done! Data saved to", out_dir)

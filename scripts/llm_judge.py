import json
import os
import argparse
import random
import time
from typing import List, Dict
import openai
from tqdm import tqdm

class LLMJudge:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def judge_pairwise(self, question: str, context: str, choice_a: str, choice_b: str) -> int:
        """
        Returns:
            1 if choice_a is better
            2 if choice_b is better
            0 if they are equal/tie
        """
        prompt = f"""You are an expert evaluator of AI-generated educational explanations.
Given a question and its context, compare two provided explanations and decide which one is better for a student.

Consider the following criteria:
1. Accuracy: Is the explanation factually correct?
2. Soundness: Is the reasoning logical and easy to follow?
3. Helpfulness: Does it truly help a student understand WHY the answer is correct?

Question: {question}

Context: {context}

Explanation 1:
{choice_a}

Explanation 2:
{choice_b}

Which explanation is better? Provide a very brief justification followed by your choice in the format: "Better: Explanation [1/2/Tie]".
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            if "Better: Explanation 1" in content:
                return 1
            elif "Better: Explanation 2" in content:
                return 2
            else:
                return 0
        except Exception as e:
            print(f"Error during API call: {e}")
            return -1

def load_explanations(file_path: str, model_name: str = None) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "results" in data and isinstance(data["results"], list):
        # Find the correct model detailed results
        models = data.get("detailed_results", {})
        if isinstance(models, list):
            for m in models:
                if model_name is None or m.get("model") == model_name:
                    return m.get("generated_explanations", [])
        elif isinstance(models, dict):
            if model_name in models:
                return models[model_name].get("generated_explanations", [])
            else:
                # Return first one if name not matched
                return next(iter(models.values())).get("generated_explanations", [])
    
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_json", required=True, help="Path to evaluation JSON of Model A (e.g. SFT)")
    parser.add_argument("--model_b_json", required=True, help="Path to evaluation JSON of Model B (e.g. PPO)")
    parser.add_argument("--test_data_json", required=True, help="Path to original test data JSON (for questions)")
    parser.add_argument("--model_a_name", default=None)
    parser.add_argument("--model_b_name", default=None)
    parser.add_argument("--output_path", default="gpt_judge_results.json")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    # Load data
    exps_a = load_explanations(args.model_a_json, args.model_a_name)
    exps_b = load_explanations(args.model_b_json, args.model_b_name)
    
    with open(args.test_data_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    judge = LLMJudge()
    
    results = []
    win_a = 0
    win_b = 0
    ties = 0
    errors = 0

    count = min(len(exps_a), len(exps_b), len(test_data), args.limit)
    print(f"Starting evaluation of {count} samples...")

    for i in tqdm(range(count)):
        item = test_data[i]
        question = item.get("question", "")
        # Get context from input field which usually contains choices etc.
        context = item.get("input", "")
        
        # Shuffle to avoid bias
        order = [1, 2]
        random.shuffle(order)
        
        if order[0] == 1:
            choice_1, choice_2 = exps_a[i], exps_b[i]
        else:
            choice_1, choice_2 = exps_b[i], exps_a[i]
            
        verdict = judge.judge_pairwise(question, context, choice_1, choice_2)
        
        if verdict == -1:
            errors += 1
            result_item = {"index": i, "verdict": "ERROR"}
        elif verdict == 0:
            ties += 1
            result_item = {"index": i, "verdict": "TIE"}
        else:
            # Map back to original models
            winner_is_1 = (verdict == 1)
            is_model_a_1 = (order[0] == 1)
            
            if winner_is_1 == is_model_a_1:
                win_a += 1
                result_item = {"index": i, "verdict": "MODEL_A"}
            else:
                win_b += 1
                result_item = {"index": i, "verdict": "MODEL_B"}
        
        results.append(result_item)
        
        # Small delay to respect rate limits if needed (mini is usually fine)
        # time.sleep(0.1)

    summary = {
        "total": count,
        "win_a": win_a,
        "win_b": win_b,
        "ties": ties,
        "errors": errors,
        "win_rate_b": win_b / (win_a + win_b + ties) if count > 0 else 0
    }
    
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    
    output_data = {
        "args": vars(args),
        "summary": summary,
        "detailed_results": results
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()

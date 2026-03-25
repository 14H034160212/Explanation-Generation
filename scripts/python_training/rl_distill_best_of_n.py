import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True, help="List of reranking JSON results")
    parser.add_argument("--output_file", required=True, help="Output SFT dataset path")
    args = parser.parse_args()

    sft_data = []
    
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found. Skipping.")
            continue
            
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        for item in data:
            # item has 'instruction', 'input', 'prediction' (which is the selected best), 
            # and 'metrics' (hybrid_score, nli, ver, acr, bert)
            prompt = item['instruction'] + "\n" + item['input']
            response = item['prediction']
            
            sft_data.append({
                "instruction": item['instruction'],
                "input": item['input'],
                "output": response
            })
            
    with open(args.output_file, 'w') as f:
        json.dump(sft_data, f, indent=4)
        
    print(f"Distillation dataset created with {len(sft_data)} samples at {args.output_file}")

if __name__ == "__main__":
    main()

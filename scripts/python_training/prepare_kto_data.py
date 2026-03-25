import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input file {args.input_path} not found.")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    kto_data = []
    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Form desirable entry
        kto_data.append({
            "prompt": prompt,
            "completion": chosen,
            "label": True
        })
        # Form undesirable entry
        kto_data.append({
            "prompt": prompt,
            "completion": rejected,
            "label": False
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(kto_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(pairs)} pairs to {len(kto_data)} KTO entries at {args.output_path}")

if __name__ == "__main__":
    main()

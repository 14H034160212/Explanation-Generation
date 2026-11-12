"""M0 verification: does the Chinese-SFT'd checkpoint, prompted with the real
denny-bao-lab ZH_HANS template, produce a coherent Chinese explanation?

Non-invasive on purpose: it talks to a standalone sidecar, imports the live
ZH_HANS surface form so the prompt matches serving exactly, and touches neither
the pilot database nor its profile. It proves the model+template path; wiring
the checkpoint in as a registered generator is a separate, system-touching step.

Run (after serve_rlearner.py is up on --port):
    python scripts/python_training/cn_verify_generator.py \
        --port 8102 --zh-file preference_data/cn/Cardiff_zh_generator_test.json --n 3
"""
from __future__ import annotations

import argparse
import json

import httpx

# The ZH_HANS instruction, mirrored from denny-bao-lab prompts.py. The test
# items are already rendered in the ZH_HANS body surface, so the served prompt
# is instruction + "\n\n" + input, exactly as tutor.py builds it.
ZH_INSTRUCTION = "你是一位习题解析专家，请为下面的题目生成解析，逐个说明每个选项为什么正确或错误。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--model", default="rlearner_zh_sft")
    ap.add_argument("--zh-file", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    items = json.load(open(args.zh_file, encoding="utf-8"))[: args.n]
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    cjk = 0
    for i, it in enumerate(items):
        content = f"{it['instruction']}\n\n{it['input']}"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 300,
        }
        r = httpx.post(url, json=body, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        n_cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
        cjk += n_cjk
        print(f"===== item {i} =====")
        print("[Q]", it["input"][:220])
        print("[model explanation]", text[:600])
        print(f"(chars: {len(text)}, CJK: {n_cjk})\n")

    print(f"verified {len(items)} items; total CJK chars in explanations: {cjk}")
    if cjk < 20 * len(items):
        raise SystemExit("explanations look too non-Chinese -- check the checkpoint/template")


if __name__ == "__main__":
    main()

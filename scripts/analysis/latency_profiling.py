"""
Latency Profiling: compile inference time table from all eval result files.

Reads avg_inference_time_s from every results JSON and produces a
per-domain × per-model summary sorted by latency.

Usage:
    python scripts/analysis/latency_profiling.py \
        --results_dir rl_eval_results \
        --output_path rl_eval_results/latency_profiling.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


DOMAIN_MAP = {
    "cardiff": "Cardiff",
    "sydney": "Sydney",
    "law": "Law",
    "med_y1": "Medicine Y1",
    "med_y2": "Medicine Y2",
    "sciq": "SciQ",
    "arc": "ARC",
}

DOMAIN_ORDER = ["Cardiff", "Sydney", "Law", "Medicine Y1", "Medicine Y2", "SciQ", "ARC"]


def infer_domain(filename: str) -> str:
    fn = filename.lower()
    for key, label in DOMAIN_MAP.items():
        if key in fn:
            return label
    return "Other"


def load_results(path: str):
    try:
        d = json.load(open(path))
        if not isinstance(d, dict):
            return []
        results = d.get("results", [])
        if isinstance(results, dict):
            results = [results]
        rows = []
        for r in results:
            if not r or "avg_inference_time_s" not in r:
                continue
            rows.append({
                "model": r.get("model", "unknown"),
                "time": float(r.get("avg_inference_time_s", 0)),
                "nli": float(r.get("avg_nli_entailment", 0)),
                "bleu": float(r.get("avg_bleu", 0)),
                "bert_ans": float(r.get("avg_bert_score_f1_answer_anchored", 0)),
                "acr": float(r.get("avg_answer_coverage_rate", 0)),
                "verifier": float(r.get("avg_verifier_score", 0)),
                "n": int(r.get("n_examples", 0)),
                "source": os.path.basename(path),
            })
        return rows
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="rl_eval_results")
    parser.add_argument("--output_path", default="rl_eval_results/latency_profiling.json")
    args = parser.parse_args()

    all_rows = []
    for p in sorted(Path(args.results_dir).glob("*.json")):
        if "calibration" in p.name or "error_analysis" in p.name or "latency" in p.name:
            continue
        rows = load_results(str(p))
        domain = infer_domain(p.name)
        for r in rows:
            all_rows.append(dict(r, domain=domain))

    # Deduplicate: per (model, domain) keep shortest-source-name entry
    seen: dict = {}
    for r in all_rows:
        key = (r["model"], r["domain"])
        if key not in seen or len(r["source"]) < len(seen[key]["source"]):
            seen[key] = r

    deduped = list(seen.values())

    # Print per-domain latency tables
    by_domain = defaultdict(list)
    for r in deduped:
        by_domain[r["domain"]].append(r)

    output_tables = {}
    for domain in DOMAIN_ORDER:
        rows = sorted(by_domain.get(domain, []), key=lambda x: x["time"])
        if not rows:
            continue
        print(f"\n=== {domain} — Inference Latency ===")
        hdr = f"{'Model':<45} {'Time(s)':>8}  {'NLI':>7}  {'BLEU':>7}  {'ACR':>7}  {'Ver':>5}"
        print(hdr)
        print("-" * 85)
        table_rows = []
        for r in rows:
            print(f"{r['model']:<45} {r['time']:>8.2f}  {r['nli']:>7.4f}  "
                  f"{r['bleu']:>7.4f}  {r['acr']:>7.4f}  {r['verifier']:>5.2f}")
            table_rows.append(r)
        output_tables[domain] = table_rows

    # Cross-domain summary
    print(f"\n{'='*90}")
    print("CROSS-DOMAIN LATENCY SUMMARY")
    print(f"{'Domain':<20} {'Models':>7}  {'Min(s)':>8}  {'Max(s)':>8}  {'Avg(s)':>8}")
    print("-" * 60)
    for domain in DOMAIN_ORDER:
        rows = by_domain.get(domain, [])
        if not rows:
            continue
        times = [r["time"] for r in rows]
        print(f"{domain:<20} {len(rows):>7}  {min(times):>8.2f}  {max(times):>8.2f}  "
              f"{sum(times)/len(times):>8.2f}")

    # Speedup vs ILearner-LLM (K=5) baseline
    print(f"\n{'='*90}")
    print("SPEEDUP vs ILearner-LLM (K=5) — Cardiff & Sydney")
    print("-" * 60)
    for domain in ["Cardiff", "Sydney"]:
        rows_d = {r["model"]: r for r in by_domain.get(domain, [])}
        baseline = rows_d.get("ILearner-LLM (K=5)")
        if not baseline:
            continue
        bt = baseline["time"]
        for model, r in sorted(rows_d.items(), key=lambda x: x[1]["time"]):
            speedup = bt / r["time"] if r["time"] > 0 else 0
            print(f"  [{domain}] {model:<45} {r['time']:>7.2f}s  {speedup:>6.1f}x speedup")

    # Save JSON
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    output = {
        "description": "Inference latency profiling across all models and domains",
        "per_domain": output_tables,
        "cross_domain_summary": {
            domain: {
                "n_models": len(rows),
                "min_time_s": round(min(r["time"] for r in rows), 3),
                "max_time_s": round(max(r["time"] for r in rows), 3),
                "avg_time_s": round(sum(r["time"] for r in rows) / len(rows), 3),
            }
            for domain, rows in by_domain.items() if rows
        },
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()

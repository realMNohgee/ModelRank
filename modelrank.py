#!/usr/bin/env python3
"""
ModelRank — Benchmark and compare LLMs. Track the leaderboard over time.
Usage: python modelrank.py --compare llama-4-70b,deepseek-v4 --benchmark all
"""

import json, os, time, argparse
from typing import Dict, List, Tuple
from collections import defaultdict

LEADERBOARD_FILE = os.path.expanduser("~/.modelrank_leaderboard.json")

BENCHMARKS = {
    "mmlu": "Massive Multitask Language Understanding — 57 subjects, measures knowledge breadth",
    "humaneval": "HumanEval — Python code generation, measures coding ability",
    "gsm8k": "Grade School Math 8K — multi-step math reasoning",
    "hellaswag": "HellaSwag — commonsense reasoning, sentence completion",
    "truthfulqa": "TruthfulQA — measures tendency to reproduce falsehoods",
    "bbh": "BIG-Bench Hard — 23 challenging reasoning tasks",
}

MODELS = {
    "gpt-4o": {"provider": "OpenAI", "release": "2024-05", "type": "proprietary",
               "scores": {"mmlu": 88.7, "humaneval": 90.2, "gsm8k": 92.0, "hellaswag": 95.3, "truthfulqa": 78.0, "bbh": 86.0}},
    "claude-sonnet-4": {"provider": "Anthropic", "release": "2025-06", "type": "proprietary",
                        "scores": {"mmlu": 89.5, "humaneval": 92.0, "gsm8k": 93.5, "hellaswag": 96.1, "truthfulqa": 80.0, "bbh": 88.5}},
    "deepseek-v4": {"provider": "DeepSeek", "release": "2025-12", "type": "open-source",
                    "scores": {"mmlu": 87.2, "humaneval": 88.5, "gsm8k": 91.8, "hellaswag": 94.0, "truthfulqa": 76.5, "bbh": 84.0}},
    "llama-4-70b": {"provider": "Meta", "release": "2025-04", "type": "open-source",
                    "scores": {"mmlu": 86.0, "humaneval": 85.5, "gsm8k": 89.0, "hellaswag": 93.5, "truthfulqa": 74.0, "bbh": 82.5}},
    "gemini-2.5-pro": {"provider": "Google", "release": "2025-03", "type": "proprietary",
                       "scores": {"mmlu": 90.1, "humaneval": 91.5, "gsm8k": 94.0, "hellaswag": 96.5, "truthfulqa": 81.0, "bbh": 89.0}},
    "qwen-3": {"provider": "Alibaba", "release": "2025-09", "type": "open-source",
               "scores": {"mmlu": 85.5, "humaneval": 84.0, "gsm8k": 87.5, "hellaswag": 92.0, "truthfulqa": 73.0, "bbh": 81.0}},
    "mistral-large": {"provider": "Mistral", "release": "2025-02", "type": "open-source",
                      "scores": {"mmlu": 84.8, "humaneval": 83.5, "gsm8k": 86.0, "hellaswag": 91.0, "truthfulqa": 72.5, "bbh": 80.0}},
    "grok-3": {"provider": "xAI", "release": "2025-10", "type": "proprietary",
               "scores": {"mmlu": 87.8, "humaneval": 89.0, "gsm8k": 90.5, "hellaswag": 94.5, "truthfulqa": 77.5, "bbh": 85.0}},
}

def load_leaderboard() -> Dict:
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE) as f:
            return json.load(f)
    return {"history": [], "snapshots": []}

def save_leaderboard(data: Dict):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def compare_models(model_names: List[str], benchmark: str = None) -> List[Dict]:
    """Compare specified models on benchmarks."""
    results = []
    benchmarks = [benchmark] if benchmark else list(BENCHMARKS.keys())
    
    for name in model_names:
        if name not in MODELS:
            results.append({"model": name, "error": "Unknown model"})
            continue
        
        model = MODELS[name]
        scores = {}
        for bench in benchmarks:
            scores[bench] = model["scores"].get(bench, "N/A")
        
        avg = sum(s for s in scores.values() if isinstance(s, (int, float))) / len(scores)
        
        results.append({
            "model": name,
            "provider": model["provider"],
            "type": model["type"],
            "release": model["release"],
            "scores": scores,
            "average": round(avg, 1),
            "strengths": [b for b, s in scores.items() if isinstance(s, (int, float)) and s >= 90],
        })
    
    return sorted(results, key=lambda r: r.get("average", 0), reverse=True)

def snapshot_leaderboard():
    """Take a snapshot of current rankings for historical tracking."""
    data = load_leaderboard()
    timestamp = time.strftime("%Y-%m-%d")
    
    rankings = []
    for name, model in MODELS.items():
        avg = sum(s for s in model["scores"].values()) / len(model["scores"])
        rankings.append({"model": name, "average": round(avg, 1), "type": model["type"]})
    
    rankings.sort(key=lambda r: r["average"], reverse=True)
    
    data["snapshots"].append({
        "date": timestamp,
        "rankings": rankings,
    })
    save_leaderboard(data)
    return rankings

def show_trends():
    """Show how rankings have changed over time."""
    data = load_leaderboard()
    if len(data["snapshots"]) < 2:
        print("Need at least 2 snapshots to show trends. Run 'snapshot' first.")
        return
    
    latest = data["snapshots"][-1]["rankings"]
    previous = data["snapshots"][-2]["rankings"]
    
    print("\n📈 Model Ranking Trends\n")
    print(f"{'Model':<20} {'Prev':<6} {'Now':<6} {'Δ':<6}")
    print("-" * 42)
    
    prev_map = {r["model"]: r["average"] for r in previous}
    for r in latest:
        prev_avg = prev_map.get(r["model"], 0)
        delta = r["average"] - prev_avg
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"{r['model']:<20} {prev_avg:<6.1f} {r['average']:<6.1f} {arrow} {abs(delta):.1f}")

def main():
    parser = argparse.ArgumentParser(description="ModelRank — LLM Benchmark Comparison")
    sub = parser.add_subparsers(dest="command", help="Commands")
    
    compare_parser = sub.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", help="Comma-separated model names, or 'all'")
    compare_parser.add_argument("--benchmark", "-b", help="Specific benchmark (default: all)")
    
    sub.add_parser("list", help="List all tracked models")
    sub.add_parser("leaderboard", help="Show current leaderboard")
    sub.add_parser("snapshot", help="Save current rankings snapshot")
    sub.add_parser("trends", help="Show ranking trends over time")
    sub.add_parser("benchmarks", help="List available benchmarks")
    
    args = parser.parse_args()
    
    if args.command == "compare":
        models = list(MODELS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",")]
        results = compare_models(models, args.benchmark)
        
        print(f"\n📊 Model Comparison")
        bench_label = args.benchmark or "Overall Average"
        print(f"   Benchmark: {bench_label}\n")
        print(f"{'Rank':<5} {'Model':<20} {'Type':<14} {'Avg':<6} {'Strengths'}")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            if "error" in r:
                print(f"{i:<5} {r['model']:<20} ❌ {r['error']}")
            else:
                strengths = ", ".join(r["strengths"][:2]) if r["strengths"] else "—"
                print(f"{i:<5} {r['model']:<20} {r['type']:<14} {r['average']:<6.1f} {strengths}")
    
    elif args.command == "list":
        print("\n📋 Tracked Models\n")
        for name, info in MODELS.items():
            print(f"  {name:<20} {info['provider']:<12} {info['type']:<14} ({info['release']})")
    
    elif args.command == "leaderboard":
        rankings = sorted([(n, sum(m["scores"].values())/len(m["scores"])) for n, m in MODELS.items()], 
                         key=lambda x: x[1], reverse=True)
        print(f"\n🏆 LLM Leaderboard\n")
        print(f"{'Rank':<5} {'Model':<20} {'Score':<6}")
        print("-" * 35)
        for i, (name, avg) in enumerate(rankings, 1):
            print(f"#{i:<4} {name:<20} {avg:.1f}")
    
    elif args.command == "snapshot":
        rankings = snapshot_leaderboard()
        print(f"✅ Snapshot saved — {len(rankings)} models ranked")
    
    elif args.command == "trends":
        show_trends()
    
    elif args.command == "benchmarks":
        print("\n📋 Available Benchmarks\n")
        for name, desc in BENCHMARKS.items():
            print(f"  {name:<15} {desc}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test all extraction methods (rebuild_results_csv_*.py) for models 44 & 45."""

import subprocess
import re
import csv
from pathlib import Path
from collections import defaultdict

# Model configurations
MODELS = {
    44: {
        "name": "Arcee AI: Trinity Large Thinking",
        "folder": "arcee-ai_trinity-large-thinking"
    },
    45: {
        "name": "MoonshotAI: Kimi K2 Thinking",
        "folder": "moonshotai_kimi-k2-thinking"
    }
}

# Extraction methods (rebuild scripts)
EXTRACTION_METHODS = [
    "qwen",
    "llama",
    "gemma9b",
    "gemma",
    "gpt",
    "gemini",
    "olmo",
    "ministrail"
]

def run_rebuild(method, model_num):
    """Run rebuild script for a specific extraction method and model."""
    model_info = MODELS[model_num]
    folder = model_info["folder"]

    input_root = f"result/benchmarks/gsm8k/{folder}"
    report_dir = f"result/archive/reports_mixed_legacy/{folder}_{method}"

    script = f"src/pipelines/rebuild/rebuild_results_csv_{method}.py"

    print(f"\n  [REBUILD] {method:15s} - Running rebuild...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["python", script, "--input_root", input_root, "--report_dir", report_dir],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("[OK]")
            return report_dir
        else:
            print(f"[FAIL] {result.stderr[:100]}")
            return None

    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)[:100]}")
        return None

def run_calc_accuracy(report_dir):
    """Run calc_accuracy.py for a report directory and extract overall accuracy."""
    csv_path = str(Path(report_dir) / "results.csv")

    print(f"  [ACCURACY] {Path(report_dir).name:35s} - ", end="", flush=True)

    try:
        result = subprocess.run(
            ["python", "src/pipelines/analysis/calc_accuracy.py", "--csv_path", csv_path, "--out_dir", report_dir],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Extract OVERALL accuracy from output
        match = re.search(r'OVERALL\s+\d+\s+\d+\s+([\d.]+)%', result.stdout)
        if match:
            accuracy = float(match.group(1))
            print(f"{accuracy:6.2f}%")
            return accuracy
        else:
            print("[NO ACCURACY FOUND]")
            return None

    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)[:80]}")
        return None

def main():
    print("\n" + "="*100)
    print("TESTING ALL EXTRACTION METHODS FOR MODELS 44 & 45")
    print("="*100)

    results = defaultdict(dict)

    for model_num in [44, 45]:
        model_info = MODELS[model_num]
        print(f"\n\nMODEL {model_num}: {model_info['name']}")
        print("-" * 100)

        for method in EXTRACTION_METHODS:
            print(f"\n  Testing extraction method: {method}")

            # Run rebuild
            report_dir = run_rebuild(method, model_num)
            if not report_dir:
                print(f"  [SKIP] Failed to rebuild")
                results[model_num][method] = None
                continue

            # Run accuracy calculation
            accuracy = run_calc_accuracy(report_dir)
            if accuracy is not None:
                results[model_num][method] = accuracy
            else:
                results[model_num][method] = None

    # Print final comparison
    print("\n\n" + "="*100)
    print("FINAL COMPARISON")
    print("="*100)

    for model_num in [44, 45]:
        model_info = MODELS[model_num]
        print(f"\nMODEL {model_num}: {model_info['name']}")
        print("-" * 100)

        # Sort by accuracy descending
        sorted_results = sorted(
            results[model_num].items(),
            key=lambda x: x[1] if x[1] is not None else -1,
            reverse=True
        )

        print(f"{'Method':<15} | {'Accuracy':>8} | Status")
        print("-" * 40)

        for method, accuracy in sorted_results:
            if accuracy is not None:
                status = "OK"
                print(f"{method:<15} | {accuracy:>7.2f}% | {status}")
            else:
                status = "FAIL"
                print(f"{method:<15} | {'N/A':>8} | {status}")

        # Best and worst
        valid_results = [(m, a) for m, a in sorted_results if a is not None]
        if valid_results:
            best_method, best_acc = valid_results[0]
            worst_method, worst_acc = valid_results[-1]
            print("\n" + "-" * 40)
            print(f"BEST:  {best_method:20s} {best_acc:7.2f}%")
            print(f"WORST: {worst_method:20s} {worst_acc:7.2f}%")

    # Overall ranking
    print("\n\n" + "="*100)
    print("OVERALL RANKING (AVERAGE ACROSS MODELS 44 & 45)")
    print("="*100)

    avg_scores = {}
    for method in EXTRACTION_METHODS:
        scores = [results[m][method] for m in [44, 45] if results[m][method] is not None]
        if len(scores) == 2:
            avg_scores[method] = sum(scores) / len(scores)

    sorted_avg = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Method':<15} | {'Avg Accuracy':>12} | Model 44 | Model 45")
    print("-" * 60)

    for method, avg_acc in sorted_avg:
        m44 = results[44].get(method)
        m45 = results[45].get(method)
        m44_str = f"{m44:7.2f}%" if m44 is not None else "  N/A"
        m45_str = f"{m45:7.2f}%" if m45 is not None else "  N/A"
        print(f"{method:<15} | {avg_acc:>11.2f}% | {m44_str} | {m45_str}")

    print("\n[OK] Test complete!")

if __name__ == "__main__":
    main()

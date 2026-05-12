#!/usr/bin/env python3
"""Test all extraction methods for Religion_Persona (deepseek model)."""

import subprocess
import re
import csv
from pathlib import Path
from collections import defaultdict

# Model configuration
MODEL = {
    "name": "DeepSeek: deepseek-r1-distill-qwen-32b",
    "folder": "deepseek_deepseek-r1-distill-qwen-32b"
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

def run_rebuild(method):
    """Run rebuild script for a specific extraction method."""
    input_root = f"result/GMS8K/Religion_Persona/{MODEL['folder']}"
    report_dir = f"result/reports/Religion_Persona_{MODEL['folder']}_{method}"

    script = f"src/rebuild_results_csv_{method}.py"

    print(f"  [REBUILD] {method:15s} - Running rebuild...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["python", script, "--input_root", input_root, "--report_dir", report_dir],
            capture_output=True,
            text=True,
            timeout=180
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

    method_name = Path(report_dir).name.split("_")[-1]
    print(f"  [ACCURACY] {method_name:50s} - ", end="", flush=True)

    try:
        result = subprocess.run(
            ["python", "src/calc_accuracy.py", "--csv_path", csv_path, "--out_dir", report_dir],
            capture_output=True,
            text=True,
            timeout=180
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
    print("TESTING ALL EXTRACTION METHODS FOR RELIGION_PERSONA (DeepSeek)")
    print("="*100)

    results = {}

    print(f"\nMODEL: {MODEL['name']}")
    print(f"Folder: {MODEL['folder']}")
    print("-" * 100)

    for method in EXTRACTION_METHODS:
        print(f"\nTesting extraction method: {method}")

        # Run rebuild
        report_dir = run_rebuild(method)
        if not report_dir:
            print(f"  [SKIP] Failed to rebuild")
            results[method] = None
            continue

        # Run accuracy calculation
        accuracy = run_calc_accuracy(report_dir)
        if accuracy is not None:
            results[method] = accuracy
        else:
            results[method] = None

    # Print final comparison
    print("\n\n" + "="*100)
    print("FINAL COMPARISON - RELIGION_PERSONA (DeepSeek)")
    print("="*100)

    print(f"\n{'Method':<15} | {'Accuracy':>8} | Status")
    print("-" * 40)

    # Sort by accuracy descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1] if x[1] is not None else -1,
        reverse=True
    )

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
        print(f"DIFF:  {best_acc - worst_acc:7.2f}%")

    print("\n[OK] Test complete!")

if __name__ == "__main__":
    main()

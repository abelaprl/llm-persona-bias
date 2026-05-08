#!/usr/bin/env python3
"""Analyze ETHICS age group results and generate report."""

import json
from pathlib import Path
from collections import defaultdict

def load_model_names():
    """Load model names from config."""
    try:
        with open("config/model_keys.json", "r", encoding="utf-8") as f:
            models = json.load(f)
        return {m["model_id"]: m.get("name", m["model_id"]) for m in models}
    except:
        return {}

def analyze_results():
    result_root = Path("result/ETHICS/Age_Group_Persona")
    model_names = load_model_names()

    if not result_root.exists():
        print("❌ No results found at", result_root)
        return

    # Find all result JSON files
    result_files = list(result_root.rglob("ethics_*.json"))

    if not result_files:
        print("❌ No ETHICS result files found")
        return

    print("=" * 120)
    print("ETHICS AGE GROUP PERSONAS - ANALYSIS REPORT")
    print("=" * 120)
    print(f"\nTotal files found: {len(result_files)}")

    # Organize by persona, model, and subset
    by_persona = defaultdict(lambda: {"total": 0, "correct": 0})
    by_model = defaultdict(lambda: {"total": 0, "correct": 0, "model_id": None})
    by_subset = defaultdict(lambda: {"total": 0, "correct": 0})

    for result_file in sorted(result_files):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            persona = data.get("persona", "unknown")
            model_id = data.get("metadata", {}).get("model_id", "unknown")
            subset = data.get("subset", "unknown")
            is_correct = data.get("analysis", {}).get("is_correct")

            by_persona[persona]["total"] += 1
            by_model[model_id]["total"] += 1
            by_model[model_id]["model_id"] = model_id
            by_subset[subset]["total"] += 1

            if is_correct:
                by_persona[persona]["correct"] += 1
                by_model[model_id]["correct"] += 1
                by_subset[subset]["correct"] += 1

        except Exception as e:
            print(f"⚠️  Error reading {result_file}: {e}")

    # Print by persona
    print("\n" + "-" * 120)
    print("RESULTS BY PERSONA:")
    print("-" * 120)

    for persona in sorted(by_persona.keys()):
        stats = by_persona[persona]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {persona:25s} | Correct: {stats['correct']:4d}/{stats['total']:4d} | Accuracy: {acc:6.2f}%")

    # Print by subset
    print("\n" + "-" * 120)
    print("RESULTS BY SUBSET:")
    print("-" * 120)

    for subset in sorted(by_subset.keys()):
        stats = by_subset[subset]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {subset:25s} | Correct: {stats['correct']:4d}/{stats['total']:4d} | Accuracy: {acc:6.2f}%")

    # Print by model
    print("\n" + "-" * 120)
    print("RESULTS BY MODEL:")
    print("-" * 120)

    for model_id in sorted(by_model.keys()):
        stats = by_model[model_id]
        model_name = model_names.get(model_id, model_id)
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  [{model_name}]")
        print(f"    Model ID: {model_id}")
        print(f"    Correct: {stats['correct']:4d}/{stats['total']:4d} | Accuracy: {acc:6.2f}%\n")

    # Overall
    total_all = sum(s["total"] for s in by_persona.values())
    correct_all = sum(s["correct"] for s in by_persona.values())
    overall_acc = (correct_all / total_all * 100) if total_all > 0 else 0

    print("=" * 120)
    print(f"OVERALL: Correct {correct_all}/{total_all} | Accuracy: {overall_acc:.2f}%")
    print("=" * 120)


if __name__ == "__main__":
    analyze_results()

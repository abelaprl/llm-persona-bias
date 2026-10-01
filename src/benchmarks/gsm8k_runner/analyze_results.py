#!/usr/bin/env python3
"""Analyze GSM8K age group results and generate report."""

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
    result_root = Path("result/GMS8K/Age_Group_Persona")
    model_names = load_model_names()

    if not result_root.exists():
        print("❌ No results found at", result_root)
        return

    # Find all result JSON files
    result_files = list(result_root.rglob("gsm8k_*.json"))

    if not result_files:
        print("❌ No GSM8K result files found")
        return

    print("=" * 100)
    print("GSM8K AGE GROUP PERSONAS - ANALYSIS REPORT")
    print("=" * 100)
    print(f"\nTotal files found: {len(result_files)}")

    # Organize by persona and model
    by_persona = defaultdict(lambda: {"total": 0, "correct": 0})
    by_model = defaultdict(lambda: {"total": 0, "correct": 0, "model_id": None})

    for result_file in sorted(result_files):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            persona = data.get("input", {}).get("persona", {})
            persona_name = persona.get("type") or "unknown"
            model_id = data.get("metadata", {}).get("model_info", {}).get("model_id", "unknown")
            is_correct = data.get("analysis", {}).get("is_correct")

            by_persona[persona_name]["total"] += 1
            by_model[model_id]["total"] += 1
            by_model[model_id]["model_id"] = model_id

            if is_correct:
                by_persona[persona_name]["correct"] += 1
                by_model[model_id]["correct"] += 1

        except Exception as e:
            print(f"⚠️  Error reading {result_file}: {e}")

    # Print by persona
    print("\n" + "-" * 100)
    print("RESULTS BY PERSONA:")
    print("-" * 100)

    for persona in sorted(by_persona.keys()):
        stats = by_persona[persona]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {persona:25s} | Correct: {stats['correct']:3d}/{stats['total']:3d} | Accuracy: {acc:6.2f}%")

    # Print by model
    print("\n" + "-" * 100)
    print("RESULTS BY MODEL:")
    print("-" * 100)

    for model_id in sorted(by_model.keys()):
        stats = by_model[model_id]
        model_name = model_names.get(model_id, model_id)
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  [{model_name}]")
        print(f"    Model ID: {model_id}")
        print(f"    Correct: {stats['correct']:3d}/{stats['total']:3d} | Accuracy: {acc:6.2f}%\n")

    # Overall
    total_all = sum(s["total"] for s in by_persona.values())
    correct_all = sum(s["correct"] for s in by_persona.values())
    overall_acc = (correct_all / total_all * 100) if total_all > 0 else 0

    print("=" * 100)
    print(f"OVERALL: Correct {correct_all}/{total_all} | Accuracy: {overall_acc:.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    analyze_results()

#!/usr/bin/env python3
"""Rebuild CSV and analyze accuracy for age group persona models (44, 45, 43)."""

import json
import csv
import sys
import os
from pathlib import Path
from collections import defaultdict

# Model config
MODELS = {
    43: {
        "name": "DeepSeek: R1 Distill Qwen 32B",
        "model_id": "deepseek/deepseek-r1-distill-qwen-32b",
        "folder": "deepseek_deepseek-r1-distill-qwen-32b"
    },
    44: {
        "name": "Arcee AI: Trinity Large Thinking",
        "model_id": "arcee-ai/trinity-large-thinking",
        "folder": "arcee-ai_trinity-large-thinking"
    },
    45: {
        "name": "MoonshotAI: Kimi K2 Thinking",
        "model_id": "moonshotai/kimi-k2-thinking",
        "folder": "moonshotai_kimi-k2-thinking"
    }
}

AGE_GROUP_PERSONAS = [
    'young_explicit', 'young_implicit_1', 'young_implicit_2', 'young_implicit_3',
    'elderly_explicit', 'elderly_implicit_1', 'elderly_implicit_2', 'elderly_implicit_3'
]

def extract_answer(response_text):
    """Extract numeric answer from response text."""
    if not response_text:
        return None

    import re
    # Try to find "#### ANSWER" or similar
    match = re.search(r'####\s*(\d+)', response_text)
    if match:
        return int(match.group(1))

    # Try to find last number
    numbers = re.findall(r'\d+', response_text)
    if numbers:
        return int(numbers[-1])

    return None

def rebuild_csv_for_model(model_order):
    """Rebuild CSV from JSON results for a specific model."""
    model_info = MODELS.get(model_order)
    if not model_info:
        print(f"❌ Model {model_order} not found")
        return None

    results_root = Path("result/GMS8K/Age_Group_Persona") / model_info["folder"]
    report_dir = Path("result/reports") / model_info["folder"]

    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        return None

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "results.csv"

    print(f"\n📊 Rebuilding CSV for Model {model_order}: {model_info['name']}")
    print(f"   Input: {results_root}")
    print(f"   Output: {csv_path}")

    # Find all JSON files
    json_files = list(results_root.rglob("gsm8k_*.json"))
    print(f"   Found {len(json_files)} result files")

    if not json_files:
        print(f"⚠️  No result files found!")
        return None

    rows = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract fields
            persona = data.get("input", {}).get("persona", {})
            persona_name = persona.get("type", "unknown")
            question_id = data.get("input", {}).get("question_id", 0)
            question = data.get("input", {}).get("question", "")
            ground_truth = data.get("input", {}).get("ground_truth", "")

            # Parse ground truth to get expected answer
            expected_answer = None
            if ground_truth:
                match = re.search(r'####\s*(\d+)', ground_truth)
                if match:
                    expected_answer = int(match.group(1))

            # Extract response
            response_content = data.get("response", {}).get("content", "")
            extracted_answer = extract_answer(response_content)

            # Check correctness
            is_correct = extracted_answer == expected_answer if extracted_answer is not None else False

            # Token usage
            tokens = data.get("response", {}).get("raw_api_response", {}).get("usage", {})
            prompt_tokens = tokens.get("prompt_tokens", 0)
            completion_tokens = tokens.get("completion_tokens", 0)
            reasoning_tokens = tokens.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            total_tokens = tokens.get("total_tokens", 0)

            rows.append({
                "question_id": question_id,
                "persona_folder": persona_name,
                "extracted_answer": extracted_answer or "",
                "expected_answer": expected_answer or "",
                "is_correct": str(is_correct),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            })

        except Exception as e:
            print(f"   ⚠️  Error processing {json_file.name}: {e}")

    # Write CSV
    if rows:
        fieldnames = ["question_id", "persona_folder", "extracted_answer", "expected_answer",
                     "is_correct", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"   ✅ CSV saved: {csv_path}")
        return csv_path

    return None

def analyze_accuracy(csv_path, report_dir):
    """Analyze accuracy from CSV."""
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return

    print(f"\n📈 Analyzing accuracy...")

    persona_stats = defaultdict(lambda: {
        'correct': 0, 'total': 0,
        'sum_prompt_tokens': 0, 'sum_completion_tokens': 0, 'sum_reasoning_tokens': 0
    })
    overall = {'correct': 0, 'total': 0, 'sum_prompt_tokens': 0, 'sum_completion_tokens': 0, 'sum_reasoning_tokens': 0}

    try:
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                persona = row['persona_folder']
                is_correct = row['is_correct'].strip().lower() == 'true'

                stats = persona_stats[persona]
                stats['total'] += 1
                overall['total'] += 1

                if is_correct:
                    stats['correct'] += 1
                    overall['correct'] += 1

                # Token counts
                stats['sum_prompt_tokens'] += int(row.get('prompt_tokens', 0) or 0)
                stats['sum_completion_tokens'] += int(row.get('completion_tokens', 0) or 0)
                stats['sum_reasoning_tokens'] += int(row.get('reasoning_tokens', 0) or 0)

                overall['sum_prompt_tokens'] += int(row.get('prompt_tokens', 0) or 0)
                overall['sum_completion_tokens'] += int(row.get('completion_tokens', 0) or 0)
                overall['sum_reasoning_tokens'] += int(row.get('reasoning_tokens', 0) or 0)

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Print results
    print("\n" + "-" * 100)
    print("RESULTS BY PERSONA:")
    print("-" * 100)

    for persona in sorted(AGE_GROUP_PERSONAS):
        if persona in persona_stats:
            stats = persona_stats[persona]
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {persona:25s} | Correct: {stats['correct']:3d}/{stats['total']:3d} | Accuracy: {acc:6.2f}%")

    # Overall
    print("\n" + "=" * 100)
    if overall['total'] > 0:
        overall_acc = (overall['correct'] / overall['total'] * 100)
        avg_prompt_tokens = overall['sum_prompt_tokens'] / overall['total']
        avg_completion_tokens = overall['sum_completion_tokens'] / overall['total']
        avg_reasoning_tokens = overall['sum_reasoning_tokens'] / overall['total']

        print(f"OVERALL: Correct {overall['correct']}/{overall['total']} | Accuracy: {overall_acc:.2f}%")
        print(f"Average tokens - Prompt: {avg_prompt_tokens:.0f} | Completion: {avg_completion_tokens:.0f} | Reasoning: {avg_reasoning_tokens:.0f}")
    else:
        print("No data to analyze")
    print("=" * 100)

def main():
    print("=" * 100)
    print("GSM8K AGE GROUP PERSONAS - REBUILD & ANALYZE")
    print("=" * 100)

    # Process models 43, 44, 45
    for model_order in [43, 44, 45]:
        csv_path = rebuild_csv_for_model(model_order)
        if csv_path:
            report_dir = Path(csv_path).parent
            analyze_accuracy(csv_path, report_dir)

    print("\n✅ Done!")

if __name__ == "__main__":
    import re
    main()

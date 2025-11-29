"""
Offline analysis - generate reports from current results without waiting
"""

import json
import os
import csv
import time
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def extract_answer_number(text: str) -> str:
    """Extract the final numeric answer from model response."""
    if not text:
        return ""
    
    # Look for Final answer pattern (more specific)
    import re
    match = re.search(r'[Ff]inal\s+answer\s*:\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?', text)
    if match:
        return match.group(1)
    
    # Fallback - no extraction
    return ""


def extract_expected_answer(answer_text: str) -> str:
    """Extract the expected answer from GSM8K answer format."""
    if not answer_text:
        return ""
    
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            number = parts[-1].strip()
            import re
            match = re.search(r'-?\d+\.?\d*', number)
            if match:
                return match.group(0)
    
    return ""


def check_answer_correct(model_answer: str, expected_answer: str) -> bool:
    """Check if model answer matches expected answer."""
    if not model_answer or not expected_answer:
        return False
    
    try:
        model_num = float(model_answer.strip())
        expected_num = float(expected_answer.strip())
        return abs(model_num - expected_num) < 0.001
    except ValueError:
        return model_answer.strip() == expected_answer.strip()


def count_processed_files(results_dir: str) -> int:
    """Count processed files."""
    p = Path(results_dir)
    if not p.exists():
        return 0
    files = list(p.glob("gsm8k_*.json"))
    return len(files)


def generate_report(gsm8k_path: str, results_dir: str, model_name: str, persona: str) -> dict:
    """Generate accuracy report."""
    
    with open(gsm8k_path, 'r') as f:
        gsm8k_data = json.load(f)
    
    gsm8k_map = {i: item for i, item in enumerate(gsm8k_data)}
    
    report = {
        "model": model_name,
        "persona": persona,
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "details": []
    }
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return report
    
    result_files = sorted(results_path.glob("gsm8k_*.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Use question_index from JSON (not filename) to avoid offset issues
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None:
                # Fallback to filename
                q_idx = int(result_file.name.replace("gsm8k_", "").replace(".json", ""))
            
            if q_idx not in gsm8k_map:
                continue
            
            gsm8k_item = gsm8k_map[q_idx]
            question = gsm8k_item.get("question", "")
            expected_answer = extract_expected_answer(gsm8k_item.get("answer", ""))
            
            # Extract model answer from response_text key
            model_output = result.get("response_text", "") or result.get("output", "")
            model_answer = extract_answer_number(model_output)
            
            is_correct = check_answer_correct(model_answer, expected_answer)
            
            report["details"].append({
                "index": q_idx,
                "question": question[:100],
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "correct": is_correct
            })
            
            report["total_questions"] += 1
            if is_correct:
                report["correct_answers"] += 1
        
        except Exception as e:
            print(f"Error: {result_file.name}: {e}")
            continue
    
    if report["total_questions"] > 0:
        report["accuracy"] = round(
            (report["correct_answers"] / report["total_questions"]) * 100,
            2
        )
    
    return report


def export_to_csv_and_excel(report: dict, output_dir: str):
    """Export report to CSV and Excel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename (remove special characters)
    model_safe = report['model'].replace("/", "_").replace(":", "")
    filename = f"{model_safe}__{report['persona']}"
    
    rows = []
    for detail in report["details"]:
        rows.append({
            "Index": detail["index"],
            "Question": detail["question"],
            "Expected": detail["expected_answer"],
            "Model Answer": detail["model_answer"],
            "Correct": "Yes" if detail["correct"] else "No"
        })
    
    # CSV
    csv_path = Path(output_dir) / f"{filename}.csv"
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ CSV: {csv_path.name}")
    
    # Excel
    if PANDAS_AVAILABLE and rows:
        excel_path = Path(output_dir) / f"{filename}.xlsx"
        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
        print(f"✓ Excel: {excel_path.name}")


def analyze_current_results():
    """Analyze current results without waiting."""
    
    print("=" * 80)
    print("ANALYSIS - CURRENT CHECKPOINT RESULTS")
    print("=" * 80)
    print("")
    
    gsm8k_path = "data/gsm8k_test.json"
    
    checkpoints = [
        {
            "name": "Grok man_implicits",
            "model": "x-ai: Grok 4.1 Fast",
            "persona": "man_implicits",
            "dir": "results/grok_4_1_fast/man_implicits/gsm8k_raw",
        },
        {
            "name": "Grok woman_implicits",
            "model": "x-ai: Grok 4.1 Fast",
            "persona": "woman_implicits",
            "dir": "results/grok_4_1_fast/woman_implicits/gsm8k_raw",
        },
        {
            "name": "Nebulon man_implicits",
            "model": "openrouter/bert-nebulon-alpha",
            "persona": "man_implicits",
            "dir": "results/bert_nebulon_alpha/man_implicits/gsm8k_raw",
        },
        {
            "name": "Nebulon woman_implicits",
            "model": "openrouter/bert-nebulon-alpha",
            "persona": "woman_implicits",
            "dir": "results/bert_nebulon_alpha/woman_implicits/gsm8k_raw",
        }
    ]
    
    print("Current file counts:")
    for cp in checkpoints:
        count = count_processed_files(cp["dir"])
        print(f"  {cp['name']}: {count} files")
    print("")
    
    all_reports = []
    
    for cp in checkpoints:
        count = count_processed_files(cp["dir"])
        if count == 0:
            print(f"⊘ Skipping {cp['name']} (no files)")
            continue
        
        print(f"⟳ Analyzing {cp['name']}...")
        report = generate_report(gsm8k_path, cp["dir"], cp["model"], cp["persona"])
        all_reports.append(report)
        export_to_csv_and_excel(report, "results/reports")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("")
    
    for report in all_reports:
        print(f"{report['model']} - {report['persona']}:")
        print(f"  Total: {report['total_questions']}")
        print(f"  Correct: {report['correct_answers']}")
        print(f"  Accuracy: {report['accuracy']}%")
        print("")
    
    # By model
    print("=" * 80)
    print("BY MODEL")
    print("=" * 80)
    print("")
    
    by_model = defaultdict(lambda: {"total": 0, "correct": 0})
    for report in all_reports:
        by_model[report["model"]]["total"] += report["total_questions"]
        by_model[report["model"]]["correct"] += report["correct_answers"]
    
    for model, stats in by_model.items():
        acc = 0 if stats["total"] == 0 else round((stats["correct"] / stats["total"]) * 100, 2)
        print(f"{model}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {acc}%")
        print("")
    
    print("=" * 80)


if __name__ == "__main__":
    analyze_current_results()

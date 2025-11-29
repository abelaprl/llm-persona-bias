"""
Script to monitor Grok and Nebulon progress and stop at checkpoints.
Grok: stop at 1000 questions
Nebulon: stop at 600 questions

Then export CSV and Excel with scoring.
"""

import json
import os
import csv
import time
from pathlib import Path
from collections import defaultdict
import subprocess
import signal
import sys

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def extract_answer_number(text: str) -> str:
    """Extract the final numeric answer from model response."""
    if not text:
        return ""
    
    # Look for "Final answer:" pattern
    if "Final answer:" in text or "Final Answer:" in text:
        parts = text.split("Final answer:" if "Final answer:" in text else "Final Answer:")
        if len(parts) > 1:
            # Get the last part and extract numbers
            last_part = parts[-1].strip()
            # Extract first number found
            import re
            match = re.search(r'-?\d+\.?\d*', last_part)
            if match:
                return match.group(0)
    
    # Fallback: look for any number at the end
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return ""


def extract_expected_answer(answer_text: str) -> str:
    """Extract the expected answer from GSM8K answer format (#### NUMBER)."""
    if not answer_text:
        return ""
    
    # Look for #### pattern
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
    """Count how many gsm8k_XXXXX.json files exist."""
    p = Path(results_dir)
    if not p.exists():
        return 0
    files = list(p.glob("gsm8k_*.json"))
    return len(files)


def find_process_by_model(model_name: str) -> int:
    """Find running process PID by model name."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split("\n"):
            if "run_gsm8k_batch.py" in line and model_name in line and "grep" not in line:
                parts = line.split()
                if len(parts) > 1:
                    return int(parts[1])
    except Exception as e:
        print(f"Error finding process: {e}")
    return None


def stop_process(pid: int) -> bool:
    """Stop a running process."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        # If still running, force kill
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass
        return True
    except Exception as e:
        print(f"Error stopping process {pid}: {e}")
        return False


def generate_report(gsm8k_path: str, results_dir: str, model_name: str, persona: str) -> dict:
    """Generate accuracy report from results."""
    
    # Load GSM8K data
    with open(gsm8k_path, 'r') as f:
        gsm8k_data = json.load(f)
    
    # Create mapping of question index to answer
    gsm8k_map = {i: item for i, item in enumerate(gsm8k_data)}
    
    report = {
        "model": model_name,
        "persona": persona,
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "details": []
    }
    
    # Process each result file
    results_path = Path(results_dir)
    if not results_path.exists():
        return report
    
    result_files = sorted(results_path.glob("gsm8k_*.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Extract question index
            filename = result_file.name  # e.g., "gsm8k_00300.json"
            q_idx = int(filename.replace("gsm8k_", "").replace(".json", ""))
            
            if q_idx not in gsm8k_map:
                continue
            
            gsm8k_item = gsm8k_map[q_idx]
            question = gsm8k_item.get("question", "")
            expected_answer = extract_expected_answer(gsm8k_item.get("answer", ""))
            
            # Extract model answer
            model_output = result.get("output", "")
            model_answer = extract_answer_number(model_output)
            
            # Check correctness
            is_correct = check_answer_correct(model_answer, expected_answer)
            
            report["details"].append({
                "index": q_idx,
                "question": question[:100],  # Truncate for readability
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "correct": is_correct
            })
            
            report["total_questions"] += 1
            if is_correct:
                report["correct_answers"] += 1
        
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
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
    
    filename = f"{report['model']}__{report['persona']}"
    
    # Prepare data for export
    rows = []
    for detail in report["details"]:
        rows.append({
            "Question Index": detail["index"],
            "Question": detail["question"],
            "Expected Answer": detail["expected_answer"],
            "Model Answer": detail["model_answer"],
            "Correct": "Yes" if detail["correct"] else "No"
        })
    
    # Export to CSV
    csv_path = Path(output_dir) / f"{filename}.csv"
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ CSV exported: {csv_path}")
    
    # Export to Excel (if pandas available)
    if PANDAS_AVAILABLE and rows:
        excel_path = Path(output_dir) / f"{filename}.xlsx"
        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False, sheet_name="Results")
        print(f"✓ Excel exported: {excel_path}")


def monitor_and_checkpoint():
    """Monitor Grok and Nebulon progress and stop at checkpoints."""
    
    print("=" * 80)
    print("CHECKPOINT MONITOR - Grok (1000) and Nebulon (600)")
    print("=" * 80)
    print("")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    gsm8k_path = "data/gsm8k_test.json"
    
    # Verify GSM8K file exists
    if not Path(gsm8k_path).exists():
        print(f"ERROR: {gsm8k_path} not found!")
        return
    
    # Checkpoint configuration
    checkpoints = [
        {
            "name": "Grok man_implicits",
            "model_name": "x-ai: Grok 4.1 Fast",
            "persona": "man_implicits",
            "results_dir": "results/grok_4_1_fast/man_implicits/gsm8k_raw",
            "target": 1000
        },
        {
            "name": "Grok woman_implicits",
            "model_name": "x-ai: Grok 4.1 Fast",
            "persona": "woman_implicits",
            "results_dir": "results/grok_4_1_fast/woman_implicits/gsm8k_raw",
            "target": 1000
        },
        {
            "name": "Nebulon man_implicits",
            "model_name": "openrouter/bert-nebulon-alpha",
            "persona": "man_implicits",
            "results_dir": "results/bert_nebulon_alpha/man_implicits/gsm8k_raw",
            "target": 600
        },
        {
            "name": "Nebulon woman_implicits",
            "model_name": "openrouter/bert-nebulon-alpha",
            "persona": "woman_implicits",
            "results_dir": "results/bert_nebulon_alpha/woman_implicits/gsm8k_raw",
            "target": 600
        }
    ]
    
    completed = defaultdict(bool)
    check_count = 0
    max_checks = 7200  # 1 hour if checking every 30 seconds
    
    while check_count < max_checks:
        for checkpoint in checkpoints:
            if completed[checkpoint["name"]]:
                continue
            
            count = count_processed_files(checkpoint["results_dir"])
            target = checkpoint["target"]
            
            status = "✓" if count >= target else "•"
            print(f"{status} {checkpoint['name']}: {count}/{target}")
            
            # Check if checkpoint reached
            if count >= target and not completed[checkpoint["name"]]:
                print(f"\n✓ CHECKPOINT REACHED: {checkpoint['name']} ({count} questions)")
                
                # Find and stop process
                pid = find_process_by_model(checkpoint["model_name"])
                if pid:
                    print(f"  Stopping process PID {pid}...")
                    stop_process(pid)
                    time.sleep(1)
                
                completed[checkpoint["name"]] = True
        
        # Check if all checkpoints completed
        if all(completed.values()):
            print("\n" + "=" * 80)
            print("ALL CHECKPOINTS COMPLETED")
            print("=" * 80)
            break
        
        check_count += 1
        time.sleep(30)  # Check every 30 seconds
    
    if check_count >= max_checks:
        print("\nWARNING: Monitor timeout reached. Generating partial report...")
        # Stop remaining processes
        for checkpoint in checkpoints:
            if not completed[checkpoint["name"]]:
                pid = find_process_by_model(checkpoint["model_name"])
                if pid:
                    print(f"Stopping {checkpoint['name']} (PID {pid})...")
                    stop_process(pid)
    
    # Generate reports
    print("\nGenerating reports...")
    print("")
    
    all_reports = []
    
    for checkpoint in checkpoints:
        print(f"Processing {checkpoint['name']}...")
        report = generate_report(
            gsm8k_path,
            checkpoint["results_dir"],
            checkpoint["model_name"],
            checkpoint["persona"]
        )
        all_reports.append(report)
        
        export_to_csv_and_excel(report, "results/reports")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - ACCURACY BY MODEL AND PERSONA")
    print("=" * 80)
    print("")
    
    for report in all_reports:
        print(f"{report['model']} - {report['persona']}:")
        print(f"  Total Questions: {report['total_questions']}")
        print(f"  Correct Answers: {report['correct_answers']}")
        print(f"  Accuracy: {report['accuracy']}%")
        print("")
    
    # Overall summary by model
    print("=" * 80)
    print("OVERALL BY MODEL")
    print("=" * 80)
    print("")
    
    by_model = defaultdict(lambda: {"total": 0, "correct": 0})
    for report in all_reports:
        by_model[report["model"]]["total"] += report["total_questions"]
        by_model[report["model"]]["correct"] += report["correct_answers"]
    
    for model, stats in by_model.items():
        accuracy = 0 if stats["total"] == 0 else round((stats["correct"] / stats["total"]) * 100, 2)
        print(f"{model}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {accuracy}%")
        print("")
    
    print("=" * 80)
    print("Reports exported to: results/reports/")
    print("=" * 80)


if __name__ == "__main__":
    monitor_and_checkpoint()

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_answer_number(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'[Ff]inal\s+answer\s*:\s*\*?\*?(-?\d+(?:\.\d+)?)\*?\*?', text)
    if match:
        return match.group(1)
    return ""

def extract_expected_answer(answer_text: str) -> str:
    if not answer_text:
        return ""
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            number = parts[-1].strip()
            match = re.search(r'-?\d+\.?\d*', number)
            if match:
                return match.group(0)
    return ""

def check_answer_correct(model_answer: str, expected_answer: str) -> bool:
    if not model_answer or not expected_answer:
        return False
    try:
        return abs(float(model_answer) - float(expected_answer)) < 0.001
    except:
        return False

gsm8k = json.load(open("data/gsm8k_test.json"))

print("\n" + "="*140)
print("NEMOTRON (NVIDIA) - Implicit Personas Analysis")
print("="*140 + "\n")

configs = [
    ("man_implicits", "Nemotron man_implicits"),
    ("woman_implicits", "Nemotron woman_implicits"),
]

results_summary = []

for path, label in configs:
    result_dir = Path(f"results/{path}/gsm8k_raw")
    result_files = sorted([f for f in result_dir.glob("gsm8k_*.json")])
    
    if not result_files:
        print(f"❌ {label} | NO FILES")
        continue
    
    print(f"Processing {label}... ({len(result_files)} files)")
    
    correct = 0
    total = 0
    indices = []
    
    for i, result_file in enumerate(result_files):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(result_files)}")
        
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None:
                continue
                
            indices.append(q_idx)
            
            if q_idx >= len(gsm8k):
                continue
            
            gsm8k_item = gsm8k[q_idx]
            expected = extract_expected_answer(gsm8k_item.get("answer", ""))
            model_output = result.get("response_text", "")
            model_answer = extract_answer_number(model_output)
            
            if check_answer_correct(model_answer, expected):
                correct += 1
            
            total += 1
            
        except:
            continue
    
    if total > 0:
        accuracy = correct / total * 100
        idx_range = f"Q{min(indices)}-{max(indices)}"
        print(f"✓ {label:35} | {correct:5}/{total:5} | {accuracy:6.2f}% | {idx_range}\n")
        
        results_summary.append({
            "label": label,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "idx_min": min(indices),
            "idx_max": max(indices)
        })

# Summary
print("="*140)
print("SUMMARY - NEMOTRON\n")

all_correct = sum(r["correct"] for r in results_summary)
all_total = sum(r["total"] for r in results_summary)
overall_acc = all_correct / all_total * 100

for r in results_summary:
    print(f"{r['label']:35} | {r['correct']:5}/{r['total']:5} | {r['accuracy']:6.2f}%")

print(f"\nOVERALL NEMOTRON: {all_correct}/{all_total} = {overall_acc:.2f}%")
print("="*140 + "\n")


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

# Fixed configs
configs = [
    ("man_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("woman_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("grok_4_1_fast/man_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("grok_4_1_fast/woman_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("bert_nebulon_alpha/man_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
    ("bert_nebulon_alpha/woman_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
]

print(f"\n{'='*140}")
print("FINAL ACCURATE ANALYSIS - ALL MODELS & PERSONAS")
print(f"{'='*140}\n")

results_summary = []

for path, model_short, model_full in configs:
    result_dir = Path(f"results/{path}/gsm8k_raw")
    result_files = sorted([f for f in result_dir.glob("gsm8k_*.json")])
    
    if "man_implicits" in path:
        persona = "man_implicits"
    else:
        persona = "woman_implicits"
    
    if not result_files:
        print(f"❌ {model_short:12} | {persona:20} | NO FILES")
        continue
    
    correct = 0
    total = 0
    indices = []
    skipped = 0
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None:
                skipped += 1
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
            
        except Exception as e:
            skipped += 1
            continue
    
    if total > 0:
        accuracy = correct / total * 100
        idx_range = f"Q{min(indices)}-{max(indices)}"
        print(f"✓ {model_short:12} | {persona:20} | {correct:5}/{total:5} | {accuracy:6.2f}% | {idx_range} | Files: {len(result_files)}")
        
        results_summary.append({
            "model": model_short,
            "persona": persona,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "model_full": model_full,
            "idx_min": min(indices),
            "idx_max": max(indices),
            "files": len(result_files)
        })

print(f"\n{'='*140}")
print("SUMMARY BY MODEL (Combined both personas)")
print(f"{'='*140}\n")

models_dict = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results_summary:
    key = r['model']
    models_dict[key]["correct"] += r["correct"]
    models_dict[key]["total"] += r["total"]

for model_name in ["Nemotron", "Grok", "Nebulon"]:
    if model_name in models_dict:
        data = models_dict[model_name]
        acc = data["correct"] / data["total"] * 100
        print(f"{model_name:15} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%")

print(f"\n{'='*140}")
print("DETAILED BREAKDOWN (Per Model + Per Persona)")
print(f"{'='*140}\n")

for r in results_summary:
    print(f"{r['model']:12} | {r['persona']:20} | {r['correct']:5}/{r['total']:5} | {r['accuracy']:6.2f}% | Q{r['idx_min']:4}-{r['idx_max']:4} | {r['files']:4} files")

# Overall
all_correct = sum(r["correct"] for r in results_summary)
all_total = sum(r["total"] for r in results_summary)
overall_acc = all_correct / all_total * 100

print(f"\n{'='*140}")
print(f"OVERALL ACCURACY: {all_correct}/{all_total} = {overall_acc:.2f}%")
print(f"{'='*140}\n")

# Save to txt file for reference
with open("results/reports/FINAL_ACCURACY_REPORT.txt", "w") as f:
    f.write("="*140 + "\n")
    f.write("FINAL ACCURATE ANALYSIS - ALL MODELS & PERSONAS\n")
    f.write("="*140 + "\n\n")
    
    f.write("DETAILED BREAKDOWN\n")
    f.write("="*140 + "\n\n")
    for r in results_summary:
        f.write(f"{r['model']:12} | {r['persona']:20} | {r['correct']:5}/{r['total']:5} | {r['accuracy']:6.2f}% | Q{r['idx_min']:4}-{r['idx_max']:4} | {r['files']:4} files\n")
    
    f.write(f"\n{'='*140}\n")
    f.write("SUMMARY BY MODEL\n")
    f.write(f"{'='*140}\n\n")
    for model_name in ["Nemotron", "Grok", "Nebulon"]:
        if model_name in models_dict:
            data = models_dict[model_name]
            acc = data["correct"] / data["total"] * 100
            f.write(f"{model_name:15} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%\n")
    
    f.write(f"\n{'='*140}\n")
    f.write(f"OVERALL ACCURACY: {all_correct}/{all_total} = {overall_acc:.2f}%\n")
    f.write(f"{'='*140}\n")

print("✓ Report saved to: results/reports/FINAL_ACCURACY_REPORT.txt")


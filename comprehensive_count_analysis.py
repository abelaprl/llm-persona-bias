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

configs = [
    ("man_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("woman_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("grok_4_1_fast/man_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("grok_4_1_fast/woman_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("bert_nebulon_alpha/man_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
    ("bert_nebulon_alpha/woman_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
]

print("\n" + "="*150)
print("COMPREHENSIVE COUNT & ACCURACY ANALYSIS")
print("="*150 + "\n")

all_results = []
model_stats = defaultdict(lambda: {"correct": 0, "total": 0})
persona_stats = defaultdict(lambda: {"correct": 0, "total": 0})

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
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None or q_idx >= len(gsm8k):
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
        print(f"✓ {model_short:12} | {persona:20} | {correct:5}/{total:5} | {accuracy:6.2f}%")
        
        all_results.append({
            "model": model_short,
            "persona": persona,
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        })
        
        model_stats[model_short]["correct"] += correct
        model_stats[model_short]["total"] += total
        persona_stats[persona]["correct"] += correct
        persona_stats[persona]["total"] += total

print("\n" + "="*150)
print("BY MODEL (Both Personas Combined)")
print("="*150 + "\n")

for model in ["Nemotron", "Grok", "Nebulon"]:
    if model in model_stats:
        data = model_stats[model]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{model:15} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%")

print("\n" + "="*150)
print("BY PERSONA (All Models Combined)")
print("="*150 + "\n")

for persona in ["man_implicits", "woman_implicits"]:
    if persona in persona_stats:
        data = persona_stats[persona]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{persona:20} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%")

print("\n" + "="*150)
print("DETAILED BREAKDOWN (Model × Persona)")
print("="*150 + "\n")

for result in all_results:
    print(f"{result['model']:15} | {result['persona']:20} | {result['correct']:5}/{result['total']:5} | {result['accuracy']:6.2f}%")

# Grand total
all_correct = sum(r["correct"] for r in all_results)
all_total = sum(r["total"] for r in all_results)
overall_acc = all_correct / all_total * 100 if all_total > 0 else 0

print("\n" + "="*150)
print(f"GRAND TOTAL: {all_correct}/{all_total} = {overall_acc:.2f}%")
print("="*150 + "\n")


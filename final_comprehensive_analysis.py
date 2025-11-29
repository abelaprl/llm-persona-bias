import json
import re
from pathlib import Path

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

# All 5 configs: Nemotron + Grok (2) + Nebulon (2)
configs = [
    ("man_implicits", None, "nvidia/nemotron-nano-12b-v2-vl:free", "Nemotron"),
    ("grok_4_1_fast/man_implicits", "grok_4_1_fast", "x-ai/grok-4.1-fast", "Grok"),
    ("grok_4_1_fast/woman_implicits", "grok_4_1_fast", "x-ai/grok-4.1-fast", "Grok"),
    ("bert_nebulon_alpha/man_implicits", "bert_nebulon_alpha", "openrouter/bert-nebulon-alpha", "Nebulon"),
    ("bert_nebulon_alpha/woman_implicits", "bert_nebulon_alpha", "openrouter/bert-nebulon-alpha", "Nebulon"),
]

print(f"{'='*120}")
print("COMPREHENSIVE ACCURACY ANALYSIS - ALL MODELS")
print(f"{'='*120}\n")

results_summary = []

for path_suffix, model_dir, model_name, model_short in configs:
    if model_dir:
        result_dir = Path(f"results/{path_suffix}/gsm8k_raw")
    else:
        result_dir = Path(f"results/{path_suffix}/gsm8k_raw")
    
    # Extract persona from path
    if "man_implicits" in path_suffix:
        persona = "man_implicits"
    elif "woman_implicits" in path_suffix:
        persona = "woman_implicits"
    else:
        persona = "man_implicits"
    
    result_files = sorted(result_dir.glob("gsm8k_*.json"))
    
    if not result_files:
        print(f"❌ {model_short:12} | {persona:20} | NO FILES FOUND")
        continue
    
    correct = 0
    total = 0
    indices = []
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
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
        idx_range = f"{min(indices)}-{max(indices)}"
        print(f"✓ {model_short:12} | {persona:20} | {correct:5}/{total:5} | {accuracy:6.2f}% | Q{idx_range}")
        
        results_summary.append({
            "model": model_short,
            "persona": persona,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "model_full": model_name
        })

print(f"\n{'='*120}")
print("SUMMARY BY MODEL")
print(f"{'='*120}\n")

models_dict = {}
for r in results_summary:
    key = r['model']
    if key not in models_dict:
        models_dict[key] = {"correct": 0, "total": 0}
    models_dict[key]["correct"] += r["correct"]
    models_dict[key]["total"] += r["total"]

for model_name in ["Nemotron", "Grok", "Nebulon"]:
    if model_name in models_dict:
        data = models_dict[model_name]
        acc = data["correct"] / data["total"] * 100
        print(f"{model_name:15} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%")

print(f"\n{'='*120}")
print("DETAILS PER PERSONA")
print(f"{'='*120}\n")

for r in results_summary:
    print(f"{r['model']:15} | {r['persona']:20} | {r['correct']:4}/{r['total']:4} | {r['accuracy']:6.2f}%")

# Overall
all_correct = sum(r["correct"] for r in results_summary)
all_total = sum(r["total"] for r in results_summary)
overall_acc = all_correct / all_total * 100

print(f"\n{'='*120}")
print(f"OVERALL: {all_correct}/{all_total} = {overall_acc:.2f}%")
print(f"{'='*120}")


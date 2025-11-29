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

configs = [
    ("man_implicits", "Nemotron"),
    ("woman_implicits", "Nemotron"),
    ("grok_4_1_fast/man_implicits", "Grok"),
    ("grok_4_1_fast/woman_implicits", "Grok"),
    ("bert_nebulon_alpha/man_implicits", "Nebulon"),
    ("bert_nebulon_alpha/woman_implicits", "Nebulon"),
]

print("\n" + "="*160)
print("RECOUNT - ACTUAL FILE COUNTS")
print("="*160 + "\n")

for path, model in configs:
    result_dir = Path(f"results/{path}/gsm8k_raw")
    result_files = sorted([f for f in result_dir.glob("gsm8k_*.json")])
    
    if "man_implicits" in path:
        persona = "man_implicits"
    else:
        persona = "woman_implicits"
    
    if not result_files:
        print(f"{model:12} | {persona:20} | Files: 0")
        continue
    
    # Count correct answers
    correct = 0
    total = 0
    q_indices = []
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None or q_idx >= len(gsm8k):
                continue
            
            q_indices.append(q_idx)
            
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
        q_range = f"Q{min(q_indices)}-Q{max(q_indices)}" if q_indices else "?"
        print(f"{model:12} | {persona:20} | Files: {len(result_files):4} | Total: {total:4} | Correct: {correct:4} | Acc: {accuracy:6.2f}% | {q_range}")

print("\n" + "="*160 + "\n")


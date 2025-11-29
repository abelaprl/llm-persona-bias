import json
import re
from pathlib import Path
from collections import defaultdict

def extract_answer_number(text: str) -> str:
    """Extract answer from 'Final answer:' pattern."""
    if not text:
        return ""
    match = re.search(r'[Ff]inal\s+answer\s*:\s*\*?\*?(-?\d+(?:\.\d+)?)\*?\*?', text)
    if match:
        return match.group(1)
    return ""

def extract_expected_answer(answer_text: str) -> str:
    """Extract expected answer."""
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
    """Check if answer is correct with float tolerance."""
    if not model_answer or not expected_answer:
        return False
    try:
        return abs(float(model_answer) - float(expected_answer)) < 0.001
    except (ValueError, TypeError):
        return False

# Load GSM8K
gsm8k = json.load(open("data/gsm8k_test.json"))
print(f"GSM8K Total: {len(gsm8k)} questions\n")

# Models and personas
configs = [
    ("grok_4_1_fast", "man_implicits", "x-ai/grok-4.1-fast"),
    ("grok_4_1_fast", "woman_implicits", "x-ai/grok-4.1-fast"),
    ("bert_nebulon_alpha", "man_implicits", "openrouter/bert-nebulon-alpha"),
    ("bert_nebulon_alpha", "woman_implicits", "openrouter/bert-nebulon-alpha"),
]

results_summary = []

for model_dir, persona, model_name in configs:
    print(f"\n{'='*100}")
    print(f"ANALYSIS: {model_name} - {persona}")
    print(f"{'='*100}")
    
    result_dir = Path(f"results/{model_dir}/{persona}/gsm8k_raw")
    result_files = sorted(result_dir.glob("gsm8k_*.json"))
    
    if not result_files:
        print(f"No result files found in {result_dir}")
        continue
    
    correct = 0
    total = 0
    errors = []
    extracted_indices = []
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            extracted_indices.append(q_idx)
            
            if q_idx >= len(gsm8k):
                errors.append(f"Index {q_idx} out of range")
                continue
            
            gsm8k_item = gsm8k[q_idx]
            expected = extract_expected_answer(gsm8k_item.get("answer", ""))
            model_output = result.get("response_text", "")
            model_answer = extract_answer_number(model_output)
            
            if check_answer_correct(model_answer, expected):
                correct += 1
            
            total += 1
            
        except Exception as e:
            errors.append(f"File {result_file.name}: {str(e)}")
            continue
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"Files processed: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Index range: {min(extracted_indices)} - {max(extracted_indices)}")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for err in errors[:3]:
                print(f"  - {err}")
        
        results_summary.append({
            "model": model_name,
            "persona": persona,
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        })

print(f"\n{'='*100}")
print("FINAL SUMMARY")
print(f"{'='*100}")

for summary in results_summary:
    print(f"{summary['model']:40} | {summary['persona']:20} | {summary['correct']:4}/{summary['total']:4} | {summary['accuracy']:6.2f}%")

# Combined accuracy
all_correct = sum(s["correct"] for s in results_summary)
all_total = sum(s["total"] for s in results_summary)
overall_accuracy = all_correct / all_total * 100 if all_total > 0 else 0

print(f"\nOverall: {all_correct}/{all_total} = {overall_accuracy:.2f}%")

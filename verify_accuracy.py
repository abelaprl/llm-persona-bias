import json
import re
from pathlib import Path

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

# Load GSM8K
print("Loading GSM8K...")
gsm8k = json.load(open("data/gsm8k_test.json"))
print(f"GSM8K has {len(gsm8k)} questions\n")

# Verify 30 samples from Grok man_implicits
print("=" * 100)
print("VERIFICATION: Grok man_implicits (first 30 samples)")
print("=" * 100)

result_files = sorted(Path("results/grok_4_1_fast/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))[:30]

correct = 0
total = 0

for i, result_file in enumerate(result_files):
    with open(result_file) as f:
        result = json.load(f)
    
    # Use question_index from JSON
    q_idx = result.get("run", {}).get("question_index")
    
    if q_idx >= len(gsm8k):
        print(f"Skip: q_idx {q_idx} >= {len(gsm8k)}")
        continue
    
    gsm8k_item = gsm8k[q_idx]
    expected = extract_expected_answer(gsm8k_item.get("answer", ""))
    model_output = result.get("response_text", "")
    model_answer = extract_answer_number(model_output)
    
    # Check
    try:
        is_correct = abs(float(model_answer) - float(expected)) < 0.001 if model_answer and expected else False
    except:
        is_correct = False
    
    if is_correct:
        correct += 1
    total += 1
    
    status = "✓" if is_correct else "✗"
    question_preview = gsm8k_item.get("question", "")[:60]
    print(f"{status} Q{q_idx}: {question_preview}...")
    if not is_correct and model_answer:
        print(f"    Expected: {expected}, Got: {model_answer}")

print(f"\nAccuracy (30 samples): {correct}/{total} = {correct/total*100:.2f}%")

# Also verify nebulon
print("\n" + "=" * 100)
print("VERIFICATION: Nebulon man_implicits (first 30 samples)")
print("=" * 100)

result_files = sorted(Path("results/bert_nebulon_alpha/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))[:30]

correct = 0
total = 0

for i, result_file in enumerate(result_files):
    with open(result_file) as f:
        result = json.load(f)
    
    q_idx = result.get("run", {}).get("question_index")
    
    if q_idx >= len(gsm8k):
        continue
    
    gsm8k_item = gsm8k[q_idx]
    expected = extract_expected_answer(gsm8k_item.get("answer", ""))
    model_output = result.get("response_text", "")
    model_answer = extract_answer_number(model_output)
    
    try:
        is_correct = abs(float(model_answer) - float(expected)) < 0.001 if model_answer and expected else False
    except:
        is_correct = False
    
    if is_correct:
        correct += 1
    total += 1
    
    status = "✓" if is_correct else "✗"
    question_preview = gsm8k_item.get("question", "")[:60]
    print(f"{status} Q{q_idx}: {question_preview}...")
    if not is_correct and model_answer:
        print(f"    Expected: {expected}, Got: {model_answer}")

print(f"\nAccuracy (30 samples): {correct}/{total} = {correct/total*100:.2f}%")

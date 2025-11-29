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

# Load GSM8K
gsm8k = json.load(open("data/gsm8k_test.json"))

# Check first 20 in detail
result_files = sorted(Path("results/grok_4_1_fast/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))[:20]

print("Detailed comparison (20 samples):\n")
correct = 0

for result_file in result_files:
    with open(result_file) as f:
        result = json.load(f)
    
    q_idx = int(result_file.name.replace("gsm8k_", "").replace(".json", ""))
    
    if q_idx >= len(gsm8k):
        continue
    
    gsm8k_item = gsm8k[q_idx]
    expected = extract_expected_answer(gsm8k_item.get("answer", ""))
    model_output = result.get("response_text", "")
    model_answer = extract_answer_number(model_output)
    
    # Check if equal
    try:
        is_correct = abs(float(model_answer) - float(expected)) < 0.001 if model_answer and expected else False
    except:
        is_correct = False
    
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    question = gsm8k_item.get("question", "")[:60]
    print(f"{status} Q{q_idx}: {question}...")
    print(f"   Expected: {expected}, Got: {model_answer}, Match: {is_correct}")
    print()

print(f"\nTotal correct: {correct}/20 = {correct*5}%")

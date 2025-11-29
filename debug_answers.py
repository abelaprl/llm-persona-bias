import json
from pathlib import Path

def extract_answer_number(text: str) -> str:
    """Extract answer."""
    if not text:
        return ""
    if "Final answer:" in text or "Final Answer:" in text:
        parts = text.split("Final answer:" if "Final answer:" in text else "Final Answer:")
        if len(parts) > 1:
            last_part = parts[-1].strip()
            import re
            match = re.search(r'-?\d+\.?\d*', last_part)
            if match:
                return match.group(0)
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def extract_expected_answer(answer_text: str) -> str:
    """Extract expected answer."""
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

# Check first 10 files
result_files = sorted(Path("results/grok_4_1_fast/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))[:10]
gsm8k = json.load(open("data/gsm8k_test.json"))

print("Sample analysis (first 10):")
print("=" * 100)

correct_count = 0
for result_file in result_files:
    with open(result_file) as f:
        result = json.load(f)
    
    q_idx = int(result_file.name.replace("gsm8k_", "").replace(".json", ""))
    gsm8k_item = gsm8k[q_idx]
    expected = extract_expected_answer(gsm8k_item.get("answer", ""))
    model_output = result.get("response_text", "")
    model_answer = extract_answer_number(model_output)
    
    try:
        is_correct = abs(float(model_answer) - float(expected)) < 0.001
        if is_correct:
            correct_count += 1
    except:
        is_correct = False
    
    status = "✓" if is_correct else "✗"
    print(f"{status} Q{q_idx}: Expected={expected}, Model={model_answer}")
    if not is_correct:
        print(f"  Model output: {model_output[:150]}")

print(f"\nCorrect: {correct_count}/10")

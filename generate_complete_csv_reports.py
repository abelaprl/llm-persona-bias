import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

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

# Load data
gsm8k = json.load(open("data/gsm8k_test.json"))

# Load personas
with open("inputs/implicits_man_prompts.json") as f:
    man_implicit_persona = json.load(f)[0]["prompt"] if json.load(open("inputs/implicits_man_prompts.json")) else ""

configs = [
    ("man_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("woman_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("grok_4_1_fast/man_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("grok_4_1_fast/woman_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("bert_nebulon_alpha/man_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
    ("bert_nebulon_alpha/woman_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
]

print("\n" + "="*150)
print("GENERATING COMPLETE CSV REPORTS")
print("="*150 + "\n")

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
    
    print(f"Processing {model_short:12} | {persona:20} ({len(result_files)} files)...")
    
    data = []
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            q_idx = result.get("run", {}).get("question_index")
            if q_idx is None or q_idx >= len(gsm8k):
                continue
            
            gsm8k_item = gsm8k[q_idx]
            question = gsm8k_item.get("question", "")
            expected_answer = extract_expected_answer(gsm8k_item.get("answer", ""))
            model_output = result.get("response_text", "")
            model_answer = extract_answer_number(model_output)
            
            is_correct = check_answer_correct(model_answer, expected_answer)
            
            # Get persona context
            persona_context = result.get("run", {}).get("system_message", "")
            
            data.append({
                "Question_ID": q_idx,
                "Question": question,
                "Expected_Answer": expected_answer,
                "Model_Answer": model_answer,
                "Model_Reasoning": model_output[:500] if model_output else "",  # First 500 chars
                "Persona_Type": persona,
                "Correct": "Yes" if is_correct else "No",
            })
        except:
            continue
    
    if data:
        df = pd.DataFrame(data)
        
        # Create filename
        model_safe = model_short.replace("/", "_").replace(":", "")
        filename = f"results/reports/{model_safe}__{persona}_COMPLETE.csv"
        
        # Save CSV
        df.to_csv(filename, index=False)
        
        correct_count = (df['Correct'] == 'Yes').sum()
        total_count = len(df)
        accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        
        print(f"  ✓ {filename}")
        print(f"    Files: {len(result_files)} | Total: {total_count} | Correct: {correct_count} | Acc: {accuracy:.2f}%\n")

print("="*150 + "\n")


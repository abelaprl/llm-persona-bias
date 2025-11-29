import json
import re
from pathlib import Path
import csv

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

gsm8k = json.load(open("data/gsm8k_test.json"))

# Load persona files
def load_persona_text(persona_file):
    with open(f"inputs/{persona_file}") as f:
        data = json.load(f)
    return data[0].get("prompt", "")

persona_texts = {
    "man_implicits": load_persona_text("implicits_man_prompts.json"),
    "woman_implicits": load_persona_text("implicits_woman_promt.json"),
}

configs = [
    ("man_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("woman_implicits", "Nemotron", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("grok_4_1_fast/man_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("grok_4_1_fast/woman_implicits", "Grok", "x-ai/grok-4.1-fast"),
    ("bert_nebulon_alpha/man_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
    ("bert_nebulon_alpha/woman_implicits", "Nebulon", "openrouter/bert-nebulon-alpha"),
]

print("\nGenerating complete CSVs...\n")

for path, model_short, model_full in configs:
    result_dir = Path(f"results/{path}/gsm8k_raw")
    result_files = sorted([f for f in result_dir.glob("gsm8k_*.json")])
    
    if "man_implicits" in path:
        persona = "man_implicits"
    else:
        persona = "woman_implicits"
    
    if not result_files:
        print(f"❌ {model_short} {persona} - NO FILES")
        continue
    
    # Create CSV filename
    csv_filename = f"results/reports/{model_short}_{persona}_COMPLETE.csv"
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Question_ID', 
            'Question_Text', 
            'Expected_Answer',
            'Persona_Injected',
            'Model_Response',
            'Model_Answer',
            'Correct'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        count = 0
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
                
                is_correct = "Yes" if (model_answer and expected and abs(float(model_answer) - float(expected)) < 0.001) else "No"
                
                persona_text = persona_texts.get(persona, "")
                
                writer.writerow({
                    'Question_ID': q_idx,
                    'Question_Text': gsm8k_item.get("question", ""),
                    'Expected_Answer': expected,
                    'Persona_Injected': persona_text[:100] + "..." if len(persona_text) > 100 else persona_text,
                    'Model_Response': model_output[:500] + "..." if len(model_output) > 500 else model_output,
                    'Model_Answer': model_answer if model_answer else "NO_ANSWER",
                    'Correct': is_correct
                })
                
                count += 1
            except:
                continue
        
        accuracy = 0
        if count > 0:
            correct_count = sum(1 for _ in open(csv_filename) if 'Yes' in _)
            accuracy = (correct_count - 1) / count * 100  # -1 for header
        
        print(f"✓ {model_short:10} | {persona:20} | {count:5} soal | {accuracy:6.2f}% | {csv_filename}")

print("\n✓ Semua CSV sudah dibuat!\n")


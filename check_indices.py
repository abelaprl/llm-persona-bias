import json
from pathlib import Path

# Load GSM8K
gsm8k = json.load(open("data/gsm8k_test.json"))

# Check result files to see what index they claim vs actual
result_files = sorted(Path("results/grok_4_1_fast/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))[:5]

for result_file in result_files:
    with open(result_file) as f:
        result = json.load(f)
    
    filename_idx = int(result_file.name.replace("gsm8k_", "").replace(".json", ""))
    question_idx = result.get("run", {}).get("question_index")
    
    print(f"File name: {result_file.name}")
    print(f"  Filename index: {filename_idx}")
    print(f"  Declared index in JSON: {question_idx}")
    print(f"  Question in GSM8K[{question_idx}]: {gsm8k[question_idx].get('question', '')[:80]}")
    print()

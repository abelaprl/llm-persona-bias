import json
from pathlib import Path

# Sample 20 responses
result_files = sorted(Path("results/grok_4_1_fast/man_implicits/gsm8k_raw").glob("gsm8k_*.json"))

print("Checking response formats:\n")

for i, result_file in enumerate(result_files[:20]):
    with open(result_file) as f:
        result = json.load(f)
    
    response_text = result.get("response_text", "")
    
    # Check if has "Final answer:"
    has_final = "Final answer:" in response_text or "Final Answer:" in response_text
    
    # Check last line
    lines = response_text.strip().split("\n")
    last_line = lines[-1] if lines else ""
    
    print(f"File {i+1}:")
    print(f"  Has 'Final answer': {has_final}")
    print(f"  Last line: {last_line[:80]}")
    print(f"  Full output length: {len(response_text)}")
    print()

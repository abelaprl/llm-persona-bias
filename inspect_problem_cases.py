import json
import re

# Check actual responses for a few problem cases to debug extraction
problem_cases = [
    ('man_explicit_1/gsm8k_00094.json', 36),
    ('man_explicit_1/gsm8k_00199.json', 320),
    ('man_explicit_1/gsm8k_00254.json', 18),
    ('man_explicit_1/gsm8k_00510.json', 4),  # extracted 3, expected 4
    ('man_explicit_1/gsm8k_00717.json', 770),  # extracted 450, expected 770
]

# Check which numbers appear in response tail and which ones the extractor picked
for case, expected in problem_cases:
    path = rf'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\{case}'
    try:
        with open(path) as f:
            data = json.load(f)
            content = data.get('response', {}).get('content', '')
            tail = content[-400:] if len(content) > 400 else content
            
            # Find all numbers in tail
            nums = re.findall(r'\d+', tail)
            
            print(f'\n{case.split("/")[1]}: Expected {expected}')
            print(f'  Numbers in tail: {nums}')
            print(f'  Tail:\n  {tail}')
    except Exception as e:
        print(f'{case}: ERROR - {e}')

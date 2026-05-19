import json, sys
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer, normalize_number
import re

NUMBER_RE = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

cases_to_check = [
    ('man_explicit_1', 'gsm8k_00038', 2),
    ('man_explicit_1', 'gsm8k_00066', 36),
    ('man_explicit_1', 'gsm8k_00158', 1500),
    ('man_explicit_1', 'gsm8k_00189', 4),
    ('man_explicit_1', 'gsm8k_00289', 3),
    ('woman_explicit_1', 'gsm8k_00108', 3),
]

for persona, qid, expected in cases_to_check:
    path = f'result/meta-llama_llama-3.3-70b-instruct/{persona}/{qid}.json'
    d = json.load(open(path, encoding='utf-8'))
    content = d['response']['content']
    result, method = extract_final_answer(content)
    status = "OK" if result == expected else "WRONG"
    print(f"{status} {persona}/{qid}: extracted={result}, method={method}, expected={expected}")

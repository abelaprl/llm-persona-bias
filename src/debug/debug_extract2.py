import json, sys
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

cases = [
    ('man_explicit_1', 'gsm8k_00227'),
    ('man_explicit_1', 'gsm8k_00277'),
    ('man_explicit_1', 'gsm8k_00626'),
    ('man_explicit_1', 'gsm8k_01148'),
    ('man_explicit_1', 'gsm8k_01311'),
    ('woman_explicit_1', 'gsm8k_00108'),
    ('woman_explicit_1', 'gsm8k_00414'),
    ('woman_explicit_1', 'gsm8k_00490'),
    ('woman_explicit_1', 'gsm8k_00523'),
    ('woman_explicit_1', 'gsm8k_01064'),
    ('man_explicit_1', 'gsm8k_00031'),
    ('man_explicit_1', 'gsm8k_00038'),
    ('man_explicit_1', 'gsm8k_00042'),
    ('man_explicit_1', 'gsm8k_00066'),
    ('man_explicit_1', 'gsm8k_00091'),
    ('man_explicit_1', 'gsm8k_00096'),
    ('man_explicit_1', 'gsm8k_00158'),
    ('man_explicit_1', 'gsm8k_00189'),
    ('man_explicit_1', 'gsm8k_00289'),
]
for persona, qid in cases:
    try:
        path = f'result/meta-llama_llama-3.3-70b-instruct/{persona}/{qid}.json'
        d = json.load(open(path, encoding='utf-8'))
        content = d['response']['content']
        result, method = extract_final_answer(content)
        print(f"{persona}/{qid}: extracted={result}, method={method}")
    except Exception as e:
        print(f"ERROR {persona}/{qid}: {e}")

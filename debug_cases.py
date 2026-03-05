import json
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
]
for persona, qid in cases:
    try:
        path = f'result/meta-llama_llama-3.3-70b-instruct/{persona}/{qid}.json'
        d = json.load(open(path, encoding='utf-8'))
        content = d['response']['content']
        lines = content.splitlines()
        print(f"\n=== {persona}/{qid} - last 8 lines ===")
        for l in lines[-8:]:
            print(f"  {repr(l)}")
    except Exception as e:
        print(f"  ERROR: {e}")

import json, sys
sys.path.insert(0, 'src')

cases = [
    ('man_explicit_1', 'gsm8k_00038', 2, 'should be 2'),
    ('man_explicit_1', 'gsm8k_00042', 200, 'should be 200'),
    ('man_explicit_1', 'gsm8k_00066', 36, 'should be 36'),
    ('man_explicit_1', 'gsm8k_00091', 225, 'should be 225'),
    ('man_explicit_1', 'gsm8k_00096', 40, 'should be 40'),
    ('man_explicit_1', 'gsm8k_00158', 1500, 'should be 1500'),
    ('man_explicit_1', 'gsm8k_00189', 4, 'should be 4'),
    ('man_explicit_1', 'gsm8k_00289', 3, 'should be 3'),
    ('woman_explicit_1', 'gsm8k_00490', -10, 'should be -10'),
]
for persona, qid, expected, note in cases:
    try:
        path = f'result/meta-llama_llama-3.3-70b-instruct/{persona}/{qid}.json'
        d = json.load(open(path, encoding='utf-8'))
        content = d['response']['content']
        lines = content.splitlines()
        print(f"\n=== {persona}/{qid} {note} ===")
        for l in lines[-10:]:
            print(f"  {repr(l)}")
    except Exception as e:
        print(f"ERROR {persona}/{qid}: {e}")

import sys; sys.path.insert(0,'src')
from rebuild_results_csv import extract_final_answer
import json, pathlib

cases = [
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00066.json', 36),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00196.json', 12),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00562.json', 180),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00107.json', 72),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00309.json', 6),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00031.json', 109),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00388.json', 36),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00556.json', 2),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00710.json', 3000),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00760.json', 60000),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_00805.json', 20),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_01101.json', 5),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_01135.json', 7),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_01249.json', 750),
    ('result/google_gemini-2.5-flash/man_explicit_1/gsm8k_01265.json', 210),
    ('result/google_gemini-2.5-flash/man_implicit_1/gsm8k_00640.json', 216),
    ('result/google_gemini-2.5-flash/man_implicit_1/gsm8k_01289.json', 34),
]

for path, truth in cases:
    d = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    resp = d.get('response','')
    val, rule = extract_final_answer(resp)
    status = 'OK' if (val is not None and abs(float(val)-float(truth))<0.001) else 'WRONG'
    name = pathlib.Path(path).name
    persona = path.split('/')[2]
    print(f"[{status}] {persona}/{name}: extracted={val} truth={truth} rule={rule}")
    if status == 'WRONG':
        tail = resp[-300:]
        print(f"  tail: {repr(tail)}")
        print()

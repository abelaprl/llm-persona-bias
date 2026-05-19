"""Inspect specific still-wrong cases to understand patterns."""
import sys, json, os, re
sys.path.insert(0, 'src')
import importlib
import rebuild_results_csv_llama
importlib.reload(rebuild_results_csv_llama)
from rebuild_results_csv_llama import extract_final_answer, normalize_number
import pandas as pd

# Focus on specific cases to understand patterns
cases = [
    ('man_explicit_1', 'gsm8k_00019', 7),    # "eat 7 dozen eggs in 4 weeks" → 4
    ('man_explicit_1', 'gsm8k_00086', 44),   # gene's 44 quilt blocks → 11
    ('man_explicit_1', 'gsm8k_00092', 28),   # 18 pets → 28
    ('man_explicit_1', 'gsm8k_00538', 188),  # 31+50+45+62=188 → 62
    ('man_explicit_1', 'gsm8k_00539', 18),   # 72-54=18 vlogs → 72
    ('man_explicit_1', 'gsm8k_00556', 2),    # $80/$40=2 weeks → 80
    ('man_explicit_1', 'gsm8k_00099', 5),    # 25-20=5 cars → 25
    ('man_explicit_1', 'gsm8k_00139', 105),  # → 30
    ('woman_implicit_2', 'gsm8k_00477', 5),  # 20/4=5 books → 300
    ('woman_implicit_2', 'gsm8k_00550', 398), # "has 398 toys" → 5
]

base = 'result/meta-llama_llama-3.3-70b-instruct'

for persona, qid, gold in cases:
    fpath = os.path.join(base, persona, qid + '.json')
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)
    content = d['response']['content']
    result = extract_final_answer(content)
    extracted = result[0] if result else None
    
    lines = content.splitlines()
    last_lines = [l for l in lines if l.strip()][-10:]
    print('='*70)
    print('%s/%s: gold=%s extracted=%s' % (persona, qid, gold, extracted))
    print('--- Last 10 non-empty lines ---')
    for l in last_lines:
        print('  |', l[:120])

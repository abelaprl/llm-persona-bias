"""
Deep analysis of remaining extraction errors for both llama models.
Shows pattern distributions, samples, and compares models.
"""
import pandas as pd
import json
import re
import os
import sys

sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

MODELS = {
    '8b': {
        'csv': r'result/archive/reports_mixed_legacy\meta-llama_llama-3.1-8b-instruct\results.csv',
        'root': r'result\meta-llama_llama-3.1-8b-instruct',
    },
    '70b': {
        'csv': r'result/archive/reports_mixed_legacy\meta-llama_llama-3.3-70b-instruct\results.csv',
        'root': r'result\meta-llama_llama-3.3-70b-instruct',
    },
}

def gold_in_tail(row):
    tail = str(row.get('text_tail', '') or '')
    gold = str(row.get('ground_truth_answer', '') or '')
    if not gold or not tail or gold.strip() in ('nan', ''):
        return False
    try:
        g = float(gold)
        nums = re.findall(r'-?\d+(?:\.\d+)?', tail)
        return any(abs(float(n) - g) < 0.01 for n in nums)
    except Exception:
        return False

def classify_pattern(row):
    tail = str(row.get('text_tail', '') or '')
    tl = tail.lower()
    extracted = str(row.get('extracted_answer', '') or '')
    gold = str(row.get('ground_truth_answer', '') or '')

    # Step-numbered algebraic solution (e.g. "1. ..., 2. ..., so Jan has 3x6=18")
    if re.search(r'\d\s*[x×*]\s*\d|\d\s*/\s*\d', tail) and re.search(r'\d\s*=\s*\d', tail):
        return 'algebraic_steps'
    # Equation in final line
    if re.search(r'\d\s*=\s*\d', tail) and not re.search(r'\n.*\n', tail):
        return 'eq_in_line'
    # Multiple additions: "300 + 312 + 150 = 762"
    if re.search(r'\d\s*[+]\s*\d.*[+]\s*\d', tail):
        return 'multi_add'
    # "some total = X" conclusion in prose
    if re.search(r'\b(total|altogether|in all|combined|overall)\b', tl) and re.search(r'\d', tail):
        return 'total_keyword'
    # Currency as answer (dollar sign)
    if re.search(r'[\$]\s*\d', tail):
        return 'currency_present'
    # Needs to / will take
    if re.search(r'\bneeds?\s+to\b|\bwill\s+take\b|\bit\s+takes?\b', tl):
        return 'need_take'
    # For N [time-unit] trailing context
    if re.search(r'\bfor\s+\d+\s+(?:days?|weeks?|months?|hours?|years?)\b', tl):
        return 'trailing_context'
    # Parallel and: "6 apples and 4 oranges"
    if re.search(r'\d+\s+\w+\s+and\s+\d+\s+\w+', tl):
        return 'parallel_and'
    # "more/less than" relative quantity
    if re.search(r'\b(more|less|fewer|extra)\s+than\b|\b(more|fewer|extra)\b', tl):
        return 'relative_quantity'
    return 'unknown'


for model_name, cfg in MODELS.items():
    print(f'\n{"="*70}')
    print(f'MODEL: {model_name}')
    print(f'{"="*70}')
    df = pd.read_csv(cfg['csv'])
    wrong = df[df['is_correct'] == False].copy()
    print(f'Total rows: {len(df)},  Wrong: {len(wrong)} ({len(wrong)/len(df)*100:.2f}%)')

    wrong['gold_in_tail'] = wrong.apply(gold_in_tail, axis=1)
    possible = wrong[wrong['gold_in_tail']].copy()
    print(f'Possible extraction errors (gold in tail): {len(possible)}')
    print(f'Genuine model errors (gold NOT in tail):   {len(wrong) - len(possible)}')

    possible['pattern'] = possible.apply(classify_pattern, axis=1)
    counts = possible['pattern'].value_counts()
    print('\nPattern distribution:')
    for pat, n in counts.items():
        print(f'  {pat:<22} {n:>4}')

    print('\nSample tails per pattern:')
    for pat in counts.index[:6]:
        sample = possible[possible['pattern'] == pat].head(3)
        print(f'\n--- {pat} ({counts[pat]} cases) ---')
        for _, row in sample.iterrows():
            file_rel = str(row.get('file', ''))
            json_path = os.path.join(cfg['root'], file_rel)
            content = ''
            if os.path.exists(json_path):
                with open(json_path, encoding='utf-8') as f:
                    d = json.load(f)
                content = d.get('response', {}).get('content', '') or ''
            new_result = extract_final_answer(content) if content else (None, None)
            print(f"  file={file_rel}  gold={row['ground_truth_answer']}"
                  f"  old={row['extracted_answer']}  new={new_result[0]}  reason={row['extraction_reason']}")
            print(f"  tail: {str(row['text_tail'])[-200:]!r}")

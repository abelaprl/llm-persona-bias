"""Deep inspect remaining 8b extraction errors - show full responses for all patterns."""
import pandas as pd, json, re, os, sys
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

CSV_PATH = r'result\reports\meta-llama_llama-3.1-8b-instruct\possible_extraction_errors_8b.csv'
JSON_ROOT = r'result\meta-llama_llama-3.1-8b-instruct'

df = pd.read_csv(CSV_PATH)

def classify_pattern(row):
    tail = str(row.get('text_tail', '') or '')
    tail_lower = tail.lower()
    if re.search(r'\d\s*[+\-xX*/]\s*\d|\d\s*=\s*\d', tail):
        return 'eq_in_line'
    if re.search(r'\bwill\s+take\b|\bneeds?\s+to\b|\bit\s+takes?\b', tail_lower):
        return 'need_take'
    if re.search(r'\bfor\s+\d+\s+(?:days?|weeks?|months?|hours?)\b', tail_lower):
        return 'trailing_context'
    if re.search(r'\btotal\b|\boverall\b|\bcombined\b', tail_lower):
        return 'total_keyword'
    if re.search(r'[\$]\s*\d', tail):
        return 'currency_present'
    return 'unknown'

df['pattern'] = df.apply(classify_pattern, axis=1)

# Re-run extraction and collect still-wrong cases
wrong_cases = []
for _, row in df.iterrows():
    file_rel = str(row.get('file', ''))
    json_path = os.path.join(JSON_ROOT, file_rel)
    gold = row['ground_truth_answer']
    if not os.path.exists(json_path):
        continue
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    content = d.get('response', {}).get('content', '') or ''
    result, reason = extract_final_answer(content)
    try:
        correct = result is not None and abs(float(result) - float(gold)) < 0.01
    except Exception:
        correct = False
    if not correct:
        wrong_cases.append({
            'file': file_rel,
            'gold': gold,
            'got': result,
            'reason': reason,
            'pattern': row['pattern'],
            'old_extracted': row['extracted_answer'],
            'tail': content[-400:]
        })

print(f'Still wrong: {len(wrong_cases)} / {len(df)}')
print()

# Group by pattern
from collections import defaultdict
by_pattern = defaultdict(list)
for c in wrong_cases:
    by_pattern[c['pattern']].append(c)

for pat in ['eq_in_line', 'total_keyword', 'unknown', 'currency_present', 'leading_number', 'need_take']:
    cases = by_pattern[pat]
    if not cases:
        continue
    print(f'\n{"="*65}')
    print(f'PATTERN: {pat}  ({len(cases)} still wrong)')
    print(f'{"="*65}')
    for c in cases[:3]:
        print(f'  file: {c["file"]}')
        print(f'  gold={c["gold"]}  old={c["old_extracted"]}  now={c["got"]}  reason={c["reason"]}')
        print('  TAIL (last 400):')
        for line in c['tail'].split('\n'):
            print('  | ' + line)
        print()

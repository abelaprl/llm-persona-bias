"""Deep dive into specific wrong cases for 8b model — show full responses."""
import pandas as pd
import json
import re
import os
import sys

sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

CSV_PATH = r'result\reports\meta-llama_llama-3.1-8b-instruct\possible_extraction_errors_8b.csv'
JSON_ROOT = r'result\meta-llama_llama-3.1-8b-instruct'

df = pd.read_csv(CSV_PATH)


def classify_pattern(row):
    tail = str(row.get('text_tail', '') or '')
    tail_lower = tail.lower()
    if re.match(r'^\s*\d', tail_lower):
        return 'leading_number'
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
print('Pattern counts:')
print(df['pattern'].value_counts().to_string())
print()

for pat in ['eq_in_line', 'total_keyword', 'leading_number', 'unknown']:
    print()
    print('=' * 65)
    print(f'PATTERN: {pat}  ({len(df[df["pattern"] == pat])} cases)')
    print('=' * 65)
    for _, row in df[df['pattern'] == pat].head(2).iterrows():
        file_rel = str(row.get('file', '')).replace('\\\\', os.sep).replace('\\', os.sep)
        json_path = os.path.join(JSON_ROOT, file_rel)
        content = ''
        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                d = json.load(f)
            content = d.get('response', {}).get('content', '') or ''
        print(f'File={file_rel}  gold={row["ground_truth_answer"]}'
              f'  extracted={row["extracted_answer"]}  reason={row["extraction_reason"]}')
        print('--- RESPONSE (last 600 chars) ---')
        print(content[-600:])
        print(f'--- RE-EXTRACT: {extract_final_answer(content)} ---')
        print()

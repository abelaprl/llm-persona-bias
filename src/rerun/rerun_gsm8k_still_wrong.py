import os
import re
import csv
import json
import pandas as pd
import unicodedata

def find_last_boxed(text):
    stack = []
    last_boxed = None
    i = 0
    while i < len(text):
        if text[i:i+7] == '\\boxed{':
            start = i + 7
            stack = [start-1]
            i = start
            while i < len(text):
                if text[i] == '{':
                    stack.append(i)
                elif text[i] == '}':
                    stack.pop()
                    if not stack:
                        last_boxed = text[start:i]
                        break
                i += 1
        else:
            i += 1
    return last_boxed

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace('{,}', ',').replace('\\,', ',')
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\u2009', ' ')
    s = s.replace('\u202f', ' ')
    s = s.replace('\u00a0', ' ')
    s = re.sub(r'\\$|\\$', '', s)
    s = s.replace('$', '')
    return s

def extract_number_with_trace(text):
    text_norm = normalize_text(text)
    trace = {'picked_by_rule': 'NONE', 'picked_span': '', 'failure_reason': ''}
    boxed = find_last_boxed(text_norm)
    if boxed:
        num = extract_last_number(boxed)
        if num is not None:
            trace['picked_by_rule'] = 'BOXED'
            trace['picked_span'] = f'\\boxed{{{boxed}}}'
            return num, trace
    answer_lines = []
    for line in text_norm.splitlines():
        if re.search(r'(answer|final|thus|therefore)\s*[:：]', line, re.I):
            answer_lines.append(line)
    if answer_lines:
        num = extract_last_number(answer_lines[-1])
        if num is not None:
            trace['picked_by_rule'] = 'ANSWER_LINE'
            trace['picked_span'] = answer_lines[-1]
            return num, trace
    for line in reversed(text_norm.splitlines()):
        if re.fullmatch(r'\s*[-+]?\d[\d,]*\.?\d*\s*', line):
            num = extract_last_number(line)
            if num is not None:
                trace['picked_by_rule'] = 'STANDALONE_LINE'
                trace['picked_span'] = line
                return num, trace
    num = extract_last_number(text_norm)
    if num is not None:
        trace['picked_by_rule'] = 'LAST_NUMBER'
        trace['picked_span'] = '...'
        return num, trace
    trace['failure_reason'] = 'no digits found' if not re.search(r'\d', text_norm) else 'format not matched'
    return None, trace

def extract_last_number(s):
    s = normalize_text(s)
    matches = list(re.finditer(r'[-+]?\d[\d,]*\.?\d*', s))
    if not matches:
        return None
    last = matches[-1].group(0).replace(',', '').rstrip('.')
    try:
        return float(last)
    except Exception:
        return None

def safe_float(x):
    try:
        return float(str(x).replace(',', '').replace(' ', ''))
    except Exception:
        return None

def round_compare(a, b):
    try:
        if a is None or b is None:
            return False
        if abs(a - int(a)) < 1e-6 and abs(b - int(b)) < 1e-6:
            return int(round(a)) == int(round(b))
        return abs(a - b) < 1e-6
    except Exception:
        return False

def get_json_path(row):
    if 'file' in row and isinstance(row['file'], str) and row['file'].endswith('.json'):
        return row['file']
    persona = row.get('persona', 'unknown')
    model = row.get('model', 'unknown')
    qid = int(row['question_id'])
    return f'result/{model}/{persona}/gsm8k_{qid:05d}.json'

def main():
    df = pd.read_csv('result/reports/allenai_olmo-3-32b-think/examples_still_wrong.csv')
    rerun_rows = []
    for idx, row in df.iterrows():
        reason = None
        if row.get('error_category', '') == 'no_number_found':
            reason = 'no_number_found'
        elif row.get('is_truncated', False) or str(row.get('is_truncated', '')).lower() == 'true':
            reason = 'truncated'
        if reason:
            rerun_rows.append({
                'question_id': row['question_id'],
                'persona': row.get('persona', ''),
                'model': row.get('model', 'allenai_olmo-3-32b-think'),
                'json_path': get_json_path(row),
                'reason': reason
            })
    rerun_plan = pd.DataFrame(rerun_rows)
    rerun_plan.to_csv('rerun_plan.csv', index=False)
    rerun_results = []
    for _, plan in rerun_plan.iterrows():
        json_path = plan['json_path']
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        response = data.get('response_content', '')
        gt = data.get('ground_truth_answer', data.get('answer', None))
        num, trace = extract_number_with_trace(response)
        is_trunc = data.get('is_truncated', False) or (data.get('completion_tokens', 0) == 2048)
        is_correct = round_compare(num, safe_float(gt))
        rerun_results.append({
            'question_id': plan['question_id'],
            'gt': gt,
            'new_response': response,
            'new_extracted': num,
            'new_is_correct': is_correct,
            'was_truncated': is_trunc,
            'picked_by_rule': trace['picked_by_rule'],
            'picked_span': trace['picked_span'],
            'failure_reason': trace.get('failure_reason', '')
        })
    rerun_results_df = pd.DataFrame(rerun_results)
    rerun_results_df.to_csv('rerun_results.csv', index=False)
    print('Rerun plan and results written to rerun_plan.csv and rerun_results.csv')

if __name__ == '__main__':
    main()

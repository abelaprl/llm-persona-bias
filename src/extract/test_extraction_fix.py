"""Test a sample of the possible_extraction_errors.csv cases to see how many are fixed."""
import sys, json, os, re
sys.path.insert(0, 'src')
import importlib
import rebuild_results_csv_llama
importlib.reload(rebuild_results_csv_llama)
from rebuild_results_csv_llama import extract_final_answer, normalize_number
import pandas as pd

df = pd.read_csv('possible_extraction_errors.csv')

base = 'result/meta-llama_llama-3.3-70b-instruct'
fixed = 0
still_wrong = 0
not_found = 0
same_wrong = 0

wrong_cases = []

for _, row in df.iterrows():
    persona = row['persona_folder']
    qid = row['qid']
    gold_raw = row['ground_truth_answer']
    old_extracted = row['extracted_answer']
    
    fpath = os.path.join(base, persona, qid + '.json')
    if not os.path.exists(fpath):
        not_found += 1
        continue
    
    try:
        with open(fpath, encoding='utf-8') as f:
            d = json.load(f)
        content = d['response']['content']
        result = extract_final_answer(content)
        if result is None:
            new_extracted = None
        else:
            new_extracted, _ = result
        
        gold = normalize_number(str(gold_raw))
        new_num = normalize_number(str(new_extracted)) if new_extracted is not None else None
        old_num = normalize_number(str(old_extracted)) if pd.notna(old_extracted) else None
        
        if new_num is not None and gold is not None and abs(new_num - gold) < 0.01:
            fixed += 1
        elif new_num == old_num:
            same_wrong += 1
            wrong_cases.append((persona, qid, gold, old_num, new_num, row['extraction_reason']))
        else:
            still_wrong += 1
            wrong_cases.append((persona, qid, gold, old_num, new_num, row['extraction_reason']))
    except Exception as e:
        not_found += 1

total = fixed + still_wrong + same_wrong
print('=== Results on %d cases ===' % total)
print('FIXED (now correct):  %d (%.1f%%)' % (fixed, 100*fixed/total if total else 0))
print('SAME WRONG:           %d (%.1f%%)' % (same_wrong, 100*same_wrong/total if total else 0))
print('DIFFERENT WRONG:      %d (%.1f%%)' % (still_wrong, 100*still_wrong/total if total else 0))
print('NOT FOUND:            %d' % not_found)
print()
print('=== Sample still-wrong cases (first 30) ===')
for persona, qid, gold, old_n, new_n, reason in wrong_cases[:30]:
    print('  %s/%s: gold=%s old=%s new=%s [%s]' % (persona, qid, gold, old_n, new_n, reason))

"""Categorize remaining extraction failures by pattern."""
import sys, json, os, re
sys.path.insert(0, 'src')
import importlib
import rebuild_results_csv_llama as m
importlib.reload(m)
import pandas as pd

df = pd.read_csv('possible_extraction_errors.csv')
base = 'result/meta-llama_llama-3.3-70b-instruct'

NUMBER_RE = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
MARKER_RE = re.compile(r'(Final Answer|Answer|Therefore|Thus|Hence|So)\b', re.I)

categories = {}
still_wrong = []

for _, row in df.iterrows():
    persona, qid = row['persona_folder'], row['qid']
    gold_raw = row['ground_truth_answer']
    old_extracted = row['extracted_answer']
    fpath = os.path.join(base, persona, qid + '.json')
    if not os.path.exists(fpath): continue
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)
    content = d['response']['content']
    result = m.extract_final_answer(content)
    extracted = result[0] if result else None
    
    gold_n = m.normalize_number(str(gold_raw))
    ext_n = m.normalize_number(str(extracted)) if extracted is not None else None
    if ext_n is not None and gold_n is not None and abs(ext_n - gold_n) < 0.01:
        continue  # already fixed
    
    # Find the last marker line
    lines = content.splitlines()
    marker_line = ''
    for line in reversed(lines):
        if MARKER_RE.search(line):
            marker_line = line.strip()
            break
    
    # Categorize
    cat = 'unknown'
    if '=' in marker_line:
        # "= N unit" at end
        if re.search(r'=\s*' + NUMBER_RE + r'\s*\w+\s*$', marker_line):
            cat = 'eq_N_unit_end'
        else:
            cat = 'eq_in_line'
    elif re.search(r'\b(more|fewer|less|longer|shorter|extra|additional|remaining)\b', marker_line, re.I):
        cat = 'relative_quantity'
    elif re.search(r'\b(total|altogether|combined|together|in all)\b', marker_line, re.I):
        cat = 'total_keyword'
    elif re.search(r'\b(need|needs|take|takes|will\s+take)\b', marker_line, re.I):
        cat = 'need_take'
    elif re.search(r'\bfor\s+\d+\s+\w+', marker_line, re.I):
        cat = 'trailing_context'
    
    categories.setdefault(cat, []).append((persona, qid, gold_raw, extracted, marker_line))

print('=== PATTERN CATEGORIES ===')
for cat, cases in sorted(categories.items(), key=lambda x: -len(x[1])):
    print('\n[%s] (%d cases)' % (cat, len(cases)))
    for persona, qid, gold, ext, line in cases[:8]:
        print('  %s/%s gold=%s got=%s | %s' % (persona, qid, gold, ext, line[:120]))

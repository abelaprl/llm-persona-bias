import sys, re
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import normalize_number, extract_final_answer
import json

# Check directly what _extract_answer_number does for this sentence
with open('result/meta-llama_llama-3.3-70b-instruct/man_explicit_1/gsm8k_00289.json', encoding='utf-8') as f:
    d = json.load(f)
content = d['response']['content']
lines = content.splitlines()
last_lines = [l for l in lines[-20:]]
print('Last 20 lines:')
for l in last_lines:
    print(repr(l))
print()

# Simulate Rule 3 processing
NUMBER_RE = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
marker_re = re.compile(r'(Final Answer|(?<![a-zA-Z-])Answer|So the answer is|Therefore|Thus|Hence|(?<![a-zA-Z])So(?!\s+the\s+answer\s+is))', re.I)

for line in reversed(last_lines):
    m = marker_re.search(line)
    if m:
        print(f'Found marker "{m.group()}" in line: {repr(line)}')
        parts = marker_re.split(line, maxsplit=1)
        search_text = ''.join(parts[2:]) if len(parts) >= 3 else line
        if re.match(r'^:\s+\*\*', search_text):
            search_text = re.sub(r'^[:\s]+', '', search_text)
        else:
            search_text = re.sub(r'^[\*:\s]+', '', search_text)
        search_text = re.sub(r'^[,;]\s*', '', search_text)
        print(f'  -> search_text: {repr(search_text)}')
        matched_marker_lower = (parts[1].lower().strip() if len(parts) >= 2 else '')
        is_weak = matched_marker_lower in ('thus', 'hence', 'therefore', 'so')
        print(f'  -> is_weak: {is_weak}')
        break  # just check the last occurrence

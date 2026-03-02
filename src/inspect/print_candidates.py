from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extractor34
import re
import json
p = Path('result/openai_gpt-oss-120b/man_explicit_1/gsm8k_00344.json')
data = json.loads(p.read_text(encoding='utf-8'))
text = extractor34.normalize_text(data.get('response', {}).get('content') or '')
print('TEXT:\n', text)

candidates = []
for m in re.finditer(r'([0-9]{1,3}(?:(?:,|\{,\}|\\,|\u2009|\u202f)\d{3})+)', text):
    tok = m.group(1)
    v = extractor34.parse_number_token(tok)
    candidates.append(('group_any', tok, v, m.start()))

for m in re.finditer(r'\\boxed\{([^}]+)\}', text):
    tok = m.group(1)
    v = extractor34.parse_number_token(tok)
    candidates.append(('boxed', tok, v, m.start()))

for m in re.finditer(r'####\s*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]*)', text):
    tok = m.group(1)
    v = extractor34.parse_number_token(tok)
    candidates.append(('hash', tok, v, m.start()))

lines = text.split('\n')
last_lines = lines[-20:]
offset_base = len(lines) - len(last_lines)
line_starts = []
idx = 0
for ln in lines:
    line_starts.append(idx)
    idx += len(ln) + 1

for i, line in enumerate(last_lines):
    global_line_index = offset_base + i
    start_offset = line_starts[global_line_index]
    if re.search(r'\bAnswer\b|\banswer\b|\bThe answer is\b', line):
        for m in re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', line):
            tok = m.group(1)
            v = extractor34.parse_number_token(tok)
            candidates.append(('answer_line', tok, v, start_offset + m.start()))

last6 = last_lines[-6:]
for i, line in enumerate(last6):
    global_line_index = offset_base + (len(last_lines) - len(last6)) + i
    start_offset = line_starts[global_line_index]
    for m in re.finditer(r'\*\*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)\*\*', line):
        tok = m.group(1)
        v = extractor34.parse_number_token(tok)
        candidates.append(('bold', tok, v, start_offset + m.start()))

for i, line in enumerate(last6):
    global_line_index = offset_base + (len(last_lines) - len(last6)) + i
    start_offset = line_starts[global_line_index]
    m = re.search(r'=\s*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)\s*$', line)
    if m:
        tok = m.group(1)
        v = extractor34.parse_number_token(tok)
        candidates.append(('equals', tok, v, start_offset + m.start(1)))

for i, line in enumerate(last_lines):
    global_line_index = offset_base + i
    start_offset = line_starts[global_line_index]
    for m in re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', line):
        tok = m.group(1)
        pre = line[:m.start()].lower()
        if m.start() > 0 and line[m.start()-1] in '-–—' and (m.start() == 1 or line[m.start()-2].isspace()):
            tok = tok.lstrip('+-')
        bad_context = any(w in pre for w in [' per ', ' each', 'price', 'cost', 'subtotal', ' rate ', 'apiece', 'per day', 'per week', 'per month', 'per year'])
        v = extractor34.parse_number_token(tok)
        candidates.append(('token', tok, v, start_offset + m.start(), bad_context))

print('\nCANDIDATES:')
for c in candidates:
    print(c)

# Also show which numeric tokens all_nums regex picks for final fallback
all_nums = list(re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', text))
print('\nALL NUM TOKENS:')
for m in all_nums:
    print(m.group(1), '->', extractor34.parse_number_token(m.group(1)))

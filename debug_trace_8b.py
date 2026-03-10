"""Debug trace for specific failing cases in 8b model."""
import sys, json, os, re
sys.path.insert(0, 'src')

# Patch to trace internal Rule 3 logic
import rebuild_results_csv_llama as efa_mod

ROOT = r'result\meta-llama_llama-3.1-8b-instruct'

def load_content(persona, qid):
    path = os.path.join(ROOT, persona, qid + '.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d['response']['content']

# ---- Test 1: gsm8k_00854 (gold=15) ----
print('='*60)
print('TEST: gsm8k_00854 (gold=15)')
content = load_content('man_explicit_1', 'gsm8k_00854')
tail = content[-1200:] if len(content) > 1200 else content
last_lines = tail.splitlines()[-20:]
print('Last 20 lines:')
for i, ln in enumerate(last_lines):
    print(f'  [{i:02d}] {repr(ln[:100])}')
print()

# Try _extract_answer_number on the key sentences
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

# Manually call on specific snippet
snippet_854 = "it would take 15 hours to clean one-half of the rooms at the KozyInn Motel."
print('_extract_answer_number test:', snippet_854)
result = extract_final_answer(snippet_854)
print('Result:', result)
print()

print('Full extract_final_answer:')
result = extract_final_answer(content)
print('Result:', result)
print()

# ---- Test 2: gsm8k_00877 (gold=41) ----
print('='*60)
print('TEST: gsm8k_00877 (gold=41)')
content2 = load_content('man_implicit_1', 'gsm8k_00877')
tail2 = content2[-1200:] if len(content2) > 1200 else content2
last_lines2 = tail2.splitlines()[-20:]
print('Last 20 lines:')
for i, ln in enumerate(last_lines2):
    print(f'  [{i:02d}] {repr(ln[:100])}')
print()
result2 = extract_final_answer(content2)
print('Full extract result:', result2)
print()

# ---- Test 3: gsm8k_00155 (gold=272) ----
print('='*60)
print('TEST: gsm8k_00155 (gold=272)')
content3 = load_content('man_explicit_1', 'gsm8k_00155')
tail3 = content3[-1200:] if len(content3) > 1200 else content3
last_lines3 = tail3.splitlines()[-20:]
print('Last 20 lines:')
for i, ln in enumerate(last_lines3):
    print(f'  [{i:02d}] {repr(ln[:100])}')
print()
result3 = extract_final_answer(content3)
print('Full extract result:', result3)
print()

# ---- Test 4: gsm8k_00803 (gold=6) ----
print('='*60)
print('TEST: gsm8k_00803 (gold=6)')
content4 = load_content('man_explicit_1', 'gsm8k_00803')
tail4 = content4[-1200:] if len(content4) > 1200 else content4
last_lines4 = tail4.splitlines()[-20:]
print('Last 20 lines:')
for i, ln in enumerate(last_lines4):
    print(f'  [{i:02d}] {repr(ln[:100])}')
print()
result4 = extract_final_answer(content4)
print('Full extract result:', result4)

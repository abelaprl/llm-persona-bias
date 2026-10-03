"""
Analyze all no_number_found cases from results.csv.
Categorize by tail pattern to identify what rules are missing.
"""
import csv, re
from collections import Counter

csv_path = r'e:\Tugas Akhir\TugasAkhir\result/archive/reports_mixed_legacy\openai_gpt-oss-20b\results.csv'

cases = []
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['error_category'].strip() == 'no_number_found':
            cases.append({
                'persona': row['persona_folder'],
                'file': row['file'],
                'gt': row['ground_truth_answer'],
                'tail': row['text_tail'],
            })

print(f"Total no_number_found: {len(cases)}\n")

# Categorize by tail pattern
def categorize(tail):
    t = tail.strip()
    if re.search(r'\\boxed\{[^}]*\}', t):
        return 'boxed_with_unit'       # \boxed{64\text{ minutes}}
    if re.search(r'\*\*[^*]+\*\*', t):
        return 'bold_sentence'         # **Igor lost 2 beakers.**
    if re.search(r'\\]\s*$', t):
        return 'latex_block_end'       # ends with \]
    if re.search(r'\\\[.*\d', t):
        return 'latex_inline_num'      # \[ ... number ... \]
    if re.search(r'confirming|matches|verified|checks? out', t, re.I):
        return 'verification_sentence'
    if re.search(r'\d+\s*(years?|months?|days?|hours?|minutes?|seconds?|people|persons?|students?|children|apples?|books?|coins?|bags?|boxes?|rooms?|miles?|km|meters?|feet|inches?|pounds?|kg|grams?|liters?|gallons?|dollars?|cents?|percent|%|oz|ounces?|trees?|flowers?|birds?|dogs?|cats?|pens?|TVs?|cars?|shirts?|pieces?|batches?|cookies?|beakers?|popcorn|snowballs?|squares?|homes?|bugs?|petals?|bars?|gallons?|mph|mph|fence|charge)', t, re.I):
        return 'number_with_unit_word'
    if re.search(r'^\d+\s*$', t.strip()):
        return 'plain_number_only'
    if re.search(r'\d', t):
        return 'has_number_other'
    return 'no_number_at_all'

cat_counter = Counter()
examples = {}
for c in cases:
    cat = categorize(c['tail'])
    cat_counter[cat] += 1
    if cat not in examples:
        examples[cat] = []
    if len(examples[cat]) < 5:
        examples[cat].append(c)

print("=== Category breakdown ===")
for cat, count in cat_counter.most_common():
    print(f"  {cat:<30} {count:>4}")

print()
for cat, count in cat_counter.most_common():
    print(f"\n{'='*60}")
    print(f"CATEGORY: {cat} ({count} cases)")
    print(f"{'='*60}")
    for ex in examples[cat]:
        print(f"  gt={ex['gt']!r:>8} | tail: {ex['tail']!r}")

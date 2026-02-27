import csv
import sys
import os
from collections import defaultdict

csv_path = r'e:\Tugas Akhir\TugasAkhir\result\mistralai_ministral-8b-2512\results.csv'
out_dir  = r'e:\Tugas Akhir\TugasAkhir\result\mistralai_ministral-8b-2512'

PERSONA_ORDER = [
    'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3',
    'woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3'
]

persona_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'no_extract': 0, 'truncated': 0})
overall = {'correct': 0, 'total': 0, 'no_extract': 0, 'truncated': 0}

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        persona = row['persona_folder']
        is_correct = row['is_correct'].strip().lower() == 'true'
        error_cat  = row['error_category'].strip()
        persona_stats[persona]['total'] += 1
        overall['total'] += 1
        if is_correct:
            persona_stats[persona]['correct'] += 1
            overall['correct'] += 1
        if error_cat == 'no_number_found':
            persona_stats[persona]['no_extract'] += 1
            overall['no_extract'] += 1
        if error_cat == 'truncated_output':
            persona_stats[persona]['truncated'] += 1
            overall['truncated'] += 1

# ── Print to console ──────────────────────────────────────────────────────────
header = f"{'Persona':<22} {'Correct':>7} {'Total':>6} {'Accuracy':>9} {'No-Extract':>11} {'Truncated':>10}"
sep    = '-' * 70
print(header)
print(sep)
rows_out = []
for p in PERSONA_ORDER:
    s   = persona_stats[p]
    acc = s['correct'] / s['total'] * 100 if s['total'] else 0
    line = f"{p:<22} {s['correct']:>7} {s['total']:>6} {acc:>8.2f}% {s['no_extract']:>11} {s['truncated']:>10}"
    print(line)
    rows_out.append({'persona': p, 'correct': s['correct'], 'total': s['total'],
                     'accuracy': round(acc, 4),
                     'no_extract': s['no_extract'], 'truncated': s['truncated']})
print(sep)
acc_all = overall['correct'] / overall['total'] * 100 if overall['total'] else 0
print(f"{'OVERALL':<22} {overall['correct']:>7} {overall['total']:>6} {acc_all:>8.2f}% {overall['no_extract']:>11} {overall['truncated']:>10}")

# ── Write accuracy.csv ────────────────────────────────────────────────────────
acc_csv = os.path.join(out_dir, 'accuracy.csv')
with open(acc_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['persona','correct','total','accuracy','no_extract','truncated'])
    writer.writeheader()
    writer.writerows(rows_out)
    writer.writerow({'persona': 'OVERALL', 'correct': overall['correct'],
                     'total': overall['total'], 'accuracy': round(acc_all, 4),
                     'no_extract': overall['no_extract'], 'truncated': overall['truncated']})
print(f"\nWrote {acc_csv}")

# ── Write per-persona result CSVs ─────────────────────────────────────────────
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    persona_rows = defaultdict(list)
    for row in reader:
        persona_rows[row['persona_folder']].append(row)

for p in PERSONA_ORDER:
    rows = persona_rows[p]
    if not rows:
        continue
    out_path = os.path.join(out_dir, f'results_{p}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    n_correct = sum(1 for r in rows if r['is_correct'].strip().lower() == 'true')
    n_total   = len(rows)
    print(f"  Wrote {out_path}  ({n_correct}/{n_total} correct)")

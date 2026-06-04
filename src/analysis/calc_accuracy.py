import csv
import sys
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv_path',
    default=r'e:\Tugas Akhir\TugasAkhir\result\reports\arcee-ai_trinity-large-thinking_llama\results.csv'
)
parser.add_argument(
    '--out_dir',
    default=r'e:\Tugas Akhir\TugasAkhir\result\reports\arcee-ai_trinity-large-thinking_llama'
)
parser.add_argument(
    '--max_rows',
    type=int,
    default=400,
    help='Max rows per persona to include in accuracy calculation (default: 400)'
)
args = parser.parse_args()

csv_path = args.csv_path
out_dir = args.out_dir

# For Qwen reports, use legacy denominator to reduce truncation/no-extract bias.
is_qwen_report = 'qwen_' in (csv_path + ' ' + out_dir).lower()

PERSONA_ORDER = [
    'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3',
    'woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3',
    'neutral'
]

# Add token fields to the stats dictionary
persona_stats = defaultdict(lambda: {
    'correct': 0, 'total': 0, 'no_extract': 0, 'truncated': 0,
    'sum_prompt_tokens': 0, 'sum_completion_tokens': 0,
    'sum_reasoning_tokens': 0, 'sum_total_tokens': 0
})
overall = defaultdict(lambda: 0)
# Helper to safely convert token values to int
def to_int(val):
    if val is None or val == '':
        return 0
    try:
        # Handle potential float strings like '123.0'
        return int(float(val))
    except (ValueError, TypeError):
        return 0

# ── First pass: collect stats and open persona CSV writers ──────────────────
persona_writers = {}
persona_files = {}
fieldnames_global = None
detected_personas = set()

try:
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames_global = reader.fieldnames

        # First pass: detect all personas in the CSV
        for row in reader:
            persona = row['persona_folder']
            detected_personas.add(persona)
finally:
    pass

# Use detected personas, sorted alphabetically
PERSONA_ORDER = sorted(list(detected_personas))

# Track row count per persona to cap at max_rows each
persona_row_count = defaultdict(lambda: 0)
MAX_ROWS_PER_PERSONA = args.max_rows

# Second pass: process data and write per-persona CSVs
persona_writers = {}
persona_files = {}

try:
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames_global = reader.fieldnames

        # Open all persona CSV writers upfront
        for p in PERSONA_ORDER:
            out_path = os.path.join(out_dir, f'results_{p}.csv')
            file_obj = open(out_path, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames_global, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            persona_writers[p] = writer
            persona_files[p] = file_obj

        # Stream through CSV, collecting stats and writing to persona files
        for row in reader:
            persona = row['persona_folder']
            
            # Cap each persona at 400 rows
            persona_row_count[persona] += 1
            if persona_row_count[persona] > MAX_ROWS_PER_PERSONA:
                continue
            
            is_correct = row['is_correct'].strip().lower() == 'true'
            error_cat  = row.get('error_category', '').strip()

            # fallback: infer no_extract from empty extracted_answer
            if not error_cat:
                extracted = row.get('extracted_answer', '').strip()
                if not extracted or extracted.lower() in ('', 'none', 'null'):
                    error_cat = 'no_number_found'

            # Update counts
            stats = persona_stats[persona]
            stats['total'] += 1
            overall['total'] += 1

            if is_correct:
                stats['correct'] += 1
                overall['correct'] += 1
            if error_cat == 'no_number_found':
                stats['no_extract'] += 1
                overall['no_extract'] += 1
            if error_cat == 'truncated_output':
                stats['truncated'] += 1
                overall['truncated'] += 1

            # Sum token usage
            stats['sum_prompt_tokens'] += to_int(row.get('prompt_tokens'))
            stats['sum_completion_tokens'] += to_int(row.get('completion_tokens'))
            stats['sum_reasoning_tokens'] += to_int(row.get('reasoning_tokens'))
            stats['sum_total_tokens'] += to_int(row.get('total_tokens'))

            overall['sum_prompt_tokens'] += to_int(row.get('prompt_tokens'))
            overall['sum_completion_tokens'] += to_int(row.get('completion_tokens'))
            overall['sum_reasoning_tokens'] += to_int(row.get('reasoning_tokens'))
            overall['sum_total_tokens'] += to_int(row.get('total_tokens'))

            # Write row to persona-specific CSV
            if persona in persona_writers:
                persona_writers[persona].writerow(row)
finally:
    # Close all persona file writers
    for p in PERSONA_ORDER:
        if p in persona_files:
            persona_files[p].close()


# ── Print to console ──────────────────────────────────────────────────────────
total_label = 'Total(OK)' if is_qwen_report else 'Total'
header = f"{'Persona':<22} {'Correct':>7} {total_label:>9} {'Accuracy':>9} {'No-Extract':>11} {'Truncated':>10}"
sep    = '-' * 70
print(header)
print(sep)
rows_out = []
for p in PERSONA_ORDER:
    s   = persona_stats[p]
    if s['total'] == 0: continue
    total_ok = max(s['total'] - s['no_extract'] - s['truncated'], 0)
    denom = total_ok if is_qwen_report else s['total']
    acc = (s['correct'] / denom * 100) if denom else 0
    
    # Calculate averages
    avg_prompt = s['sum_prompt_tokens'] / s['total']
    avg_completion = s['sum_completion_tokens'] / s['total']
    avg_reasoning = s['sum_reasoning_tokens'] / s['total']
    avg_total = s['sum_total_tokens'] / s['total']

    line = f"{p:<22} {s['correct']:>7} {denom:>9} {acc:>8.2f}% {s['no_extract']:>11} {s['truncated']:>10}"
    print(line)
    
    rows_out.append({
        'persona': p, 'correct': s['correct'],
        'total': s['total'],
        'total_ok': total_ok,
        'accuracy': round(acc, 4),
        'no_extract': s['no_extract'], 'truncated': s['truncated'],
        'sum_prompt_tokens': s['sum_prompt_tokens'],
        'sum_completion_tokens': s['sum_completion_tokens'],
        'sum_reasoning_tokens': s['sum_reasoning_tokens'],
        'sum_total_tokens': s['sum_total_tokens'],
        'avg_prompt_tokens': round(avg_prompt, 2),
        'avg_completion_tokens': round(avg_completion, 2),
        'avg_reasoning_tokens': round(avg_reasoning, 2),
        'avg_total_tokens': round(avg_total, 2)
    })

print(sep)
overall_total_ok = max(overall['total'] - overall['no_extract'] - overall['truncated'], 0)
overall_denom = overall_total_ok if is_qwen_report else overall['total']
acc_all = (overall['correct'] / overall_denom * 100) if overall_denom else 0
print(f"{'OVERALL':<22} {overall['correct']:>7} {overall_denom:>9} {acc_all:>8.2f}% {overall['no_extract']:>11} {overall['truncated']:>10}")

# ── Write accuracy.csv ────────────────────────────────────────────────────────
acc_csv = os.path.join(out_dir, 'accuracy.csv')
fieldnames_acc = [
    'persona','correct','total','total_ok','accuracy','no_extract','truncated',
    'sum_prompt_tokens', 'sum_completion_tokens', 'sum_reasoning_tokens', 'sum_total_tokens',
    'avg_prompt_tokens', 'avg_completion_tokens', 'avg_reasoning_tokens', 'avg_total_tokens'
]

with open(acc_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_acc)
    writer.writeheader()
    writer.writerows(rows_out)
    
    # Calculate overall averages
    avg_prompt_all = overall['sum_prompt_tokens'] / overall['total'] if overall['total'] else 0
    avg_completion_all = overall['sum_completion_tokens'] / overall['total'] if overall['total'] else 0
    avg_reasoning_all = overall['sum_reasoning_tokens'] / overall['total'] if overall['total'] else 0
    avg_total_all = overall['sum_total_tokens'] / overall['total'] if overall['total'] else 0

    writer.writerow({
        'persona': 'OVERALL', 'correct': overall['correct'],
        'total': overall['total'], 'total_ok': overall_total_ok, 'accuracy': round(acc_all, 4),
        'no_extract': overall['no_extract'], 'truncated': overall['truncated'],
        'sum_prompt_tokens': overall['sum_prompt_tokens'],
        'sum_completion_tokens': overall['sum_completion_tokens'],
        'sum_reasoning_tokens': overall['sum_reasoning_tokens'],
        'sum_total_tokens': overall['sum_total_tokens'],
        'avg_prompt_tokens': round(avg_prompt_all, 2),
        'avg_completion_tokens': round(avg_completion_all, 2),
        'avg_reasoning_tokens': round(avg_reasoning_all, 2),
        'avg_total_tokens': round(avg_total_all, 2)
    })
print(f"\nWrote {acc_csv}")

# ── Print per-persona result CSV locations ──────────────────────────────────
for p in PERSONA_ORDER:
    s = persona_stats[p]
    if s['total'] > 0:
        out_path = os.path.join(out_dir, f'results_{p}.csv')
        n_correct = s['correct']
        n_total = s['total']
        print(f"  Wrote {out_path}  ({n_correct}/{n_total} correct)")


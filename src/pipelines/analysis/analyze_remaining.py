"""
Analyze remaining no_number_found and truncated_output cases.
Show full response tail for each case.
"""
import csv, re
from collections import Counter

csv_path = r'e:\Tugas Akhir\TugasAkhir\result\reports\openai_gpt-oss-20b\results.csv'

no_extract = []
truncated  = []

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cat = row['error_category'].strip()
        if cat == 'no_number_found':
            no_extract.append(row)
        elif cat == 'truncated_output':
            truncated.append(row)

print(f"no_number_found : {len(no_extract)}")
print(f"truncated_output: {len(truncated)}")

print("\n" + "="*70)
print("NO_NUMBER_FOUND cases (full tail):")
print("="*70)
for i, r in enumerate(no_extract):
    print(f"\n[{i+1}] persona={r['persona_folder']} | file={r['file'].split(chr(92))[-1]}")
    print(f"     gt={r['ground_truth_answer']!r}")
    print(f"     finish={r['finish_reason']} | trunc_meta={r['truncated_by_metadata']} | trunc_content={r['truncated_by_content']}")
    print(f"     tail: {r['text_tail']!r}")

print("\n" + "="*70)
print("TRUNCATED_OUTPUT cases (full tail + finish_reason):")
print("="*70)
for i, r in enumerate(truncated):
    print(f"\n[{i+1}] persona={r['persona_folder']} | file={r['file'].split(chr(92))[-1]}")
    print(f"     gt={r['ground_truth_answer']!r} | extracted={r['extracted_answer']!r} | correct={r['is_correct']}")
    print(f"     finish={r['finish_reason']} | completion_tokens={r['completion_tokens']}")
    print(f"     tail: {r['text_tail']!r}")

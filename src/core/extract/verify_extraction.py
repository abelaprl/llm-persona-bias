import pandas as pd
import json

# Read CSV to verify extracted values
csv_path = r'E:\Tugas Akhir\TugasAkhir\result/archive/reports_mixed_legacy\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)

# Check the problematic cases mentioned
problem_cases = [
    ('man_explicit_1', 'gsm8k_00094'),
    ('man_explicit_1', 'gsm8k_00199'),
    ('man_explicit_1', 'gsm8k_00254'),
]

for persona, qid in problem_cases:
    row = df[(df['persona_folder'] == persona) & (df['question_id'] == qid)]
    if len(row) > 0:
        row = row.iloc[0]
        # Extract expected number from ground_truth (the #### number)
        gt = str(row['ground_truth'])
        import re
        m = re.search(r'####\s*(-?\d+)', gt)
        expected = m.group(1) if m else 'unknown'
        
        print(f'{qid}: extracted={row["extracted_answer"]} expected={expected}')
        print(f'  extraction_reason: {row["extraction_reason"]}')
        print()

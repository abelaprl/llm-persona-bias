import pandas as pd
import re

csv_path = r'E:\Tugas Akhir\TugasAkhir\result/archive/reports_mixed_legacy\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)

# Problem cases from user data to check extraction method
problem_cases = [
    ('man_explicit_1', 'gsm8k_00094'),
    ('man_explicit_1', 'gsm8k_00199'),
    ('man_explicit_1', 'gsm8k_00254'),
    ('man_explicit_1', 'gsm8k_00510'),
    ('woman_explicit_1', 'gsm8k_00094'),
    ('woman_explicit_1', 'gsm8k_00254'),
]

print('Problem cases analysis:')
for persona, qid in problem_cases:
    rows = df[(df['persona_folder'] == persona) & (df['question_id'] == qid)]
    if len(rows) > 0:
        row = rows.iloc[0]
        # Extract expected from ground_truth
        m = re.search(r'####\s*(-?\d+)', str(row['ground_truth']))
        expected = m.group(1) if m else 'UNKNOWN'
        
        extracted = str(row['extracted_answer'])
        reason = row['extraction_reason']
        is_correct = row['is_correct']
        
        print(f'\n{qid} ({persona}):')
        print(f'  Expected: {expected}, Extracted: {extracted}')
        print(f'  Reason: {reason}, Correct: {is_correct}')

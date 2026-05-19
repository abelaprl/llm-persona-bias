import pandas as pd
import re

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-3-27b-it\results.csv'
df = pd.read_csv(csv_path)

# Extract question ID from file column
df['qid'] = df['file'].str.extract(r'(gsm8k_\d+)')

# Expand problem cases based on user data
problem_qids = ['gsm8k_00094', 'gsm8k_00199', 'gsm8k_00254', 'gsm8k_00510', 'gsm8k_00529', 'gsm8k_00717', 'gsm8k_00754']
problem_personas = ['man_explicit_1', 'woman_explicit_1']

print('Problem cases analyzed:')
for persona in problem_personas[:1]:  # Just first persona for brevity
    for qid in problem_qids:
        rows = df[(df['persona_folder'] == persona) & (df['qid'] == qid)]
        if len(rows) > 0:
            row = rows.iloc[0]
            extracted = row["extracted_answer"]
            correct = row["is_correct"]
            status = "OK" if correct else "WRONG"
            print(f'{qid}: extracted={extracted} [{status}]')


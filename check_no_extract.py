import pandas as pd
import json

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)
no_extract = df[df['error_category'] == 'no_number_found']

print(f'Total no_extract cases: {len(no_extract)}')
print(f'\nFirst 3 cases:\n')

for idx in range(min(3, len(no_extract))):
    row = no_extract.iloc[idx]
    print(f'Case {idx+1}:')
    print(f'  File: {row["file"]}')
    print(f'  Extracted: "{row["extracted_answer"]}"')
    print(f'  Ground truth (first 100 chars): {row["ground_truth"][:100]}')
    print()

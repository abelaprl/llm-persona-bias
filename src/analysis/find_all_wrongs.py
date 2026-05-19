import pandas as pd

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-3-27b-it\results.csv'
df = pd.read_csv(csv_path)

# Extract question ID from file column
df['qid'] = df['file'].str.extract(r'(gsm8k_\d+)')

# Find all WRONG cases
wrong_cases = df[df['is_correct'] == False]

print(f'Total wrong cases: {len(wrong_cases)}')
print(f'\nFirst 20 wrong cases:')
for idx, row in wrong_cases.head(20).iterrows():
    print(f'{row["qid"]} ({row["persona_folder"]}): extracted={row["extracted_answer"]}')

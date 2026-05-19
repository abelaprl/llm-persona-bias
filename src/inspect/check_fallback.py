import pandas as pd

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)

# Check for cases extracted via fallback
fallback = df[df['extraction_reason'] == 'model_refusal_fallback']
print(f'Fallback extractions: {len(fallback)}')

if len(fallback) > 0:
    # Count correct vs incorrect
    correct_fb = len(fallback[fallback['is_correct'] == 'True'])
    print(f'Correct: {correct_fb} / {len(fallback)}')
    print(f'\nFirst 3 fallback cases:')
    for idx, row in fallback.head(3).iterrows():
        print(f'  {row["question_id"]}: extracted={row["extracted_answer"]} expected={row["ground_truth"][:30]}')
else:
    print('No fallback extractions found')
    print(f'  extraction_reason values: {df["extraction_reason"].unique()[:10]}')

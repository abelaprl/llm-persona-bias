import json
import pandas as pd

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)
no_extract = df[df['error_category'] == 'no_number_found']

# Take 5 cases from different personas to inspect
sample_indices = [0, 8, 15, 25, 35]

for idx in sample_indices[:3]:  # Just show first 3
    row = no_extract.iloc[idx]
    file = row['file']
    path = rf'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\{file}'
    
    try:
        with open(path) as f:
            data = json.load(f)
            content = data.get('response', {}).get('content', '')
            
            print(f'\n{"="*80}')
            print(f'Question: {row["question_id"]} | {row["persona_folder"]}')
            print(f'Expected answer: {row["ground_truth"][-20:]}')  # Show last part with #### number
            print(f'Response length: {len(content)} chars')
            print(f'Last 600 chars:')
            print(content[-600:])
    except Exception as e:
        print(f'{file}: ERROR - {e}')

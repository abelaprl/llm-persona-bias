import json
import re
import pandas as pd

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-2-9b-it\results.csv'
df = pd.read_csv(csv_path)
no_extract = df[df['error_category'] == 'no_number_found']

# Inspect first few cases more carefully  
sample_indices = [0, 1, 2, 3, 4]

for idx in sample_indices:
    row = no_extract.iloc[idx]
    file = row['file']
    path = rf'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\{file}'
    
    try:
        with open(path) as f:
            data = json.load(f)
            content = data.get('response', {}).get('content', '')
            
            # Find all numbers in the response
            numbers = re.findall(r'\d+(?:\.\d+)?', content)
            
            # Extract expected answer from ground_truth (the #### number part)
            gt = row['ground_truth']
            gt_match = re.search(r'####\s*([\d.]+)', gt)
            expected = gt_match.group(1) if gt_match else 'unknown'
            
            print(f'\nCase: {row["question_id"]} ({row["persona_folder"]})')
            print(f'  Expected: {expected}')
            print(f'  Numbers found: {str(numbers[:10])}...' if len(numbers) > 10 else f'  Numbers found: {numbers}')
            print(f'  Last part: ...{content[-150:]}')
    except Exception as e:
        print(f'{file}: ERROR - {e}')

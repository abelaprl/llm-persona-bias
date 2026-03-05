import csv
from collections import defaultdict

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-2-9b-it\results.csv'

cases = []
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['error_category'].strip() == 'no_number_found':
            cases.append({
                'persona': row['persona_folder'],
                'question_id': row['question_id'],
                'file': row['file'],
                'extracted': row['extracted_answer'],
                'ground_truth': row['ground_truth']
            })

# Group by persona
by_persona = defaultdict(list)
for c in cases:
    by_persona[c['persona']].append(c)

for persona in sorted(by_persona.keys()):
    print(f'\n{persona}: {len(by_persona[persona])} cases')
    for c in sorted(by_persona[persona], key=lambda x: x['question_id']):
        print(f'  {c["question_id"]}: extracted="{c["extracted"]}" expected="{c["ground_truth"]}"')

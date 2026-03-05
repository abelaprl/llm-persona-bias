import pandas as pd
csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\google_gemma-3-27b-it\results.csv'
df = pd.read_csv(csv_path)

# Check specific cases that should have wrong extractions
problem_qids = ['gsm8k_00094', 'gsm8k_00199', 'gsm8k_00254', 'gsm8k_00510']
problem_personas = ['man_explicit_1', 'woman_explicit_1']

print('Problem cases in google_gemma-3-27b-it:')
for persona in problem_personas:
    for qid in problem_qids:
        rows = df[(df['persona_folder'] == persona) & (df['question_id'] == qid)]
        if len(rows) > 0:
            row = rows.iloc[0]
            print(f'  {qid} ({persona}): extracted={row["extracted_answer"]}, reason={row["extraction_reason"]}')
        else:
            print(f'  {qid} ({persona}): NOT FOUND')

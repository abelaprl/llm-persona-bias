import csv

csv_path = r'E:\Tugas Akhir\TugasAkhir\result\reports\openai_gpt-oss-120b\results.csv'

no_extract = []
with open(csv_path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['error_category'] == 'no_number_found':
            no_extract.append(row)

print(f"Total no_number_found cases: {len(no_extract)}\n")
print("="*100)

for i, r in enumerate(no_extract, 1):
    print(f"\n### Case {i}: {r['question_id']} ({r['persona_folder']})")
    print(f"Ground Truth: {r['ground_truth_answer']}")
    print(f"Extraction reason: {r['extraction_reason']}")
    print(f"\n--- Last 350 characters of response ---")
    print(r['response_content_full'][-350:])
    print("="*100)

import csv

IN_CSV = r'result/reports/openai_gpt-oss-20b/results.csv'
OUT_CSV = r'result/reports/openai_gpt-oss-20b/wrong_answers_only.csv'

with open(IN_CSV, encoding='utf-8') as fin, open(OUT_CSV, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row.get('is_correct', '').strip().lower() == 'false':
            writer.writerow(row)
print(f"Done. Wrong answers written to {OUT_CSV}")

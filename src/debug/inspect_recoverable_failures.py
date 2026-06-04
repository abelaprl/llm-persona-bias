"""Show full response text for all 42 non-empty no_number_found failures."""
import csv, os

csv_path = r'result/reports/GSM8K/_tmp_rebuild_results_csv_ministrail/results.csv'

rows = []
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ec = row.get('error_category', '') or ''
        content = row.get('response_content_full', '') or ''
        if ec.strip() == 'no_number_found' and content.strip():
            rows.append(row)

out_lines = [f'Non-empty no_number_found: {len(rows)}\n']

for i, row in enumerate(rows):
    content = row.get('response_content_full', '') or ''
    gt = row.get('ground_truth_answer', '')
    persona = row.get('persona_folder', '')
    qid = row.get('file', '')
    lines = content.strip().splitlines()
    last5 = '\n'.join(lines[-5:])
    out_lines.append(f'--- #{i+1} | GT={gt} | persona={persona} | {qid}')
    out_lines.append('LAST 5 LINES:')
    out_lines.append(last5)
    out_lines.append('')

with open('result/reports/ministrail_recoverable_42.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print(f'Written {len(rows)} cases to result/reports/ministrail_recoverable_42.txt')

import csv
import random

IN_CSV = r'result/archive/reports_mixed_legacy/openai_gpt-oss-20b/wrong_answers_only.csv'
OUT_SAMPLE = r'result/archive/reports_mixed_legacy/openai_gpt-oss-20b/wrong_answers_sample_50.csv'
OUT_PREVIEW = r'result/archive/reports_mixed_legacy/openai_gpt-oss-20b/wrong_answers_preview.csv'
SAMPLE_N = 50
PREVIEW_LEN = 300

rows = []
with open(IN_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

count = len(rows)
if count == 0:
    print(f"No wrong-answer rows found in {IN_CSV}")
else:
    # Deterministic sample
    random.seed(0)
    sample = rows if count <= SAMPLE_N else random.sample(rows, SAMPLE_N)
    with open(OUT_SAMPLE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        for r in sample:
            writer.writerow(r)
    # Preview file with truncated response preview column
    preview_field = 'response_preview'
    with open(OUT_PREVIEW, 'w', newline='', encoding='utf-8') as f:
        fieldnames = reader.fieldnames + [preview_field]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            full = r.get('response_content_full', '') or ''
            r[preview_field] = (full[:PREVIEW_LEN] + '...') if len(full) > PREVIEW_LEN else full
            writer.writerow(r)
    print(f"Wrote sample {len(sample)} -> {OUT_SAMPLE}")
    print(f"Wrote preview {len(rows)} -> {OUT_PREVIEW}")

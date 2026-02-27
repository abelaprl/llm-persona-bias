import csv
import re

IN_CSV = r'result/reports/openai_gpt-oss-20b/results.csv'
OUT_CSV = r'result/reports/openai_gpt-oss-20b/wrong_answers_smart_extract.csv'

# Helper to extract the best candidate number based on user heuristics
# - Prefer number after last '='
# - If not found, prefer bold (**number**)
# - If not found, prefer number near 'answer' or 'final'
def smart_extract(row):
    text = row['response_content_full']
    # 1. After last '='
    eq_matches = list(re.finditer(r'=\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)', text))
    if eq_matches:
        num = eq_matches[-1].group(1)
        return num.strip(), 'eq_rhs'
    # 2. Bold number
    bold_matches = list(re.finditer(r'\*\*\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)\s*\*\*', text))
    if bold_matches:
        num = bold_matches[-1].group(1)
        return num.strip(), 'bold'
    # 3. Near answer/final
    answer_match = re.search(r'(answer|final)[^\n\r:]*[:\-]?\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)', text, re.I)
    if answer_match:
        num = answer_match.group(2)
        return num.strip(), 'near_answer'
    # 4. Fallback: last number
    num_matches = list(re.finditer(r'([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)', text))
    if num_matches:
        num = num_matches[-1].group(1)
        return num.strip(), 'last_number'
    return '', 'none'

with open(IN_CSV, encoding='utf-8') as fin, open(OUT_CSV, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames + ['smart_extracted_answer', 'smart_extraction_reason']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        if row.get('is_correct', '').strip().lower() == 'false':
            smart_ans, smart_reason = smart_extract(row)
            row['smart_extracted_answer'] = smart_ans
            row['smart_extraction_reason'] = smart_reason
            writer.writerow(row)
print(f"Done. Wrong answers with smart extraction written to {OUT_CSV}")

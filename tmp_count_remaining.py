import csv
import re
from collections import Counter

ids = {
    '00016','00031','00045','00066','00099','00107','00120','00175','00196','00237','00279','00297','00305','00309','00325','00361','00388','00423','00541','00556','00562','00590','00639','00640','00660','00690','00710','00730','00760','00800','00803','00805','00811','00877','00990','01004','01027','01028','01044','01066','01081','01096','01101','01135','01224','01249','01265','01289'
}

p = 'result/reports/google_gemini-2.5-flash_tmp/results.csv'
wrong = []
total = 0
with open(p, encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        m = re.search(r'gsm8k_(\d+)\.json', row['file'])
        if not m or m.group(1) not in ids:
            continue
        total += 1
        if row['is_correct'].strip().lower() != 'true':
            wrong.append((
                m.group(1), row['persona_folder'], row['extracted_answer'],
                row['ground_truth_answer'], row['extraction_reason'],
                row['ground_truth_answer'].strip() in row['response_content_full']
            ))

print('total rows in listed IDs =', total)
print('wrong rows in listed IDs =', len(wrong))
print('gt present among wrong =', sum(1 for w in wrong if w[5]), '/', len(wrong))
print('top qids:', Counter(w[0] for w in wrong).most_common(25))
for w in wrong[:30]:
    print(w)

import csv, os
infile = os.path.join('result','reports','qwen_qwen3-next-80b-a3b-thinking','results.csv')
outfile = 'rerun_plan.csv'
count = 0
with open(infile, encoding='utf-8') as inf, open(outfile, 'w', encoding='utf-8', newline='') as outf:
    reader = csv.DictReader(inf)
    writer = csv.writer(outf)
    writer.writerow(['json_path'])
    for r in reader:
        if r.get('error_category','').strip() in ('truncated_output','no_number_found'):
            f = r['file'].replace('\\','/')
            full = os.path.join('result','qwen_qwen3-next-80b-a3b-thinking', f).replace('\\','/')
            writer.writerow([full])
            count += 1
print('WROTE', outfile, 'with', count, 'rows')

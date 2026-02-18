import json
from pathlib import Path
import csv

MODEL = 'qwen_qwen3-next-80b-a3b-thinking'
RESULTS_ROOT = Path('result') / MODEL
OUTPUT_CSV = Path('result/reports') / f'scan_{MODEL}_empty_trunc_correctpath.csv'

if not RESULTS_ROOT.exists():
    print('Model results folder not found:', RESULTS_ROOT)
    raise SystemExit(1)

persona_folders = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
summary = []

overall = {'total':0,'empty_content':0,'success_false':0,'truncated_length':0}

for persona in sorted(persona_folders):
    files = list(persona.glob('*.json'))
    total = len(files)
    empty = 0
    success_false = 0
    truncated = 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            # skip invalid
            continue
        response = data.get('response', {})
        # success flag
        success = response.get('success')
        if success in (False, 'False', 'false'):
            success_false += 1
        # content
        content = response.get('content')
        if content is None or (isinstance(content, str) and content.strip()==''):
            empty += 1
        # check raw_api_response finish_reason
        raw = response.get('raw_api_response')
        fin_len = False
        if isinstance(raw, dict):
            choices = raw.get('choices') or []
            if choices and isinstance(choices, list):
                ch = choices[0]
                fr = ch.get('finish_reason')
                native = ch.get('native_finish_reason')
                if fr == 'length' or native == 'length':
                    fin_len = True
        if fin_len:
            truncated += 1

    summary.append({'persona': persona.name, 'total': total, 'empty_content': empty, 'success_false': success_false, 'truncated_length': truncated})
    overall['total'] += total
    overall['empty_content'] += empty
    overall['success_false'] += success_false
    overall['truncated_length'] += truncated

# write CSV
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['persona','total_files','empty_content','success_false','truncated_length'])
    for r in summary:
        writer.writerow([r['persona'], r['total'], r['empty_content'], r['success_false'], r['truncated_length']])
    writer.writerow(['TOTAL', overall['total'], overall['empty_content'], overall['success_false'], overall['truncated_length']])

# print concise summary
print('Scan complete. Summary:')
for r in summary:
    print(f"- {r['persona']}: total={r['total']}, empty={r['empty_content']}, success_false={r['success_false']}, truncated={r['truncated_length']}")
print(f"TOTAL: total={overall['total']}, empty={overall['empty_content']}, success_false={overall['success_false']}, truncated={overall['truncated_length']}")
print('\nWrote CSV to', OUTPUT_CSV)

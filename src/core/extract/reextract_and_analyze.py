import csv
import json
from pathlib import Path
import sys

# add src to path to import extractor
sys.path.insert(0, str(Path('src').resolve()))
from generate_analysis_csv import extract_answer_from_response

# Config
reports_csv = Path('result/reports/qwen_qwen3-next-80b-a3b-thinking/man_explicit_1/results.csv')
results_root = Path('results [january]/qwen_qwen3-next-80b-a3b-thinking')
output_count = 100

def normalize_persona(csv_persona):
    return csv_persona.rsplit('_', 1)[0]

if not reports_csv.exists():
    print('Reports CSV not found:', reports_csv)
    raise SystemExit(1)

rows = []
with reports_csv.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        is_correct = row.get('is_correct')
        if is_correct in ('True', 'true', '1', '1.0'):
            correct = True
        else:
            correct = False
        if not correct:
            rows.append(row)
        # also include numeric mismatch
        else:
            # if both present but differ
            ext = row.get('extracted_answer')
            gt = row.get('ground_truth_answer')
            try:
                if ext not in ('', None) and gt not in ('', None):
                    if abs(float(ext) - float(gt)) > 0.01:
                        rows.append(row)
            except:
                pass
        if len(rows) >= output_count:
            break

print(f'Will analyze {len(rows)} rows')

results = []
for i, row in enumerate(rows, start=1):
    qid = int(row['question_id'])
    csv_persona = row['persona_folder']
    persona_folder = normalize_persona(csv_persona)
    json_name = f'gsm8k_{qid:05d}.json'
    json_path = results_root / persona_folder / json_name

    item = {
        'index': i,
        'question_id': qid,
        'csv_extracted': row.get('extracted_answer'),
        'ground_truth': row.get('ground_truth_answer'),
        'json_path': str(json_path),
        'status': None,
        'reextracted': None,
        'note': None
    }

    if not json_path.exists():
        item['status'] = 'json_missing'
        results.append(item)
        continue

    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
    except Exception as e:
        item['status'] = 'json_invalid'
        item['note'] = str(e)
        results.append(item)
        continue

    # get response content
    response = data.get('response', {})
    content = response.get('content')
    if content is None or (isinstance(content, str) and content.strip()==''):
        item['status'] = 'empty_content'
        results.append(item)
        continue

    # re-extract
    try:
        parsed = extract_answer_from_response(content)
    except Exception as e:
        item['status'] = 'extractor_error'
        item['note'] = str(e)
        results.append(item)
        continue

    item['reextracted'] = parsed

    # compare to ground truth
    try:
        gt_val = float(row.get('ground_truth_answer')) if row.get('ground_truth_answer') not in (None, '', 'nan') else None
    except:
        gt_val = None

    if parsed is None:
        item['status'] = 'parse_fail'
    else:
        if gt_val is None:
            item['status'] = 'no_ground_truth'
        else:
            try:
                if abs(parsed - gt_val) < 0.01:
                    item['status'] = 'parsed_match'
                else:
                    item['status'] = 'parsed_mismatch'
            except:
                item['status'] = 'parsed_mismatch'

    results.append(item)

# summarize
from collections import Counter
counts = Counter([r['status'] for r in results])
print('\nSummary counts:')
for k,v in counts.items():
    print(f'  {k}: {v}')

print('\nSamples:')
for r in results[:50]:
    print('---')
    print(f"[{r['index']}] QID={r['question_id']} status={r['status']}")
    print(f"  csv_extracted={r['csv_extracted']} ground_truth={r['ground_truth']} reextracted={r['reextracted']}")
    print(f"  json={r['json_path']}")

# write detailed CSV
out = Path('scripts/reextract_analysis_output.csv')
with out.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['index','question_id','status','csv_extracted','reextracted','ground_truth','json_path','note'])
    for r in results:
        writer.writerow([r['index'], r['question_id'], r['status'], r['csv_extracted'], r['reextracted'], r['ground_truth'], r['json_path'], r.get('note')])

print('\nWrote detailed CSV to', out)
print('DONE')

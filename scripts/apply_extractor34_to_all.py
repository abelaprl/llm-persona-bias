import os
import json
import csv
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('extractor34', str(Path(__file__).parent.parent / 'extractor34.py'))
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

root = Path('result') / 'openai_gpt-oss-120b'
report = []
total = 0
fixed = 0
still_none = 0

def parse_gt(gt_text):
    if not gt_text:
        return None
    import re
    m = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', gt_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None

for persona_dir in sorted(root.iterdir()):
    if not persona_dir.is_dir():
        continue
    for jf in sorted(persona_dir.glob('gsm8k_*.json')):
        total += 1
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            report.append({'file': str(jf), 'error': f'read_error: {e}'})
            continue

        content = data.get('response', {}).get('content')
        # try reasoning field if content empty
        if not content:
            raw = data.get('response', {}).get('raw_api_response', {})
            choices = raw.get('choices', []) if raw else []
            if choices:
                msg = choices[0].get('message', {})
                content = msg.get('reasoning', '') or msg.get('content', '')
        gt_text = data.get('input', {}).get('ground_truth', '')
        gt = parse_gt(gt_text)

        ex = extractor.extract_final_answer(content)
        is_correct = None
        if ex is None:
            still_none += 1
            is_correct = None
        else:
            if gt is None:
                is_correct = None
            else:
                is_correct = (ex == gt)
                if is_correct:
                    fixed += 1

        # update analysis
        data.setdefault('analysis', {})
        data['analysis']['extracted_answer'] = ex
        data['analysis']['is_correct'] = is_correct

        # write back
        try:
            with open(jf, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            report.append({'file': str(jf), 'error': f'write_error: {e}'})
            continue

        report.append({'file': str(jf), 'gt': gt, 'extracted': ex, 'is_correct': is_correct})

# write summary JSON and CSV
out_json = Path('apply_extractor34_report.json')
with open(out_json, 'w', encoding='utf-8') as of:
    json.dump({'total': total, 'fixed_count': fixed, 'still_none': still_none, 'rows': report}, of, ensure_ascii=False, indent=2)

out_csv = Path('apply_extractor34_report.csv')
with open(out_csv, 'w', encoding='utf-8', newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['file','gt','extracted','is_correct'])
    for r in report:
        w.writerow([r.get('file'), r.get('gt'), r.get('extracted'), r.get('is_correct')])

print('Done. Total files:', total)
print('Fixed count (extracted present):', fixed)
print('Still None extracted:', still_none)
print('Wrote', out_json, 'and', out_csv)

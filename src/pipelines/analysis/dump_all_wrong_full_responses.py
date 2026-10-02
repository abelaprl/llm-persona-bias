#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
result_dir = root / 'result'
out_file = root / '.codex_tmp' / 'all_mismatches_full_responses.json'

mismatches = []
count = 0
for p in result_dir.rglob('*.json'):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    analysis = data.get('analysis', {})
    if not analysis:
        continue
    is_correct = analysis.get('is_correct')
    if is_correct is False:
        entry = {
            'file': str(p).replace('\\', '/').split(str(root) + '/')[-1],
            'gt': None,
            'extracted': analysis.get('extracted_answer'),
            'is_correct': False,
            'response_content': None,
        }
        # ground truth may be in input.ground_truth
        gt = data.get('input', {}).get('ground_truth')
        if isinstance(gt, (int, float)):
            entry['gt'] = float(gt)
        else:
            # try to parse numeric from string if present at end like "#### 123"
            if isinstance(gt, str):
                # look for last number token
                import re
                m = re.search(r"(\d[\d,\. ]*)\s*$", gt)
                if m:
                    s = m.group(1).replace(',', '')
                    try:
                        entry['gt'] = float(s)
                    except Exception:
                        entry['gt'] = None
        # response content
        resp = data.get('response', {}).get('content')
        if resp is None:
            # try raw_api_response -> choices[0].message.content
            raw = data.get('response', {}).get('raw_api_response', {})
            try:
                choices = raw.get('choices') or []
                if choices and isinstance(choices, list):
                    msg = choices[0].get('message', {})
                    resp = msg.get('content')
            except Exception:
                resp = None
        entry['response_content'] = resp
        mismatches.append(entry)
        count += 1

out = {'count': count, 'mismatches': mismatches}
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {out_file} with {count} mismatches')

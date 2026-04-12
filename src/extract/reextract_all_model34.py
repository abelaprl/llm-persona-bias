import json
import os
import pathlib
import importlib.util

# load extractor module
spec = importlib.util.spec_from_file_location('extractor34', str(pathlib.Path(__file__).parent.parent / 'extractor34.py'))
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

root = os.path.join('result', 'openai_gpt-oss-120b')
personas = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

stats = {}
all_issues = []

for persona in personas:
    ppath = os.path.join(root, persona)
    files = sorted([f for f in os.listdir(ppath) if f.endswith('.json')])
    s = {'total': 0, 'correct': 0, 'wrong': 0, 'no_extract': 0, 'truncated': 0, 'length_limit': 0}
    for fname in files:
        fpath = os.path.join(ppath, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception:
            continue
        s['total'] += 1
        gt_text = d.get('input', {}).get('ground_truth', '')
        gt = None
        m = None
        if gt_text:
            import re
            m = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', gt_text)
            if m:
                try:
                    gt = float(m.group(1).replace(',', ''))
                except:
                    gt = None

        content = d.get('response', {}).get('content', '')
        raw = d.get('response', {}).get('raw_api_response', {})
        choices = raw.get('choices', []) if raw else []
        fr = choices[0].get('finish_reason','') if choices else ''
        if fr == 'length':
            s['length_limit'] += 1

        ex = extractor.extract_final_answer(content)
        if ex is None:
            s['no_extract'] += 1
            all_issues.append({'persona': persona, 'file': fname, 'issue': 'NO_EXTRACT', 'gt': gt, 'extracted': None, 'detail': content[-200:]})
        elif gt is not None and ex == gt:
            s['correct'] += 1
        else:
            s['wrong'] += 1
            if len([i for i in all_issues if i.get('issue') == 'WRONG']) < 200:
                all_issues.append({'persona': persona, 'file': fname, 'issue': 'WRONG', 'gt': gt, 'extracted': ex, 'detail': content[-200:]})
    stats[persona] = s

total = {'total': 0, 'correct': 0, 'wrong': 0, 'no_extract': 0, 'length_limit': 0}
for p,s in stats.items():
    for k in total:
        if k in s:
            total[k] += s[k]

report = {'stats': stats, 'total': total, 'issues_sample': all_issues}
with open('reextract34_report.json', 'w', encoding='utf-8') as of:
    json.dump(report, of, ensure_ascii=False, indent=2)

with open('reextract34_mismatches_top200.txt', 'w', encoding='utf-8') as mf:
    for it in all_issues[:200]:
        mf.write(f"{it['persona']}/{it['file']}\tGT:{it['gt']}\tEX:{it['extracted']}\n")

print('Wrote reextract34_report.json and reextract34_mismatches_top200.txt')

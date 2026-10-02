import json
from pathlib import Path
from collections import defaultdict

REPORT = Path('apply_extractor34_report.json')
OUT = Path('.codex_tmp') / 'analyze_apply_report.json'

with open(REPORT, 'r', encoding='utf-8') as f:
    report = json.load(f)

rows = report.get('rows', []) if isinstance(report, dict) else report

cats = defaultdict(list)
counts = defaultdict(int)

for r in rows:
    gt = r.get('gt')
    ex = r.get('extracted')
    if ex is None:
        cats['no_extract'].append(r)
        counts['no_extract'] += 1
        continue
    if gt is None:
        cats['no_gt'].append(r)
        counts['no_gt'] += 1
        continue
    if r.get('is_correct'):
        counts['correct'] += 1
        continue
    # now wrong
    counts['wrong'] += 1
    # numeric analysis
    try:
        ex_f = float(ex)
        gt_f = float(gt)
    except Exception:
        cats['parse_error'].append(r)
        counts['parse_error'] += 1
        continue
    # avoid divide by zero
    if ex_f == 0:
        cats['zero_extracted'].append(r)
        counts['zero_extracted'] += 1
        continue
    ratio = gt_f / ex_f if ex_f != 0 else None
    ar = abs(ratio) if ratio is not None else None
    # scale categories
    if ar is not None and ar >= 900 and ar <= 1100:
        cats['scale_1000'].append(r)
        counts['scale_1000'] += 1
    elif ar is not None and ar >= 90 and ar <= 110:
        cats['scale_100'].append(r)
        counts['scale_100'] += 1
    elif ar is not None and ar >= 9 and ar <= 11:
        cats['scale_10'].append(r)
        counts['scale_10'] += 1
    elif ex_f < 1 and gt_f >= 1:
        cats['tiny_fraction'].append(r)
        counts['tiny_fraction'] += 1
    elif abs(ex_f - gt_f) / max(1.0, abs(gt_f)) < 0.02:
        cats['near_match_rounding'].append(r)
        counts['near_match_rounding'] += 1
    else:
        # check if ex equals prefix digits of gt (truncation)
        s_gt = str(int(gt_f)) if abs(gt_f - int(gt_f)) < 1e-9 else str(gt_f)
        s_ex = str(int(ex_f)) if abs(ex_f - int(ex_f)) < 1e-9 else str(ex_f)
        if s_gt.startswith(s_ex):
            cats['prefix_trunc'].append(r)
            counts['prefix_trunc'] += 1
        else:
            cats['other'].append(r)
            counts['other'] += 1

summary = {'total_rows': len(rows), 'counts': dict(counts)}
# keep top 10 examples per category
examples = {k: v[:10] for k, v in cats.items()}

out = {'summary': summary, 'examples': examples}
with open(OUT, 'w', encoding='utf-8') as of:
    json.dump(out, of, ensure_ascii=False, indent=2)

print('Wrote', OUT)
print('Summary:', summary)

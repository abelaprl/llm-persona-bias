import json
import re
from collections import defaultdict
from pathlib import Path

def has_boxed(text):
    return bool(re.search(r'\\boxed\{|\\\[|\\\]', text))

def has_currency(text):
    return bool(re.search(r'[\$€£]|dollar|dollars', text, re.IGNORECASE))

def has_thousands_grouping(text):
    return bool(re.search(r'\d{1,3}(?:[,\u2009\u202f]\d{3})+', text))

def last_char_truncated(text):
    if not text: return True
    t = text.strip()
    return t[-1] not in '.!?)]}\"\'0123456789'

def classify_row(row):
    tags = []
    if row.get('extracted') is None:
        tags.append('no_extract')
    else:
        gt = row.get('gt')
        ex = row.get('extracted')
        if isinstance(gt, (int,float)) and isinstance(ex, (int,float)) and gt != 0:
            ratio = ex/gt
            if ratio < 0.01:
                tags.append('tiny_fraction')
            elif ratio < 0.1:
                tags.append('small_fraction')
            elif ratio > 10:
                tags.append('much_larger')
            else:
                tags.append('other_ratio')
            # detect scale factors roughly
            for f in [10,100,1000,10000]:
                try:
                    if abs((gt/ex) - f) < 0.01:
                        tags.append(f'scale_{f}')
                except Exception:
                    pass
    return tags

def main():
    rep = json.load(open('apply_extractor34_report.json', 'r', encoding='utf-8'))
    rows = rep.get('rows', [])
    wrongs = [r for r in rows if r.get('is_correct') is False or (r.get('is_correct') is None and r.get('gt') is not None)]

    counts = defaultdict(int)
    examples = defaultdict(list)

    for r in wrongs:
        f = r.get('file')
        path = Path(f)
        content = ''
        finish_reason = ''
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            content = d.get('response', {}).get('content', '') or ''
            raw = d.get('response', {}).get('raw_api_response', {})
            choices = raw.get('choices', []) if raw else []
            if choices:
                finish_reason = choices[0].get('finish_reason','')
        except Exception:
            content = r.get('file') + ' (read error)'

        tags = classify_row(r)
        if has_boxed(content):
            tags.append('boxed')
        if has_currency(content):
            tags.append('currency')
        if has_thousands_grouping(content):
            tags.append('thousands_grouping')
        if finish_reason == 'length' or last_char_truncated(content):
            tags.append('truncated')

        if not tags:
            tags = ['other']

        for t in set(tags):
            counts[t] += 1
            if len(examples[t]) < 5:
                examples[t].append({'file': r.get('file'), 'gt': r.get('gt'), 'ex': r.get('extracted'), 'snippet': content[:200]})

    out = {'total_wrongs': len(wrongs), 'counts': dict(counts), 'examples': examples}
    with open('remaining_wrong_analysis34.json', 'w', encoding='utf-8') as of:
        json.dump(out, of, ensure_ascii=False, indent=2)

    print('Wrote remaining_wrong_analysis34.json')
    for k,v in sorted(counts.items(), key=lambda x:-x[1]):
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()

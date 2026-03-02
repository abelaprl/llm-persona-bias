import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'all_mismatches_full_responses.json'
OUT = ROOT / 'scripts' / 'mismatch_examples_full_responses.json'

def to_review_entry(item):
    f = item.get('file', '')
    # make path relative starting from 'result' if present
    idx = f.lower().find('result')
    if idx != -1:
        rel = f[idx:]
    else:
        rel = f
    rel = rel.replace('/', '\\').replace('\\\\', '\\')

    return {
        'file': rel,
        'gt': item.get('gt'),
        'extracted': item.get('extracted'),
        'is_correct': item.get('is_correct', False),
        'response_content': item.get('response_content', '')
    }

def main():
    with open(SRC, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mismatches = data.get('mismatches', [])
    review = [to_review_entry(m) for m in mismatches]

    out = {'review_examples': review}
    with open(OUT, 'w', encoding='utf-8') as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)

    print(f'Wrote {OUT} with {len(review)} examples')

if __name__ == '__main__':
    main()

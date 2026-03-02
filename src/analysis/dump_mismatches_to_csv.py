import json
import csv
import re
import os

INFILE = os.path.join('scripts', 'all_mismatches_full_responses.json')
OUTFILE = os.path.join('scripts', 'all_mismatches_compact.csv')

def extract_persona(path):
    if not path:
        return ''
    p = path.replace('\\', '/')
    m = re.search(r'/result/[^/]+/([^/]+)/', p)
    if m:
        return m.group(1)
    # fallback: try to take the directory just before the filename
    parts = p.split('/')
    if len(parts) >= 2:
        return parts[-2]
    return ''

def snippet(text, length=200):
    if not text:
        return ''
    s = ' '.join(text.split())
    return s[:length]

def main():
    with open(INFILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mismatches = data.get('mismatches', [])

    with open(OUTFILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'persona', 'gt', 'extracted', 'is_correct', 'snippet'])
        for m in mismatches:
            fp = m.get('file', '')
            persona = extract_persona(fp)
            gt = m.get('gt', '')
            ext = m.get('extracted', '')
            ok = m.get('is_correct', False)
            sn = snippet(m.get('response_content', ''))
            writer.writerow([fp, persona, gt, ext, ok, sn])

    print(f'Wrote {OUTFILE} with {len(mismatches)} rows')

if __name__ == '__main__':
    main()

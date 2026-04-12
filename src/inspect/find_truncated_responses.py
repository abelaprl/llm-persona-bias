import json
import re
import os

INFILE = os.path.join('scripts', 'all_mismatches_full_responses.json')
OUTFILE = os.path.join('scripts', 'truncated_mismatches.json')

def looks_truncated(s: str) -> bool:
    if not s:
        return True
    t = s.strip()
    # Common signs of truncation
    if t.endswith('...'):
        return True
    if re.search(r"\\boxed\{[^{]*$", t):
        return True
    if t.endswith('=') or t.endswith('+') or t.endswith('-') or t.endswith('*') or t.endswith('/'):
        return True
    # Unclosed LaTeX environments
    if t.count('\\[') != t.count('\\]'):
        return True
    if t.count('\\(') != t.count('\\)'):
        return True
    # Unbalanced braces (simple heuristic)
    if t.count('{') > t.count('}'):
        return True
    # If there is no numeric token near the end (last 50 chars)
    tail = t[-200:]
    if not re.search(r"\d", tail):
        return True
    # otherwise consider not truncated
    return False

def snippet(s, n=200):
    if not s:
        return ''
    return ' '.join(s.split())[:n]

def main():
    with open(INFILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mismatches = data.get('mismatches', [])
    truncated = []
    for m in mismatches:
        rc = m.get('response_content', '')
        if looks_truncated(rc):
            truncated.append({
                'file': m.get('file'),
                'gt': m.get('gt'),
                'extracted': m.get('extracted'),
                'snippet': snippet(rc, 400),
            })
    with open(OUTFILE, 'w', encoding='utf-8') as f:
        json.dump({'count': len(truncated), 'truncated': truncated}, f, ensure_ascii=False, indent=2)
    print(f'Found {len(truncated)} likely-truncated responses; wrote {OUTFILE}')

if __name__ == '__main__':
    main()

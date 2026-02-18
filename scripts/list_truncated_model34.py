import json, os, re

MODEL_DIR = 'result/openai_gpt-oss-120b'
OUT = 'truncated_examples_model34.txt'

def is_truncated(text):
    if not text:
        return True, "EMPTY"
    text = text.strip()
    if not text:
        return True, "EMPTY"
    stripped = re.sub(r'[\*`\$\s]+$', '', text)
    stripped = re.sub(r'\\\]|\\\)$', '', stripped)
    if not stripped:
        return True, "EMPTY_AFTER_STRIP"
    last_char = stripped[-1]
    ok_endings = set('.!?)\]"\'0123456789%')
    if last_char in ok_endings:
        return False, "OK"
    if re.search(r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*\*\*?\s*$', text):
        return False, "OK_NUM_WITH_MARKUP"
    return True, f"ENDS_WITH: '{text[-30:]}'"


def main():
    truncated = []
    for persona in sorted(os.listdir(MODEL_DIR)):
        pdir = os.path.join(MODEL_DIR, persona)
        if not os.path.isdir(pdir):
            continue
        for fname in sorted([f for f in os.listdir(pdir) if f.endswith('.json')]):
            fpath = os.path.join(pdir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                continue
            content = data.get('response', {}).get('content', '')
            is_tr, info = is_truncated(content)
            if is_tr:
                gt = data.get('input', {}).get('ground_truth', '').strip()
                truncated.append((persona, fname, info, content[:300].replace('\n',' '), gt))
    with open(OUT, 'w', encoding='utf-8') as out:
        for p,f,info,sn,gt in truncated:
            out.write(f"{p}/{f} | {info} | GT: {gt} | Snippet: {sn}\n")
    print(f'Wrote {len(truncated)} truncated entries to {OUT}')

if __name__ == '__main__':
    main()

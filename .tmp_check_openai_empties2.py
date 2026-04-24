import json, re, math
from pathlib import Path

BASE = Path('result/ETHICS')
MODELS = ['openai_gpt-oss-20b','openai_gpt-oss-120b']
EMPTY_TOKENS = {'null','none','nan','na','n/a','<>','<&gt;',"''"}

def is_empty_like(v):
    if v is None:
        return True
    if isinstance(v, float):
        return math.isnan(v)
    s = str(v).strip()
    return s == '' or s.lower() in EMPTY_TOKENS

def get_content(d):
    r = d.get('response')
    if not isinstance(r, dict):
        return ''
    c = r.get('content')
    if isinstance(c, str) and c.strip():
        return c
    ra = r.get('raw_api_response')
    if isinstance(ra, dict):
        ch = ra.get('choices')
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            m = ch[0].get('message')
            if isinstance(m, dict) and isinstance(m.get('content'), str):
                return m['content']
        for k in ('content','text','response'):
            v = ra.get(k)
            if isinstance(v, str):
                return v
    elif isinstance(ra, str):
        return ra
    return ''

def short(t, n=300):
    t = (t or '').replace('\r',' ').replace('\n',' ')
    return re.sub(r'\s+', ' ', t).strip()[:n]

for model in MODELS:
    model_dir = BASE / model
    files = [p for p in model_dir.rglob('*.json') if p.parent != model_dir]
    case_a = []
    case_b = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        pred = d.get('prediction') if isinstance(d.get('prediction'), dict) else {}
        parsed = pred.get('parsed_answer')
        excuse = pred.get('excuse')
        content = get_content(d)
        if is_empty_like(parsed) and isinstance(content, str) and content.strip():
            if is_empty_like(excuse):
                case_a.append((p, content))
            else:
                case_b.append((p, content, excuse))

    print(f'=== {model} ===')
    print(f'caseA_count: {len(case_a)}')
    for p, c in case_a[:15]:
        print(f'A_SAMPLE {p.as_posix()} || {short(c, 300)}')
    print(f'caseB_count: {len(case_b)}')
    for p, c, e in case_b[:10]:
        print(f'B_SAMPLE {p.as_posix()} || excuse={short(str(e),120)} || {short(c, 300)}')

    unresolved = [c for _, c in case_a] + [c for _, c, _ in case_b]
    pats = {
        'final_answer': r'\bfinal\s+answer\b',
        'answer_colon': r'\banswer\s*[:\-]',
        'label_colon': r'\blabel\s*[:\-]',
        'prediction_colon': r'\bprediction\s*[:\-]',
        'the_answer_is': r'\bthe\s+answer\s+is\b',
        'classification_colon': r'\bclassification\s*[:\-]',
        'result_colon': r'\bresult\s*[:\-]',
        'boxed_digit': r'\\boxed\s*\{\s*[01]\s*\}',
        'bare_0_or_1': r'(?<!\d)[01](?!\d)',
        'acceptable_words': r'\b(morally\s+)?(acceptable|unacceptable|wrong|right)\b'
    }
    print('pattern_counts:')
    for name, pat in pats.items():
        rgx = re.compile(pat, re.I)
        print(f'  {name}: {sum(1 for c in unresolved if rgx.search(c or ""))}')

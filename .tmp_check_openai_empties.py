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
    if s == '':
        return True
    if s.lower() in EMPTY_TOKENS:
        return True
    return False

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
        if isinstance(ch, list) and ch:
            msg = ch[0].get('message') if isinstance(ch[0], dict) else None
            if isinstance(msg, dict):
                mc = msg.get('content')
                if isinstance(mc, str):
                    return mc
        # fallback keys
        for k in ('content','text','response'):
            v = ra.get(k)
            if isinstance(v, str):
                return v
    elif isinstance(ra, str):
        return ra
    return ''

def short(t, n=300):
    t = (t or '').replace('\r',' ').replace('\n',' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:n]

for model in MODELS:
    model_dir = BASE / model
    files = [p for p in model_dir.rglob('*.json') if p.parent != model_dir]

    case_a = []  # parsed empty, excuse empty, content non-empty
    case_b = []  # parsed empty, excuse non-empty, content non-empty

    for p in files:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        pred = d.get('prediction') if isinstance(d.get('prediction'), dict) else {}
        parsed = pred.get('parsed_answer')
        excuse = pred.get('excuse')
        content = get_content(d)
        has_content = isinstance(content, str) and content.strip() != ''
        if is_empty_like(parsed) and has_content:
            if is_empty_like(excuse):
                case_a.append((p, content))
            else:
                case_b.append((p, content, excuse))

    print(f'=== {model} ===')
    print(f'caseA_count(parsed empty + excuse empty + content non-empty): {len(case_a)}')
    for p,c in case_a[:15]:
        print(f'A_SAMPLE {p.as_posix()} || {short(c)}')

    print(f'caseB_count(parsed empty + excuse non-empty + content non-empty): {len(case_b)}')
    for p,c,e in case_b[:10]:
        es = short(str(e),120)
        print(f'B_SAMPLE {p.as_posix()} || excuse={es} || {short(c)}')

    # pattern hints from combined unresolved rows
    unresolved = [c for _,c in case_a] + [c for _,c,_ in case_b]
    pats = {
        'has_final_answer': re.compile(r'\bfinal\s+answer\b', re.I),
        'has_answer_colon': re.compile(r'\banswer\s*[:\-]', re.I),
        'has_label_colon': re.compile(r'\blabel\s*[:\-]', re.I),
        'has_prediction_colon': re.compile(r'\bprediction\s*[:\-]', re.I),
        'has_the_answer_is': re.compile(r'\bthe\s+answer\s+is\b', re.I),
        'has_classification_colon': re.compile(r'\bclassification\s*[:\-]', re.I),
        'has_result_colon': re.compile(r'\bresult\s*[:\-]', re.I),
        'has_boxed_digit': re.compile(r'\\boxed\s*\{\s*[01]\s*\}'),
        'contains_digit_0_or_1': re.compile(r'(?<!\d)[01](?!\d)'),
        'contains_acceptable_words': re.compile(r'\b(morally\s+)?(acceptable|unacceptable|wrong|right)\b', re.I),
    }
    print('pattern_counts:')
    for name, rgx in pats.items():
        cnt = sum(1 for c in unresolved if rgx.search(c or ''))
        print(f'  {name}: {cnt}')


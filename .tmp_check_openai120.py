import json, re, math
from pathlib import Path

model='openai_gpt-oss-120b'
base=Path('result/ETHICS')/model
EMPTY={'null','none','nan','na','n/a','<>','<&gt;',"''"}

def em(v):
    if v is None:
        return True
    if isinstance(v,float):
        return math.isnan(v)
    s=str(v).strip()
    return s=='' or s.lower() in EMPTY

def content(d):
    r=d.get('response')
    if not isinstance(r,dict):
        return ''
    c=r.get('content')
    if isinstance(c,str) and c.strip():
        return c
    ra=r.get('raw_api_response')
    if isinstance(ra,dict):
        ch=ra.get('choices')
        if isinstance(ch,list) and ch and isinstance(ch[0],dict):
            m=ch[0].get('message')
            if isinstance(m,dict) and isinstance(m.get('content'),str):
                return m.get('content')
        for k in ('content','text','response'):
            v=ra.get(k)
            if isinstance(v,str):
                return v
    elif isinstance(ra,str):
        return ra
    return ''

def short(t,n=300):
    t=(t or '').replace('\r',' ').replace('\n',' ')
    return re.sub(r'\s+',' ',t).strip()[:n]

files=[p for p in base.rglob('*.json') if p.parent!=base]
A=[]
B=[]
for p in files:
    try:
        d=json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    pred=d.get('prediction') if isinstance(d.get('prediction'),dict) else {}
    pa=pred.get('parsed_answer')
    ex=pred.get('excuse')
    c=content(d)
    if em(pa) and isinstance(c,str) and c.strip():
        if em(ex):
            A.append((p,c))
        else:
            B.append((p,c,ex))

print(f'=== {model} ===')
print(f'caseA_count(parsed empty + excuse empty + content non-empty): {len(A)}')
for p,c in A[:15]:
    print(f'A_SAMPLE {p.as_posix()} || {short(c)}')

print(f'caseB_count(parsed empty + excuse non-empty + content non-empty): {len(B)}')
for p,c,e in B[:10]:
    print(f'B_SAMPLE {p.as_posix()} || excuse={short(str(e),120)} || {short(c)}')

un=[c for _,c in A]+[c for _,c,_ in B]
pats={
 'has_final_answer':r'\bfinal\s+answer\b',
 'has_answer_colon':r'\banswer\s*[:\-]',
 'has_label_colon':r'\blabel\s*[:\-]',
 'has_prediction_colon':r'\bprediction\s*[:\-]',
 'has_the_answer_is':r'\bthe\s+answer\s+is\b',
 'has_classification_colon':r'\bclassification\s*[:\-]',
 'has_result_colon':r'\bresult\s*[:\-]',
 'has_boxed_digit':r'\\boxed\s*\{\s*[01]\s*\}',
 'contains_digit_0_or_1':r'(?<!\d)[01](?!\d)',
 'contains_acceptable_words':r'\b(morally\s+)?(acceptable|unacceptable|wrong|right)\b'
}
print('pattern_counts:')
for k,pat in pats.items():
    rg=re.compile(pat,re.I)
    print(f'  {k}: {sum(1 for c in un if rg.search(c or ""))}')

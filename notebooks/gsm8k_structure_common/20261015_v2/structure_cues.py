"""Whitespace-invariant literal structure cues; never reconstruct lost layout."""
import re
IDS=('step_label','numbered_sequence','labeled_bullet','hash_heading')
LABELS={'step_label':'Label Step bernomor','numbered_sequence':'Butir bernomor berurutan',
        'labeled_bullet':'Bullet dengan label tebal','hash_heading':'Subjudul bertanda #'}
STEP=re.compile(r'\bStep\s+[1-9]\d?(?:\*\*)?\s*[:\u2013\u2014-]\s*(?:\*\*)?[A-Za-z]')
STEP_REFERENCE=re.compile(r'\b(?:from|in|of|at|to|using|see|recall|per|previous|equation|result|by)\s+(?:the\s+)?$',re.I)
NUMBER=re.compile(r'(?<!\S)(?:\*\*)?([1-9]\d?)[.)](?:\*\*)?\s+(?=\S)')
BULLET=re.compile(r'(?<!\S)[*\-\u2022]\s+\*\*[A-Za-z][^*]{0,100}?(?::\*\*|\*\*\s*:)')
HEADING=re.compile(r'(?<!\S)#{1,6}\s+(?:\*\*)?[A-Za-z]')

def flatten(text):return ' '.join(text.split())

def detect(text):
    text=flatten(text)
    hits=[]
    for marker,pattern in [('step_label',STEP),('labeled_bullet',BULLET),('hash_heading',HEADING)]:
        for m in pattern.finditer(text):
            if marker=='step_label' and STEP_REFERENCE.search(text[max(0,m.start()-50):m.start()]):
                continue
            hits.append(dict(marker_id=marker,start=m.start(),end=m.end(),matched_text=m.group()))
    numbered=list(NUMBER.finditer(text));chain=[]
    def finish():
        if len(chain)>=2:
            for m in chain:
                hits.append(dict(marker_id='numbered_sequence',start=m.start(),end=m.end(),matched_text=m.group()))
    for m in numbered:
        n=int(m.group(1))
        if n==1:
            finish();chain=[m]
        elif chain and n==int(chain[-1].group(1))+1:
            chain.append(m)
        else:
            finish();chain=[]
    finish()
    return hits

def counts(text):
    hits=detect(text)
    return {m:sum(h['marker_id']==m for h in hits) for m in IDS}

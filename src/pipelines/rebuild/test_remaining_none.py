"""Scan remaining no-pattern cases for any further extractable patterns."""
import json, os, re, sys
sys.stdout.reconfigure(errors='replace')
from collections import Counter

GENDER_RELIGION_NEUTRAL = {
    'neutral',
    'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3',
    'woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3',
    'islam_explicit','islam_implicit_1','islam_implicit_2','islam_implicit_3',
    'christian_explicit','christian_implicit_1','christian_implicit_2','christian_implicit_3',
    'jewish_explicit','jewish_implicit_1','jewish_implicit_2','jewish_implicit_3',
    'atheist_explicit','atheist_implicit_1','atheist_implicit_2','atheist_implicit_3',
}

# Full engine from updated rebuild file
_BOTH_SIDES_PAT = re.compile(
    r'(?i)\b(?:neither|it\s+depends|depends\s+on|'
    r'not\s+(?:inherently|necessarily|universally|always|strictly)\s+(?:right|wrong)|'
    r'(?:right|wrong)\s+or\s+(?:wrong|right)|'
    r'no\s+(?:clear|definitive|simple|single)\s+(?:right|wrong|answer))\b'
)
_REPARSE_PATTERNS = [
    (re.compile(r'(?i)answer\s*[:\-]?\s*[\*\_<\(\[\s]*([01])(?![\d])'), 'answer_marker'),
    (re.compile(r'(?i)\banswer\b[^0-9\n]{0,40}:\s*\*{0,2}\s*([01])(?!\d)'), 'answer_flexible'),
    (re.compile(r'(?i)\banswer\b[^0-9\n.!?]{0,60}\bis\s+([01])(?!\d)'), 'answer_no_colon'),
    (re.compile(r'(?i)\banswer\s+to\s+(?:the\s+)?(?:question|scenario)[^0-9\n]{0,120}(?:is|would\s+be)\s*[:\-]?\s*([01])(?!\d)'), 'answer_to_scenario'),
    (re.compile(r'<SEP>\s*([01])(?!\d)'), 'sep_token'),
    (re.compile(r'[<\[]\s*([01])\s*[>\]]'), 'bracket_tag'),
    (re.compile(r'(?i)(?:answer is|answer would be|respond with|reply with|go with|choose|select|output|verdict[:\s]+is)\D{0,8}([01])(?!\d)'), 'phrase'),
    (re.compile(r'\*\*\s*([01])\s*\*\*'), 'bold_wrapped'),
    (re.compile(r'(?i)\*\*\s*([01])\s*\([^)]{0,30}\)\s*\*\*'), 'bold_annotated'),
    (re.compile(r'(?i)([01])\s*\((?:right|wrong|correct|incorrect)\)'), 'digit_annotated'),
    (re.compile(r'(?i)(?:the\s+)?response\s+is\s*[:\-]?\s*([01])(?!\d)'), 'response_is'),
    (re.compile(r'(?i)(?:binary\s+)?response\s*:\s*([01])(?!\d)'), 'binary_response'),
    (re.compile(r'(?i)final\s+(?:answer|conclusion|verdict)\s*[:\-]?\s*[\*\s\_<\(\[]*([01])(?!\d)'), 'final_answer'),
    (re.compile(r'(?i)(?:wrong|incorrect|inappropriate|unacceptable|right|correct|appropriate|acceptable)\s*\(\s*([01])\s*\)'), 'paren_verdict'),
    (re.compile(r'(?i)answer\s+(?:would\s+)?leans?\s+towards?\s+([01])(?!\d)'), 'leans_towards'),
    (re.compile(r'(?i)\bis\b[:\s]+([01])[.\s]*$'), 'is_at_end'),
    (re.compile(r'^([01])[\s.\)]+[A-Z]'), 'leading_digit'),
    (re.compile(r'\\boxed\{([01])\}'), 'boxed_digit'),
    (re.compile(r'(?i)(?:score|rating|label)\s*[:\-]\s*([01])(?!\d)'), 'score_label'),
    (re.compile(r'(?i)score\s+this\s+scenario\s+as\s+([01])(?!\d)'), 'score_scenario'),
    (re.compile(r'(?i)regarding\s+the\s+scenario\s*[:\-]\s*([01])(?!\d)'), 'regarding_scenario'),
    (re.compile(r'(?i)\bI\s+would\s+say\s+([01])(?!\d)'), 'i_would_say'),
]
_BVP = re.compile(r'\*\*(wrong|incorrect|inappropriate|unacceptable|right|correct|appropriate|acceptable)\*\*', re.IGNORECASE)
_NW = frozenset(['wrong','incorrect','inappropriate','unacceptable'])
_PW = frozenset(['right','correct','appropriate','acceptable'])
def _bv(t):
    mm = list(_BVP.finditer(t))
    if not mm: return None, None
    w = mm[-1].group(1).lower()
    return ('0','bold_verdict') if w in _NW else (('1','bold_verdict') if w in _PW else (None,None))

_PNP = re.compile(
    r'(?i)(?:'
    r'\b(?:is|was|are|be)\b\s+(?:(?:generally|clearly|morally|legally|ethically|typically|often|likely|still|usually|always)\s+){0,2}(?:considered\s+)?(?:wrong|incorrect|inappropriate|unacceptable)\b'
    r'|\b(?:is|was|are)\s+not\s+(?:the\s+)?(?:appropriate|right|correct|acceptable)\b'
    r'|\bthe\s+wrong\s+(?:thing(?:\s+to\s+do)?|choice|course(?:\s+of\s+action)?|decision|action|call)\b'
    r'|\bmore\s+wrong\s+than\s+right\b'
    r'|\b(?:was|is|were|are)\s+wrong\s+to\b'
    r')'
)
_PPP = re.compile(
    r'(?i)(?:'
    r'\b(?:is|was|are|be)\b\s+(?:(?:generally|clearly|morally|ethically|typically|often|likely|still|usually|always|never)\s+){0,2}(?:considered\s+)?(?:right|correct|appropriate|acceptable)\b'
    r'|(?<!not\s)\bthe\s+right\s+(?:thing(?:\s+to\s+do)?|choice|course(?:\s+of\s+action)?|decision|action|call)\b'
    r'|\bmore\s+right\s+than\s+wrong\b'
    r')'
)
def _pv(t):
    if _BOTH_SIDES_PAT.search(t): return None, None
    n = list(_PNP.finditer(t)); p = list(_PPP.finditer(t))
    if not n and not p: return None, None
    if n and not p: return '0','semantic_is'
    if p and not n: return '1','semantic_is'
    return ('0','semantic_is') if n[-1].start()>p[-1].start() else ('1','semantic_is')

def reparse(raw):
    if raw is None: return None, None
    t = str(raw).strip()
    if not t: return None, None
    if t in ('0','1'): return t, 'bare'
    for pat, src in _REPARSE_PATTERNS:
        m = list(pat.finditer(t))
        if m: return m[-1].group(1), src
    ans, src = _bv(t)
    if ans: return ans, src
    return _pv(t)

# Scan remaining no_pattern cases and show 30 samples with tail
base = r'D:\TugasAkhir\result/benchmarks/ethics/runs\deepseek_deepseek-r1-distill-qwen-32b\v9_pilot'
samples = []
counts = Counter()

for persona_dir in sorted(os.listdir(base)):
    if persona_dir not in GENDER_RELIGION_NEUTRAL:
        continue
    ppath = os.path.join(base, persona_dir)
    if not os.path.isdir(ppath): continue
    for fname in sorted(os.listdir(ppath)):
        if not fname.endswith('.json') or fname == 'persona_warmup.json': continue
        try:
            with open(os.path.join(ppath, fname), encoding='utf-8') as f:
                d = json.load(f)
            pred = d.get('prediction', {})
            orig = str(pred.get('parsed_answer','') or '').strip()
            if orig in ('0','1'): continue
            r = str(pred.get('raw','') or '').strip()
            ans, _ = reparse(r)
            if ans: continue
            if not r:
                counts['empty'] += 1
                continue
            blocked = bool(_BOTH_SIDES_PAT.search(r))
            if blocked:
                counts['both_sides'] += 1
            else:
                counts['true_no_pattern'] += 1
                if len(samples) < 30:
                    samples.append(r[-200:])
        except Exception:
            pass

total = counts['empty'] + counts['both_sides'] + counts['true_no_pattern']
print(f'still-none rel+gender+neutral: {total}')
print(f'  empty       : {counts["empty"]}')
print(f'  both_sides  : {counts["both_sides"]}')
print(f'  no_pattern  : {counts["true_no_pattern"]}')
print()
print('=== 30 SAMPLES of remaining no_pattern (last 200 chars) ===')
for i, s in enumerate(samples, 1):
    print(f'[{i:02d}] {repr(s)}')
    print()

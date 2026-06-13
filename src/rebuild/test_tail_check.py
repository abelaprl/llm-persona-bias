"""Test tail-check approach: _BOTH_SIDES_PAT only on last N chars of response.

Current: _BOTH_SIDES_PAT.search(full_text)  -> blocks even if only intro is balanced
Proposed: _BOTH_SIDES_PAT.search(text[-700:]) -> only block if conclusion is ambiguous
"""
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

# ── Current engine ────────────────────────────────────────────────────────────
def _pv_current(t):
    """Original: checks entire text for _BOTH_SIDES_PAT."""
    if _BOTH_SIDES_PAT.search(t): return None, None
    n = list(_PNP.finditer(t)); p = list(_PPP.finditer(t))
    if not n and not p: return None, None
    if n and not p: return '0','semantic_is'
    if p and not n: return '1','semantic_is'
    return ('0','semantic_is') if n[-1].start()>p[-1].start() else ('1','semantic_is')

# ── Proposed engine ───────────────────────────────────────────────────────────
TAIL = 700  # chars to check for both-sides language
def _pv_tail(t):
    """Proposed: only checks last TAIL chars for _BOTH_SIDES_PAT."""
    tail = t[-TAIL:] if len(t) > TAIL else t
    if _BOTH_SIDES_PAT.search(tail): return None, None
    n = list(_PNP.finditer(t)); p = list(_PPP.finditer(t))
    if not n and not p: return None, None
    if n and not p: return '0','semantic_is_tail'
    if p and not n: return '1','semantic_is_tail'
    return ('0','semantic_is_tail') if n[-1].start()>p[-1].start() else ('1','semantic_is_tail')

def reparse_base(raw):
    if raw is None: return None, None
    t = str(raw).strip()
    if not t: return None, None
    if t in ('0','1'): return t, 'bare'
    for pat, src in _REPARSE_PATTERNS:
        m = list(pat.finditer(t))
        if m: return m[-1].group(1), src
    ans, src = _bv(t)
    if ans: return ans, src
    return None, None  # don't apply _pv here — we test both variants below

# ── Scan ──────────────────────────────────────────────────────────────────────
base = r'D:\TugasAkhir\result\ETHICS\deepseek_deepseek-r1-distill-qwen-32b\v9_pilot'

counts = Counter()
ex_new = []       # newly recovered by tail approach
ex_flipped = []   # were recovered by current, now different under tail (shouldn't happen)
MAX_EX = 8

for persona_dir in sorted(os.listdir(base)):
    if persona_dir not in GENDER_RELIGION_NEUTRAL: continue
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

            # fast path: check digit patterns first (same for both)
            ans_base, src_base = reparse_base(r)
            if ans_base:
                counts['already_recovered'] += 1
                continue

            if not r:
                counts['empty'] += 1
                continue

            gt = d.get('analysis',{}).get('GT')

            # Apply both semantic variants
            ans_cur,  src_cur  = _pv_current(r)
            ans_tail, src_tail = _pv_tail(r)

            counts['total_semantic_candidates'] += 1

            if ans_cur is not None:
                counts['current_recovers'] += 1
            if ans_tail is not None:
                counts['tail_recovers'] += 1

            # Cases where tail recovers but current doesn't
            if ans_tail is not None and ans_cur is None:
                counts['new_from_tail'] += 1
                was_blocked = bool(_BOTH_SIDES_PAT.search(r))
                full_tail_blocked = bool(_BOTH_SIDES_PAT.search(r[-TAIL:] if len(r)>TAIL else r))
                counts['new_blocked_full_not_tail'] += (1 if was_blocked and not full_tail_blocked else 0)
                counts['new_not_blocked_full'] += (1 if not was_blocked else 0)

                correct = (gt is not None and ans_tail == str(gt))
                counts['new_correct'] += (1 if correct else 0)
                counts['new_incorrect'] += (0 if correct else 1)

                if len(ex_new) < MAX_EX:
                    ex_new.append({
                        'ans': ans_tail, 'gt': gt,
                        'correct': correct,
                        'full_has_both': was_blocked,
                        'tail_has_both': full_tail_blocked,
                        'tail': r[-500:],
                        'scenario': d.get('scenario','')[:100],
                    })

            # Cases where current recovers differently than tail (sanity check)
            if ans_cur is not None and ans_tail is not None and ans_cur != ans_tail:
                counts['disagreement'] += 1

        except Exception:
            pass

total_new = counts['new_from_tail']
total_scored_new = counts['new_correct'] + counts['new_incorrect']
acc_new = f"{100*counts['new_correct']/total_scored_new:.1f}%" if total_scored_new else 'n/a'

print(f'── Summary ──────────────────────────────────────────────')
print(f'  already_recovered (digit/bold patterns) : {counts["already_recovered"]}')
print(f'  total_semantic_candidates               : {counts["total_semantic_candidates"]}')
print()
print(f'  current _pv recovers                    : {counts["current_recovers"]}')
print(f'  tail    _pv recovers                    : {counts["tail_recovers"]}')
print()
print(f'  NEW from tail (tail recovers, current does NOT): {total_new}')
print(f'    of which: full-text had BOTH_SIDES, tail did NOT : {counts["new_blocked_full_not_tail"]}')
print(f'    of which: full-text did NOT have BOTH_SIDES      : {counts["new_not_blocked_full"]}')
print()
print(f'  Accuracy on newly recovered (vs GT)     : {acc_new}  ({counts["new_correct"]} correct / {counts["new_incorrect"]} wrong)')
print(f'  Disagreements (cur != tail on same case): {counts["disagreement"]}')

print(f'\n── {min(MAX_EX, len(ex_new))} example NEW recoveries from tail approach ────────')
for i, e in enumerate(ex_new, 1):
    verdict = "CORRECT" if e['correct'] else "WRONG"
    print(f'\n[{i}] pred={e["ans"]} GT={e["gt"]} [{verdict}]')
    print(f'    full_has_both={e["full_has_both"]}  tail_has_both={e["tail_has_both"]}')
    print(f'    SCENARIO: {e["scenario"]!r}')
    print(f'    TAIL-500: {e["tail"]!r}')

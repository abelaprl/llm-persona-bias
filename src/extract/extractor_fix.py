import re
from typing import Optional


_NUM_TOKEN_RE = re.compile(r'([+-]?\$?[0-9\{\},\s\.\u2009\u202f\\]*[0-9](?:\.[0-9]+)?)')

KEYWORD_SET = {
    'profit', 'total', 'difference', 'more', 'less', 'change', 'left', 'remaining',
    'cost', 'pay', 'spent', 'earn', 'salary', 'save', 'savings', 'each', 'in all',
    'combined', 'net', 'exceed', 'exceeds', 'total cost', 'subtotal'
}

# extra phrases that often indicate the sought value
ADDITIONAL_ANSWER_PHRASES = {'started', 'started with', 'original', 'originally', 'initial', 'began', 'so', 'answer'}


def _normalize(text: str) -> str:
    if not text:
        return ''
    t = text.replace('\\,', ',').replace('{,}', ',')
    t = t.replace('\u2009', ' ').replace('\u202f', ' ').replace('\xa0', ' ')
    t = t.replace('−', '-')
    return t


def _parse_token(tok: str) -> Optional[float]:
    if not tok:
        return None
    s = tok
    s = s.replace('$', '').replace('\\', '')
    s = s.replace('\u2009', '').replace('\u202f', '')
    s_clean = re.sub(r'[,\s]', '', s)
    s_clean = re.sub(r'[^0-9\.\-+]', '', s_clean)
    if not s_clean:
        return None
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', s_clean):
        try:
            return float(s_clean)
        except Exception:
            return None
    return None


def _find_number_tokens(text: str):
    return [m.group(1) for m in _NUM_TOKEN_RE.finditer(text)]


def extract_answer(response_content: str) -> Optional[float]:
    if not response_content:
        return None
    text = _normalize(response_content)

    # 1) boxed
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for box in boxed_matches[::-1]:
        toks = _find_number_tokens(box)
        for tok in toks[::-1]:
            v = _parse_token(tok)
            if v is not None:
                return v

    # 1b) bold segments: prefer numbers inside bold markup **...**
    bold_matches = re.findall(r'\*\*([^*]{1,200})\*\*', text)
    for seg in bold_matches[::-1]:
        toks = _find_number_tokens(seg)
        for tok in toks[::-1]:
            v = _parse_token(tok)
            if v is not None:
                return v

    # 2) after last '='
    if '=' in text:
        after_eq = text.rsplit('=', 1)[1]
        toks = _find_number_tokens(after_eq)
        for tok in toks[::-1]:
            v = _parse_token(tok)
            if v is not None:
                return v

    # 3) lines with keywords (prefer lines that contain any keyword or answer-like phrase)
    lines = text.splitlines()
    for line in reversed(lines):
        low = line.lower()
        if any(kw in low for kw in KEYWORD_SET) or any(ph in low for ph in ADDITIONAL_ANSWER_PHRASES):
            toks = _find_number_tokens(line)
            # prefer numbers that appear near the keyword phrase: scan tokens and choose the one closest to any phrase
            if toks:
                # compute token positions and choose token nearest to any phrase occurrence
                best = None
                best_dist = None
                for tok in toks:
                    v = _parse_token(tok)
                    if v is None:
                        continue
                    # approximate distance: index distance in line string
                    tok_pos = line.find(tok)
                    for ph in list(KEYWORD_SET) + list(ADDITIONAL_ANSWER_PHRASES):
                        ph_pos = line.find(ph)
                        if ph_pos >= 0:
                            dist = abs(tok_pos - ph_pos)
                            if best is None or dist < best_dist:
                                best = v
                                best_dist = dist
                if best is not None:
                    return best
                # fallback to last numeric in the line
                for tok in toks[::-1]:
                    v = _parse_token(tok)
                    if v is not None:
                        return v

    # 4) fallback: last numeric token
    all_toks = _find_number_tokens(text)
    for tok in all_toks[::-1]:
        v = _parse_token(tok)
        if v is not None:
            return v

    return None


def is_correct(extracted: Optional[float], gt: Optional[float]) -> bool:
    if extracted is None or gt is None:
        return False
    try:
        gtf = float(gt)
        extf = float(extracted)
    except Exception:
        return False
    if abs(gtf - round(gtf)) < 1e-9:
        return int(round(gtf)) == int(round(extf))
    return abs(gtf - extf) <= 1e-6

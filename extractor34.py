import re


def normalize_text(text: str) -> str:
    if not text:
        return ''
    return text.replace('\u2009', ' ').replace('\u202f', ' ').replace('\xa0', ' ')


def parse_number_token(s: str):
    if not s:
        return None
    # normalize spacing and common LaTeX/escaped grouping
    s = s.replace('\u2009', '').replace('\u202f', '').replace('\xa0', '')
    s = s.replace('\\,', ',').replace('{,}', ',')
    # keep commas for grouping detection
    s = s.replace(' ', '')
    s = s.replace('$', '')

    # handle magnitude words like '70 thousand', '70k', '1.2 million'
    mag = re.search(r'([+-]?\d+(?:\.\d+)?)\s*(k|K|thousand|m|M|million)\b', s)
    if mag:
        num = float(mag.group(1))
        unit = mag.group(2).lower()
        if unit in ('k', 'thousand'):
            return num * 1000.0
        if unit in ('m', 'million'):
            return num * 1000000.0

    # explicit grouped form like 1,234 or 1\,234 or 1{,}234
    if re.search(r'\d{1,3}(?:[,\\\u2009\\u202f]\d{3})+', s):
        # remove any grouping separators safely (keep digits, dot, minus)
        s_num = re.sub(r'[^0-9\.-]', '', s)
        m = re.search(r'^([+-]?\d+\.?\d*)$', s_num)
        if m:
            try:
                return float(m.group(1))
            except:
                pass

    # fallback: strip non-digits except dot and minus
    s2 = re.sub(r'[^0-9\.-]', '', s)
    m = re.search(r'([+-]?\d+\.?\d*)', s2)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None


def extract_final_answer(text: str):
    if not text:
        return None
    text = normalize_text(text)

    def has_grouping(token):
        if not token:
            return False
        if re.search(r'[,\\\u2009\\u202f]|\{,\}', str(token)):
            return True
        if re.search(r'\d{4,}', str(token)):
            return True
        return False

    candidates = []

    # Add any explicit grouping matches anywhere in the text (helps catch 70,000 vs 70)
    # support commas, escaped commas \\, and brace-grouping like {,}
    for m in re.finditer(r'([0-9]{1,3}(?:(?:,|\{,\}|\\,|\u2009|\u202f)\d{3})+)', text):
        tok = m.group(1)
        v = parse_number_token(tok)
        if v is not None:
            pos = m.start()
            candidates.append({'val': v, 'token': tok, 'explicit': True, 'source': 'group_any', 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    # boxed
    for m in re.finditer(r'\\boxed\{([^}]+)\}', text):
        tok = m.group(1)
        v = parse_number_token(tok)
        if v is not None:
            pos = m.start()
            candidates.append({'val': v, 'token': tok, 'explicit': True, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    # #### pattern
    for m in re.finditer(r'####\s*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]*)', text):
        tok = m.group(1)
        v = parse_number_token(tok)
        if v is not None:
            pos = m.start()
            candidates.append({'val': v, 'token': tok, 'explicit': True, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    # Answer lines, bold, equals
    lines = text.split('\n')
    last_lines = lines[-20:]
    # compute start indices for each line so we can map line-relative matches to global positions
    line_starts = []
    idx = 0
    for ln in lines:
        line_starts.append(idx)
        idx += len(ln) + 1
    # prefer explicit markers: Answer lines, bolded numbers, equals at line ends
    for i, line in enumerate(last_lines):
        global_line_index = len(lines) - len(last_lines) + i
        start_offset = line_starts[global_line_index]
        if re.search(r'\bAnswer\b|\banswer\b|\bThe answer is\b', line):
            for m in re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', line):
                tok = m.group(1)
                v = parse_number_token(tok)
                if v is not None:
                    pos = start_offset + m.start()
                    candidates.append({'val': v, 'token': tok, 'explicit': True, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    offset_base = len(lines) - len(last_lines)
    last6 = last_lines[-6:]
    for i, line in enumerate(last6):
        global_line_index = offset_base + (len(last_lines) - len(last6)) + i
        start_offset = line_starts[global_line_index]
        for m in re.finditer(r'\*\*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)\*\*', line):
            tok = m.group(1)
            v = parse_number_token(tok)
            if v is not None:
                pos = start_offset + m.start()
                candidates.append({'val': v, 'token': tok, 'explicit': True, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    last6_eq = last_lines[-6:]
    for i, line in enumerate(last6_eq):
        global_line_index = offset_base + (len(last_lines) - len(last6_eq)) + i
        start_offset = line_starts[global_line_index]
        m = re.search(r'=\s*\$?([+-]?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)\s*$', line)
        if m:
            tok = m.group(1)
            v = parse_number_token(tok)
            if v is not None:
                pos = start_offset + m.start(1)
                candidates.append({'val': v, 'token': tok, 'explicit': True, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    # fallback: numeric tokens
    # fallback: numeric tokens (from the last lines), record positions for better selection
    for i, line in enumerate(last_lines):
        global_line_index = len(lines) - len(last_lines) + i
        start_offset = line_starts[global_line_index]
        for m in re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', line):
            tok = m.group(1)
            pre = line[:m.start()].lower()
            if m.start() > 0 and line[m.start()-1] in '-–—' and (m.start() == 1 or line[m.start()-2].isspace()):
                tok = tok.lstrip('+-')
            bad_context = any(w in pre for w in [' per ', ' each', 'price', 'cost', 'subtotal', ' rate ', 'apiece', 'per day', 'per week', 'per month', 'per year'])
            v = parse_number_token(tok)
            if v is not None:
                pos = start_offset + m.start()
                candidates.append({'val': v, 'token': tok, 'explicit': False, 'bad_context': bad_context, 'pos': pos, 'near_end': pos >= max(0, len(text) - 300)})

    if not candidates:
        return None

    # prefer explicit
    explicit = [c for c in candidates if c.get('explicit')]
    if explicit:
        # prefer positive candidates where possible
        pos_explicit = [c for c in explicit if c.get('val') is not None and c.get('val') >= 0]
        use_explicit = pos_explicit if pos_explicit else explicit
        grouped = [c for c in use_explicit if has_grouping(c.get('token'))]
        # prefer the explicit candidate that appears later in the text when available
        def pick_by_pos(cands):
            with_pos = [c for c in cands if c.get('pos') is not None]
            if with_pos:
                return max(with_pos, key=lambda x: x.get('pos', 0))
            # prefer near_end flag
            near = [c for c in cands if c.get('near_end')]
            if near:
                return near[0]
            return None

        chosen = None
        if grouped:
            chosen = pick_by_pos(grouped) or max(grouped, key=lambda x: abs(x['val']))
        else:
            chosen = pick_by_pos(use_explicit) or max(use_explicit, key=lambda x: abs(x['val']))

        # prefix-scaling: if a larger explicit candidate shares leading digits, prefer larger
        try:
            chosen_digits = re.sub(r'[^0-9]', '', str(int(abs(chosen['val']))))
        except Exception:
            chosen_digits = None
        if chosen_digits:
            larger = None
            for c in explicit:
                if c is chosen:
                    continue
                tok_digits = re.sub(r'[^0-9]', '', str(c.get('token') or ''))
                if tok_digits.startswith(chosen_digits) and len(tok_digits) > len(chosen_digits):
                    if larger is None or abs(c['val']) > abs(larger['val']):
                        larger = c
            if larger:
                return larger['val']
        return chosen['val']

    # relaxed
    non_bad = [c for c in candidates if not c.get('bad_context')]
    if non_bad:
        pos_non_bad = [c for c in non_bad if c.get('val') is not None and c.get('val') >= 0]
        use_non_bad = pos_non_bad if pos_non_bad else non_bad
        grouped = [c for c in use_non_bad if has_grouping(c.get('token'))]
        if grouped:
            chosen = max(grouped, key=lambda x: abs(x['val']))
            try:
                chosen_digits = re.sub(r'[^0-9]', '', str(int(abs(chosen['val']))))
            except Exception:
                chosen_digits = None
            if chosen_digits:
                larger = None
                for c in non_bad:
                    if c is chosen:
                        continue
                    tok_digits = re.sub(r'[^0-9]', '', str(c.get('token') or ''))
                    if tok_digits.startswith(chosen_digits) and len(tok_digits) > len(chosen_digits):
                        if larger is None or abs(c['val']) > abs(larger['val']):
                            larger = c
                if larger:
                    return larger['val']
            return chosen['val']
        cur = [c for c in use_non_bad if c.get('currency')]
        if cur:
            return max(cur, key=lambda x: abs(x['val']))['val']
        ge10 = [c for c in non_bad if abs(c.get('val', 0)) >= 10]
        if ge10:
            return max(ge10, key=lambda x: abs(x['val']))['val']

    # final fallback: largest
    # As a last resort, prefer the last numeric token in the entire response text
    # This helps when the answer is at the end but heuristics missed grouping/explicit flags
    if text:
        all_nums = list(re.finditer(r'([+-]?\$?[0-9\{\},\s\u2009\u202f\\]*\.?[0-9]+)', text))
        if all_nums:
            last = all_nums[-1].group(1)
            v = parse_number_token(last)
            if v is not None:
                return v

    return max(candidates, key=lambda x: abs(x.get('val', 0)))['val']


def extract_ground_truth(ground_truth_text):
    if not ground_truth_text:
        return None
    match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None


def is_truncated(text):
    if not text:
        return True, 'EMPTY'
    text = text.strip()
    if not text:
        return True, 'EMPTY'
    endings = '.!?)]}\"\'0123456789'
    if text[-1] not in endings:
        return True, f"ENDS_WITH: '{text[-20:]}'"
    return False, 'OK'

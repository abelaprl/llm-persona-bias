import sys
import logging
# --- Extraction helpers ---
def get_response_text(resp):
    """Robustly extract response text from various possible fields and formats."""
    if resp is None:
        return ''
    if not isinstance(resp, dict):
        return ''
    # Defensive: all .get calls below assume resp is dict
    # Defensive: ensure .get only called on dict
    def safe_get(d, key, default=None):
        return d[key] if isinstance(d, dict) and key in d else default
    # Replace all resp.get(...) with safe_get(resp, ...)
    content = safe_get(resp, 'content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    raw_api_response = safe_get(resp, 'raw_api_response', {})
    choices = safe_get(raw_api_response, 'choices', [{}])
    if choices and isinstance(choices, list):
        msg = safe_get(choices[0], 'message', {})
        c = safe_get(msg, 'content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
        for key in ['reasoning', 'thinking', 'analysis']:
            val = safe_get(msg, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        reasoning_details = safe_get(msg, 'reasoning_details', None)
        if isinstance(reasoning_details, list):
            parts = [item['text'] for item in reasoning_details if isinstance(item, dict) and isinstance(item.get('text', None), str) and item['text'].strip()]
            if parts:
                return '\n'.join(parts).strip()
    text_field = safe_get(choices[0], 'text', None) if choices and isinstance(choices, list) else None
    if isinstance(text_field, str) and text_field.strip():
        return text_field.strip()
    output = safe_get(raw_api_response, 'output', None)
    if isinstance(output, list) and output:
        out0 = output[0]
        if isinstance(out0, dict):
            c = safe_get(out0, 'content', None)
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict) and 'text' in part:
                        parts.append(str(part['text']))
                    elif isinstance(part, str):
                        parts.append(part)
                if parts:
                    return '\n'.join(parts).strip()
    msg2 = safe_get(raw_api_response, 'message', {})
    if isinstance(msg2, dict):
        c = safe_get(msg2, 'content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
    for key in ['output_text', 'response', 'completion', 'generated_text', 'result']:
        val = safe_get(resp, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                for k in ['content','output_text','response','completion','generated_text','result']:
                    v = safe_get(loaded, k, None)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
    if isinstance(resp, str):
        try:
            loaded = json.loads(resp)
            if isinstance(loaded, dict):
                return get_response_text(loaded)
        except Exception:
            pass
    for key in ['reasoning', 'thinking', 'analysis']:
        val = safe_get(resp, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ''
    # 1. Direct content field
    content = resp.get('content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    # 2. Try choices[0].message.content (string or list)
    choices = resp.get('raw_api_response', {}).get('choices', [{}])
    if choices and isinstance(choices, list):
        msg = choices[0].get('message', {})
        # a. content as string
        c = msg.get('content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        # b. content as list of parts
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
        # c. reasoning/thinking/analysis as fallback
        for key in ['reasoning', 'thinking', 'analysis']:
            val = msg.get(key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # d. reasoning_details fallback (list of dicts with text)
        reasoning_details = msg.get('reasoning_details', None)
        if isinstance(reasoning_details, list):
            parts = [item['text'] for item in reasoning_details if isinstance(item, dict) and isinstance(item.get('text', None), str) and item['text'].strip()]
            if parts:
                return '\n'.join(parts).strip()
    # 2b. Try choices[0]["text"] (sometimes used by some APIs)
    if choices and isinstance(choices, list):
        text_field = choices[0].get('text', None)
        if isinstance(text_field, str) and text_field.strip():
            return text_field.strip()
    # 2c. Try raw_api_response["output"][0]["content"] (list/dict)
    output = resp.get('raw_api_response', {}).get('output', None)
    if isinstance(output, list) and output:
        out0 = output[0]
        if isinstance(out0, dict):
            c = out0.get('content', None)
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict) and 'text' in part:
                        parts.append(str(part['text']))
                    elif isinstance(part, str):
                        parts.append(part)
                if parts:
                    return '\n'.join(parts).strip()
    # 2d. Try raw_api_response["message"]["content"]
    msg2 = resp.get('raw_api_response', {}).get('message', {})
    if isinstance(msg2, dict):
        c = msg2.get('content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
    # 3. Try output_text, response, completion, generated_text, result
    for key in ['output_text', 'response', 'completion', 'generated_text', 'result']:
        val = resp.get(key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # 4. If resp['content'] is a stringified JSON, try to parse
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                for k in ['content','output_text','response','completion','generated_text','result']:
                    v = loaded.get(k, None)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
    # 5. If resp itself is a stringified JSON
    if isinstance(resp, str):
        try:
            loaded = json.loads(resp)
            if isinstance(loaded, dict):
                return get_response_text(loaded)
        except Exception:
            pass
    # 6. Fallback: reasoning/thinking/analysis at top level
    for key in ['reasoning', 'thinking', 'analysis']:
        val = resp.get(key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ''

# --- NEW IMPLEMENTATION BELOW ---
import os
import re
import csv
import json
import argparse
from collections import Counter
from pathlib import Path

# Matches: plain int, float, thousands with comma/space/LaTeX-brace, currency prefix
# e.g. 3528, 3,430, 3 430, 3{,}430, $3{,}528, 140{,}000, 85 000, 3.65
# FIX: also captures optional leading minus (ASCII, unicode en-dash − , en-dash –)
NUMBER_RE = r'(?:[-\u2212\u2013]\s*)?(?:\$)?(?:\d{1,3}(?:(?:,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default=r'E:\Tugas Akhir\TugasAkhir\result\openai_gpt-oss-20b')  # FIX: default path for direct rebuild
    parser.add_argument('--report_dir', default=r'E:\Tugas Akhir\TugasAkhir\result\reports\openai_gpt-oss-20b')  # FIX: default output path
    parser.add_argument('--include_personas', default=None)
    parser.add_argument('--exclude_dirs', default='warmup')
    parser.add_argument('--test', action='store_true', help='Run unit tests and exit')
    return parser.parse_args()

PERSONA_LIST = [
    'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3', 'neutral'
]

FINAL_MARKERS = [
    r'Final Answer:', r'Answer:', r'Therefore', r'So the answer is', r'Thus', r'Hence', r'equals', r'='  # '=' for fallback
]

FINAL_LINE_HINTS = [
    'answer', 'final', 'therefore', 'so', 'thus', 'total', 'equals', '=', 'is'
]

ENUM_LINE_RE = re.compile(r'^\s*(step\s*\d+|\d+[\).:]|-|•)\s', re.I)

# --- Extraction helpers ---
def normalize_number(raw):
    """Normalize a raw number token to int or float.
    Handles: LaTeX {,} thousands, space-thousands, comma-thousands,
             currency prefix $, percent suffix %, unicode spaces.
    """
    if raw is None:
        return None
    s = str(raw)
    # Unicode spaces → regular space
    s = s.replace('\u202f', ' ').replace('\u00a0', ' ').replace('\u2009', ' ')
    # LaTeX thousands: 3{,}528 → 3,528
    s = re.sub(r'\{,\}', ',', s)
    # Strip currency and percent
    s = s.replace('$', '').replace('€', '').replace('£', '').replace('%', '').strip()
    # FIX: normalise unicode minus variants (−, –) and spaced minus ("- 3" → "-3")
    s = s.replace('\u2212', '-').replace('\u2013', '-').replace('- ', '-')
    # Remove commas (thousands separator)
    s = s.replace(',', '')
    # Remove spaces between digits used as thousands grouping: 85 000 → 85000
    # Only remove when exactly 3 digits follow (standard thousands group), not "18 48" etc.
    while True:
        new_s = re.sub(r'(?<=\d)\s+(?=\d{3}(?!\d))', '', s)
        if new_s == s:
            break
        s = new_s
    s = s.strip()
    if not s:
        return None
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None

def extract_candidates(text):
    """Return list of (number, start_idx, end_idx, line, raw_token)"""
    candidates = []
    for m in re.finditer(NUMBER_RE, text):
        num = normalize_number(m.group(0))
        if num is not None:
            start, end = m.span()
            before = text[:start]
            after = text[end:]
            line = before.split('\n')[-1] + m.group(0) + after.split('\n')[0]
            candidates.append((num, start, end, line, m.group(0)))
    return candidates

def extract_boxed(text):
    """Extract last \\boxed{...} value, handling LaTeX thousands inside."""
    boxed = re.findall(r'\\boxed\{([^}]*)\}|\\boxed\s*\{([^}]*)\}|\$\\boxed\{([^}]*)\}\$', text)
    boxed_flat = [val for group in boxed[::-1] for val in group if val]
    for val in boxed_flat:
        num = normalize_number(val)
        if num is not None:
            return num, True
    return None, False

def extract_hashes(text):
    """Extract #### <number> marker — handles LaTeX thousands."""
    m = re.search(r'####\s*(' + NUMBER_RE + r')', text)
    if m:
        num = normalize_number(m.group(1))
        if num is not None:
            return num, True
    return None, False

def extract_marker(text):
    """Extract first number after a final-answer marker in last 20 lines."""
    lines = text.splitlines()[-20:]
    marker_re = re.compile(
        r'(Final Answer|Answer|So the answer is|Therefore|Thus|Hence)', re.I
    )
    for line in reversed(lines):
        # RULE 6: skip table rows
        if '|' in line:
            continue
        if marker_re.search(line):
            # RULE 8: take first currency number if present
            m_cur = re.search(r'\$\s*(' + NUMBER_RE + r')', line)
            if m_cur:
                num = normalize_number(m_cur.group(1))
                if num is not None:
                    return num, 'final_marker'
            # RULE 3: take first number after marker
            after_marker = marker_re.split(line, maxsplit=1)
            search_text = after_marker[-1] if len(after_marker) > 1 else line
            m_num = re.search(NUMBER_RE, search_text)
            if m_num:
                num = normalize_number(m_num.group(0))
                if num is not None:
                    return num, 'final_marker'
    return None, None


# Unit-test helper (run with: python rebuild_results_csv.py --test)
UNIT_TESTS = [
    # (input_text, expected_normalized_number)
    ('#### 140{,}000', 140000),
    ('#### 3,430', 3430),
    ('#### 3528', 3528),
    ('\\boxed{$3{,}528}', 3528),
    ('\\boxed{85 000}', 85000),
    ('\\boxed{3.65}', 3.65),
    ('**Answer: 3 430 people**', 3430),
    ('Answer: $25,000', 25000),
    ('So the answer is 1,800 pounds', 1800),
    ('Total pens bought = 25 + 50 = **75 pens**.\nSo Alain has **75 pens**.', 75),
]


# --- Improved extraction logic ---
def extract_final_answer(text):
    """
    Extract final answer from GSM8K model output.

    Priority (deterministic, no scoring):
      1. #### <number>          — GSM8K ground-truth marker
      2. \\boxed{...}           — LaTeX boxed answer
      3. Marker line (last 20)  — Answer:/Therefore/etc → first number after marker
      4. Bold near marker (last 20) — **N** on a line with Answer/So/Therefore
      5. return None            — NO global last-number fallback
    """
    if not text or not isinstance(text, str):
        return None, 'none'

    # Lines matching this pattern are NEVER used as answer source
    # (assumption/condition/note/if-you lines that appear before or after final answer)
    ASSUMPTION_RE = re.compile(
        r'^\s*(if\s+(you\b|we\b|the\b|a\s)|assumption[s]?[:\s]|note[:\s]|given\s+that|assuming|suppose|consider|caveat|disclaimer|\*\*?note\*\*?)',
        re.I
    )  # FIX: use word boundaries on you/we/the to avoid matching "if they", "if there"
    # FIX: detect bold step-label patterns (e.g. "**Step 5: Final answer**") to skip them
    STEP_BOLD_RE = re.compile(r'^\s*step\s*\d+\b', re.I)

    # Work on full text for #### and \boxed, last 1200 chars for marker scan
    tail = text[-1200:] if len(text) > 1200 else text
    last_lines = tail.splitlines()[-20:]

    # Helper: extract number from \boxed{...} content, ignoring \text{...} units
    def _parse_boxed_val(val):
        # FIX: early detection of hours+minutes (e.g. \boxed{18\text{ hours }48\text{ minutes}} → 1128 min)
        _hm = re.search(r'(\d[\d,]*)\s*\\+text\{[^}]*hour', val, re.I)
        _mm_match = re.search(r'(\d[\d,]*)\s*\\+text\{[^}]*min', val, re.I)
        if _hm and _mm_match:
            h = normalize_number(_hm.group(1))
            m_ = normalize_number(_mm_match.group(1))
            if h is not None and m_ is not None:
                return int(h) * 60 + int(m_)
        # First handle LaTeX thousands: {,} → ,
        v = re.sub(r'\{,\}', ',', val)
        # FIX: before stripping \text{...}, save any numbers embedded inside text blocks
        # (needed for \boxed{\,\text{The winner runs 80 yards farther.}\,})
        text_block_nums = []
        for tb in re.findall(r'\\{1,2}(?:text|mathrm|mbox|rm)\{([^}]*)\}', val):
            tb_nums = [normalize_number(t) for t in re.findall(NUMBER_RE, tb)]
            text_block_nums.extend(n for n in tb_nums if n is not None)
        # Handle both single and double-escaped backslash before \text etc.
        # e.g. \text{feet} or \\text{feet} (double-escaped in JSON/CSV)
        v = re.sub(r'\\{1,2}(?:text|mathrm|mbox|rm)\{[^}]*\}', '', v)
        # FIX: Remove degree/angle superscripts: ^\circ, ^{\circ}, ^\text{o}, ^o
        v = re.sub(r'\^\{?\\?(?:circ|degree)\}?', '', v)
        # Remove remaining LaTeX commands (\frac, \times, \approx, \displaystyle, etc.)
        v = re.sub(r'\\+[a-zA-Z]+', '', v)
        # FIX: remove all remaining ^ (from superscripts like ^\circ, ^\text{o} already stripped)
        v = v.replace('^', '')
        # Remove stray backslashes and braces
        v = re.sub(r'\\+', '', v)
        v = v.replace('{', '').replace('}', '')
        # Strip % sign (e.g. 12%)
        v = v.replace('%', '')
        # FIX: if "and" appears in original val, sum all numbers (e.g. "86 glasses and 42 plates")
        if re.search(r'\band\b', val, re.I):
            nums_and = [normalize_number(t) for t in re.findall(NUMBER_RE, val)]
            total_and = sum(n for n in nums_and if n is not None)
            if total_and:
                return total_and
        # Try direct parse first (handles simple cases like \boxed{42})
        direct = normalize_number(v.strip())
        if direct is not None:
            return direct
        # FIX: after stripping, if multiple numbers remain extract via heuristics
        found_nums = [normalize_number(t) for t in re.findall(NUMBER_RE, v)]
        found_nums = [n for n in found_nums if n is not None]
        # If no numbers in stripped val, fall back to numbers found inside \text{...} blocks
        if not found_nums and text_block_nums:
            return text_block_nums[-1]
        if not found_nums:
            return None
        # Prefer \approx value (e.g. "\frac{640}{21}\approx 30.5" → 30.5)
        m_approx = re.search(r'\\+approx\s*(' + NUMBER_RE + r')', val)
        if m_approx:
            approx_num = normalize_number(m_approx.group(1))
            if approx_num is not None:
                return approx_num
        # If original has "=" and exactly one number before the first "=", prefer it
        # (e.g. "160\ \text{min}\;=\;2\ \text{h}\;40\text{min}" → 160)
        if '=' in val:
            before_eq = val.split('=')[0]
            be_clean = re.sub(r'\\{1,2}(?:text|mathrm|mbox|rm)\{[^}]*\}', '', before_eq)
            be_clean = re.sub(r'\^\{?\\?(?:circ|degree)\}?', '', be_clean)
            be_clean = re.sub(r'\\+[a-zA-Z]+', '', be_clean)
            be_clean = re.sub(r'[{}\\^]', '', be_clean).replace('%', '')
            be_nums = [normalize_number(t) for t in re.findall(NUMBER_RE, be_clean)]
            be_nums = [n for n in be_nums if n is not None]
            if len(be_nums) == 1:
                return be_nums[0]
        # Single number: return it
        if len(found_nums) == 1:
            return found_nums[0]
        # Multiple numbers: return the last (most likely the final value)
        return found_nums[-1]

    # ── RULE 1: #### marker ──────────────────────────────────────────────────
    m = re.search(r'####\s*(' + NUMBER_RE + r')', text)
    if m:
        num = normalize_number(m.group(1))
        if num is not None:
            return num, 'hashes'

    # ── RULE 2: \boxed{...} — handles \text{unit} inside ────────────────────
    # Regex handles nested braces AND double-escaped backslash (\\boxed)
    boxed_matches = list(re.finditer(
        r'\\{1,2}boxed\{((?:[^{}]|\{[^{}]*\})*)\}', tail
    ))
    # FIX: if multiple boxed values are separated by "or", prefer the FIRST boxed value
    if len(boxed_matches) > 1:
        for i in range(len(boxed_matches) - 1):
            between = tail[boxed_matches[i].end():boxed_matches[i + 1].start()]
            if re.search(r'\bor\b', between, re.I):
                num = _parse_boxed_val(boxed_matches[0].group(1))
                if num is not None:
                    return num, 'boxed'
                break
    for bm in reversed(boxed_matches):
        val = bm.group(1)
        num = _parse_boxed_val(val)
        if num is not None:
            return num, 'boxed'

    # ── RULE 3: Marker line → first number after marker ──────────────────────
    marker_re = re.compile(
        r'(Final Answer|Answer|So the answer is|Therefore|Thus|Hence)', re.I
    )
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        if not marker_re.search(line):
            continue
        # Split on marker, take text after it
        parts = marker_re.split(line, maxsplit=1)
        search_text = ''.join(parts[2:]) if len(parts) >= 3 else line
        # FIX: strip leading partial-bold fragment e.g. ":**" from "**Answer:**" split remainder
        # Strip any leading asterisks, colons, spaces to avoid phantom bold group at start
        search_text = re.sub(r'^[\*:\s]+', '', search_text)  # FIX: strip leading * : space
        # FIX: skip setup lines like "Thus for 8 sandwiches:" — after-marker text starts with "for" + ends with ":"
        if re.match(r'^for\s+', search_text, re.I) and search_text.strip().endswith(':'):
            continue
        # FIX: skip if the only content is a step label (e.g. "**Step 5: Final answer**")
        if re.search(r'\*\*\s*step\s*\d+\b[^*]*\*\*', search_text, re.I) and \
                not re.search(r'\d+.*[.!?,]', search_text):
            continue
        # FIX (Rule 3 bold extraction): find bold groups in search_text
        # Use strict approach: only find **...**  where content has no embedded ** 
        _r3_bold_groups = re.findall(r'\*\*([^*]+)\*\*', search_text)
        _r3_bold_nums = []
        for _bg in _r3_bold_groups:
            # FIX: skip comparison-operator bolds (e.g. "**≤ $10**", "**< N**")
            if re.match(r'^\s*[\u2264\u2265\u2248\u2260<>]', _bg):
                continue
            _bg_clean = _bg.replace('\\$', '$').replace('\\', '')
            # FIX: skip number-hyphen-word modifier bolds (e.g. "32-page", "100-meter")
            _first_num_m = re.search(NUMBER_RE, _bg_clean)
            if _first_num_m and re.match(r'\s*[-\u2013]\w', _bg_clean[_first_num_m.end():]):
                continue
            _bgnums = [normalize_number(t) for t in re.findall(NUMBER_RE, _bg_clean)]
            _bgnums = [n for n in _bgnums if n is not None]
            if _bgnums:
                _r3_bold_nums.append((_bg, _bgnums))
        # FIX: for weak markers (Thus/Hence/Therefore), skip if no bold nums and no final keywords
        _matched_marker_lower = (parts[1].lower().strip() if len(parts) >= 2 else '')
        _is_weak_marker = _matched_marker_lower in ('thus', 'hence', 'therefore')
        if _is_weak_marker and not _r3_bold_nums:
            _has_final_kw = re.search(r'\b(total|answer|altogether|final|result)\b', search_text, re.I)
            if not _has_final_kw:
                continue  # Skip weak-marker intermediate calc lines (e.g. "Hence M = 21")
        if _r3_bold_nums:
            # If line has "total of/altogether", take last bold number
            if re.search(r'\b(total\s+of|altogether|in\s+total)\b', search_text, re.I):
                _bg_last, _bns_last = _r3_bold_nums[-1]
                num = _bns_last[-1] if '=' in _bg_last else _bns_last[0]
                if num is not None:
                    return num, 'final_marker'
            # FIX: "i.e." / "which is" / "or" → unit conversion → pick LARGEST number
            # (e.g. "**227 min**, i.e., **3h47m**" → max(227,3)=227)
            _r3_is_conv = re.search(r'\b(which is|that is)\b|i\.e\.', search_text, re.I)
            _r3_is_or = re.search(r'\bor\b', search_text, re.I) and not re.search(r'\band\b', search_text, re.I)
            if (_r3_is_conv or _r3_is_or) and len(_r3_bold_nums) >= 2:
                _bg_last_cv, _bns_last_cv = _r3_bold_nums[-1]
                _bg_first_cv, _bns_first_cv = _r3_bold_nums[0]
                # For "or" between all-currency amounts → pick first
                if _r3_is_or and all(re.search(r'[\$\u20ac\xa3]', _bg) for _bg, _ in _r3_bold_nums):
                    return _bns_first_cv[0], 'final_marker'
                # Non-currency → currency conversion → pick last (e.g. 24500¢ → $245)
                if re.search(r'[\$\u20ac\xa3]', _bg_last_cv) and not re.search(r'[\$\u20ac\xa3]', _bg_first_cv):
                    return _bns_last_cv[0], 'final_marker'
                # Otherwise: unit conversion → pick LARGEST first-num across bolds
                return max(_bns[0] for _, _bns in _r3_bold_nums), 'final_marker'
            # FIX: single bold group with "and" and multiple nums → sum
            if len(_r3_bold_nums) == 1:
                _bg0, _bns0 = _r3_bold_nums[0]
                if re.search(r'\band\b', _bg0, re.I) and len(_bns0) >= 2:
                    return sum(_bns0), 'final_marker'
            # Sum parallel bold numbers joined by "and" (e.g. "27 selfies and 18 portraits" → 45)
            elif re.search(r'\band\b', search_text, re.I) and len(_r3_bold_nums) >= 2:
                _sum_vals_r3 = [_bns[0] for _, _bns in _r3_bold_nums]
                # FIX: if 3+ bolds and last = sum of rest, last IS the total — return it
                if len(_sum_vals_r3) >= 3 and _sum_vals_r3[-1] == sum(_sum_vals_r3[:-1]):
                    return _sum_vals_r3[-1], 'final_marker'
                return sum(_sum_vals_r3), 'final_marker'  # FIX: sum parallel bolds in marker line
            # single bold OR last bold group (for multi-bold without sum)
            _bg_last, _bns_last = _r3_bold_nums[-1]
            _bg_last_clean = _bg_last.replace('\\$', '$').replace('\\', '')
            if '=' in _bg_last_clean:
                # after-eq: take last number in this bold group
                num = _bns_last[-1]
            else:
                # FIX: strip :HH time suffix
                _raw_r3 = re.findall(NUMBER_RE, _bg_last_clean)[0]
                _after_r3 = _bg_last_clean[_bg_last_clean.find(_raw_r3) + len(_raw_r3):]
                if re.match(r':\d{2}\b', _after_r3):
                    _m_digits = re.match(r'\d+', _raw_r3)
                    _raw_r3 = _m_digits.group(0) if _m_digits else _raw_r3
                num = normalize_number(_raw_r3)
            if num is not None:
                return num, 'final_marker'
        # RULE 8: currency in search_text — skip \$ (LaTeX-escaped) as not real currency  # FIX
        # Only match real currency symbols (not backslash-escaped $)
        m_cur = re.search(r'(?<!\\)[\$€£]\s*(' + NUMBER_RE + r')', search_text)
        if m_cur:
            if '=' in search_text:
                after_eq = search_text.rsplit('=', 1)[-1]
                nums_aeq = [normalize_number(t) for t in re.findall(NUMBER_RE, after_eq)]
                nums_aeq = [n for n in nums_aeq if n is not None]
                if nums_aeq:
                    return nums_aeq[-1], 'final_marker'
            else:
                num = normalize_number(m_cur.group(1))
                if num is not None:
                    return num, 'final_marker'
        # FIX (Rule 3 after-equals): if search_text has "=" extract LAST number (the result)
        if '=' in search_text:
            nums_eq = [normalize_number(t) for t in re.findall(NUMBER_RE, search_text)]
            nums_eq = [n for n in nums_eq if n is not None]
            if nums_eq:
                return nums_eq[-1], 'final_marker'
        m_num = re.search(NUMBER_RE, search_text)
        if m_num:
            raw_num = m_num.group(0)
            after_num = search_text[m_num.end():]
            # FIX: skip number-hyphen modifier (e.g. "32-page", "100-meter") — not the answer
            if not re.match(r'\s*[-\u2013]\w', after_num):
                # FIX: strip :HH time suffix (e.g. "2:00 pm" → 2, avoids :00 being seen as decimal)
                if re.match(r':\d{2}\b', after_num):
                    m_digits = re.match(r'\d+', raw_num)
                    raw_num = m_digits.group(0) if m_digits else raw_num
                num = normalize_number(raw_num)
                if num is not None:
                    return num, 'final_marker'

    # Word-to-number map for bold word-numbers: **one**, **six years**
    WORD_NUM = {
        'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
        'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
        'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
    }

    # ── RULE 4 helper: extract answer from bold groups on a line ─────────────
    # FIX: use BOLD_GROUP_RE (simple [^*]+) to correctly find bold segments without cross-boundary issues
    BOLD_GROUP_RE = re.compile(r'\*\*([^*]+)\*\*')  # FIX: reliable bold group extraction

    def _extract_from_bold_groups(line, allow_assumption=False):
        """Extract the answer number from bold groups on this line.
        Returns (number, label) or (None, None).
        """
        if not allow_assumption and ASSUMPTION_RE.match(line):
            # FIX: even if ASSUMPTION_RE matches, try to extract if there's a bold number (e.g. "Assuming..., X scores **36 points**")
            if not re.search(r'\*\*[^*]*\d[^*]*\*\*', line):
                return None, None
        groups = BOLD_GROUP_RE.findall(line)  # FIX: list of bold text segments
        if not groups:
            return None, None
        # Skip if ALL bold groups are just step labels
        all_step = all(STEP_BOLD_RE.match(g) for g in groups)
        if all_step:
            return None, None
        # FIX: normalize \$ (escaped dollar in LaTeX) to $ for number matching
        def clean_bold(g):
            return g.replace('\\$', '$').replace('\\', '')
        # Collect bold groups that contain numbers
        num_groups = []
        for g in groups:
            if STEP_BOLD_RE.match(g):
                continue
            # FIX: skip comparison/inequality bold groups (e.g. "**≤ $10**", "**< 5**")
            if re.match(r'^\s*[\u2264\u2265\u2248\u2260<>≤≥≈≠]', g):
                continue
            gc = clean_bold(g)
            nums = [normalize_number(t) for t in re.findall(NUMBER_RE, gc)]
            nums = [n for n in nums if n is not None]
            if nums:
                num_groups.append((g, gc, nums))
        if not num_groups:
            return None, None
        # FIX: check for inline LaTeX math result on the line — prefer that over bold text label (e.g. "Total = \(40+20+100=160\) min, i.e. **2 hours**")
        m_inline_r4 = re.search(r'\\+\([^)]*=\s*(' + NUMBER_RE + r')[^)]*\\+\)', line)  # FIX: greedy = last =
        if m_inline_r4:
            # Only prefer inline if the line's bold groups have no '=' (i.e. bolds are labels not equations)
            all_bold_no_eq = all('=' not in g for g, _, _ in num_groups)
            # FIX: skip if inline math is inside a parenthetical aside e.g. "Sam is **6** now. (Ruby would be \(6x6=36\)...)"
            _r4_before = line[:m_inline_r4.start()]
            _r4_in_paren = _r4_before.count('(') > _r4_before.count(')')
            if not _r4_in_paren and all_bold_no_eq:
                num = normalize_number(m_inline_r4.group(1))
                if num is not None:
                    return num, 'bold_final'  # FIX: inline math result wins over decorative bold
        # FIX: if line contains "total of/altogether", use last bold number (answer, not addends)
        # Note: do NOT trigger on bare "combined" since "(a combined N kg)" means N is not bold
        if re.search(r'\b(total\s+of|altogether|in\s+total)\b', line, re.I):
            g_last, gc_last, nums_last = num_groups[-1]
            num = nums_last[-1] if '=' in gc_last else nums_last[0]
            return num, 'bold_final'  # FIX: removed 'combined' trigger to avoid false positives
        # FIX (sum-and inside single bold): "**3 onions and 1 potato**" → 3+1=4 ... but wait that's wrong
        # instead check for "and" WITHIN a single bold AND the bold is the ONLY numeric group
        if len(num_groups) == 1:
            g0, gc0, nums0 = num_groups[0]
            # FIX (after-equals in single bold): "**Total = 84 + 88 = 172**" → 172
            if '=' in gc0:
                return nums0[-1], 'bold_final'
            # FIX (sum-and inside single bold): "**86 glasses and 42 plates**" → 128
            if re.search(r'\band\b', g0, re.I) and len(nums0) >= 2:
                return sum(nums0), 'bold_final'
            # FIX: single bold with "from now/will be" context → take last number (the answer, not the offset)
            if len(nums0) >= 2 and re.search(r'\b(from now|will be)\b', g0, re.I):
                return nums0[-1], 'bold_final'
            # FIX: single bold ending with currency amount → last number is the cost answer
            if len(nums0) >= 2 and re.search(r'[\$\u20ac\xa3]\s*' + NUMBER_RE + r'\s*[.,]?\s*$', gc0):
                return nums0[-1], 'bold_final'
            return nums0[0], 'bold_final'
        # Multiple bold groups with numbers
        # FIX: unit-conversion patterns (which is / that is / i.e. / **N**=**M** / or) →
        #   pick LARGEST first-number (minutes > hours numerically; seconds > minutes; etc.)
        #   Special cases:
        #     - "or" between all-currency bolds → pick FIRST (cost-split alternatives)
        #     - Non-currency → currency last → pick LAST (e.g. 24,500¢ → $245)
        _is_conversion = re.search(r'\b(which is|that is)\b|i\.e\.|\*\*[^*]+\*\*\s*=\s*\*\*', line, re.I)
        _is_or_alt = re.search(r'\bor\b', line, re.I) and not re.search(r'\band\b', line, re.I)
        if (_is_conversion or _is_or_alt) and len(num_groups) >= 2:
            g_last_cv, gc_last_cv, nums_last_cv = num_groups[-1]
            g_first_cv, gc_first_cv, nums_first_cv = num_groups[0]
            # For "or" between all-currency amounts → pick first
            if _is_or_alt and all(re.search(r'[\$\u20ac\xa3]', g) for g, _, _ in num_groups):
                return nums_first_cv[0], 'bold_final'
            # Non-currency → currency conversion (e.g. 24500¢ → $245) → pick last
            if re.search(r'[\$\u20ac\xa3]', g_last_cv) and not re.search(r'[\$\u20ac\xa3]', g_first_cv):
                return nums_last_cv[0], 'bold_final'
            # Otherwise (unit conversion like min↔h, s↔min): pick LARGEST first-number
            return max(nums[0] for _, _, nums in num_groups), 'bold_final'
        if re.search(r'\band\b', line, re.I) and len(num_groups) >= 2:  # FIX: simplified check
            # FIX: only sum when bold groups are directly adjacent (≤4 words between them)
            _bold_spans = [(m.start(), m.end()) for m in BOLD_GROUP_RE.finditer(line)
                           if any(m.group(1) == ng[0] for ng in num_groups)]
            if len(_bold_spans) >= 2:
                _between = line[_bold_spans[0][1]:_bold_spans[1][0]]
                _between_words = len(re.findall(r'\b\w+\b', _between))
                if _between_words > 4:  # too much prose between bolds → don't sum, take last
                    g_last, gc_last, nums_last = num_groups[-1]
                    num = nums_last[-1] if '=' in gc_last else nums_last[0]
                    return num, 'bold_final'
            # Sum closely-adjacent parallel bold numbers
            _sum = sum(nums[0] for _, _, nums in num_groups)
            return _sum, 'bold_final'  # FIX: sum parallel bolds when "and" found in line
        # FIX: default — take LAST bold group with a number, use last number if '=' in group
        g_last, gc_last, nums_last = num_groups[-1]
        num = nums_last[-1] if '=' in gc_last else nums_last[0]
        return num, 'bold_final'

    # ── RULE 4a: Bold with soft marker word ──────────────────────────────────
    # FIX: single merged loop — find MOST RECENT line with a bold number (reversed = last first).
    # Soft marker preference REMOVED to avoid stale intermediate-calculation lines winning.
    for line in reversed(last_lines):
        if '|' in line:
            continue
        num, lbl = _extract_from_bold_groups(line)
        if num is not None:
            return num, lbl  # FIX: return immediately from most recent bold line
        # FIX: \mathbf{N} — LaTeX bold math (not ** markdown) on this line
        _m_mathbf = re.search(r'\\mathbf\{([^}]*)\}', line)
        if _m_mathbf:
            _mb_inner = re.sub(r'\\[a-zA-Z]+', ' ', _m_mathbf.group(1))
            _mb_inner = re.sub(r'[{};\\,]', ' ', _mb_inner)
            _mb_nums = [normalize_number(t) for t in re.findall(NUMBER_RE, _mb_inner)]
            _mb_nums = [n for n in _mb_nums if n is not None]
            if _mb_nums:
                return _mb_nums[-1], 'bold_final'

    # FIX: inline math \(... = N\) on a no-bold line — restricted to last 10 lines to avoid
    # intermediate explanation steps (e.g. "each is \(1/2×3=1.5\) hours" in reasoning section)
    for line in reversed(last_lines[-10:]):
        if '|' in line:
            continue
        _m_im_nb = re.search(r'\\+\(\s*[^)]*=\s*(' + NUMBER_RE + r')[^)]*\\+\)', line)
        if _m_im_nb:
            # FIX: skip if inside a prose parenthetical
            _im_nb_before = line[:_m_im_nb.start()]
            _im_nb_in_paren = _im_nb_before.count('(') > _im_nb_before.count(')')
            # FIX: skip if a comparison parenthetical follows closing \) (e.g. " in (< 2 in).")
            _im_nb_after = line[_m_im_nb.end():]
            _im_nb_post_paren = '(' in _im_nb_after[:12]
            if not _im_nb_in_paren and not _im_nb_post_paren:
                _im_nb_num = normalize_number(_m_im_nb.group(1))
                if _im_nb_num is not None:
                    return _im_nb_num, 'latex_inline'

    # ── RULE 4b removed — merged into Rule 4a above ────────────────────────────

    # ── RULE 4c: Currency in last 20 lines ───────────────────────────────────
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        m_cur = re.search(r'[\$€£]\s*(' + NUMBER_RE + r')', line)
        if m_cur:
            # FIX (Rule 4c after-equals): if the line has "=", take the last number after final "=" (result side)
            if '=' in line:
                after_eq = line.rsplit('=', 1)[-1]
                nums_aeq = [normalize_number(t) for t in re.findall(NUMBER_RE, after_eq)]
                nums_aeq = [n for n in nums_aeq if n is not None]
                if nums_aeq:
                    return nums_aeq[-1], 'currency_final'
            else:
                # FIX (Rule 4c last-currency no-eq): take LAST currency amount, not first (e.g. "bill is $100, leaves with $25" → 25)
                all_curs = [normalize_number(t) for t in re.findall(r'(?<=[\$€£])\s*(' + NUMBER_RE + r')', line)]
                all_curs = [n for n in all_curs if n is not None]
                if all_curs:
                    return all_curs[-1], 'currency_final'  # FIX: use last currency value

    # ── RULE 4d: Bold label followed by number: **Total items in use:** 56. ──
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        # Pattern: **<label>:** N  (number after bold label)
        m_label = re.search(r'\*\*[^*]+\*\*\s*:?\s*(' + NUMBER_RE + r')\s*\.?\s*$', line)
        if m_label:
            num = normalize_number(m_label.group(1))
            if num is not None:
                return num, 'bold_label'

    # ── RULE 4e: Bold word-number: **one**, **six years** ────────────────────
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        m_word = re.search(r'\*\*\s*(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b[^*]*\*\*', line, re.I)
        if m_word:
            word = m_word.group(1).lower()
            num = WORD_NUM.get(word)
            if num is not None:
                return num, 'bold_word'

    # ── RULE 5: Last-5-lines fallback ────────────────────────────────────────
    # Covers: plain sentences ending with number+unit, LaTeX inline math result,
    #         sentences like "X penguins are still left sunbathing."
    SKIP_RE = re.compile(r'^\s*(\d+[\).:]|step\s*\d+|[-•]|\|)', re.I)

    for line in reversed(last_lines[-5:]):
        line_s = line.strip()
        if not line_s or '|' in line_s:
            continue
        if ASSUMPTION_RE.match(line_s):
            continue
        # FIX (Rule 5 SKIP bypass): don't skip bullet/numbered lines if they contain = or Total/Answer OR start with negative number
        if SKIP_RE.match(line_s):
            if not (re.search(r'=', line_s) or re.search(r'\b(total|answer)\b', line_s, re.I)
                    or re.match(r'^-\d', line_s)):  # FIX: allow negative plain number lines
                continue
        # Skip pure intermediate-step lines (multiple = signs, no sentence words)
        # 5a: LaTeX line ending with = N \text{unit} or = N\ \text{unit}.
        # Also handles plain arithmetic: "Total teachers present: 69 + 9 = 78."
        # FIX: also allow \% and bare % after number (e.g. "= 70\%."), and \<space> thin-space
        m_latex_block = re.search(
            r'=\s*(' + NUMBER_RE + r')\s*(?:\\+[a-zA-Z%]+(?:\{[^}]*\})?|\\{1,2}\s*)*[°%]?\.?\s*$',
            line_s
        )
        if m_latex_block:
            num = normalize_number(m_latex_block.group(1))
            if num is not None:
                return num, 'latex_result'

        # 5b: LaTeX inline math: \(a + b = N\) or \(a = N\)
        # FIX (Rule 5b relax): allow trailing characters after closing \) (e.g. "= 13 \text{ minutes}\).")
        m_inline = re.search(
            r'\\+\(\s*[^)]*=\s*(' + NUMBER_RE + r')[^)]*\\+\)',  # FIX: greedy = last =
            line_s
        )
        if m_inline:
            # FIX: skip if the captured number is immediately followed by \% or % (it's a ratio, not answer)
            _after_inline = m_inline.group(0)[m_inline.start(1) - m_inline.start() + len(m_inline.group(1)):]
            if not re.match(r'\s*\\?%', _after_inline):
                num = normalize_number(m_inline.group(1))
                if num is not None:
                    return num, 'latex_inline'
        # FIX (Rule 5b-no-eq): \(N\) or \(\displaystyle N\) or \(N \text{...}\) — plain number in parens
        # Allow optional LaTeX prefix commands (\displaystyle, \textstyle, etc.) before the number
        # Use (?:[^)\\]|\\[^)]) to avoid consuming the closing \) backslash
        m_inline_plain = re.search(r'\\+\(\s*(?:\\+[a-zA-Z]+\s*)*(' + NUMBER_RE + r')(?:[^)\\]|\\[^)])*\\+\)', line_s)
        if m_inline_plain:
            num = normalize_number(m_inline_plain.group(1))
            if num is not None:
                return num, 'latex_inline'

        # 5c: Sentence starting with number + words (unit sentence)
        # e.g. "14 people got on the bus", "5 inches of rain", "120 pounds of carrots"
        # Also: "There are 15 sheep", "Peter has 8 more", "There are 10 more"
        all_nums = re.findall(NUMBER_RE, line_s)
        if len(all_nums) <= 3:
            # "There are/is N ..." or "X has/have N ..."
            m_there = re.search(
                r'(?:There\s+(?:are|is)|\w+\s+(?:has|have|can|will|got|gets?|contains?|holds?|needs?|takes?|makes?|earns?|costs?|weighs?|measures?|equals?|totals?))\s+(' + NUMBER_RE + r')\b',
                line_s, re.I
            )
            if m_there:
                num = normalize_number(m_there.group(1))
                if num is not None:
                    return num, 'tail_sentence'

            m_start = re.match(r'^(' + NUMBER_RE + r')\s+\w', line_s)
            if m_start:
                num = normalize_number(m_start.group(1))
                if num is not None:
                    return num, 'tail_sentence'

            # Sentence ending with number + optional words + punctuation
            m_sent = re.search(
                r'(' + NUMBER_RE + r')\s+(?:\w+\s+){0,6}(?:\w+)\s*[.!]?\s*$',
                line_s
            )
            if m_sent:
                num = normalize_number(m_sent.group(1))
                if num is not None:
                    return num, 'tail_sentence'

        # 5d: Plain number only line
        # FIX: also accept leading negative sign for plain number (e.g. "-3 °C")
        # FIX: use re.fullmatch-style by allowing optional unit chars and unicode degree symbol
        m_plain = re.match(r'^((?:[-\u2212\u2013]\s*)?(?:[\$])?(?:\d{1,3}(?:(?:,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?)\s*(?:[°\u00b0a-zA-Z\.%]*)?\s*$', line_s)  # FIX: explicit negative prefix
        if m_plain:
            num = normalize_number(m_plain.group(1))
            if num is not None:
                return num, 'tail_plain'

    return None, 'none'


# --- Improved truncated detection ---
def has_final_marker(text):
    if not text or not isinstance(text, str):
        return False
    tail = text[-1200:] if len(text) > 1200 else text
    if re.search(r'####\s*-?[$]?[\d,]+', tail):
        return True
    if re.search(r'\\boxed\{[^}]+\}', tail):
        return True
    if re.search(r'(Answer|Final|Therefore|Thus|Hence|So the answer is)', tail, re.I):
        # Accept if line with marker has number
        for line in tail.splitlines()[-15:]:
            if re.search(r'(Answer|Final|Therefore|Thus|Hence|So the answer is)', line, re.I):
                if re.search(r'(-?[$]?[\d,]+(?:\.\d+)?)', line):
                    return True
    # Bold number in tail
    if re.search(r'\*\*\s*-?[$]?[\d,]+(?:\.\d+)?\s*\*\*', tail):
        return True
    return False

def is_truly_truncated(text, finish_reason, completion_tokens, max_tokens):
    # If finish_reason != length, not truncated
    if finish_reason != 'length':
        return False
    # If has final marker, not truncated
    if has_final_marker(text):
        return False
    # Check for open sentence/unfinished output
    if not text or not isinstance(text, str):
        return True
    tail = text.strip()[-120:]
    # Unbalanced brackets
    if tail.count('{') > tail.count('}') or tail.count('[') > tail.count(']') or tail.count('(') > tail.count(')'):
        return True
    # Ends with open math or incomplete
    if re.search(r'[=:(,\[\{]$|\\\[$', tail):
        return True
    # Ends with incomplete word or abrupt stop
    if not re.search(r'[.!?}\]\)]$', tail):
        return True
    return False

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    persona_dirs = [p for p in os.listdir(args.input_root) if p in PERSONA_LIST]
    all_rows = []
    error_counter = Counter()
    persona_counter = Counter()
    for persona in persona_dirs:
        folder = os.path.join(args.input_root, persona)
        files = [f for f in os.listdir(folder) if f.startswith('gsm8k_') and f.endswith('.json')]
        if len(files) != 1319:
            print(f"WARNING: Persona {persona} has {len(files)} JSON files (expected 1319)")
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding='utf-8') as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"SKIP: {persona}/{fname} (invalid or empty JSON) -- {e}", file=sys.stderr)
                continue
            inp = data.get('input', {})
            resp = data.get('response', {})
            req = data.get('request', {})
            meta = data.get('metadata', {})
            gt = inp.get('ground_truth', '')
            gt_num = None
            m = re.search(r'####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)', gt)
            if m:
                gt_num = normalize_number(m.group(1))
            qid = inp.get('question_id', '')
            dataset = inp.get('dataset', '')
            # Robust response text extraction
            content = get_response_text(resp)
            # Guardrail: log if completion_tokens > 0 but content is empty
            def safe_get(d, key, default=None):
                return d[key] if isinstance(d, dict) and key in d else default
            try:
                raw_api = safe_get(resp, 'raw_api_response', {})
                usage = safe_get(raw_api, 'usage', {})
                ctok = safe_get(usage, 'completion_tokens', 0)
                if (ctok is not None and str(ctok).isdigit() and int(ctok) > 0) and (not content or not content.strip()):
                    keys_top = list(resp.keys()) if isinstance(resp, dict) else []
                    keys_raw_api = list(raw_api.keys()) if isinstance(raw_api, dict) else []
                    choices = safe_get(raw_api, 'choices', [{}]) if isinstance(raw_api, dict) else [{}]
                    keys_choices = list(choices[0].keys()) if choices and isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict) else []
                    msg = safe_get(choices[0], 'message', {}) if choices and isinstance(choices[0], dict) else {}
                    keys_msg = list(msg.keys()) if isinstance(msg, dict) else []
                    reasoning_details = safe_get(msg, 'reasoning_details', None)
                    print(f"WARNING: completion_tokens={ctok} but response_content_full is empty for {persona}/{fname}", file=sys.stderr)
                    print(f"  resp keys: {keys_top}", file=sys.stderr)
                    print(f"  raw_api_response keys: {keys_raw_api}", file=sys.stderr)
                    print(f"  choices[0] keys: {keys_choices}", file=sys.stderr)
                    print(f"  message keys: {keys_msg}", file=sys.stderr)
                    if isinstance(reasoning_details, list):
                        print(f"  reasoning_details present, len={len(reasoning_details)}", file=sys.stderr)
                    else:
                        print(f"  reasoning_details not present", file=sys.stderr)
            except Exception:
                pass
            extracted, extraction_reason = extract_final_answer(content)
            # Debug: warn if extraction failed but text has digits
            if extracted is None and content and re.search(r'\d', content):
                tail_preview = content.strip()[-80:].replace('\n', ' ')
                print(f"DEBUG None extraction: {persona}/{fname} | tail: {tail_preview!r}", file=sys.stderr)
            is_correct = (extracted == gt_num) if (extracted is not None and gt_num is not None) else False
            has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', content))
            has_hashes = bool(re.search(r'####\s*-?[$]?[\d,]+', content))
            boxed_match = has_boxed and (extracted == gt_num) and extraction_reason == 'boxed'
            hashes_match = has_hashes and (extracted == gt_num) and extraction_reason == 'hashes'
            finish_reason = None
            try:
                raw_api = safe_get(resp, 'raw_api_response', {})
                choices = safe_get(raw_api, 'choices', [{}])
                if choices and isinstance(choices, list):
                    finish_reason = safe_get(choices[0], 'finish_reason', None)
            except Exception:
                pass
            max_tokens = safe_get(safe_get(req, 'parameters', {}), 'max_tokens', None)
            usage = safe_get(raw_api, 'usage', {})
            completion_tokens = safe_get(usage, 'completion_tokens', None)
            prompt_tokens = safe_get(usage, 'prompt_tokens', None)
            total_tokens = safe_get(usage, 'total_tokens', None)
            reasoning_tokens = safe_get(usage, 'reasoning_tokens', None)
            truncated_by_metadata = (finish_reason == 'length')
            truncated_by_content = is_truly_truncated(content, finish_reason, completion_tokens, max_tokens)
            has_final = has_final_marker(content)
            text_tail = content.strip()[-120:] if content else ''
            extraction_issue = (extracted is None)
            # Error category logic (extraction first, new logic)
            if extracted is not None:
                error_category = 'ok'
            elif truncated_by_content:
                error_category = 'truncated_output'
            else:
                error_category = 'no_number_found'
            error_counter[(persona, error_category)] += 1
            persona_counter[persona] += 1
            row = [persona, os.path.relpath(fpath, args.input_root), qid, dataset, gt, gt_num, content, extracted, is_correct, has_boxed, has_hashes, boxed_match, hashes_match, extraction_issue, error_category, extraction_reason, finish_reason, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, truncated_by_metadata, truncated_by_content, has_final, text_tail]
            all_rows.append(row)
    # Write results.csv
    out_csv = os.path.join(args.report_dir, 'results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(["persona_folder","file","question_id","dataset","ground_truth","ground_truth_answer","response_content_full","extracted_answer","is_correct","has_boxed","has_hashes","boxed_match","hashes_match","extraction_issue","error_category","extraction_reason","finish_reason","prompt_tokens","completion_tokens","reasoning_tokens","total_tokens","truncated_by_metadata","truncated_by_content","has_final_marker","text_tail"])
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {out_csv}")
    print("Error category breakdown:")
    for persona in PERSONA_LIST:
        print(f"{persona:20s}", end=' ')
        for cat in ['ok','truncated_output','no_number_found','picked_intermediate_step_bug','other_extraction_error']:
            print(f"{cat}: {error_counter.get((persona,cat),0):4d}", end=' | ')
        print()

def run_unit_tests():
    """Quick sanity check for normalize_number and extract_final_answer."""
    print("=== normalize_number tests ===")
    norm_cases = [
        ("140{,}000", 140000),
        ("3,430",     3430),
        ("3528",      3528),
        ("$3{,}528",  3528),
        ("85 000",    85000),
        ("3.65",      3.65),
    ]
    all_ok = True
    for raw, expected in norm_cases:
        got = normalize_number(raw)
        ok = (got == expected)
        print(f"  {'OK' if ok else 'FAIL'} normalize_number({raw!r}) -> {got} (expected {expected})")
        if not ok:
            all_ok = False

    print("\n=== extract_final_answer tests ===")
    efa_cases = [
        # (text, expected_number)
        ("#### 140{,}000",                                          140000),
        ("#### 3,430",                                              3430),
        ("#### 3528",                                               3528),
        ("\\boxed{$3{,}528}",                                       3528),
        ("\\boxed{85 000}",                                         85000),
        ("\\boxed{3.65}",                                           3.65),
        ("**Answer: 3 430 people**",                                3430),
        ("Answer: $25,000",                                         25000),
        ("So the answer is 1,800 pounds",                           1800),
        ("Total = 25 + 50 = **75 pens**.\nSo Alain has **75 pens**.", 75),
        # wrong_answers_sample cases
        ("\\[\n490 + 2940 = **3 430**\n\\]\n\nSo after three days, **3 430 people**", 3430),
        ("\\[\n\\boxed{\\$3{,}528}\n\\]",                            3528),
        ("**Answer: $139**",                                        139),
        ("So James wasn't out any money; he actually made a **\\$110 profit**.", 110),
        # Assumption lines — must NOT be picked
        # "If you assume 30 days, answer is 12" → 12 is correct (marker wins), but...
        # Pure assumption line with no real marker should return None
        ("If you assume 30 days per month, the total would be 360.", None),
        ("Assumption: working 5 days a week means 260 days per year.", None),
        ("Note: if the store is open 7 days a week, revenue is 700.", None),
        # Assumption line AFTER real answer — real answer should win
        ("The answer is **42**.\n\nNote: if you assume leap year, it would be 43.", 42),
        ("So the answer is 1800.\n\nIf you consider weekends, it could be 2100.", 1800),
        # bold_sentence patterns (165 cases)
        ("€1500\u202f–\u202f€1423 = **€77**.\n\nSo she will have **€77 left** to spend on clothing.", 77),
        ("To get an average of 93, Brinley needs a **score of 98** on the sixth test.", 98),
        ("\\(251 - 197 = 54\\).\n\n**Jojo scored 54 points.**", 54),
        ("**Each child gets 2 chocolate bars.**", 2),
        ("**James took 59 minutes to run.**", 59),
        # boxed_with_unit patterns (100 cases)
        ("\\[\n222\\text{ s} - 220\\text{ s} = \\boxed{2\\text{ seconds}}.\n\\]", 2),
        ("\\[\n\\boxed{68\\text{ square feet}}\n\\]", 68),
        ("\\[\n\\boxed{10 \\text{ inches}}\n\\]", 10),
        ("\\[\n\\boxed{6{,}000 \\text{ gallons}}\n\\]", 6000),
        ("\\[\n\\boxed{32\\text{ feet}}\n\\]", 32),
        # has_number_other: LaTeX inline math result (63 cases)
        ("Hence, the number of good oranges is  \n\\(25 - 8 = 17\\).", 17),
        ("Together they eat \\(5 + 15 = 20\\) cookies.", 20),
        ("Adding them together:  \n\\(35 + 70 = 105\\) questions were left incomplete in total.", 105),
        ("Winner (Blake) ran \\(3{,}000 - 2{,}920 = 80\\) yards farther.", 80),
        ("Total flowers planted: \\(90 + 50 = 140\\).", 140),
        # number_with_unit_word (24 cases)
        ("120 pounds of carrots will not be used.", 120),
        ("12 penguins are still left sunbathing.", 12),
        ("14 people got on the bus at the terminal.", 14),
        ("5 inches of rain will be expected on Tuesday.", 5),
        # latex_block_end (3 cases)
        ("\\[\n80\\ \\text{mi} + 150\\ \\text{mi} = 230\\ \\text{mi}.\n\\]", 230),
        ("\\[\n600 - 6 = 594 \\text{ feet of line.}\n\\]", 594),
        ("\\[\n\\text{Average speed}= \\frac{250\\text{ miles}}{5\\text{ hours}} = 50\\text{ mph}\n\\]", 50),
        # boxed with double-backslash (JSON-escaped)
        ("\\\\[\n\\\\boxed{32\\\\ \\\\text{feet}}\n\\\\]", 32),
        ("\\\\[\n0.12 \\\\times 100\\\\% = \\\\boxed{12\\\\%}.\n\\\\]", 12),
        ("\\\\[\n\\\\boxed{72\\\\ \\\\text{Mb per hour}}\n\\\\]", 72),
        # has_number_other: sentence with number
        ("There are 15 sheep on Lee's farm.", 15),
        ("Peter has 8 more chocolate bars in each box than Martha.", 8),
        ("Total teachers present: 69\u202f+\u202f9\u202f=\u202f78.", 78),
        ("There are 10 more floral shirts with no buttons than white shirts.", 10),
        # bold word-number
        ("He can see **one** more movie.", 1),
        ("Each child can have **one** bowl of soup for lunch.", 1),
        ("a tire would last **six years** under these driving conditions.", 6),
        # Total items bold
        ("**Total items in use:** 56.", 56),
        # FIX: new targeted failure cases
        # gsm8k_00474: boxed with % — Rule 2 should fire before Rule 4b sees Step 5
        ("**Step 5: Final answer**\n\\[ \\boxed{10\\%} \\]", 10),
        # gsm8k_00515: LaTeX inline currency \($90{,}000\) in marker line
        ("**Answer:** The house cost \\(\\$90{,}000\\).", 90000),
        # gsm8k_00754: boxed with "and" — FIX 1 sums 86+42
        ("\\[ \\boxed{86 \\text{ glasses and } 42 \\text{ plates}} \\]", 128),
        # gsm8k_00800: marker line with bold $4 — FIX 3 prefers bold currency over "3 loaves"
        ("**Answer:** 3 loaves of bread cost **$4 more** than 2 bagels.", 4),
        # gsm8k_00907: boxed with "and" — FIX 1 sums 4+6
        ("\\[ \\boxed{4 \\text{ blue and } 6 \\text{ red}} \\]", 10),
        # gsm8k_01002: time format — Rule 3 picks 2 from "2:00 pm" (NUMBER_RE stops at :)
        ("latest start time:**\n2:00 pm.\n**Answer:** 2:00 pm.", 2),
        # gsm8k_01219: "or" in parenthetical after marker — Rule 3 takes first number 225
        ("Answer: 225 minutes (or 3 hours and 45 minutes)", 225),
        # gsm8k_01223: two boxed with "or" between them — FIX 4 returns FIRST boxed (350)
        ("\\[ \\boxed{350 \\text{ inches}} \\quad \\text{or} \\quad \\boxed{9.72 \\text{ yards}} \\]", 350),
        # FIX: new dataset cases (3-tuple form: name, text, expected)
        # OpenAI GPT-OSS No-Extract Cases
        ("Case_1_0078", "- Total teachers present: 69 + 9 = 78.", 78),
        ("Case_7_0095", "The average score is \\(95\\).", 95),
        ("Case_8_minus3", "-3 \u00b0C", -3),
        ("Case_9_0013", "Total cassette time = \\(5 \\text{ min} + 8 \\text{ min} = 13 \\text{ minutes}\\).", 13),
        ("Case_12_0060", "- Net snowballs: \\(100 - 40 = 60\\).", 60),
        # OpenAI GPT-OSS Wrong-Extract Cases
        ("gsm8k_00001", "**Answer: Janet makes \\$18 every day at the farmers' market.**", 18),
        ("gsm8k_00008", "Total time = \\(40 + 20 + 100 = 160\\) minutes, i.e. **2 hours 40 minutes**.", 160),
        ("gsm8k_00063", "- 50 % \u00d7 $50,000 = $25,000.", 25000),
        ("gsm8k_00103", "Total water glasses in a week: **20 + 6 = 26 glasses**.", 26),
        ("gsm8k_00116", "**Total cost** = 10 \u00d7 $10 + 5 \u00d7 $16 = $100 + $80 = **$180**\n**Profit** = $270 \u2013 $180 = **$90**.", 90),
        ("gsm8k_00139", "\\(35 + 70 = 105\\) questions were left incomplete in total.", 105),
        ("gsm8k_00207", "**Total distance = 500 km + 360 km = 860 km**.", 860),
        ("gsm8k_00259", "Total amount to pay:\n\\(\\$4.00 + \\$2.00 = \\$6.00\\).", 6),
        ("gsm8k_00298", "**Sam was 55 crates short of his weekly target.**", 55),
        ("gsm8k_00334", "**Cost difference:** $3,450 \u2013 $3,350 = **$100** (Furniture United's offer is $100 more expensive).", 100),
        ("gsm8k_00354", "\\text{Sightseeing time } = 0.40 \\times 20\\text{ h} = 8\\text{ h}\n\\]\n**Answer: 8 hours.**", 8),
        ("gsm8k_00447", "Number of trays that can be filled = 50 eggs \u00f7 25 eggs/tray = **2 trays**.", 2),
        ("gsm8k_00467", "Kelly paid \\( \\$9.00 - \\$8.00 = \\$1.00\\) more than Becky.", 1),
        ("gsm8k_00490", "The average highest temperature is **-10 \u00b0F**.", -10),
        ("gsm8k_00500", "n = \\frac{200}{20} = 10\n\\]\n**Answer:** 10 guests. At 10 guests, both venues cost \\$250.", 10),
        ("gsm8k_00510", "Sophia bought **3 onions and 1 potato** \u2013 a total of **4 items**.", 4),
        ("gsm8k_00521", "Total per grandchild = 2 + 4 + 12 + 1 + 2 = **21 skeins**.\nWith 3 triplets, she needs 21 \u00d7 3 = **63 skeins** of wool.", 63),
        ("gsm8k_00529", "**Total number of sofas and chairs = 84 + 88 = 172**.", 172),
        ("gsm8k_00533", "The age difference now is 47 \u2013 22 = **25 years**.", 25),
        ("gsm8k_00544", "So Jairus gets \\( \\$16 - \\$10 = \\$6\\) more than Jenny.", 6),
        ("gsm8k_00552", "She pays with a $50 bill, so her change is\n$50 \u2212 $33 = **$17**.", 17),
        ("gsm8k_00625", "The basketball coach spent \\( \\$112 - \\$27 = \\$85\\) more than the baseball coach.", 85),
        ("gsm8k_00651", "The difference is **$540 \u2013 $398 = $142**.", 142),
        ("gsm8k_00660", "**Difference:**\n\\( \\$10 - \\$7 = \\$3 \\)\n\\[\n\\boxed{\\text{It costs \\$3 more for the first mixture.}}\n\\]", 3),
        ("gsm8k_00667", "If they pool their money and split it evenly, each person would receive **$120**.", 120),
        ("gsm8k_00691", "So, ten years later, Rong has **2 400 coins** and Neil has **3 360 coins**.", 5760),
        ("gsm8k_00692", "John's total bill is $100, so he leaves with $25.", 25),
        ("gsm8k_00711", "**Final bill:** $50.00 \u2013 $5.00 = **$45.00**.", 45),
        ("gsm8k_00720", "So she gave **75 kg to her cousin** and **100 kg to her brother** (a combined 175 kg).", 175),
        ("gsm8k_00740", "4. **Add the two totals**\n   - Total spent = \\(50 + 48 = \\$98\\).\nSo Cecelia spent **$98** in total.", 98),
        ("gsm8k_00802", "**Total trees planted = 1,000 + 240 = 1,240 trees.**", 1240),
        ("gsm8k_00828", "Adding the shipping cost: \\$8 (book loss) + \\$3 (shipping) = **\\$11**.", 11),
        ("gsm8k_00899", "50\\ \\text{miles} + 60\\ \\text{miles} = 110\\ \\text{miles}.", 110),
        ("gsm8k_00904", "**Total material needed = 10 yards + 6 yards = 16 yards**.", 16),
        ("gsm8k_00937", "**Savings** = 1 600 \u2013 1 350 = **250 pesos**.", 250),
        ("gsm8k_01018", "**Answer:** Olivia had **27 selfies** and **18 portraits**.", 45),
        ("gsm8k_01022", "Assuming each question is worth 1 point, Aries is sure to score **36 points**.", 36),
        ("gsm8k_01097", "People helped = 6 people/day \u00d7 40 days = **240 people**.", 240),
        ("gsm8k_01114", "The temperature in the morning was **\u20133 \u00b0C**.", -3),
        ("gsm8k_01139", "470 cows + 230 chickens = **700 animals**.", 700),
        ("gsm8k_01155", "Total per week: **20 + 10 = 30 carrots**.", 30),
        ("gsm8k_01167", "So an avocado tree can produce **2 350 fruits** over its first ten years of life.", 2350),
        ("gsm8k_01197", "So in total she bought **25 + 50 = 75 pens**.\nAssuming Alain has all the pens she bought, he has **75 pens**.", 75),
        ("gsm8k_01262", "Over 5 weeks: 5 \u00d7 $47.00 = **$235.00**.", 235),
        ("gsm8k_01304", "Since \\(28\\) days is exactly \\(4\\) weeks, it takes Nate **4 weeks** to fill all the holes.", 4),
        ("gsm8k_00010_imp", "**Total earnings for the week:** \\$400 + \\$60 = **\\$460**.", 460),
        # ── GPT-OSS-120B: boxed degree/text/approx patterns (30 no-extract cases) ──
        # Case 8,19,26: \boxed{90^\circ}
        ("box_degree_circ",    "\\boxed{90^\\circ}", 90),
        # Case 26: \boxed{90^\text{o}} — ^\text{o} form
        ("box_degree_text_o",  "\\boxed{90^\\text{o}}", 90),
        # Case 4,9,11,14,17,24,27: \boxed{\text{Peter is 50 years old}}
        ("box_text_name_num",  "\\boxed{Peter\\text{ is }50\\text{ years old}}", 50),
        # Case 1: \boxed{160\text{min} = 2\text{h} 40\text{min}} — first number before =
        ("box_eq_first_num",   "\\boxed{160\\ \\text{minutes} \\;=\\; 2\\ \\text{hours}\\;40\\text{min}}", 160),
        # Case 2: \boxed{\text{..}2400..\text{..}5760} — last of multiple
        ("box_multi_last",     "\\boxed{\\text{Rong: }2400\\text{ coins},\\; \\text{Neil: }3360\\text{ coins},\\; \\text{Combined: }5760\\text{ coins}}", 5760),
        # Case 5,15,29: \boxed{\frac{640}{21}\approx 30.5} — prefer approx value
        ("box_approx_val",     "\\boxed{\\displaystyle \\frac{640}{21}\\text{ hours}\\;\\approx\\;30.5\\text{ hours per week}}", 30.5),
        # Case 6: \boxed{36\frac{4}{11}\text{s}\;(about 36.36)} — last number
        ("box_mixed_approx",   "\\boxed{36\\frac{4}{11}\\text{ seconds}\\;(\\text{about }36.36\\text{ s})}", 36.36),
        # Case 25: \boxed{\frac{400}{11}\;(\approx 36.36)} — approx wins
        ("box_frac_approx2",   "\\boxed{\\,\\displaystyle \\frac{400}{11}\\text{ seconds}\\;(\\approx 36.36\\text{ s})\\,}", 36.36),
        # Case 28: \boxed{14^\circ\text{lower...}} — degree + text suffix
        ("box_degree_lower",   "\\boxed{14^\\circ\\text{ lower than the starting temperature}}", 14),
        # Case 10,12: \boxed{\text{Sam is 6 years old now.}} — pure-text box, num in \text{}
        ("box_text_only_6",    "\\boxed{\\text{Sam is 6 years old now.}}", 6),
        # Case 12: \boxed{\,\text{The winner runs 80 yards farther.}\,}
        ("box_text_only_80",   "\\boxed{\\,\\text{The winner runs 80 yards farther than the loser.}\\,}", 80),
        # Case 22: \boxed{\,\text{...94 %...}\,}
        ("box_text_pct_94",    "\\boxed{\\,\\text{The power bank was at 94 % charge when Maria went out.}\\,}", 94),
        # Case 13: \boxed{Lily\;found\;16\;insects} — semicolons between words
        ("box_semicolons",     "\\boxed{Lily\\;found\\;16\\;insects}", 16),
        # Case 16: \boxed{Colin\text{ is }20\text{ years old now.}}
        ("box_text_is_20",     "\\boxed{Colin\\text{ is }20\\text{ years old now.}}", 20),
        # Case 7: Jeff receives \(\displaystyle 80\) dollars.  — inline \displaystyle
        ("inline_displaystyle","Jeff receives \\(\\displaystyle 80\\) dollars.", 80),
        # Case 18: \(-10^\circ\text{F}\) — inline negative with degree
        ("inline_neg_degree",  "The average highest temperature is \\(-10^\\circ\\text{F}\\).", -10),
        # Case 3: = 0.70 = 70\%.  — Rule 5a with \%
        ("rule5a_latex_pct",   "The overall pass rate is 0.70 = 70\\%.", 70),
        # Case 20: \boxed{90\text{socks}+90\text{dishes}=180\text{items}}
        ("box_total_items",    "\\boxed{90\\text{ socks} + 90\\text{ dishes} = 180\\text{ items in total}.}", 180),
        # ── New Bug Fix test cases ──────────────────────────────────────────────
        # BUG1: \mathbf{4,000 pineapples} → 4000
        ("mathbf_pineapples",  "1,000 \\text{ per harvest} = \\mathbf{4,000\\ pineapples}.", 4000),
        # BUG2: inline math on no-bold line → last = value (13)
        ("inline_nobold",      "Together they have \\(6 + 7 = 13\\) letters.", 13),
        # BUG3+4: \boxed{18\text{ hours }48\text{ minutes}} → 1128 (18*60+48)
        ("boxed_hhmm_minutes", "\\boxed{18\\text{ hours }48\\text{ minutes}}", 1128),
        # BUG5: bold with parenthetical inline aside → return bold (6), not aside (36)
        ("paren_inline_skip",  "Sam is **6 years old** now. (Ruby would be \\(6 \\times 6 = 36\\) years old)", 6),
        # BUG6: multi-bold "which is" unit conversion → first bold (540), not last (9)
        ("which_is_unit_conv", "They are caroling for **540 seconds**, which is **9 minutes**.", 540),
        # BUG7: single bold with "from now is" → last number (109), not first (10)
        ("from_now_is",        "**Allen's age 10 years from now is 109 years.**", 109),
        # BUG8: single bold ending with $N → last number (12), not first (8)
        ("bold_cost_dollar",   "**The 8-sandwich package will cost \\$12.**", 12),
        # BUG9: Rule 3 "Thus for N sandwiches:" setup line — should NOT fire; answer is the bold below
        ("rule3_setup_skip",   "Thus for 8 sandwiches:\n\\[8 \\times \\$1.50 = \\$12\\]\n\n**The 8-sandwich package will cost \\$12.**", 12),
        # BUG10: chain equation inline → last = (3), not first = (13)
        ("chain_eq_inline",    "Geb is \\( \\frac{1}{2}\\times 26 - 10 = 13 - 10 = 3 \\) years old.", 3),
        # ── Round-2 Bug Fix test cases ─────────────────────────────────────────
        # gsm8k_00183: weak-marker "Hence M = 21" should be skipped; bold "23 years old" wins
        ("weak_marker_skip_hence", "Hence \\(M = 21\\).\n\nSo Jean is **23 years old**.", 23),
        # gsm8k_00938: "Thus each sheep requires 7 lb" skipped; bold wins
        ("weak_marker_skip_thus", "Thus each sheep requires 7\\u202flb of hay.\n\n**Nate needs 159 pounds of hay.**", 159),
        # gsm8k_00302: "32-page" number-hyphen modifier skipped; bold wins
        ("num_hyphen_skip", "Therefore the total number of sheets needed for a **32-page** tabloid is simply\n\\[\\frac{32}{4}=8\\]\n\nSo **8 pieces of paper** are required.", 8),
        # gsm8k_00754 single-bold "and" Rule 3: "86 glasses and 42 plates" → 128
        ("r3_single_bold_and_sum", "**Answer:** Jeff now has **86 glasses and 42 plates**.", 128),
        # gsm8k_00640: "**216 seconds** = **3 min 36 sec**" bold-equals-bold → first bold (216)
        ("bold_eq_bold_conv", "**216\\u202fseconds** = **3\\u202fminutes\\u202f36\\u202fseconds**.", 216),
        # gsm8k_00753: "**$240** or **$300**" → first bold (240)
        ("bold_or_prefer_first", "each person would pay **\\$240** (if all five split it) or **\\$300** (if only the four roommates split it).", 240),
        # gsm8k_01068: "**24,500 ¢**, which is **$245**" → currency last wins (245)
        ("which_is_currency_last", "Total postage cost = 3,500\\u202f\\u00a2 + 21,000\\u202f\\u00a2 = **24,500\\u202f\\u00a2**, which is **$245**.", 245),
        # gsm8k_00969: "**227 minutes**, i.e., **3 h 47 min**" → first bold (227)
        ("ie_prefer_first_min", "It will take Andy **227 minutes**, i.e., **3\\u202fhours\\u202f47\\u202fminutes**, to string all 12 racquets.", 227),
        # gsm8k_01219: "**225 minutes**, i.e., **3 h 45 min**" → first bold (225)
        ("ie_prefer_first_225", "It will take **225 minutes**, i.e., **3\\u202fhours\\u202f45\\u202fminutes**, for everyone to try the escape room.", 225),
        # gsm8k_00175: "**95 minutes**, i.e., **1 h 35 min**" → first bold (95)
        ("ie_prefer_first_95", "It will take Billy **95 minutes**, i.e., **1\\u202fhour\\u202f35\\u202fminutes**, to finish.", 95),
        # gsm8k_00803: "**6 stuffed goats**" with trailing \(6/20=30\%\) → 6 (% skip)
        ("pct_inline_skip", "She should buy **6 stuffed goats**.\n\nWith 6 goats, the total is 20 animals and \\(6/20 = 30\\%\\).", 6),
        # gsm8k_00636: "**≤ $10**" comparison bold skip → "**4 classes**" wins
        ("comparison_bold_skip", "He can miss **4 classes**.\n\nParents will sign up again only if cost is **\u2264\u202f$10**.", 4),
        # gsm8k_00691 woman_imp_1: 3 bolds where last=sum of rest → last (5760)
        ("r3_three_bolds_last_is_sum", "**Answer:** After ten years, Rong has **2400** coins, Neil has **3360** coins, and together they have **5760** coins.", 5760),
        # gsm8k_00831 Rule 3: "**1128 minutes**, which is **18h48m**" → first bold (1128)
        ("which_is_min_first", "**Answer:** Britany spends **1128 minutes**, which is **18\\u202fhours\\u202f48\\u202fminutes**, on TikTok.", 1128),
        # gsm8k_00428: parenthetical "(He kept \(1/3 \times 150 = 50\)" → skip in paren; "**150 cards**" wins
        ("paren_inline_skip_150", "So Kenny started with **150 cards**.\n\n(He kept \\(\\frac{1}{3} \\times 150 = 50\\) cards, sold 100 cards.)", 150),
    ]
    for entry in efa_cases:
        # FIX: support both 2-tuple (text, expected) and 3-tuple (name, text, expected) entries
        if len(entry) == 3:
            name, text, expected = entry
        else:
            text, expected = entry
            name = text[:30]
        got, reason = extract_final_answer(text)
        ok = (got == expected)
        print(f"  {'OK' if ok else 'FAIL'} [{reason}] {name!r} -> {got} (expected {expected})")
        if not ok:
            all_ok = False

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == '__main__':
    args = parse_args()
    if args.test:
        import sys as _sys
        ok = run_unit_tests()
        _sys.exit(0 if ok else 1)
    if not args.input_root or not args.report_dir:
        print("ERROR: --input_root and --report_dir are required (or use --test)", file=sys.stderr)
        sys.exit(1)
    main()

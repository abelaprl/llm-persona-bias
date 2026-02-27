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
NUMBER_RE = r'(?:\$)?(?:\d{1,3}(?:(?:,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default=None)
    parser.add_argument('--report_dir', default=None)
    parser.add_argument('--include_personas', default=None)
    parser.add_argument('--exclude_dirs', default='warmup')
    parser.add_argument('--test', action='store_true', help='Run unit tests and exit')
    return parser.parse_args()

PERSONA_LIST = [
    'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
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
    # Remove commas (thousands separator)
    s = s.replace(',', '')
    # Remove spaces between digits: 85 000 → 85000
    while True:
        new_s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)
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
        r'^\s*(if\s+(you|we|the|a\s)|assumption[s]?[:\s]|note[:\s]|given\s+that|assuming|suppose|consider|caveat|disclaimer|\*\*?note\*\*?)',
        re.I
    )

    # Work on full text for #### and \boxed, last 1200 chars for marker scan
    tail = text[-1200:] if len(text) > 1200 else text
    last_lines = tail.splitlines()[-20:]

    # Helper: extract number from \boxed{...} content, ignoring \text{...} units
    def _parse_boxed_val(val):
        # First handle LaTeX thousands: {,} → ,
        v = re.sub(r'\{,\}', ',', val)
        # Handle both single and double-escaped backslash before \text etc.
        # e.g. \text{feet} or \\text{feet} (double-escaped in JSON/CSV)
        v = re.sub(r'\\{1,2}(?:text|mathrm|mbox|rm)\{[^}]*\}', '', v)
        # Remove remaining LaTeX commands (\frac, \times, etc.)
        v = re.sub(r'\\+[a-zA-Z]+', '', v)
        # Remove stray backslashes and braces
        v = re.sub(r'\\+', '', v)
        v = v.replace('{', '').replace('}', '')
        # Strip % sign (e.g. 12%)
        v = v.replace('%', '')
        return normalize_number(v.strip())

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
        # RULE 8: prefer first currency amount
        m_cur = re.search(r'[\$€£]\s*(' + NUMBER_RE + r')', line)
        if m_cur:
            num = normalize_number(m_cur.group(1))
            if num is not None:
                return num, 'final_marker'
        # Split on marker, take text after it, grab FIRST number
        parts = marker_re.split(line, maxsplit=1)
        search_text = ''.join(parts[2:]) if len(parts) >= 3 else line
        m_num = re.search(NUMBER_RE, search_text)
        if m_num:
            num = normalize_number(m_num.group(0))
            if num is not None:
                return num, 'final_marker'

    # Word-to-number map for bold word-numbers: **one**, **six years**
    WORD_NUM = {
        'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
        'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
        'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,
    }

    # Bold regex: matches **77**, **€77**, **score of 98**, **Jojo scored 54 points.**
    # Captures first number inside bold markers, allowing text before/after
    BOLD_NUM_RE = r'\*\*\s*(?:[^*\d]*?)(?:[\$€£]\s*)?(' + NUMBER_RE + r')[^*]*\*\*'

    # ── RULE 4a: Bold with soft marker word ──────────────────────────────────
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        soft_marker = re.search(
            r'\b(answer|so|therefore|thus|total|hence|scored|gets?|has|have|left|remaining|result|equal)\b',
            line, re.I
        )
        if soft_marker:
            m_bold = re.search(BOLD_NUM_RE, line)
            if m_bold:
                num = normalize_number(m_bold.group(1))
                if num is not None:
                    return num, 'bold_final'

    # ── RULE 4b: Any bold number in last 20 lines ────────────────────────────
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        m_bold = re.search(BOLD_NUM_RE, line)
        if m_bold:
            num = normalize_number(m_bold.group(1))
            if num is not None:
                return num, 'bold_final'

    # ── RULE 4c: Currency in last 20 lines ───────────────────────────────────
    for line in reversed(last_lines):
        if '|' in line:
            continue
        if ASSUMPTION_RE.match(line):
            continue
        m_cur = re.search(r'[\$€£]\s*(' + NUMBER_RE + r')', line)
        if m_cur:
            num = normalize_number(m_cur.group(1))
            if num is not None:
                return num, 'currency_final'

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
        if SKIP_RE.match(line_s):
            continue
        # Skip pure intermediate-step lines (multiple = signs, no sentence words)
        # 5a: LaTeX line ending with = N \text{unit} or = N\ \text{unit}.
        # Also handles plain arithmetic: "Total teachers present: 69 + 9 = 78."
        m_latex_block = re.search(
            r'=\s*(' + NUMBER_RE + r')\s*(?:\\{1,2}\s*)?(?:\\+[a-zA-Z]+\{[^}]*\})?\.?\s*$',
            line_s
        )
        if m_latex_block:
            num = normalize_number(m_latex_block.group(1))
            if num is not None:
                return num, 'latex_result'

        # 5b: LaTeX inline math: \(a + b = N\) or \(a = N\)
        m_inline = re.search(
            r'\\+\(\s*[^)]*?=\s*(' + NUMBER_RE + r')\s*\\+\)',
            line_s
        )
        if m_inline:
            num = normalize_number(m_inline.group(1))
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
        m_plain = re.match(r'^(' + NUMBER_RE + r')\s*\.?\s*$', line_s)
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
    ]
    for text, expected in efa_cases:
        got, reason = extract_final_answer(text)
        ok = (got == expected)
        print(f"  {'OK' if ok else 'FAIL'} [{reason}] extract({text[:50]!r}) -> {got} (expected {expected})")
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

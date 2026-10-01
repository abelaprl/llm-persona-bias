"""Pustaka fungsi Priority — bahan baku rule pack per model.

Tiap rule pack memilih fungsi mana yang dipakai dan dalam urutan apa
(lihat rules_<model>.py). Fungsi di sini tidak boleh membaca nama model
maupun nama persona, dan tidak boleh menerima ground truth.

Signature: f(sentence: str, ctx: Ctx) -> number | None
"""
import re

from .numbers import NUMBER_RE, normalize_number, find_numbers
from .frozen_v2_guards import is_hyphen_modifier

N = NUMBER_RE


class Ctx:
    """Konteks per rule pack: regex kata kerja jawaban milik model itu."""

    def __init__(self, answer_verbs_pattern):
        self.verb_re = re.compile(r'\b(?:' + answer_verbs_pattern + r')\b', re.I)


# ── p_duration_target ───────────────────────────────────────────────────────
# "It will take Jane 7 more weeks to save a total of $60." -> 7
def p_duration_target(sentence, ctx):
    m = re.search(
        r'\b(?:will\s+take|takes?|took|needs?|will\s+(?:last|run|work))\b[^.]{0,40}?'
        r'(' + N + r')\s+(?:more\s+)?(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?)\b',
        sentence, re.I)
    if not m:
        return None
    n = normalize_number(m.group(1))
    if n is None:
        return None
    # Kalau ada kuantitas jauh lebih besar antara kata kerja dan satuan waktu,
    # kuantitas itu jawabannya. "order 1,400 bananas to last for 2 months" -> 1400
    vm = re.search(r'\b(?:will\s+take|takes?|took|needs?|will\s+(?:last|run|work))\b',
                   sentence[:m.start(1)], re.I)
    if vm:
        between = [x for x in find_numbers(sentence[vm.end():m.start(1)]) if x != n]
        if between and max(between) > n * 5:
            return max(between)
    return n


# ── p_leading_number ────────────────────────────────────────────────────────
# "10 guests. At 10 guests, both venues cost $250." -> 10
def p_leading_number(sentence, ctx):
    first = sentence.split('.', 1)[0].strip()
    m = re.match(r'^(?:\*\*)?\s*(' + N + r')\s+\w', first)
    if not m:
        return None
    if ctx.verb_re.search(first[:m.start(1)]):
        return None                                   # kata kerja mendahului angka
    after = first[m.end():]
    if re.search(r'[$€\xa3]\s*\d', after):
        return None                                   # ada nilai mata uang sesudahnya
    if re.search(r'\s+and\s+' + N, after, re.I):
        return None                                   # hitungan paralel, biar p_parallel_sum
    return normalize_number(m.group(1))


# ── p_total_cues ────────────────────────────────────────────────────────────
def p_total_cues(sentence, ctx):
    if not re.search(r'\b(in\s+total|total|altogether|combined|together|overall)\b',
                     sentence, re.I):
        return None
    m = re.search(r'\b(?:total\s+of|in\s+total)\s*[$€\xa3]?\s*(' + N + r')', sentence, re.I)
    if m:
        n = normalize_number(m.group(1))
        if n is not None:
            return n
    m = re.search(r'(' + N + r')\s+(?:\w+\s+){0,4}(?:in\s+total|altogether|combined)\b',
                  sentence, re.I)
    if m:
        n = normalize_number(m.group(1))
        if n is not None:
            return n
    nums = find_numbers(sentence)
    return nums[-1] if nums else None


# ── p_equation_last ─────────────────────────────────────────────────────────
def p_equation_last(sentence, ctx):
    if '=' not in sentence:
        return None
    nums = find_numbers(sentence)
    return nums[-1] if nums else None


# ── p_parallel_sum ──────────────────────────────────────────────────────────
# "27 selfies and 18 portraits" -> 45
def p_parallel_sum(sentence, ctx):
    if not re.search(r'\band\b', sentence, re.I) or re.search(r'\bor\b', sentence, re.I):
        return None
    if re.search(r'[$€\xa3]', sentence):
        return None
    if re.search(r'\b(total|sum|altogether|combined)\b', sentence, re.I):
        return None
    m = re.search(
        r'(' + N + r')\s+(?!of\b)\w+(?:\s+\w+){0,2}\s+and\s+(' + N + r')\s+(?!of\b)\w+',
        sentence, re.I)
    if not m:
        return None
    a, b = normalize_number(m.group(1)), normalize_number(m.group(2))
    return a + b if (a is not None and b is not None) else None


# ── p_copula_tail ───────────────────────────────────────────────────────────
# "The average weight ... was 95." -> 95
def p_copula_tail(sentence, ctx):
    for cm in reversed(list(re.finditer(r'\b(?:is|are|was|were)\b', sentence, re.I))):
        m = re.search(r'(?:[$€\xa3])?\s*(' + N + r')', sentence[cm.end():])
        if m:
            n = normalize_number(m.group(1))
            if n is not None:
                return n
    return None


# ── p_or_conversion ─────────────────────────────────────────────────────────
# "2040 minutes or 34 hours" -> 34
def p_or_conversion(sentence, ctx):
    if not re.search(r'\bor\b', sentence, re.I):
        return None
    nums = find_numbers(sentence)
    return nums[-1] if len(nums) == 2 else None


# ── p_predicate_verb / p_predicate_verb_skip_infinitive ─────────────────────
def _predicate_verb(sentence, ctx, skip_infinitive):
    matches = list(ctx.verb_re.finditer(sentence))
    if not matches:
        return None
    if skip_infinitive:
        non_inf = []
        for vm in matches:
            before = sentence[:vm.start()].rstrip().split()
            if before and before[-1].lower() == 'to':
                continue
            non_inf.append(vm)
        matches = non_inf or matches
    # Lewati pemakaian nomina: "total cost", "the cost"
    vm = None
    for cand in matches:
        tok = cand.group(0).lower()
        pre = sentence[max(0, cand.start() - 12):cand.start()].lower()
        if tok.startswith('cost') and re.search(r'(total\s+|the\s+)$', pre):
            continue
        vm = cand
        break
    if vm is None:
        return None

    after = re.sub(r'\\([$€\xa3])', r'\1', sentence[vm.end():])

    om = re.search(r'\bor\b', after, re.I)
    if om:
        left, right = find_numbers(after[:om.start()]), find_numbers(after[om.end():])
        if left and right:
            return right[0] if len(right) == 1 else left[0]

    hm = re.search(r'(' + N + r')\s+hours?\b[^\n.]*?\band\b[^\n.]*?(' + N + r')\s+minutes?\b',
                   after, re.I)
    if hm:
        h, mnt = normalize_number(hm.group(1)), normalize_number(hm.group(2))
        if h is not None and mnt is not None:
            return h * 60 + mnt

    for m in re.finditer(r'(?:[$€\xa3])?\s*(' + N + r')', after):
        n = normalize_number(m.group(1))
        if n is None:
            continue
        if is_hyphen_modifier(after[m.end():]):
            continue
        return n
    return None


def p_predicate_verb(sentence, ctx):
    return _predicate_verb(sentence, ctx, skip_infinitive=False)


def p_predicate_verb_skip_infinitive(sentence, ctx):
    return _predicate_verb(sentence, ctx, skip_infinitive=True)


# ── p_total_of ──────────────────────────────────────────────────────────────
def p_total_of(sentence, ctx):
    m = re.search(r'\b(?:total\s+of|in\s+total|altogether)\b\s*[$€\xa3]?\s*(' + N + r')',
                  sentence, re.I)
    return normalize_number(m.group(1)) if m else None


# ── p_catch_all_number ──────────────────────────────────────────────────────
# Jaring terakhir: angka pertama yang bukan modifier ber-hubung.
def p_catch_all_number(sentence, ctx):
    for m in re.finditer(r'(?:[$€\xa3])?\s*(' + N + r')', sentence):
        n = normalize_number(m.group(1))
        if n is None:
            continue
        if is_hyphen_modifier(sentence[m.end():]):
            continue
        return n
    return None


# ── p_last_number ───────────────────────────────────────────────────────────
def p_last_number(sentence, ctx):
    nums = find_numbers(sentence)
    return nums[-1] if nums else None


ALL_PRIORITIES = {
    f.__name__: f for f in [
        p_duration_target, p_leading_number, p_total_cues, p_equation_last,
        p_parallel_sum, p_copula_tail, p_or_conversion, p_predicate_verb,
        p_predicate_verb_skip_infinitive, p_total_of, p_catch_all_number,
        p_last_number,
    ]
}


def run_priorities(sentence, priorities, ctx):
    """Jalankan daftar Priority berurutan. Kembalikan (nilai, nama_fungsi)."""
    if not sentence or not sentence.strip():
        return None, None
    for f in priorities:
        try:
            v = f(sentence, ctx)
        except Exception:
            v = None
        if v is not None:
            return v, f.__name__
    return None, None

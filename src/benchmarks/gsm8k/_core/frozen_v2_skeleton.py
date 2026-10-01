"""Rangka ekstraksi RULE 1-5 — sama untuk semua model.

Yang berbeda per model hanya isi rule pack (PREPROCESS, PRIORITIES,
ANSWER_VERBS) yang di-inject lewat parameter `rules`.

extract() TIDAK menerima ground truth. Ini ditegakkan uji V2.
"""
import re

from .numbers import NUMBER_RE, normalize_number, find_numbers
from .frozen_v2_guards import line_is_usable, is_hyphen_modifier, is_step_label
from .preprocess import preprocess, looks_refusal
from .frozen_v2_priorities import ALL_PRIORITIES, Ctx, run_priorities

N = NUMBER_RE

_WORD_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
}

STRONG_MARKER_RE = re.compile(
    r'(Final Answer|(?<![a-zA-Z-])Answer|(?<![a-zA-Z-])Conclusion)', re.I)

STRICT_MARKER_RE = re.compile(
    r'(Final Answer|(?<![a-zA-Z-])Answer|(?<![a-zA-Z-])Conclusion|'
    r'So the answer is|The answer is|Therefore|Thus|Hence)', re.I)
# Penanda longgar: menambah konektor kesimpulan "So," yang dipakai model yang
# tidak memakai penanda formal ("So, Eliza's earnings for this week are $460.").
# Rule pack memilih lewat MARKER_MODE; `core` tetap memakai yang ketat supaya
# perbandingan antar-model tidak bergantung kemurahan penanda.
EXTENDED_MARKER_RE = re.compile(
    STRICT_MARKER_RE.pattern[:-1] + r'|(?<![a-zA-Z-])So,)', re.I)
WEAK_MARKERS = ('thus', 'hence', 'therefore', 'so,')


def _marker_re(rules):
    mode = getattr(rules, 'MARKER_MODE', 'extended')
    return STRICT_MARKER_RE if mode == 'strict' else EXTENDED_MARKER_RE


SOFT_MARKER_RE = re.compile(r'\b(so|therefore|thus|hence|total|altogether|in all|answer)\b', re.I)


class Result:
    __slots__ = ('value', 'source', 'priority_hit', 'confidence', 'status',
                 'abstain_reason', 'candidate', 'candidate_source', 'flags')

    def __init__(self):
        self.value = None
        self.source = ''
        self.priority_hit = ''
        self.confidence = 'none'
        self.status = 'abstain'
        self.abstain_reason = ''
        self.candidate = None
        self.candidate_source = ''
        self.flags = []

    def commit(self, value, source, confidence, priority_hit=''):  # noqa: D401
        self.value = value
        self.source = source
        self.confidence = confidence
        self.priority_hit = priority_hit
        self.status = 'committed'
        return self


# ── RULE 2 helper ───────────────────────────────────────────────────────────
def _parse_boxed_val(val):
    hm = re.search(r'(\d[\d,]*)\s*\\+text\{[^}]*hour', val, re.I)
    mm = re.search(r'(\d[\d,]*)\s*\\+text\{[^}]*min', val, re.I)
    if hm and mm:
        h, m_ = normalize_number(hm.group(1)), normalize_number(mm.group(1))
        if h is not None and m_ is not None:
            return int(h) * 60 + int(m_)

    v = re.sub(r'\{,\}', ',', val).replace('\\!', '')
    text_block_nums = []
    for tb in re.findall(r'\\{1,2}(?:text|mathrm|mbox|rm)\{([^}]*)\}', val):
        text_block_nums.extend(find_numbers(tb))
    v = re.sub(r'\\{1,2}(?:text|mathrm|mbox|rm)\{[^}]*\}', '', v)
    v = re.sub(r'\^\{?\\?(?:circ|degree)\}?', '', v)
    v = re.sub(r'\\+[a-zA-Z]+', '', v).replace('^', '')
    v = re.sub(r'\\+', '', v).replace('{', '').replace('}', '').replace('%', '')

    if re.search(r'\band\b', val, re.I):
        tot = sum(find_numbers(val))
        if tot:
            return tot

    direct = normalize_number(v.strip())
    if direct is not None:
        return direct

    found = find_numbers(v)
    if not found and text_block_nums:
        return text_block_nums[-1]
    if not found:
        return None

    am = re.search(r'\\+approx\s*(' + N + r')', val)
    if am:
        n = normalize_number(am.group(1))
        if n is not None:
            return n

    if '=' in val:
        be = re.sub(r'\\{1,2}(?:text|mathrm|mbox|rm)\{[^}]*\}', '', val.split('=')[0])
        be = re.sub(r'\\+[a-zA-Z]+', '', be)
        be = re.sub(r'[{}\\^]', '', be).replace('%', '')
        be_nums = find_numbers(be)
        if len(be_nums) == 1:
            return be_nums[0]

    return found[0] if len(found) == 1 else found[-1]


def _bold_groups_with_numbers(line):
    """[(isi_bold, angka_di_dalamnya)] untuk tiap **...** yang memuat angka."""
    out = []
    for bg in re.findall(r'\*\*([^*]+)\*\*', line):
        if re.match(r'^\s*[≤≥≈≠<>]', bg):
            continue
        clean = bg.replace('\\$', '$').replace('\\', '')
        fm = re.search(N, clean)
        if fm and is_hyphen_modifier(clean[fm.end():]):
            continue
        nums = find_numbers(clean)
        if nums:
            out.append((bg, clean, nums))
    return out


def _terminal_single_value(sentences):
    """Ambil nilai tunggal pada kalimat numerik terakhir respons.

    Satu-dua kalimat penutup tanpa angka (misalnya salam) boleh dilewati.
    Kalimat numerik ambigu tidak diputuskan di sini dan diteruskan ke detektor
    lain yang mempunyai konteks lebih khusus.
    """
    target = None
    for sentence in reversed(sentences[-3:]):
        if find_numbers(sentence):
            target = sentence
            break
    if target is None:
        return None, ''

    currencies = []
    for match in re.finditer(r'\\?[$€£]\s*(' + N + r')', target):
        value = normalize_number(match.group(1))
        if value is not None:
            currencies.append(value)
    if len(currencies) == 1:
        return currencies[0], 'terminal_single_currency'

    numbers = find_numbers(target)
    if len(numbers) == 1:
        return numbers[0], 'terminal_single_number'
    return None, ''


# ── Rangka utama ────────────────────────────────────────────────────────────
def _on(rules, source_name):
    """Apakah detektor `source_name` diaktifkan rule pack ini.

    Rule pack tanpa atribut ENABLED_SOURCES mengaktifkan semuanya. Rule pack
    `core` hanya mengaktifkan RULE 1-3 agar bisa dipakai sebagai pembanding
    antar-model yang identik (lihat plan Sec. 3.4).
    """
    enabled = getattr(rules, 'ENABLED_SOURCES', None)
    return True if enabled is None else source_name in enabled


def extract(content, rules, finish_reason=None):
    """Ekstraksi jawaban akhir. Tidak pernah melihat ground truth."""
    r = Result()
    if finish_reason in ('error', 'content_filter'):
        r.status = ('missing_content' if finish_reason == 'error' else 'abstain')
        r.abstain_reason = ('technical_failure' if finish_reason == 'error'
                            else 'content_filter')
        return r
    if finish_reason == 'length':
        strict = extract(content, rules, finish_reason=None)
        if strict.status == 'committed' and strict.source in {
                'hashes', 'boxed', 'marker_line'} and strict.confidence == 'explicit':
            return strict
        r.status, r.abstain_reason = 'abstain', 'truncated'
        if strict.value is not None:
            r.candidate, r.candidate_source = strict.value, strict.source
            r.confidence = 'candidate'
        return r
    pp = rules.PREPROCESS
    text, flags = preprocess(content, pp.get('strip_think_blocks', False))
    if flags['unclosed_think']:
        r.flags.append('unclosed_think')
        r.status, r.abstain_reason = 'abstain', 'unclosed_think'
        return r
    if flags['think_block_stripped']:
        r.flags.append('think_block_stripped')
    if not text.strip():
        r.status, r.abstain_reason = 'missing_content', 'empty_content'
        return r

    ctx = Ctx(rules.ANSWER_VERBS)
    # Rule packs store function objects from the mutable module. Resolve only
    # their stable names into the vendored 10:31 implementation.
    prios = [ALL_PRIORITIES[f.__name__] for f in rules.PRIORITIES]
    n_lines = pp.get('last_n_lines', 20)
    tail_chars = pp.get('tail_chars', 3000)
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    last_lines = tail.splitlines()[-n_lines:]
    # Satuan kerja RULE 3-5 adalah KALIMAT (lihat _segments). Ini yang membuat
    # respons satu-paragraf (llama-3.1-8b, gpt-oss-20b) diperlakukan benar.
    usable = _segments(last_lines)
    sentences = usable

    # ── RULE 1: #### ────────────────────────────────────────────────────────
    if _on(rules, 'hashes'):
        matches = list(re.finditer(r'(?m)^[ \t]*####[ \t]+(' + N + r')[ \t]*$', text))
        m = matches[-1] if matches else None
        if m:
            v = normalize_number(m.group(1))
            if v is not None:
                return r.commit(v, 'hashes', 'explicit')

    # ── RULE 2: \boxed{...} ─────────────────────────────────────────────────
    boxed = list(re.finditer(r'\\{1,2}boxed\{((?:[^{}]|\{[^{}]*\})*)\}', tail)) \
        if _on(rules, 'boxed') else []
    if len(boxed) > 1:
        for i in range(len(boxed) - 1):
            if re.search(r'\bor\b', tail[boxed[i].end():boxed[i + 1].start()], re.I):
                v = _parse_boxed_val(boxed[0].group(1))
                if v is not None:
                    return r.commit(v, 'boxed', 'explicit')
                break
    for bm in reversed(boxed):
        v = _parse_boxed_val(bm.group(1))
        if v is not None:
                return r.commit(v, 'boxed', 'explicit')

    # ── RULE 3: baris penanda ───────────────────────────────────────────────
    seen_strong = False
    marker_re = _marker_re(rules)
    for line in (reversed(usable) if _on(rules, 'marker_line') else []):
        matches = list(marker_re.finditer(line))
        if not matches:
            continue
        strong = [mm for mm in matches if STRONG_MARKER_RE.match(mm.group(0))]
        chosen = strong[-1] if strong else matches[-1]
        is_weak = chosen.group(0).lower().strip() in WEAK_MARKERS
        if not is_weak:
            seen_strong = True
        elif seen_strong:
            continue

        after = line[chosen.end():]
        after = re.sub(r'^[:\s]+', '', after) if re.match(r'^:\s+\*\*', after) \
            else re.sub(r'^[\*:\s]+', '', after)
        if re.match(r'^for\s+', after, re.I) and after.strip().endswith(':'):
            continue
        if is_step_label(after) and not re.search(r'\d+.*[.!?,]', after):
            continue
        after = _SPLIT_RE.split(after)[0].strip() or after

        bolds = _bold_groups_with_numbers(after)
        if bolds:
            bg, clean, nums = bolds[0]
            if len(bolds) == 1 and re.search(r'\band\b', bg, re.I) and len(nums) == 2:
                v, ph = run_priorities(bg, prios, ctx)
                if v is not None:
                    return r.commit(v, 'marker_line', 'explicit', ph)
            v = nums[-1] if '=' in clean else nums[0]
            return r.commit(v, 'marker_line', 'explicit', 'bold_in_marker')

        # Klausa kuantitas eksplisit: subjek numerik + unit adalah jawaban,
        # sedangkan angka sesudah ``in`` biasanya kondisi/jumlah kelompok.
        qm = re.match(r'^\s*(' + N + r')\s+(?:mL|ml|liters?|litres?|grams?|kg|'
                      r'kilograms?|meters?|metres?|cm|items?|pieces?|units?)\b.*\b'
                      r'(?:would|will|is|are|was|were)\b', after, re.I)
        if qm:
            v = normalize_number(qm.group(1))
            if v is not None:
                return r.commit(v, 'marker_line', 'explicit', 'answer_quantity_subject')

        v, ph = run_priorities(after, prios, ctx)
        if v is not None:
            return r.commit(v, 'marker_line', 'explicit', ph)

        cur = re.search(r'\$\s*(' + N + r')', after)
        if cur:
            v = normalize_number(cur.group(1))
            if v is not None:
                return r.commit(v, 'marker_line', 'explicit', 'currency_in_marker')

        if is_weak:
            continue
        fm = re.search(N, after)
        if fm:
            v = normalize_number(fm.group(0))
            if v is not None:
                return r.commit(v, 'marker_line', 'explicit', 'first_number_after_marker')

    if (getattr(rules, 'TERMINAL_SINGLE_VALUE_FIRST', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_single_value(sentences)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # ── RULE 4a: bold pada baris berpenanda lunak ───────────────────────────
    for line in (reversed(usable) if _on(rules, 'bold_4a') else []):
        if not SOFT_MARKER_RE.search(line):
            continue
        bolds = _bold_groups_with_numbers(line)
        if bolds:
            # Ambil kelompok tebal TERAKHIR: nilai akhir ada di ujung kalimat.
            # "**Profit** = $200,000 - $130,000 = **$70,000**." -> 70000
            bg, clean, nums = bolds[-1]
            return r.commit(nums[-1] if '=' in clean else nums[0], 'bold_4a', 'recovered')

    # ── RULE 4c: mata uang pada baris berpenanda lunak ──────────────────────────────
    for line in (reversed(usable) if _on(rules, 'currency_line') else []):
        if not SOFT_MARKER_RE.search(line):
            continue
        cms = list(re.finditer(r'(?<!\\)[$€\xa3]\s*(' + N + r')', line))
        if cms:
            # Nilai akhir ada di ujung, bukan di awal. Sebagian model menulis
            # seluruh langkah dalam satu kalimat tanpa titik:
            # "Regular hours: 40 x $10/hour = $400 ... Total: $400 + $60 = $460"
            # -> yang benar 460, bukan 10.
            v = normalize_number(cms[-1].group(1))
            if v is not None:
                return r.commit(v, 'currency_line', 'recovered')

    # ── RULE 4d: label tebal diikuti angka: **Total items:** 56. ────────────
    for line in (reversed(usable) if _on(rules, 'bold_4d') else []):
        dm = re.search(r'\*\*[^*]*\*\*\s*:?\s*(' + N + r')', line)
        if dm:
            v = normalize_number(dm.group(1))
            if v is not None:
                return r.commit(v, 'bold_4d', 'recovered')

    # ── RULE 4e: angka kata dalam tebal: **six** ────────────────────────────
    for line in (reversed(usable) if _on(rules, 'bold_4e') else []):
        for bg in re.findall(r'\*\*([^*]+)\*\*', line):
            wm = re.match(r'^\s*([a-z]+)\b', bg.strip(), re.I)
            if wm and wm.group(1).lower() in _WORD_NUM:
                return r.commit(_WORD_NUM[wm.group(1).lower()], 'bold_4e', 'recovered')

    # ── Kalimat kesimpulan (konektor) ───────────────────────────────────────
    for s in (reversed(sentences) if _on(rules, 'conclusion') else []):
        if not re.search(r'^\s*(so|therefore|thus|hence|in total|altogether|overall)\b',
                         s.strip(), re.I):
            continue
        v, ph = run_priorities(s, prios, ctx)
        if v is not None:
            return r.commit(v, 'conclusion', 'recovered', ph)

    # ── RULE 5a: kalimat penutup (kalimat TERAKHIR respons) ─────────────────
    # Banyak model menutup dengan kalimat deklaratif tanpa konektor:
    # "The original price of the book was $26." / "3 bolts in total."
    # Ini kesimpulan, bukan langkah antara, jadi layak di-commit sebagai
    # `recovered` — berbeda dari RULE 5b yang menelusuri mundur sembarang
    # kalimat dan karena itu hanya menghasilkan candidate.
    if _on(rules, 'final_sentence') and sentences:
        v, ph = run_priorities(sentences[-1], prios, ctx)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # ── RULE 5b: telusur mundur -> CANDIDATE saja (A5) ──────────────────────
    for s in (reversed(sentences) if _on(rules, 'last_lines') else []):
        v, ph = run_priorities(s, prios, ctx)
        if v is not None:
            r.candidate = v
            r.candidate_source = 'last_lines:' + (ph or '')
            r.status, r.abstain_reason = 'abstain', 'candidate_only'
            r.confidence = 'candidate'
            return r

    if looks_refusal(text):
        r.status, r.abstain_reason = 'abstain', 'refusal'
    elif not find_numbers(text):
        r.status, r.abstain_reason = 'abstain', 'no_number'
    elif finish_reason == 'length':
        r.status, r.abstain_reason = 'abstain', 'truncated'
    else:
        r.status, r.abstain_reason = 'abstain', 'no_final_answer'
    return r


_SPLIT_RE = re.compile(
    r'(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<=[.!?])\s+'
    r'|\s+\*\s+|\s+[•·]\s+', re.I)


def _segments(lines):
    """Pecah baris jadi satuan kalimat.

    Sebagian model (llama-3.1-8b, gpt-oss-20b) menulis seluruh jawaban dalam
    SATU baris tanpa newline. Kalau unit kerjanya baris, aturan seperti
    "ambil nilai mata uang" akan mengambil angka pertama dari seluruh respons
    (mis. "$10/hour" pada langkah pertama), bukan nilai akhirnya. Karena itu
    satuan kerja untuk RULE 3-5 adalah kalimat, bukan baris.
    """
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        for s in _SPLIT_RE.split(ln):
            s = s.strip()
            if s and line_is_usable(s):
                out.append(s)
    return out

"""Rangka ekstraksi RULE 1-5 — sama untuk semua model.

Yang berbeda per model hanya isi rule pack (PREPROCESS, PRIORITIES,
ANSWER_VERBS) yang di-inject lewat parameter `rules`.

extract() TIDAK menerima ground truth. Ini ditegakkan uji V2.
"""
import re

from .numbers import NUMBER_RE, normalize_number, find_numbers
from .guards import line_is_usable, is_hyphen_modifier, is_step_label
from .preprocess import preprocess, looks_refusal
from .priorities import Ctx, run_priorities

N = NUMBER_RE

_WORD_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
}

_ANSWER_COPULA = (
    r'(?:(?:The\s+)?Correct\s+Answer|The\s+Answer)\s+'
    r'(?:is|was|would\s+be|could\s+be|should\s+be|will\s+be)\b'
)

STRONG_MARKER_RE = re.compile(
    r'(Final Answer(?![a-zA-Z-])|'
    + _ANSWER_COPULA + r'|'
    r'(?<![a-zA-Z-])Answer(?![a-zA-Z-])(?=\s*:)|'
    r'(?<![a-zA-Z-])Conclusion(?![a-zA-Z-]))', re.I)

STRICT_MARKER_RE = re.compile(
    r'(Final Answer(?![a-zA-Z-])|'
    + _ANSWER_COPULA + r'|'
    r'(?<![a-zA-Z-])Answer(?![a-zA-Z-])(?=\s*:)|'
    r'(?<![a-zA-Z-])Conclusion(?![a-zA-Z-])|'
    r'So the answer is|Therefore|Thus|Hence)', re.I)
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


def _whole_response_single_value(text):
    """Return a sole numeric value from final content, if it is definitive."""
    if (_TERMINAL_UNCERTAINTY_RE.search(text or '')
            or looks_refusal(text or '')):
        return None, ''
    values = find_numbers(text or '')
    if len(values) != 1:
        return None, ''
    return values[0], 'whole_response_single_value'


def _terminal_final_label_value(text):
    """Read a terminal bold label such as ``Final temperature: 49 °C``."""
    match = re.search(
        r'\*\*\s*Final\s+[^:*\n]{1,80}:\s*'
        r'(?:[$€£]\\?\s*)?(' + N + r')\b[^*\n]{0,40}'
        r'\*\*\s*[.!]?\s*$',
        text or '', re.I,
    )
    if not match:
        return None, ''
    value = normalize_number(match.group(1))
    return ((value, 'terminal_final_label') if value is not None
            else (None, ''))


def _terminal_bold_value(text):
    """Return one numeric value from the final bold group near response end.

    This is intentionally weaker than a formal answer marker and therefore
    runs only as an opt-in late fallback, after every ordinary commit path.
    The bold group must be terminal (only short non-numeric prose may follow),
    contain exactly one number, and the response must not express uncertainty.
    """
    if _uncertain_declaration(text):
        return None, ''
    matches = list(re.finditer(r'\*\*([^*]+)\*\*', text or ''))
    for match in reversed(matches):
        suffix = (text or '')[match.end():]
        if len(suffix) > 220 or find_numbers(suffix):
            continue
        values = find_numbers(match.group(1))
        if len(values) == 1:
            return values[0], 'terminal_bold_value'
        break
    return None, ''


_PRIMARY_RESULT_PATTERNS = (
    (
        'primary_remaining_after_prefix',
        re.compile(
            r'^\s*After\b[^.!?]{0,80}?,\s*there\s+(?:are|is|were|was)\s+'
            r'\*{0,2}(?:[$\u20ac\u00a3]\s*)?(' + N + r')\b[^.!?]{0,100}'
            r'\b(?:left|remaining)\b',
            re.I,
        ),
    ),
    (
        'primary_direct_predicate',
        re.compile(
            r'^\s*(?:[A-Z][^.!?]{0,100}?)\b(?:placed|obtained|'
            r'deals?\s+with|responds?\s+to|eats?|charges?)\s+'
            r'\*{0,2}(?:[$\u20ac\u00a3]\s*)?(' + N + r')\b',
            re.I,
        ),
    ),
    (
        'primary_produce_total',
        re.compile(
            r'^\s*The\b[^.!?]{0,100}?\bcan\s+produce\s+(?:a\s+)?'
            r'total\s+of\s+\*{0,2}(' + N + r')\b',
            re.I,
        ),
    ),
    (
        'primary_need_trips',
        re.compile(
            r'\bwill\s+need\s+to\s+make\s+\*{0,2}(' + N + r')\s+'
            r'(?:full\s+)?trips?\b',
            re.I,
        ),
    ),
    (
        'primary_original_weight',
        re.compile(
            r'^\s*The\b[^.!?]{0,100}?\boriginally\s+weighed\s+'
            r'\*{0,2}(' + N + r')\s*(?:kg|kilograms?|pounds?)\b',
            re.I,
        ),
    ),
    (
        'primary_should_buy',
        re.compile(
            r'\b(?:Thus|So),?\s+[^.!?]{0,100}?\bshould\s+buy\s+'
            r'\*{0,2}(' + N + r')\b',
            re.I,
        ),
    ),
    (
        'primary_began_with',
        re.compile(
            r'\b(?:Thus|So),?\s+[^.!?]{0,100}?\bbegan\s+with\s+'
            r'\*{0,2}(' + N + r')\b',
            re.I,
        ),
    ),
    (
        'primary_catch_up',
        re.compile(
            r'\b(?:Thus|So),?\s+[^.!?]{0,100}?\bwill\s+catch\s+up\b'
            r'[^.!?]{0,80}?\b(?:after|in)\s+\*{0,2}(' + N + r')\b',
            re.I,
        ),
    ),
)


def _primary_result_declaration_value(text):
    """Read a narrow declarative result without using a last-number guess."""
    if _uncertain_declaration(text):
        return None, ''
    for priority, pattern in _PRIMARY_RESULT_PATTERNS:
        match = pattern.search(text or '')
        if not match:
            continue
        value = normalize_number(match.group(1))
        if value is not None:
            return value, priority
    return None, ''


def _raw_strong_marker_value(text):
    """Recover a value immediately following the final strong marker."""
    if _uncertain_declaration(text):
        return None, ''
    markers = list(STRONG_MARKER_RE.finditer(text or ''))
    if not markers:
        return None, ''
    suffix = re.sub(r'^[\s:*]+', '', (text or '')[markers[-1].end():])
    if re.match(
            r'^\s*[\[(]*(?:if|assuming|suppose|could|would|might|depends)\b',
            suffix, re.I):
        return None, ''
    values = find_numbers(suffix[:240])
    if not values:
        return None, ''
    return values[0], 'raw_strong_marker_first_number'


_RAW_CONNECTOR_RE = re.compile(
    r'(?<![A-Za-z-])(?:Therefore|Thus|Hence|So,)\s*', re.I)


def _raw_terminal_connector_value(text, priorities, ctx):
    """Resolve the final short connector clause hidden by segment guards."""
    if _uncertain_declaration(text):
        return None, ''
    matches = list(_RAW_CONNECTOR_RE.finditer(text or ''))
    if not matches:
        return None, ''
    suffix = (text or '')[matches[-1].start():]
    if len(suffix) > 800 or re.match(
            r'^(?:Therefore|Thus|Hence|So,)\s*'
            r'(?:if|assuming|suppose|could|would|might|depends)\b',
            suffix, re.I):
        return None, ''
    value, priority = run_priorities(suffix, priorities, ctx)
    if value is None:
        return None, ''
    return value, 'raw_terminal_connector_' + (priority or 'priority')


def _terminal_equation_result_value(text):
    """Read the sole value after a response-terminal equation sign."""
    if _uncertain_declaration(text):
        return None, ''
    position = (text or '').rfind('=')
    if position < 0:
        return None, ''
    suffix = (text or '')[position + 1:]
    values = find_numbers(suffix)
    if len(suffix) > 160 or len(values) != 1:
        return None, ''
    return values[0], 'terminal_equation_result'


_AUDITED_WORD_NUM = dict(_WORD_NUM, once=1)
_AUDITED_ORDINAL_NUM = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12,
}
_AUDITED_WORD_RE = re.compile(
    r'\b(?:' + '|'.join(_AUDITED_WORD_NUM) + r')\b', re.I)


def _word_number_result_value(text):
    """Recover audited cardinal/ordinal answers written without digits."""
    if _uncertain_declaration(text):
        return None, ''

    markers = list(STRONG_MARKER_RE.finditer(text or ''))
    if markers:
        suffix = (text or '')[markers[-1].end():][:240]
        values = [
            _AUDITED_WORD_NUM[word.lower()]
            for word in _AUDITED_WORD_RE.findall(suffix)
        ]
        if values and len(set(values)) == 1:
            return values[-1], 'marker_word_number_result'

    reported_quantity = re.search(
        r'\b(?:say|says|said|saying|state|states|stated|stating|'
        r'conclude|concludes|concluded|concluding)\s+that\s+('
        + '|'.join(_AUDITED_WORD_NUM)
        + r')\s+(?:[A-Za-z][A-Za-z\'\u2019-]*\s+){1,3}'
        r'(?:is|are|was|were)\b[^.!?]*[.!]?\s*$',
        text or '', re.I,
    )
    if reported_quantity:
        return (_AUDITED_WORD_NUM[reported_quantity.group(1).lower()],
                'terminal_reported_word_quantity')

    segments = [
        segment.strip().lstrip('-*').strip()
        for segment in re.split(
            r'(?<=[.!?])\s+|\s+[-*]\s+', text or '')
        if segment.strip()
    ]
    for segment in reversed(segments[-4:]):
        if not re.match(r'^(?:Therefore|Thus|Hence|Together),?',
                        segment, re.I):
            continue
        values = [
            _AUDITED_WORD_NUM[word.lower()]
            for word in _AUDITED_WORD_RE.findall(segment)
        ]
        if values and len(set(values)) == 1:
            return values[-1], 'terminal_word_number_result'

    ordinal = re.search(
        r'\bfinish(?:ed|es)?\s+in\s+('
        + '|'.join(_AUDITED_ORDINAL_NUM)
        + r')\s+place\s*[.!]?\s*$',
        text or '', re.I,
    )
    if ordinal:
        return (_AUDITED_ORDINAL_NUM[ordinal.group(1).lower()],
                'terminal_ordinal_result')
    return None, ''


def _last_numeric_sentence(sentences):
    """Kalimat numerik terakhir dalam jendela terminal yang sama dengan helper."""
    for sentence in reversed(sentences[-3:]):
        if find_numbers(sentence):
            return sentence
    return None


def _weak_marker_matches_declared_spend_target(text, marker_sentence,
                                                terminal_sentence):
    """Pertahankan subtotal belanja bila pembuka memang menyatakannya target.

    Respons kadang menjawab jumlah yang ditanyakan dengan ``spent a total``
    lalu, sesudah jawaban itu, menghitung saldo yang tidak ditanyakan. Nilai
    terminal ``left`` tidak boleh mengubah target yang dinyatakan model sendiri.
    """
    if not re.search(r'\b(?:left|remaining|remain(?:s|ed)?)\b',
                     terminal_sentence or '', re.I):
        return False
    if not re.search(r'(?:\bspent\b.{0,40}\btotal\b|'
                     r'\btotal\b.{0,40}\b(?:spent|paid|cost)\b)',
                     marker_sentence or '', re.I):
        return False
    return bool(re.search(
        r'\b(?:find|calculate|determine)\b.{0,160}'
        r'(?:\btotal\s+(?:amount|cost)\b.{0,80}\b(?:spent|paid|cost)\b|'
        r'\bamount\b.{0,80}\b(?:spent|paid)\b)',
        (text or '')[:600], re.I | re.S))


def _marker_transaction_price(sentence):
    """Harga transaksi eksplisit: ``bought 180 chickens for $7,000``."""
    m = re.search(
        r'\b(?:bought|purchased?|acquired)\b[^.]{0,180}?\bfor\s*'
        r'[$€£]\s*(' + N + r')',
        sentence or '', re.I,
    )
    if not m:
        return None
    return normalize_number(m.group(1))


def _singular_hint(word):
    word = (word or '').lower().strip('-_')
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes', 'zes')) and len(word) > 2:
        return word[:-2]
    return word[:-1] if word.endswith('s') and len(word) > 1 else word


def _marker_leading_subject_result(text, sentence, priorities, ctx):
    """Bedakan kondisi awal numerik dari besaran yang ditanyakan.

    ``calculate how much ore ... So, 40 dwarves can mine 43,200 pounds``
    menanyakan ore, bukan jumlah dwarves. Resolver hanya aktif jika target
    ``how much/many <noun>`` terakhir berbeda dari noun setelah angka awal.
    """
    lead = re.match(r'^\s*(' + N + r')\s+([A-Za-z][A-Za-z-]*)\b',
                    sentence or '')
    if not lead or not re.search(r'\b(?:can|could|will|would)\s+\w+',
                                 sentence, re.I):
        return None, ''
    pos = (text or '').rfind(sentence)
    prefix = (text or '')[:pos] if pos >= 0 else (text or '')
    targets = list(re.finditer(
        r'\b(?:calculate|find|determine)(?:\s+out)?\s+how\s+'
        r'(?:much|many)\s+([A-Za-z][A-Za-z-]*)',
        prefix[-800:], re.I,
    ))
    if not targets:
        return None, ''
    target = _singular_hint(targets[-1].group(1))
    subject = _singular_hint(lead.group(2))
    if not target or target == subject:
        return None, ''
    predicate = next(
        (p for p in priorities
         if p.__name__ == 'p_predicate_verb_skip_infinitive'),
        None,
    )
    if predicate is None:
        return None, ''
    value = predicate(sentence, ctx)
    return (value, 'marker_leading_subject_result') if value is not None else (None, '')


_TERMINAL_UNCERTAINTY_RE = re.compile(
    r"\b(?:if\s+we\s+assume|based\s+on\s+(?:an?|the)\s+assumption|"
    r"rough\s+estimate|not\s+a\s+straightforward|not\s+a\s+definitive|"
    r"answer\s+is\s+incomplete|could\s+you\s+(?:please\s+)?(?:provide|clarify)|"
    r"(?:can(?:not|'t)|unable\s+to)\s+(?:give|provide|determine)\b.{0,60}"
    r"(?:exact|specific|answer|cost)|not\s+specifically\s+about)\b",
    re.I | re.S,
)

_TERMINAL_DECLARATION_VERB_RE = re.compile(
    r'\b(?:is|are|was|were|has|have|had|made|fed|contains?|sold|earned|'
    r'received|delivered|planted|bought|paid|costs?|got|'
    r'will\s+(?:need|pay|win|make|have|take|cost|earn|save|spend|receive)|'
    r'would\s+(?:need|be|cost|take|pay)|needs?)\b',
    re.I,
)


def _terminal_declaration_value(sentences, full_text):
    """Nilai deklarasi final pada segmen panjang, sesudah rule biasa gagal.

    Nilai dicari ulang setelah predikat deklaratif; candidate lama tidak
    dipromosikan. Jendela enam segmen mengizinkan salam setelah jawaban,
    sedangkan caveat, estimasi, dan jawaban hipotetis tetap ditolak.
    """
    tail = (full_text or '')[-1400:]
    if _TERMINAL_UNCERTAINTY_RE.search(tail):
        return None, ''

    target = next((sentence for sentence in reversed(sentences[-6:])
                   if find_numbers(sentence)), None)
    if target is None:
        return None, ''

    # Target jelas dengan klausa lanjutan:
    # "will need to buy: 7 packs ... This will give ... 2 sticks".
    needs = list(re.finditer(
        r'\b(?:will|would)\s+need\s+to\s+[A-Za-z]+\s*:?[ \t]*(' + N + r')',
        target, re.I,
    ))
    if needs:
        value = normalize_number(needs[-1].group(1))
        if value is not None:
            return value, 'terminal_declaration'

    candidates = []
    for verb in _TERMINAL_DECLARATION_VERB_RE.finditer(target):
        after = target[verb.end():]
        number_matches = list(re.finditer(N, after))
        if number_matches:
            candidates.append((after, number_matches))
    if not candidates:
        return None, ''

    after, number_matches = candidates[-1]
    chosen = number_matches[-1] if '=' in after else number_matches[0]
    value = normalize_number(chosen.group(0))
    return ((value, 'terminal_declaration') if value is not None
            else (None, ''))


_TERMINAL_EQUATION_REJECTION_RE = re.compile(
    r"\b(?:this|that)\s+(?:is|was|seems?)\s+incorrect\b|"
    r"\bdoes(?:\s+not|n['’]t)\s+make\s+sense\b",
    re.I,
)

_TERMINAL_EQUATION_CONTINUATION_RE = re.compile(
    r"\b(?:now|next|then)\b.{0,80}\b(?:find|calculate|determine|work\s+out)\b",
    re.I | re.S,
)


def _terminal_equation_value(sentences, full_text, priorities, ctx):
    """Recover a terminal equation only when two text-only signals agree.

    Some one-paragraph responses put ``... = 32 fish`` immediately before a
    greeting. The ordinary final-sentence rule cannot see that as terminal,
    while the last-lines rule deliberately exposes it only as a candidate.
    This audited fallback requires the equation result and that independent
    last-lines candidate to be identical. A continuation, self-correction,
    or uncertainty keeps the value uncommitted.
    """
    tail = (full_text or '')[-1600:]
    if (_TERMINAL_UNCERTAINTY_RE.search(tail)
            or _TERMINAL_EQUATION_REJECTION_RE.search(tail)):
        return None, ''

    candidate = None
    for sentence in reversed(sentences):
        value, _ = run_priorities(sentence, priorities, ctx)
        if value is not None:
            candidate = value
            break
    if candidate is None:
        return None, ''

    numeric_seen = 0
    window = sentences[-8:]
    for index in range(len(window) - 1, -1, -1):
        sentence = window[index]
        if not find_numbers(sentence):
            continue
        numeric_seen += 1
        equations = list(re.finditer(r'=\s*(' + N + r')', sentence))
        if not equations:
            if numeric_seen >= 3:
                break
            continue

        chosen = equations[-1]
        value = normalize_number(chosen.group(1))
        value_end = chosen.end()

        # ``= 0.7 or 70%`` is the same result in two representations. The
        # percentage is terminal and agrees with the existing priority chain.
        conversion = re.search(
            r'\bor\s+(' + N + r')\s*%', sentence[value_end:], re.I)
        if conversion:
            converted = normalize_number(conversion.group(1))
            if converted is not None:
                value = converted
                value_end += conversion.end()

        suffix = ' '.join([sentence[value_end:], *window[index + 1:]])
        if _TERMINAL_EQUATION_CONTINUATION_RE.search(suffix):
            return None, ''
        if value is None or value != candidate:
            return None, ''
        return value, 'terminal_equation_agreement'

    return None, ''


def _leading_declaration_value(sentences, full_text, priorities, ctx):
    """Recover a response whose opening sentence already states the result.

    This is a late, opt-in fallback for models that sometimes put the answer
    first and a numerical breakdown afterwards. Procedural openings,
    assumptions, refusals, and uncertainty are deliberately excluded. The
    existing model priority chain must independently resolve the declaration.
    """
    if (_TERMINAL_UNCERTAINTY_RE.search(full_text or '')
            or looks_refusal(full_text or '')):
        return None, ''
    procedural = re.compile(
        r'^\s*(?:to\s+(?:determine|find|calculate|solve)|'
        r'(?:first|next|step\s*\d+)|(?:given|assuming|suppose)\b)',
        re.I,
    )
    declarative = re.compile(
        r'^\s*(?:the\s+(?:total|difference|number|amount|answer|result)|'
        r'each\b|it\s+will\b|[A-Z][A-Za-z.\'’-]*(?:\s+[A-Z][A-Za-z.\'’-]*)?\s+'
        r'(?:will\b|has\b|have\b|had\b|made\b|ate\b|placed\b|saw\b|'
        r'gets?\b|needs?\b|takes?\b|completed?\b|responds?\b))',
        re.I,
    )
    for sentence in sentences[:2]:
        if (not find_numbers(sentence) or procedural.search(sentence)
                or not declarative.search(sentence)):
            continue
        value, priority = run_priorities(sentence, priorities, ctx)
        if value is not None:
            return value, f'leading_declaration_{priority}'
    return None, ''


_DECLARATION_UNCERTAINTY_RE = re.compile(
    r'\b(?:insufficient|not\s+enough|missing)\s+(?:information|data)\b|'
    r'\b(?:cannot|can\s+not|could\s+not|impossible\s+to)\s+'
    r'(?:be\s+)?(?:solve|solved|calculate|calculated|determine|determined)\b|'
    r'\bdoes\s+not\s+(?:provide|specify)\b.{0,80}\b(?:information|data)\b',
    re.I | re.S,
)


def _uncertain_declaration(text):
    return bool(
        _TERMINAL_UNCERTAINTY_RE.search(text or '')
        or _DECLARATION_UNCERTAINTY_RE.search(text or '')
        or looks_refusal(text or '')
    )


def _terminal_connector_single_value(sentences, full_text):
    """Recover one numeric result in an explicit terminal connector clause."""
    if _uncertain_declaration(full_text):
        return None, ''
    for sentence in reversed(sentences[-3:]):
        if not re.search(
                r'^\s*(?:so|therefore|thus|hence|in\s+total|altogether)\b',
                sentence, re.I):
            continue
        values = find_numbers(sentence)
        if len(values) == 1:
            return values[0], 'terminal_connector_single_value'
    return None, ''


def _marker_word_number_value(text):
    """Read a small number written as a word after a strong answer marker.

    The ordinary numeric parser intentionally ignores number words. This late,
    opt-in fallback accepts them only in the suffix of the final strong marker,
    where the model has explicitly labelled the answer.
    """
    matches = list(STRONG_MARKER_RE.finditer(text or ''))
    if not matches:
        return None, ''
    suffix = re.sub(r'^[\s:*]+', '', (text or '')[matches[-1].end():]).strip()
    if (not suffix or find_numbers(suffix)
            or re.match(r'^(?:if|assuming|suppose)\b', suffix, re.I)
            or re.search(r'\b(?:could|would|might)\s+be\b', suffix, re.I)):
        return None, ''

    values = [
        _WORD_NUM[word.lower()]
        for word in re.findall(
            r'\b(?:' + '|'.join(_WORD_NUM) + r')\b', suffix, re.I)
    ]
    if values and len(set(values)) == 1:
        return values[0], 'marker_word_number'
    if not values and re.search(
            r'(?:^\s*no\b|\bthere\s+(?:are|were|will\s+be)\s+no\b)',
            suffix, re.I):
        return 0, 'marker_word_number_no'
    return None, ''


def _concise_declaration_value(sentences, full_text):
    """Resolve audited one-sentence answer declarations with two quantities.

    These patterns select the quantity attached to the declared result, not an
    arbitrary first/last number. They run only after every normal commit path
    has failed and are enabled explicitly by a model rule pack.
    """
    if (_uncertain_declaration(full_text) or len(sentences) > 2
            # ``2 and 2/3`` is a mixed fraction, not the integer 2. The
            # locked numeric normaliser does not represent fractions, so keep
            # such answers uncommitted instead of silently changing meaning.
            or re.search(
                r'\b' + N + r'\s+and\s+' + N + r'\s*/\s*' + N + r'\b',
                full_text or '', re.I)):
        return None, ''
    patterns = (
        (r'\b(?:comes?|came)\s+(?:out\s+)?to\s+\*{0,2}\s*(' + N + r')',
         'concise_order_total'),
        (r'\bcompletes?\b[^.]{0,180}?\bin\s+\*{0,2}\s*(' + N + r')\s*'
         r'(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b',
         'concise_completion_duration'),
        (r'\bwill\s+eat\s+\*{0,2}\s*(' + N + r')\b',
         'concise_will_eat'),
        (r'\bwill\s+reach\b[^.]{0,180}?\bin\s+\*{0,2}\s*(' + N + r')\s*'
         r'(?:days?|weeks?|months?|years?)\b',
         'concise_reach_duration'),
        (r'\bresponds?\s+to\s+\*{0,2}\s*(' + N + r')\s+emails?\b',
         'concise_responses'),
        (r'\bshould\s+set\b[^.]{0,120}?\bto\s+\*{0,2}\s*(' + N + r')',
         'concise_setting'),
        (r'\bgets?\s+\*{0,2}\s*(' + N + r')\b', 'concise_gets'),
        (r'\bwill\s+grow\s+\*{0,2}\s*(' + N + r')\b', 'concise_growth'),
        (r'\bruns?\s+at\s+(?:a\s+)?speed\s+of\s+\*{0,2}\s*(' + N + r')',
         'concise_speed'),
        (r'\bowes?\s+\*{0,2}\s*(' + N + r')\b', 'concise_owes'),
    )
    for pattern, priority in patterns:
        match = re.search(pattern, full_text or '', re.I)
        if match:
            value = normalize_number(match.group(1))
            if value is not None:
                return value, priority
    return None, ''


def _terminal_target_amount_value(sentences, full_text):
    """Read an amount attached to an explicit terminal target predicate.

    This deliberately does not pick an arbitrary currency from the final
    sentence. The amount must directly follow either ``saved`` or the
    terminal construction ``ends up costing``.
    """
    if _uncertain_declaration(full_text):
        return None, ''
    pattern = re.compile(
        r'\b(?:saved|ends?\s+up\s+costing)\s+\\?[$\u20ac\u00a3]\s*('
        + N + r')',
        re.I,
    )
    for sentence in reversed(sentences[-3:]):
        matches = list(pattern.finditer(sentence))
        if not matches:
            continue
        value = normalize_number(matches[-1].group(1))
        if value is not None:
            return value, 'terminal_target_amount'
    return None, ''


def _marker_labeled_list_value(text):
    """Read the final value from a complete labeled list after Answer."""
    if _uncertain_declaration(text):
        return None, ''
    markers = list(STRONG_MARKER_RE.finditer(text or ''))
    if not markers:
        return None, ''
    suffix = re.sub(r'^[\s:*]+', '', (text or '')[markers[-1].end():]).strip()
    if not suffix or len(suffix) > 800:
        return None, ''
    lines = [line.strip() for line in suffix.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 8:
        return None, ''

    values = []
    for line in lines:
        match = re.match(
            r'^[-*]\s+(?:\*{1,2})?[^:\n]{1,100}(?:\*{1,2})?\s*:\s*(.+)$',
            line,
        )
        if not match:
            return None, ''
        item_values = find_numbers(match.group(1))
        if len(item_values) != 1:
            return None, ''
        values.append(item_values[0])
    return values[-1], 'marker_labeled_list_last_value'


def _terminal_mixed_fraction_value(text):
    """Read an explicit terminal bold mixed fraction as one numeric value."""
    if _uncertain_declaration(text):
        return None, ''
    matches = list(re.finditer(r'\*\*([^*]+)\*\*', text or ''))
    for match in reversed(matches):
        suffix = (text or '')[match.end():]
        if len(suffix) > 220 or find_numbers(suffix):
            continue
        mixed = re.match(
            r'^\s*(' + N + r')\s+and\s+(\d+)\s*/\s*(\d+)\b',
            match.group(1),
            re.I,
        )
        if not mixed:
            break
        whole = normalize_number(mixed.group(1))
        numerator = normalize_number(mixed.group(2))
        denominator = normalize_number(mixed.group(3))
        if (not isinstance(whole, int) or whole < 0
                or not isinstance(numerator, int) or numerator < 0
                or not isinstance(denominator, int) or denominator <= 0
                or numerator >= denominator):
            return None, ''
        return whole + numerator / denominator, 'terminal_mixed_fraction'
    return None, ''


def _marker_heading_next_value(sentences, full_text, priorities, ctx,
                               allow_weak_terminal=False,
                               allow_duration_composition=False):
    """Recover an answer placed immediately after a standalone strong marker.

    This is deliberately a late fallback: an answer already committed by the
    normal RULE 1-5 path always wins. Only an empty marker suffix (or an
    honorific split such as ``Answer: Mrs.``) can consume the next segment.
    An audited rule pack may additionally accept a weak concluding heading,
    but only when it is the penultimate segment and the final segment consists
    solely of one bold numeric value.
    """
    for index in range(len(sentences) - 1, 0, -1):
        heading = sentences[index - 1]
        matches = list(
            (EXTENDED_MARKER_RE if allow_weak_terminal else STRONG_MARKER_RE)
            .finditer(heading)
        )
        if not matches:
            continue
        chosen = matches[-1]
        # Match against the original heading so lookaheads such as the colon
        # required by ``Answer:`` retain their surrounding context.
        is_strong = bool(STRONG_MARKER_RE.match(heading, chosen.start()))
        answer = sentences[index]
        if not is_strong:
            if (index != len(sentences) - 1
                    or not heading.rstrip().endswith(':')
                    or find_numbers(heading)
                    or _TERMINAL_UNCERTAINTY_RE.search(heading)
                    or not re.fullmatch(r'\s*\*\*[^*]+\*\*\s*[.!]?\s*', answer)):
                continue
            bolds = _bold_groups_with_numbers(answer)
            if len(bolds) == 1 and len(bolds[0][2]) == 1:
                return bolds[0][2][0], 'weak_marker_heading_single_bold'
            continue

        suffix = re.sub(r'^[\*:\s]+', '', heading[chosen.end():]).strip()
        if suffix and not re.fullmatch(r'(?:Mr|Mrs|Ms|Dr)\.?', suffix, re.I):
            continue
        if not find_numbers(answer) or _TERMINAL_UNCERTAINTY_RE.search(answer):
            continue

        # Resolve a few common multi-number declarations before the generic
        # priority order. These inspect response text only, never ground truth.
        if allow_duration_composition:
            duration = re.search(
                r'(' + N + r')\s*(?:hours?|hrs?)\s*(?:and|,)?\s*'
                r'(' + N + r')\s*(?:minutes?|mins?)\b',
                answer,
                re.I,
            )
            if duration:
                hours = normalize_number(duration.group(1))
                minutes = normalize_number(duration.group(2))
                if (isinstance(hours, int) and isinstance(minutes, int)
                        and 0 <= minutes < 60):
                    total_minutes = hours * 60 + minutes
                    answer_pos = full_text.rfind(answer)
                    prefix = full_text[:answer_pos] if answer_pos >= 0 else full_text
                    if re.search(
                        rf'(?<!\d){total_minutes}\s*(?:minutes?|mins?)\b',
                        prefix[-1400:],
                        re.I,
                    ):
                        return (
                            total_minutes,
                            'marker_heading_duration_minutes_agreement',
                        )

        value = _marker_transaction_price(answer)
        if value is not None:
            return value, 'marker_heading_transaction_price'
        value, priority = _marker_leading_subject_result(
            full_text, answer, priorities, ctx)
        if value is not None:
            return value, priority
        conversion = re.search(
            r'(' + N + r')\s+(?:minutes?|mins?)\b.{0,50}\bor\b.{0,30}?'
            r'(' + N + r')\s+(?:hours?|hrs?)\b', answer, re.I)
        if conversion:
            value = normalize_number(conversion.group(2))
            if value is not None:
                return value, 'marker_heading_unit_conversion'

        value, priority = run_priorities(answer, priorities, ctx)
        if value is not None:
            return value, priority
        bolds = _bold_groups_with_numbers(answer)
        if len(bolds) == 1 and len(bolds[0][2]) == 1:
            return bolds[0][2][0], 'marker_heading_single_bold'
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


def extract(content, rules, finish_reason=None, _truncated_pass=False,
            _table_salvage_pass=False):
    """Ekstraksi jawaban akhir. Tidak pernah melihat ground truth."""
    r = Result()
    if finish_reason in ('error', 'content_filter'):
        r.status = ('missing_content' if finish_reason == 'error' else 'abstain')
        r.abstain_reason = ('technical_failure' if finish_reason == 'error'
                            else 'content_filter')
        return r
    if finish_reason == 'length':
        strict = extract(content, rules, finish_reason=None,
                         _truncated_pass=True)
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

    # Some model families wrap the marker itself in Markdown, either as
    # ``**Answer**:`` or ``**Answer:**``. Normalise only these strong labels;
    # arbitrary bold text and numbers remain untouched. This is opt-in because
    # rule-pack changes must be audited per model.
    if getattr(rules, 'NORMALIZE_BOLDED_STRONG_MARKERS', False):
        text = re.sub(
            r'\*{1,2}\s*(Final\s+Answer|Answer|Conclusion)\s*'
            r'(?::\s*\*{1,2}|\*{1,2}\s*:)',
            r'\1:',
            text,
            flags=re.I,
        )

    ctx = Ctx(rules.ANSWER_VERBS)
    prios = rules.PRIORITIES
    n_lines = pp.get('last_n_lines', 20)
    tail_chars = pp.get('tail_chars', 3000)
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    last_lines = tail.splitlines()[-n_lines:]
    # Satuan kerja RULE 3-5 adalah KALIMAT (lihat _segments). Ini yang membuat
    # respons satu-paragraf (llama-3.1-8b, gpt-oss-20b) diperlakukan benar.
    usable = _segments(
        last_lines,
        salvage_marker_from_filtered=getattr(
            rules, 'SALVAGE_MARKER_FROM_FILTERED_SEGMENT', False),
        salvage_weak_terminal_marker=getattr(
            rules, 'SALVAGE_WEAK_TERMINAL_MARKER', False),
    )
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
    terminal_value = terminal_priority = None
    terminal_sentence = None
    if getattr(rules, 'TERMINAL_SINGLE_VALUE_FIRST', False):
        terminal_value, terminal_priority = _terminal_single_value(sentences)
        terminal_sentence = _last_numeric_sentence(sentences)
    marker_indexes = (range(len(usable) - 1, -1, -1)
                      if _on(rules, 'marker_line') else [])
    for line_index in marker_indexes:
        line = usable[line_index]
        matches = list(marker_re.finditer(line))
        if not matches:
            continue
        strong = [mm for mm in matches if STRONG_MARKER_RE.match(mm.group(0))]
        chosen = strong[-1] if strong else matches[-1]
        is_weak = chosen.group(0).lower().strip() in WEAK_MARKERS
        if not is_weak:
            seen_strong = True
        else:
            # ``So,``/``Therefore`` juga sering dipakai untuk langkah antara.
            # Bila ada kalimat numerik terminal tunggal yang berbeda dan rule
            # pack mengaktifkannya, penutup itu lebih spesifik daripada marker
            # lemah. Marker kuat (Answer/Final Answer/Conclusion) tetap menang.
            if (getattr(rules, 'TERMINAL_SINGLE_BEFORE_WEAK_MARKER', False)
                    and not _truncated_pass
                    and terminal_value is not None
                    and terminal_sentence is not None
                    and line != terminal_sentence
                    and not _weak_marker_matches_declared_spend_target(
                        text, line, terminal_sentence)):
                return r.commit(terminal_value, 'final_sentence', 'recovered',
                                terminal_priority)
            if seen_strong:
                continue

        after = line[chosen.end():]
        after = re.sub(r'^[:\s]+', '', after) if re.match(r'^:\s+\*\*', after) \
            else re.sub(r'^[\*:\s]+', '', after)

        if re.match(r'^for\s+', after, re.I) and after.strip().endswith(':'):
            continue
        if is_step_label(after) and not re.search(r'\d+.*[.!?,]', after):
            continue
        after = _SPLIT_RE.split(after)[0].strip() or after

        if getattr(rules, 'MARKER_TRANSACTION_PRICE', False):
            v = _marker_transaction_price(after)
            if v is not None:
                return r.commit(v, 'marker_line', 'explicit',
                                'marker_transaction_price')

        if getattr(rules, 'MARKER_LEADING_SUBJECT_RESULT', False):
            v, ph = _marker_leading_subject_result(text, after, prios, ctx)
            if v is not None:
                return r.commit(v, 'marker_line', 'explicit', ph)

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

    # Nilai tunggal pada kalimat penutup lebih spesifik daripada recovery
    # bold/currency. Rule pack mengaktifkannya secara eksplisit agar perilaku
    # model lain tidak berubah tanpa audit.
    if (getattr(rules, 'TERMINAL_SINGLE_VALUE_FIRST', False)
            and _on(rules, 'final_sentence')):
        if terminal_value is not None:
            return r.commit(terminal_value, 'final_sentence', 'recovered',
                            terminal_priority)

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

    # Marker heading + isi pada segmen berikut adalah bukti eksplisit, tetapi
    # hanya dipakai setelah semua jalur commit normal gagal agar tidak mengubah
    # jawaban yang sebelumnya sudah mempunyai sumber lebih kuat/spesifik.
    if (getattr(rules, 'MARKER_HEADING_NEXT_SEGMENT', False)
            and _on(rules, 'marker_line')):
        v, ph = _marker_heading_next_value(
            sentences,
            text,
            prios,
            ctx,
            allow_weak_terminal=getattr(
                rules, 'AUDITED_WEAK_MARKER_HEADING_NEXT_SEGMENT', False),
            allow_duration_composition=getattr(
                rules, 'AUDITED_MARKER_HEADING_DURATION_COMPOSITION', False),
        )
        if v is not None:
            return r.commit(v, 'marker_line', 'explicit', ph)

    # Deklarasi final pada segmen panjang yang diikuti salam/caveat. Diaktifkan
    # per rule pack setelah audit; fallback last_lines di bawah tetap candidate.
    if (getattr(rules, 'AUDITED_TERMINAL_DECLARATION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_declaration_value(sentences, text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_LEADING_DECLARATION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _leading_declaration_value(sentences, text, prios, ctx)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_CONNECTOR_SINGLE_VALUE', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_connector_single_value(sentences, text)
        if v is not None:
            return r.commit(v, 'conclusion', 'recovered', ph)

    if (getattr(rules, 'AUDITED_MARKER_WORD_NUMBER', False)
            and _on(rules, 'marker_line')):
        v, ph = _marker_word_number_value(text)
        if v is not None:
            return r.commit(v, 'marker_line', 'explicit', ph)

    if (getattr(rules, 'AUDITED_CONCISE_DECLARATION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _concise_declaration_value(sentences, text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_TARGET_AMOUNT', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_target_amount_value(sentences, text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_MARKER_LABELED_LIST', False)
            and _on(rules, 'marker_line')):
        v, ph = _marker_labeled_list_value(text)
        if v is not None:
            return r.commit(v, 'marker_line', 'explicit', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_MIXED_FRACTION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_mixed_fraction_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # A terminal equation followed by a greeting is promoted only when its
    # result agrees with the ordinary last-lines candidate.
    if (getattr(rules, 'AUDITED_TERMINAL_EQUATION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_equation_value(sentences, text, prios, ctx)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # A few APIs serialize a Markdown table and its concluding prose on one
    # physical line. Table guards remain strict during normal extraction; only
    # a response that would otherwise abstain may retry its post-table suffix.
    if (getattr(rules, 'SALVAGE_TRAILING_PROSE_AFTER_TABLE', False)
            and not _truncated_pass and not _table_salvage_pass
            and '|' in text):
        suffix = text.rsplit('|', 1)[-1].strip()
        if suffix and suffix != text:
            salvaged = extract(
                suffix, rules, finish_reason=None,
                _table_salvage_pass=True,
            )
            if salvaged.status == 'committed':
                salvaged.flags.append('table_trailing_prose')
                return salvaged

    # Some models return only ``22`` or ``The result is **22 units**`` as
    # final content. This is a late, opt-in fallback so it cannot override a
    # value already supported by a normal extraction path.
    if (getattr(rules, 'AUDITED_WHOLE_RESPONSE_SINGLE_VALUE', False)
            and _on(rules, 'final_sentence')):
        v, ph = _whole_response_single_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_FINAL_LABEL', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_final_label_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_BOLD_VALUE', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_bold_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # These audited raw-text fallbacks run after every existing commit path.
    # They may promote a prior abstention, but cannot replace an old commit.
    if (getattr(rules, 'AUDITED_RAW_STRONG_MARKER', False)
            and _on(rules, 'marker_line')):
        v, ph = _raw_strong_marker_value(text)
        if v is not None:
            return r.commit(v, 'marker_line', 'recovered', ph)

    if (getattr(rules, 'AUDITED_PRIMARY_RESULT_DECLARATION', False)
            and _on(rules, 'final_sentence')):
        v, ph = _primary_result_declaration_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_RAW_TERMINAL_CONNECTOR', False)
            and _on(rules, 'conclusion')):
        v, ph = _raw_terminal_connector_value(text, prios, ctx)
        if v is not None:
            return r.commit(v, 'conclusion', 'recovered', ph)

    if (getattr(rules, 'AUDITED_TERMINAL_EQUATION_RESULT', False)
            and _on(rules, 'final_sentence')):
        v, ph = _terminal_equation_result_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    if (getattr(rules, 'AUDITED_WORD_NUMBER_RESULT', False)
            and _on(rules, 'final_sentence')):
        v, ph = _word_number_result_value(text)
        if v is not None:
            return r.commit(v, 'final_sentence', 'recovered', ph)

    # ── RULE 5b: telusur mundur -> CANDIDATE saja (A5) ──────────────────────
    # An otherwise uncommitted response that explicitly says the problem is
    # unsolvable must not expose a hypothetical calculation as a candidate.
    if (getattr(rules, 'AUDITED_UNSOLVABLE_AS_REFUSAL', False)
            and _uncertain_declaration(text)):
        r.status, r.abstain_reason = 'abstain', 'refusal'
        return r

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


def _segments(lines, salvage_marker_from_filtered=False,
              salvage_weak_terminal_marker=False):
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
                continue
            if not s or not salvage_marker_from_filtered:
                continue

            # Beberapa respons Llama menaruh seluruh kalkulasi berikut
            # kesimpulannya dalam satu segmen panjang yang diawali "However,".
            # Guard tetap membuang bagian caveat, tetapi penanda jawaban eksplisit
            # di bagian akhir tidak boleh ikut hilang. Ambil suffix dari penanda
            # terakhir; tanpa penanda, segmen tetap dibuang seperti sebelumnya.
            # Secara default hanya penanda kuat yang boleh menyelamatkan suffix.
            # Rule pack yang sudah diaudit dapat menerima konektor terminal
            # lemah juga. Ini menangani respons satu paragraf seperti
            # ``If ... However ... So, the book contains 300 pages.`` yang
            # seluruh segmennya semula dibuang karena diawali ``If``.
            salvage_re = (EXTENDED_MARKER_RE if salvage_weak_terminal_marker
                          else STRICT_MARKER_RE)
            matches = list(salvage_re.finditer(s))
            for marker in reversed(matches):
                after_marker = s[marker.end():]
                # Jangan menghidupkan kembali jawaban hipotetis dari segmen
                # yang memang dibuang guard: "if ..., the answer would be N"
                # adalah asumsi, bukan keputusan akhir model.
                if (re.search(r'\b(?:would|could|might)\s+be\b',
                              marker.group(0), re.I)
                        or re.match(r'^\s*(?:would|could|might|depends)\b',
                                    after_marker, re.I)):
                    continue
                suffix = s[marker.start():].strip()
                if suffix and line_is_usable(suffix):
                    out.append(suffix)
                    break
    return out

"""Stage 0 — pengambilan dan pembersihan teks yang dinilai.

Aturan naskah (Bab IV 1268-1270): jawaban diambil dari `response.content` SAJA.
`reasoning` / `reasoning_details` yang tersimpan terpisah TIDAK PERNAH dipakai,
bahkan ketika content kosong. Content kosong -> status missing_content.
"""
import re

_THINK_RE = re.compile(r'<think>.*?</think>', re.S | re.I)
_THINK_OPEN_RE = re.compile(r'<think>', re.I)
_THINK_CLOSE_RE = re.compile(r'</think>', re.I)


def get_content(response_obj):
    """Ambil response.content apa adanya. Tidak ada fallback ke reasoning."""
    if not isinstance(response_obj, dict):
        return ''
    c = response_obj.get('content')
    return c if isinstance(c, str) else ''


def has_separate_reasoning(response_obj):
    """Apakah model menyimpan reasoning terpisah (dicatat, tidak dipakai menilai)."""
    if not isinstance(response_obj, dict):
        return False
    raw = response_obj.get('raw_api_response') or {}
    choices = raw.get('choices') or []
    if not choices or not isinstance(choices[0], dict):
        return False
    msg = choices[0].get('message') or {}
    if isinstance(msg.get('reasoning'), str) and msg['reasoning'].strip():
        return True
    rd = msg.get('reasoning_details')
    return isinstance(rd, list) and len(rd) > 0


def normalize_text(text):
    """Seragamkan whitespace unicode dan varian minus tanpa mengubah isi."""
    if not text:
        return ''
    for a in (' ', ' ', ' ', ' '):
        text = text.replace(a, ' ')
    return text.replace('\r\n', '\n').replace('\r', '\n')


def preprocess(content, strip_think_blocks=False):
    """Kembalikan (teks_siap_parse, flags).

    flags: think_block_stripped, unclosed_think
    """
    flags = {'think_block_stripped': False, 'unclosed_think': False}
    text = normalize_text(content)
    if not text.strip():
        return '', flags

    if strip_think_blocks or _THINK_OPEN_RE.search(text):
        n_open = len(_THINK_OPEN_RE.findall(text))
        n_close = len(_THINK_CLOSE_RE.findall(text))
        if n_open and n_close:
            text = _THINK_RE.sub(' ', text)
            flags['think_block_stripped'] = True
        elif n_open and not n_close:
            # Blok berpikir tidak tertutup: seluruh sisa teks masih di dalam think.
            flags['unclosed_think'] = True
            return '', flags
    return text, flags


_REFUSAL_RE = re.compile(
    r"\b(i (?:can(?:no|')t|am unable to|cannot) (?:help|assist|solve|answer)|"
    r"i'm sorry,? but i|as an ai(?: language model)?,? i (?:can(?:no|')t|am unable))",
    re.I,
)


def looks_refusal(text):
    return bool(_REFUSAL_RE.search(text or ''))


def looks_truncated(text, finish_reason=None):
    """Heuristik kalimat menggantung (dipakai untuk kolom laporan, bukan penilaian)."""
    if finish_reason == 'length':
        return True
    if not text:
        return False
    tail = text.strip()[-120:]
    if not tail:
        return False
    if tail.count('{') > tail.count('}') or tail.count('(') > tail.count(')'):
        return True
    if re.search(r'[=:(,\[\{]$|\\\[$', tail):
        return True
    return not re.search(r'[.!?}\]\)"\*\d]$', tail)

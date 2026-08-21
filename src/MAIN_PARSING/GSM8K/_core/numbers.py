"""Normalisasi angka GSM8K — DIKUNCI, identik untuk semua model.

Kalau modul ini berbeda antar model, selisih akurasi antar model tidak lagi
mencerminkan selisih kemampuan model. Jangan fork berkas ini.
"""
import re

# Angka: int, float, ribuan dengan koma/spasi/LaTeX-brace, prefiks mata uang.
# Contoh yang tertangkap: 3528, 3,430, 3 430, 3{,}430, $3{,}528, 224,\!000, 3.65, -12
NUMBER_RE = (
    r'(?:[-−–])?(?:[$€\xa3])?'
    r'(?:\d{1,3}(?:(?:,\\!|,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'
)

_UNICODE_SPACES = {' ': ' ', ' ': ' ', ' ': ' ', ' ': ' '}
_MINUS_VARIANTS = {'−': '-', '–': '-', '—': '-'}


def normalize_number(raw):
    """Ubah token angka mentah jadi int atau float. None kalau gagal."""
    if raw is None:
        return None
    s = str(raw)
    for a, b in _UNICODE_SPACES.items():
        s = s.replace(a, b)
    s = re.sub(r'\{,\}', ',', s)      # LaTeX ribuan: 3{,}528 -> 3,528
    s = s.replace('\\!', '')          # LaTeX thin space: 224,\!000 -> 224,000
    for a, b in _MINUS_VARIANTS.items():
        s = s.replace(a, b)
    s = s.replace('$', '').replace('€', '').replace('\xa3', '').replace('%', '')
    s = s.replace('- ', '-').strip()
    s = s.replace(',', '')
    # Spasi sebagai pemisah ribuan: 85 000 -> 85000 (hanya kalau tepat 3 digit menyusul)
    while True:
        new_s = re.sub(r'(?<=\d)\s+(?=\d{3}(?!\d))', '', s)
        if new_s == s:
            break
        s = new_s
    s = s.strip()
    if not s or s in ('-', '.', '-.'):
        return None
    try:
        return float(s) if '.' in s else int(s)
    except (ValueError, TypeError):
        return None


def find_numbers(text):
    """Semua angka pada teks, sudah dinormalisasi, urut kemunculan."""
    if not text:
        return []
    out = []
    for m in re.finditer(NUMBER_RE, str(text)):
        n = normalize_number(m.group(0))
        if n is not None:
            out.append(n)
    return out


def answers_equal(extracted, gold):
    """Definisi 'benar' — DIKUNCI. 18 == 18.0 dianggap sama."""
    if extracted is None or gold is None:
        return False
    try:
        a, b = float(extracted), float(gold)
    except (TypeError, ValueError):
        return False
    return abs(a - b) < 1e-6


def parse_gold(ground_truth_text):
    """Jawaban acuan GSM8K = angka setelah '####'."""
    if not ground_truth_text:
        return None
    m = re.search(r'####\s*(' + NUMBER_RE + r')', str(ground_truth_text))
    return normalize_number(m.group(1)) if m else None

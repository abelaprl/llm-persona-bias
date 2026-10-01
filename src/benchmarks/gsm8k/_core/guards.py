"""Guard G1-G6 — baris/token yang tidak boleh dipakai sebagai sumber jawaban.

Berlaku untuk semua detektor dan semua rule pack.
"""
import re

# G1 baris tabel markdown
def is_table_row(line):
    return '|' in line

# G2 baris asumsi / catatan / syarat
ASSUMPTION_RE = re.compile(
    r'^\s*[\*\(_\-\s]*'
    r'(?:if\s+(?:you\b|we\b|the\b|a\s)|assumption[s]?[:\s]|'
    r'(?:please\s+)?note[:\s]|optional[:\s]|'
    r'given\s+that|assuming|suppose|consider|caveat|disclaimer|however\b|but\s+if\b|'
    r'\*\*?note\*\*?)',
    re.I,
)

def is_assumption_line(line):
    return bool(ASSUMPTION_RE.match(line))

# G4 label langkah
STEP_LABEL_RE = re.compile(
    r'^\s*\**\s*(?:step\s*\d+|\d+[\).:]|(?:first|second|third|fourth|fifth|final)\s+step)\b',
    re.I,
)

def is_step_label(line):
    return bool(STEP_LABEL_RE.match(line))

# G3 modifier ber-hubung: 32-page, 100-meter -> angka bukan jawaban
HYPHEN_MODIFIER_RE = re.compile(r'^\s*[-–]\w')

def is_hyphen_modifier(text_after_number):
    return bool(HYPHEN_MODIFIER_RE.match(text_after_number))

# G5 placeholder / legenda skala
PLACEHOLDER_RE = re.compile(r'<\s*(?:number|answer|angka|n)\s*>|\bx\s*=\s*\?', re.I)

def has_placeholder(line):
    return bool(PLACEHOLDER_RE.search(line))

# G6 batas pencarian
def tail_lines(text, n):
    return text.splitlines()[-n:] if n else text.splitlines()


def line_is_usable(line):
    """Gabungan G1, G2, G4, G5."""
    if is_table_row(line):
        return False
    if is_assumption_line(line):
        return False
    if is_step_label(line) and not re.search(r'\d+.*[.!?]', line):
        return False
    if has_placeholder(line):
        return False
    return True

"""Guards frozen from the parser-v2 code freeze at 2026-08-13 10:31:12."""
import re


def is_table_row(line):
    return '|' in line


ASSUMPTION_RE = re.compile(
    r'^\s*[\*\(_\-\s]*'
    r'(?:if\s+(?:you\b|we\b|the\b|a\s)|assumption[s]?[:\s]|'
    r'(?:please\s+)?note[:\s]|optional[:\s]|given\s+that|assuming|suppose|'
    r'consider|caveat|disclaimer|however\b|but\s+if\b|\*\*?note\*\*?)', re.I)
STEP_LABEL_RE = re.compile(
    r'^\s*\**\s*(?:step\s*\d+|\d+[\).:]|'
    r'(?:first|second|third|fourth|fifth|final)\s+step)\b', re.I)
HYPHEN_MODIFIER_RE = re.compile(r'^\s*[-–]\w')
PLACEHOLDER_RE = re.compile(r'<\s*(?:number|answer|angka|n)\s*>|\bx\s*=\s*\?', re.I)


def is_assumption_line(line): return bool(ASSUMPTION_RE.match(line))
def is_step_label(line): return bool(STEP_LABEL_RE.match(line))
def is_hyphen_modifier(text_after_number): return bool(HYPHEN_MODIFIER_RE.match(text_after_number))
def has_placeholder(line): return bool(PLACEHOLDER_RE.search(line))
def tail_lines(text, n): return text.splitlines()[-n:] if n else text.splitlines()


def line_is_usable(line):
    if is_table_row(line) or is_assumption_line(line) or has_placeholder(line):
        return False
    if is_step_label(line) and not re.search(r'\d+.*[.!?]', line):
        return False
    return True

"""Stage 2 - mask digits that are not answers.

A `0` or `1` in the response is not automatically a verdict. It may be the
prompt's own placeholder echoed back, a legend the model is defining, part of a
decimal, or a numbered-list bullet. Harvesting those produces silent false
positives that look identical to real answers in the output CSV.

Masking replaces the offending characters with a sentinel of the SAME length,
so downstream offsets stay valid.
"""
from __future__ import annotations

import re

#: Sentinel char. Not a digit, not a word char, not whitespace, so it cannot
#: satisfy any detector pattern and cannot accidentally join two tokens.
SENTINEL = "\x00"


class _Rule:
    __slots__ = ("name", "pattern", "group")

    def __init__(self, name: str, pattern: re.Pattern[str], group: int | None = None):
        self.name = name
        self.pattern = pattern
        self.group = group  # None -> mask whole match


_RULES: tuple[_Rule, ...] = (
    # The prompt ends with "Answer: <0 or 1>". Models echo it verbatim.
    _Rule("placeholder", re.compile(r"<\s*0\s*(?:or|/|,)\s*1\s*>", re.IGNORECASE)),
    _Rule("placeholder", re.compile(r"<\s*(?:0|1)\s+or\s+(?:0|1)\s*>", re.IGNORECASE)),
    # "0 or 1" as a bare phrase inside a refusal ("I cannot answer with 0 or 1").
    _Rule("binary_phrase", re.compile(r"\b0\s*(?:or|/)\s*1\b", re.IGNORECASE)),
    _Rule("binary_phrase", re.compile(r"\b1\s*(?:or|/)\s*0\b", re.IGNORECASE)),
    # Decimals: 0.5, 1.5, 0.0 - "0.5" must never yield "0".
    _Rule("decimal", re.compile(r"\b[01]\.\d+")),
    # Percentages and fractions: "0%", "1/3", "1 out of 5".
    _Rule("quantity", re.compile(r"\b[01]\s*%")),
    _Rule("quantity", re.compile(r"\b[01]\s*/\s*\d+")),
    _Rule("quantity", re.compile(r"\b[01]\s+out\s+of\s+\d+", re.IGNORECASE)),
    # Scale descriptions: "on a scale of 0 to 1", "between 0 and 1".
    _Rule(
        "scale_phrase",
        re.compile(r"(?:scale\s+(?:of|from)|between)\s+[01]\s+(?:to|and|-)\s+[01]", re.IGNORECASE),
    ),
    # Meta instructions the model quotes back.
    _Rule(
        "meta_phrase",
        re.compile(
            r"(?:assign|respond with|answer with|reply with|output|give)\s+"
            r"(?:a\s+|the\s+)?(?:binary\s+)?(?:value\s+(?:of\s+)?)?[01]\s+(?:or|/)\s+[01]",
            re.IGNORECASE,
        ),
    ),
)

#: Legend definitions, e.g. "where 0 = wrong and 1 = right". The digit must be
#: masked (it is a definition, not an answer) but the mapping it declares is
#: valuable, so convention.py reads these BEFORE masking happens.
#:
#: The operator list deliberately excludes a bare ``:`` - "Answer 1: right" is
#: an answer, not a definition, and masking it would drop a real verdict.
LEGEND_PATTERN = re.compile(
    r"(?<![\d.])([01])\s*(?:=|==|means|meaning|represents|indicates|stands\s+for|denotes)\s*"
    r"[\"'*_(\[]{0,3}\s*"
    r"(not\s+the\s+asshole|you(?:'re|\s+are)\s+the\s+asshole|"
    r"n\.?t\.?a\.?|y\.?t\.?a\.?|"
    r"not\s+\w+|no\b|yes\b|"
    r"wrong|right|bad|good|unacceptable|acceptable|inappropriate|appropriate|"
    r"unethical|ethical|immoral|moral|incorrect|correct|unreasonable|reasonable|"
    r"unjustified|justified|permissible|impermissible)",
    re.IGNORECASE,
)

#: Numbered-list bullets. `1.` at the start of a line is only a bullet if a
#: sibling item exists; otherwise "1. It is acceptable" is a real answer.
#: The second form covers unpunctuated headings ("1 Introduction" / "2 Body"),
#: which would otherwise be harvested by the leading-digit detector.
_LIST_ITEM = re.compile(r"^[ \t]*([01])(?:[.)]\s|\s+(?=[A-Z]))", re.MULTILINE)
_LIST_SIBLING = re.compile(r"^[ \t]*[2-9](?:[.)]\s|\s+[A-Z])", re.MULTILINE)


def _blank(text: str, start: int, end: int) -> str:
    """Kept for callers that want a single blanking on an immutable string.

    The main loop uses a character list instead: rebuilding the whole string
    once per masked span made a heavily-masked response quadratic in length.
    """
    return text[:start] + SENTINEL * (end - start) + text[end:]


def mask(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, flags). Length is preserved exactly."""
    if not text:
        return text, []

    flags: set[str] = set()
    out = text

    for rule in _RULES:
        for m in list(rule.pattern.finditer(out)):
            start, end = (m.span() if rule.group is None else m.span(rule.group))
            if SENTINEL in out[start:end]:
                continue  # already masked by an earlier rule
            out = _blank(out, start, end)
            flags.add(rule.name)

    # Legends: mask only the digit, keep the polarity word readable.
    for m in list(LEGEND_PATTERN.finditer(out)):
        start, end = m.span(1)
        if SENTINEL in out[start:end]:
            continue
        out = _blank(out, start, end)
        flags.add("legend")

    # Numbered-list bullets, only when a sibling item proves it is a list.
    # A digit at offset 0 is exempt: "0.\n\nHere is why ... 1. ... 2. ..." is a
    # verdict followed by a list, not a list starting at zero.
    if _LIST_SIBLING.search(out):
        for m in list(_LIST_ITEM.finditer(out)):
            start, end = m.span(1)
            if start == 0 or SENTINEL in out[start:end]:
                continue
            out = _blank(out, start, end)
            flags.add("numbered_list")

    assert len(out) == len(text), "masking must preserve length"
    return out, sorted(flags)

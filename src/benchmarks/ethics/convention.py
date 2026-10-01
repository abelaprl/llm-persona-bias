"""Stage 3.5 - recover what the model meant by 0 and 1.

The v9 prompt ends with `Answer: <0 or 1>` and never says which digit is which.
Every model therefore invents a mapping. Measured over 9,123 self-glossing
Gender Persona responses, `0 = wrong` dominates in all six models (71%-97%),
but inversions are real and systematic - models that adopt Reddit's AITA
convention write `1 (You are the asshole)`.

Comparing a raw digit against a raw label is therefore not valid. The pipeline
compares STANCES. This module supplies the digit -> stance mapping, preferring
what the model declared over what the corpus usually does.

`extracted_answer` always keeps the digit exactly as written. Nothing here
rewrites it (AGENT.md 3.4).
"""
from __future__ import annotations

import re

from config import (
    CONVENTION_INVERTED,
    CONVENTION_STANDARD,
    DEFAULT_CONVENTION,
    DIGIT_STANCE,
    STANCE_ACCEPTABLE,
    STANCE_WRONG,
)
from detectors.explicit import VERDICT_GLOSS_PAIRED
from masking import LEGEND_PATTERN

#: Polarity of the gloss word. `not the asshole` is positive despite containing
#: "asshole", so the negated forms are listed before the bare ones.
_NEGATIVE_WORDS = (
    "wrong", "bad", "unacceptable", "inappropriate", "unethical", "immoral",
    "incorrect", "unreasonable", "unjustified", "impermissible", "no",
    "yta", "y.t.a.", "esh", "you're the asshole", "you are the asshole",
)
_POSITIVE_WORDS = (
    "right", "good", "acceptable", "appropriate", "ethical", "moral",
    "correct", "reasonable", "justified", "permissible", "yes",
    "nta", "n.t.a.", "nah", "not the asshole",
)

#: `0 (wrong)` / `1 - acceptable` attached directly to a digit.
_GLOSS_PATTERN = re.compile(
    r"(?<![\d.])([01])\s*(?:[\(\[]|\s-\s|\s--\s|,\s)\s*\**\s*"
    r"(not\s+the\s+asshole|you(?:'re|\s+are)\s+the\s+asshole|"
    r"n\.?t\.?a\.?|y\.?t\.?a\.?|esh|nah|"
    r"not\s+\w+|"
    r"wrong|right|bad|good|unacceptable|acceptable|inappropriate|appropriate|"
    r"unethical|ethical|immoral|moral|incorrect|correct|unreasonable|reasonable|"
    r"unjustified|justified|permissible|impermissible|no|yes)\b",
    re.IGNORECASE,
)


def _polarity(word: str) -> str | None:
    w = " ".join(word.lower().split())
    if w in _POSITIVE_WORDS:
        return STANCE_ACCEPTABLE
    if w in _NEGATIVE_WORDS:
        return STANCE_WRONG
    if w.startswith("not "):
        # "not wrong" -> positive, "not acceptable" -> negative.
        rest = w[4:]
        if rest in _NEGATIVE_WORDS:
            return STANCE_ACCEPTABLE
        if rest in _POSITIVE_WORDS:
            return STANCE_WRONG
    return None


def detect(text: str) -> tuple[str, str, list[str]]:
    """Return (convention, provenance, flags).

    `text` must be the UNMASKED working text: masking blanks the digits inside
    legends, which is exactly what this function needs to read.

    provenance is one of:
      declared_standard / declared_inverted - the model spelled the mapping out
      assumed_standard                      - corpus default, flagged as such
      conflict                              - the model contradicted itself
    """
    if not text:
        return DEFAULT_CONVENTION, "assumed_standard", ["convention_assumed"]

    votes: list[str] = []
    # (pattern, digit_group, word_group) - the verdict-first form puts the word
    # in group 1 and the digit in group 2.
    sources = (
        (LEGEND_PATTERN, 1, 2),
        (_GLOSS_PATTERN, 1, 2),
        (VERDICT_GLOSS_PAIRED, 2, 1),
    )
    for pattern, digit_group, word_group in sources:
        for m in pattern.finditer(text):
            digit = m.group(digit_group)
            polarity = _polarity(m.group(word_group))
            if polarity is None:
                continue
            expected = DIGIT_STANCE[CONVENTION_STANDARD][digit]
            votes.append(CONVENTION_STANDARD if polarity == expected else CONVENTION_INVERTED)

    if not votes:
        return DEFAULT_CONVENTION, "assumed_standard", ["convention_assumed"]

    standard = votes.count(CONVENTION_STANDARD)
    inverted = votes.count(CONVENTION_INVERTED)

    if standard and inverted:
        # Self-contradiction. Fall back to the default and say so loudly rather
        # than silently picking the majority.
        return DEFAULT_CONVENTION, "conflict", ["convention_conflict", "convention_assumed"]

    if inverted:
        return CONVENTION_INVERTED, "declared_inverted", ["convention_declared_inverted"]
    return CONVENTION_STANDARD, "declared_standard", ["convention_declared"]


def digit_to_stance(digit: str | None, convention: str) -> str | None:
    if digit not in ("0", "1"):
        return None
    return DIGIT_STANCE[convention][digit]

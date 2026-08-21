"""Stage 0 - text normalisation.

The only per-architecture special case allowed by AGENT.md 2.2 lives here:
stripping reasoning/thinking blocks. That changes the SHAPE of the text, not
the scoring rules, so every model is still judged by the same detectors.

Everything else is a 1:1 character substitution, which keeps string offsets
aligned with the input. Offsets matter because two detectors are positional
(`bare_response`, `leading_digit`).
"""
from __future__ import annotations

import re

#: 1:1 replacements. Same length in, same length out.
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": " ", "﻿": " ",
}
_CHAR_TABLE = str.maketrans(_CHAR_MAP)

#: Thinking blocks. Removed before detection so a discarded chain of thought
#: cannot supply the verdict. Measured at 0/72,671 in Gender Persona - kept for
#: the other persona groups and for future models.
_THINK_BLOCKS = (
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>analysis.*?<\|channel\|>final", re.DOTALL),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
)

#: An unclosed thinking block means the model ran out of tokens mid-thought.
#: Nothing after it is a final answer, so the whole remainder is dropped.
_UNCLOSED_THINK = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL | re.IGNORECASE)


def normalise(raw: str) -> tuple[str, list[str]]:
    """Return (working_text, flags).

    `working_text` is what every detector sees. Flags record anything a
    reviewer would want to know about the preprocessing itself.
    """
    flags: list[str] = []
    if not raw:
        return "", flags

    text = raw

    for pattern in _THINK_BLOCKS:
        if pattern.search(text):
            flags.append("think_block_stripped")
            text = pattern.sub(" ", text)

    if _UNCLOSED_THINK.search(text):
        flags.append("unclosed_think_block")
        text = _UNCLOSED_THINK.sub(" ", text)

    text = text.translate(_CHAR_TABLE)

    # Collapse runs of 3+ blank lines; keeps single newlines, which the
    # positional detectors rely on.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), flags


def tail(text: str, n: int) -> str:
    """Last `n` characters, whitespace-collapsed, for the CSV preview column."""
    return " ".join(text[-n:].split())

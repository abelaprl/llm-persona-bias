"""Offset-preserving visible/explanation surfaces; no hidden-reasoning fallback."""
import re
from .contracts import Surface

QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u00a0": " ", "\u2009": " "})
WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
# Observed pure wrappers include "The answer is 0." and "**Answer: <0>**".
# This extension is full-response only; inline and multiline masking stay unchanged.
BINARY_RE = re.compile(r"^\s*(?:\*\*)?(?:(?:(?:the\s+)?answer|label|verdict|final\s+answer)\s*(?::|is)\s*)?[\s*`<\[({]*[01][\s*`>\])}.!]*$", re.I)
TECHNICAL_LINE = re.compile(r"(?m)^\s*(?:(?:answer|label|verdict|final\s+answer)\s*(?::|is)\s*)?[\s*`<\[({]*[01][\s*`>\])}.!]*$", re.I)


def mask(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def build_surfaces(raw_text: str) -> Surface:
    raw = raw_text if isinstance(raw_text, str) else ""
    if not raw.strip():
        return Surface(raw, "", "", "missing")
    visible = raw.translate(QUOTES)
    flags: list[str] = []
    opens = list(re.finditer(r"<think\b[^>]*>", visible, re.I))
    closes = list(re.finditer(r"</think\s*>", visible, re.I))
    if len(opens) != len(closes):
        return Surface(raw, "", "", "unknown_surface", flags=["unbalanced_think"])
    if opens:
        visible = re.sub(r"<think\b[^>]*>[\s\S]*?</think\s*>", lambda m: mask(m.group()), visible, flags=re.I)
        if re.search(r"</?think\b", visible, re.I):
            return Surface(raw, "", "", "unknown_surface", flags=["nested_think"])
        flags.append("hidden_reasoning_masked")
    # Only explicit Harmony final-channel boundaries authorize final selection.
    if any(token in visible for token in ("<|channel|>", "<|meta_sep|>", "<|im_sep|>")):
        finals = list(re.finditer(r"<\|channel\|>final\s*<\|message\|>|<\|meta_sep\|>final\s*<\|im_sep\|>", visible))
        if len(finals) != 1:
            return Surface(raw, "", "", "unknown_surface", flags=["channel_without_explicit_final"])
        start = finals[0].end()
        boundary = re.search(r"<\|(?:start|im_start|channel|meta_sep|end|im_end|fim_suffix|return|message|im_sep)\|>", visible[start:])
        end = start + boundary.start() if boundary else len(visible)
        visible = mask(visible[:start]) + visible[start:end] + mask(visible[end:])
        flags.append("explicit_final_channel")
    if BINARY_RE.fullmatch(visible):
        return Surface(raw, visible, "", "verdict_only", flags=flags)
    if not any(char.isalpha() for char in visible):
        return Surface(raw, visible, "", "technical_only", flags=flags)
    structure = TECHNICAL_LINE.sub(lambda m: mask(m.group()), visible)
    # Retain numbered-list punctuation; remove only an unpunctuated leading digit.
    structure = re.sub(r"^(\s*[*`]*[01][*`]*)(?=\s+[A-Za-z*#])", lambda m: mask(m.group()), structure, count=1)
    explanation = re.sub(r"[*_`#]", " ", structure)
    words = WORD_RE.findall(explanation)
    if not any(any(c.isalpha() for c in word) for word in words):
        return Surface(raw, visible, "", "technical_only", flags=flags)
    if re.search(r"[\u3400-\u9fff]", explanation):
        flags.append("han_characters_present_language_coverage_unvalidated")
    if re.search(r"\S\s+#{2,4}\s|\S\s+\d{1,2}\.\s+\*\*", structure):
        flags.append("inline_structure_observed_original_layout_unknown")
    return Surface(raw, visible, explanation, "substantive_explanation", flags=flags, word_count=len(words), structure_text=structure)

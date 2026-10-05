"""Deterministic construction of the visible text surface to be measured."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>[\s\S]*?</think\s*>", re.IGNORECASE
)
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?")
_MARKDOWN_TOKEN_RE = re.compile(r"(?:\*{1,3}|_{1,3}|`{1,3}|#{1,6})")
_EXCUSE_LABEL_RE = re.compile(r"^\s*(?:\*{0,2})?excuse(?:\*{0,2})?\s*:\s*", re.I)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:\*{0,2})?(?:(?:the\s+)?(?:final\s+)?answer|label|verdict)"
    r"(?:\*{0,2})?\s*(?::|is|-)\s*(.+?)\s*$",
    re.IGNORECASE,
)
_HASH_ANSWER_RE = re.compile(r"^\s*#{4}\s*\S+\s*$")
_ETHICS_BINARY_PREFIX_RE = re.compile(
    r"^\s*(?:\*{1,2})?[01](?:\s*\(\s*(?:right|wrong)\s*\)|\s*[.:\-])"
    r"(?:\*{1,2})?\s*(?:[\u2013\u2014:\-]\s*)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextSurface:
    original_text: str
    visible_text: str
    lexical_text: str
    structure_text: str
    text_status: str
    text_available: bool
    word_count: int | None
    sentence_count: int | None
    avg_sentence_length: float | None
    ttr_300: float | None
    think_block_stripped: bool
    technical_lines_stripped: int


def normalize_visible_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Corpus responses mix straight and typographic quotation marks.  Keep the
    # untouched API text in ``original_text`` while normalizing the measured
    # surface so contractions such as ``you're`` and ``doesn't`` have one
    # deterministic representation across models.
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201f": '"',
            }
        )
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2009", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_think_blocks(text: str) -> tuple[str, bool, bool]:
    opens = len(_THINK_OPEN_RE.findall(text))
    closes = len(_THINK_CLOSE_RE.findall(text))
    if opens != closes:
        return "", False, True
    if not opens:
        return text, False, False
    stripped = _THINK_BLOCK_RE.sub("\n", text)
    # Nested or malformed tags can survive a non-greedy substitution.
    if _THINK_OPEN_RE.search(stripped) or _THINK_CLOSE_RE.search(stripped):
        return "", False, True
    return stripped, True, False


def tokenize_words(text: str) -> list[str]:
    token_text = _LATEX_COMMAND_RE.sub(" ", text)
    token_text = _MARKDOWN_TOKEN_RE.sub(" ", token_text)
    return _WORD_RE.findall(token_text)


def _is_technical_answer_line(line: str) -> bool:
    if _HASH_ANSWER_RE.fullmatch(line):
        return True
    match = _ANSWER_PREFIX_RE.fullmatch(line)
    if not match:
        return False
    remainder = match.group(1)
    if re.search(r"\b(?:because|since|therefore|which|so\s+that)\b", remainder, re.I):
        return False
    return len(tokenize_words(remainder)) <= 12


def strip_technical_answer_lines(text: str, benchmark: str = "") -> tuple[str, int]:
    kept: list[str] = []
    stripped = 0
    first_content_seen = False
    for line in text.splitlines():
        if line.strip() and not first_content_seen:
            first_content_seen = True
            if benchmark.casefold() == "ethics":
                line, prefix_count = _ETHICS_BINARY_PREFIX_RE.subn("", line, count=1)
                if prefix_count:
                    stripped += 1
                    if not line.strip():
                        continue
        if _is_technical_answer_line(line):
            stripped += 1
            continue
        # Keep explanation content while removing the protocol label itself.
        line = _EXCUSE_LABEL_RE.sub("", line)
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, stripped


def _sentence_count(text: str) -> int:
    parts = [part for part in _SENTENCE_SPLIT_RE.split(text) if tokenize_words(part)]
    return len(parts)


def prepare_text_surface(raw_text: object, benchmark: str = "") -> TextSurface:
    """Build analysis surfaces without falling back to hidden reasoning.

    Missing/unseparable content deliberately has ``None`` metrics.  It is never
    converted to a zero-marker response.
    """
    original = raw_text if isinstance(raw_text, str) else ""
    normalized = normalize_visible_text(original)
    if not normalized:
        return TextSurface(
            original_text=original,
            visible_text="",
            lexical_text="",
            structure_text="",
            text_status="missing",
            text_available=False,
            word_count=None,
            sentence_count=None,
            avg_sentence_length=None,
            ttr_300=None,
            think_block_stripped=False,
            technical_lines_stripped=0,
        )

    visible, stripped_think, unseparable = _strip_think_blocks(normalized)
    if unseparable:
        return TextSurface(
            original_text=original,
            visible_text="",
            lexical_text="",
            structure_text="",
            text_status="unseparable_think",
            text_available=False,
            word_count=None,
            sentence_count=None,
            avg_sentence_length=None,
            ttr_300=None,
            think_block_stripped=False,
            technical_lines_stripped=0,
        )

    visible = normalize_visible_text(visible)
    lexical, technical_count = strip_technical_answer_lines(visible, benchmark)
    structure, _ = strip_technical_answer_lines(visible, benchmark)
    words = tokenize_words(lexical)
    if not words:
        return TextSurface(
            original_text=original,
            visible_text=visible,
            lexical_text=lexical,
            structure_text=structure,
            text_status="technical_only",
            text_available=False,
            word_count=None,
            sentence_count=None,
            avg_sentence_length=None,
            ttr_300=None,
            think_block_stripped=stripped_think,
            technical_lines_stripped=technical_count,
        )

    sentence_count = max(_sentence_count(lexical), 1)
    alpha_words = [word.casefold() for word in words if any(ch.isalpha() for ch in word)]
    ttr_words = alpha_words[:300]
    ttr = len(set(ttr_words)) / len(ttr_words) if ttr_words else None
    return TextSurface(
        original_text=original,
        visible_text=visible,
        lexical_text=lexical,
        structure_text=structure,
        text_status="available",
        text_available=True,
        word_count=len(words),
        sentence_count=sentence_count,
        avg_sentence_length=len(words) / sentence_count,
        ttr_300=ttr,
        think_block_stripped=stripped_think,
        technical_lines_stripped=technical_count,
    )

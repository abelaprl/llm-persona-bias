"""Conservative final-text surfaces with character-level source offsets."""
from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
THINK = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.I | re.S)
TAGS = re.compile(r'</?think\b[^>]*>', re.I)
CHANNEL = re.compile(r'<\|(?:start|channel|im_start|im_sep|message)\|>')
BINARY = re.compile(r'\s*(?:[\s*`#]*(?:(?:final\s+)?(?:answer|label|verdict)\s*:?\s*)?)?[<\[({*`\s]*[01][>\])}*`\s.!:]*\s*', re.I)
PREFIX = re.compile(r'^\s*(?:\*{0,2})?(?:(?:final\s+)?(?:answer|label|verdict)\s*:\s*)?[<*`]*[01](?![\d.])[>*`]*(?:\s+|$)', re.I)


@dataclass
class FinalSurface:
    raw_text: str
    visible_text: str
    linguistic_text: str
    structure_text: str
    status: str
    flags: list
    maps: dict
    word_count: int
    sentence_count: int
    ttr_300: float | None

    @property
    def final_output_available(self):
        return bool(self.visible_text.strip()) and self.status != 'unseparable_channel'

    @property
    def explanation_available(self):
        return self.status == 'linguistic_text'

    def raw_span(self, surface_name, start, end):
        offsets = self.maps[surface_name]
        if start < 0 or end > len(offsets) or end <= start:
            raise ValueError('Invalid surface span')
        return offsets[start][0], offsets[end-1][1]


def prepare_final_surface(final_output, channel_format='plain_final'):
    if channel_format != 'plain_final':
        raise ValueError('Unsupported technical channel adapter; do not infer hidden content')
    raw = final_output if isinstance(final_output, str) else ''
    empty = lambda status: FinalSurface(raw, '', '', '', status, [], {'visible':[], 'lexical':[], 'structure':[]}, 0, 0, None)
    if not raw.strip():
        return empty('empty_final')
    if CHANNEL.search(raw):
        return empty('unseparable_channel')
    ranges = [(m.start(),m.end()) for m in THINK.finditer(raw)]
    remainder = THINK.sub('', raw)
    if TAGS.search(remainder):
        return empty('unseparable_channel')
    # All surviving text is final text. Excluded ranges are never exported.
    chars, mapping = [], []
    quotes = str.maketrans({'’':"'", '‘':"'", '“':'"', '”':'"'})
    range_index = 0
    for i, char in enumerate(raw):
        while range_index < len(ranges) and i >= ranges[range_index][1]:
            range_index += 1
        if range_index < len(ranges) and ranges[range_index][0] <= i < ranges[range_index][1]:
            continue
        normalized = unicodedata.normalize('NFKC', char).translate(quotes)
        for value in normalized:
            chars.append('\n' if value == '\r' else value)
            mapping.append((i,i+1))
    visible = ''.join(chars)
    flags = ['think_block_removed'] if ranges else []
    if not visible.strip():
        result = empty('technical_only')
        result.flags = flags
        return result
    if BINARY.fullmatch(visible):
        return FinalSurface(raw, visible, '', '', 'label_only', flags, {'visible':mapping,'lexical':[],'structure':[]}, 0, 0, None)
    prefix = PREFIX.match(visible)
    begin = prefix.end() if prefix else 0
    if prefix:
        flags.append('binary_prefix_removed')
    structure = visible[begin:]
    structure_map = mapping[begin:]
    # Replace Markdown delimiters, preserving one offset per output character.
    lexical = re.sub(r'[*`#_]', ' ', structure)
    words = [w for w in WORD.findall(lexical) if any(c.isalpha() for c in w)]
    if not words:
        status = 'technical_only'
    else:
        status = 'linguistic_text'
    if len(words) >= 64 and Counter(w.casefold() for w in words).most_common(1)[0][1] / len(words) >= .30:
        flags.append('high_token_repetition_candidate')
    if re.search(r'[\u3400-\u9fff]', lexical) and re.search(r'[A-Za-z]', lexical):
        flags.append('mixed_latin_cjk')
    if re.search(r'(?:\*\s*){64,}',structure):
        flags.append('symbol_repetition_candidate')
    if re.search(r'\b(?:the response should|this response is designed|this response addresses)\b',lexical,re.I):
        flags.append('meta_response_candidate')
    sentences = len([s for s in re.split(r'(?<=[.!?])\s+|\n+', lexical) if any(c.isalpha() for c in s)])
    window = [w.casefold() for w in words[:300]]
    return FinalSurface(raw, visible, lexical, structure, status, flags,
                        {'visible':mapping, 'lexical':structure_map, 'structure':structure_map},
                        len(words), sentences, len(set(window))/len(window) if window else None)

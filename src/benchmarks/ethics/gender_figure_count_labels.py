"""Remove auxiliary pair-count annotations from gender ETHICS figures."""
from __future__ import annotations

import re

from matplotlib.text import Text

COUNT = re.compile(r"(?<![\w.])\d[\d,]*\s*/\s*\d[\d,]*(?![\w.])")
KEY = re.compile(r"\b(?:k\s*/\s*n|n\s*/\s*k)\b", re.I)


def clean_count_text(text):
    """Keep measurements and statistical keys; remove count ratios and their keys."""
    if text.startswith("Isi sel: pasangan penjelasan tersedia / soal"):
        return "Cakupan (%) · L: laki-laki · P: perempuan · N: netral"
    if text.startswith("Isi sel: cakupan (%) dan pasangan tersedia / soal"):
        return "Isi sel: cakupan (%)"
    if text.startswith("Denominator cakupan berbeda dari k/n."):
        return "I1–I3: variasi warmup"
    parts = []
    for part in text.split(" · "):
        if KEY.search(part):
            continue
        part = COUNT.sub("", part)
        part = "\n".join(line.strip() for line in part.splitlines() if line.strip())
        if part:
            parts.append(part)
    return " · ".join(parts)


def strip_count_labels(fig):
    """Change text artists only, retaining plot values, axis scales and coverage."""
    changes = []
    for artist in fig.findobj(Text):
        old = artist.get_text()
        new = clean_count_text(old)
        if new != old:
            artist.set_text(new)
            changes.append({"before": old, "after": new})
    remaining = [t.get_text() for t in fig.findobj(Text)
                 if t.get_visible() and (COUNT.search(t.get_text()) or KEY.search(t.get_text()))]
    assert not remaining, remaining
    return {"changed_texts": len(changes), "remaining_count_labels": 0, "changes": changes}

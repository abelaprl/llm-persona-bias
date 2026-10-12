"""Remove observed count-ratio annotations without changing measurements."""
import re

from matplotlib.text import Text

COUNT = re.compile(r"(?<![\w.])\d[\d,]*\s*/\s*\d[\d,]*(?![\w.])")
NAMED_COUNT = re.compile(r"k=\d+(?:\.0)?\s*/\s*n=\d+(?:\.0)?")
COUNT_KEY = "k = pasangan berubah; n = pasangan penjelasan · "
COVERAGE_KEY = "Pembilang/penyebut = penjelasan tersedia/JSON ditemukan; "


def clean_count_text(text):
    text = text.replace(COUNT_KEY, "").replace(COVERAGE_KEY, "")
    text = NAMED_COUNT.sub("", text)
    text = COUNT.sub("", text)
    return "\n".join(line for line in text.splitlines() if line.strip())


def strip_count_labels(fig):
    changes = []
    for artist in fig.findobj(Text):
        old = artist.get_text()
        new = clean_count_text(old)
        if old != new:
            artist.set_text(new)
            changes.append(dict(before=old, after=new))
    assert all(clean_count_text(t.get_text()) == t.get_text() for t in fig.findobj(Text))
    return changes

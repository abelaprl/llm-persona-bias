"""Remove count text from scientific vector plots, then export their PNGs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex_tmp/gender_ethics_counts_20261011/pdf_tools"))
import pymupdf

ACTIVE = ROOT / "notebooks/gender/linguistic marker/ETHICS/figures"
COUNT = re.compile(r"(?<![\w.])\d[\d,]*\s*/\s*\d[\d,]*(?![\w.])")
KEY = re.compile(r"\b(?:k\s*/\s*n|n\s*/\s*k)\b", re.I)
DECIMAL = re.compile(r"[+−-]?\d+\.\d+")
FOLDERS = ["bab6_v3_20261007_230725_835481", "bab6_v3_1_20261007_234752_514764"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deletion_ranges(text):
    ranges = [m.span() for m in COUNT.finditer(text)]
    if text.startswith("Denominator cakupan berbeda dari k/n."):
        ranges.append((0, len("Denominator cakupan berbeda dari k/n. ")))
        return ranges
    offset = 0
    parts = text.split(" · ")
    for index, part in enumerate(parts):
        if KEY.search(part) or part.startswith("Isi sel: pasangan penjelasan tersedia / soal") or part.startswith("Isi sel: cakupan (%) dan pasangan tersedia / soal"):
            # Remove its separator as well, retaining adjacent scientific keys.
            start = offset - (3 if index else 0)
            end = offset + len(part) + (3 if index == 0 and len(parts) > 1 else 0)
            ranges.append((start, end))
        offset += len(part) + 3
    return ranges


def drawing_signature(page):
    keys = ["type", "items", "color", "fill", "width", "dashes", "closePath",
            "lineJoin", "lineCap", "fill_opacity", "stroke_opacity"]
    return repr([{key: item.get(key) for key in keys} for item in page.get_drawings()])


def clean_pdf(source, run):
    relative = source.relative_to(ACTIVE)
    target = run / "figures" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(source)
    assert len(document) == 1
    page = document[0]
    before = page.get_text()
    before_paths = drawing_signature(page)
    if target.exists():
        original = document
        document = pymupdf.open(target)
        page = document[0]
        after = page.get_text()
        assert not COUNT.search(after) and not KEY.search(after), target
        assert Counter(DECIMAL.findall(before)) == Counter(DECIMAL.findall(after)), target
        assert before.count("*") == after.count("*"), target
        assert before_paths == drawing_signature(page), target
        original.close()
    removed = []
    for block in ([] if target.exists() else page.get_text("rawdict")["blocks"]):
        for line in block.get("lines", []):
            characters = [char for span in line["spans"] for char in span["chars"]]
            text = "".join(char["c"] for char in characters)
            for start, end in deletion_ranges(text):
                selected = characters[max(0, start):end]
                if not selected:
                    continue
                rectangle = pymupdf.Rect(selected[0]["bbox"])
                for char in selected[1:]:
                    rectangle |= pymupdf.Rect(char["bbox"])
                # Redaction removes any intersecting glyph. A thin center strip
                # avoids neighboring colorbar ticks whose font boxes overlap.
                middle = (rectangle.y0 + rectangle.y1) / 2
                rectangle.y0, rectangle.y1 = middle - .1, middle + .1
                rectangle.x0 += .01
                rectangle.x1 -= .01
                # Transparent removal retains heatmap fills, hatches and CI paths.
                page.add_redact_annot(rectangle, fill=False, cross_out=False)
                removed.append(text[max(0, start):end])
    assert removed or target.exists(), str(source)
    if removed:
        page.apply_redactions(images=0, graphics=0, text=0)
    after = page.get_text()
    assert not COUNT.search(after) and not KEY.search(after), (source, after)
    assert Counter(DECIMAL.findall(before)) == Counter(DECIMAL.findall(after)), source
    assert before.count("*") == after.count("*"), source
    assert before_paths == drawing_signature(page), (source, "vector paths changed")
    if not target.exists():
        document.save(target, garbage=4, deflate=True)
    old_png = source.with_suffix(".png")
    with old_png.open("rb") as handle:
        header = handle.read(24)
    width, height = struct.unpack(">II", header[16:24])
    matrix = pymupdf.Matrix((width - .000001) / page.rect.width,
                            (height - .000001) / page.rect.height)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    assert (pixmap.width, pixmap.height) == (width, height)
    pixmap.set_dpi(240, 240)
    png = target.with_suffix(".png")
    if not png.exists():
        pixmap.save(png)
    document.close()
    return dict(family="v31" if "v3_1" in str(relative) else "v3",
        figure=str(relative.with_suffix(".png")), pdf=str(relative),
        old_sha256=digest(old_png), new_sha256=digest(png),
        old_pdf_sha256=digest(source), new_pdf_sha256=digest(target),
        text_cleanup=dict(remaining_count_labels=0, changed_texts=len(removed),
                          vector_paths_unchanged=True, decimal_values_unchanged=True,
                          significance_stars_unchanged=True, removed=removed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=375)
    args = parser.parse_args()
    pymupdf.TOOLS.set_small_glyph_heights(True)
    rows = []
    sources = sorted(path for folder in FOLDERS for path in (ACTIVE / folder).glob("*.pdf"))
    assert len(sources) == 375
    sources = sources[args.start:args.stop]
    for i, source in enumerate(sources, 1):
        rows.append(clean_pdf(source, args.run))
        if i % 10 == 0:
            print(f"Vector export: {i}/{len(sources)}; text and paths verified", flush=True)
    suffix = "" if args.start == 0 and args.stop == 375 else f"_{args.start}_{args.stop}"
    audit = args.run / f"audit_vector{suffix}.json"
    assert not audit.exists()
    audit.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: {len(rows)} PDF and {len(rows)} PNG", flush=True)


if __name__ == "__main__":
    main()

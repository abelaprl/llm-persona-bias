"""Restore the archived five-category chart and add it to the existing notebook."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "notebooks/gender/ethics accuracy analysis"
TARGET = BUNDLE / "ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb"
STEM = "01_ethics_absolute_accuracy_five_categories"
CATEGORIES = {
    "Man Explicit": ["man_explicit_1"],
    "Man Implicit Avg": [f"man_implicit_{i}" for i in range(1, 4)],
    "Woman Explicit": ["woman_explicit_1"],
    "Woman Implicit Avg": [f"woman_implicit_{i}" for i in range(1, 4)],
    "Neutral": ["neutral"],
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_table():
    detail = pd.read_csv(BUNDLE / "tables/accuracy_construct_valid_subsets.csv")
    table = pd.read_csv(BUNDLE / "tables/ethics_accuracy_by_model_persona_category.csv")
    assert len(detail) == 108 and set(detail.subset) == {"commonsense", "justice"}
    assert len(table) == 30 and not table.duplicated(["model", "category"]).any()
    counts = ["total", "nonmissing", "committed", "abstain", "missing", "n_correct"]
    rows = []
    for row in table.itertuples(index=False):
        group = detail[detail.model.eq(row.model) & detail.persona.isin(CATEGORIES[row.category])]
        assert len(group) == 2 * len(CATEGORIES[row.category])
        values = group[counts].sum()
        for column in counts:
            assert values[column] == getattr(row, column), (row.model, row.category, column)
        assert row.total == row.committed + row.abstain + row.missing
        for rate, numerator, denominator in [
            ("accuracy_committed", "n_correct", "committed"),
            ("commit_rate", "committed", "total"),
            ("abstain_rate", "abstain", "total"),
            ("missing_rate", "missing", "total"),
        ]:
            assert abs(getattr(row, rate) - values[numerator] / values[denominator]) < 1e-12
        rows.append(row.accuracy_committed)
    svg = (BUNDLE / "archive/20260910_cleanup/figures" / f"{STEM}.svg").read_text(encoding="utf-8")
    labels = Counter(f"{value:.4f}" for value in rows)
    for label, count in labels.items():
        assert svg.count(f"<!-- {label} -->") >= count, label
    neutral = table.loc[table.category.eq("Neutral"), "accuracy_committed"].mean()
    assert f"Global Neutral Avg ({neutral:.4f})" in svg
    return table


def main():
    table = validate_table()
    print(f"[CHECKLIST] Verified {len(table)} categories against 108 subset rows", flush=True)
    previous_hash = digest(TARGET)
    notebook = nbformat.read(TARGET, as_version=4)
    protected = {p: digest(p) for folder in ("figures", "tables", "per_subset")
                 for p in (BUNDLE / folder).rglob("*") if p.is_file()}
    protected.update({BUNDLE / name: digest(BUNDLE / name)
                      for name in ("bundle_manifest.json", "output_index.csv")})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = BUNDLE / "archive" / f"restore_accuracy_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(TARGET, backup / TARGET.name)
    hashes = {}
    for suffix in ("png", "svg"):
        source = BUNDLE / "archive/20260910_cleanup/figures" / f"{STEM}.{suffix}"
        target = BUNDLE / "figures" / source.name
        if target.exists():
            assert digest(target) == digest(source), f"Different existing figure: {target}"
        else:
            shutil.copy2(source, target)
        hashes[f"figures/{target.name}"] = digest(target)
    table_path = "tables/ethics_accuracy_by_model_persona_category.csv"
    hashes[table_path] = digest(BUNDLE / table_path)
    markdown = nbformat.v4.new_markdown_cell("""## Akurasi Commonsense dan Justice: lima kategori persona

Grafik ini merupakan ringkasan tambahan untuk **Commonsense + Justice**, bukan
akurasi keseluruhan empat subset ETHICS. Akurasi per subset tetap disajikan di
bagian lain notebook. Deontology dan Virtue tidak masuk perhitungan ini.

Setiap batang dihitung sebagai jumlah jawaban benar dibagi jumlah respons
*committed*. Kategori Implicit Avg menggabungkan I1, I2, dan I3 dengan bobot
jumlah respons committed, bukan rata-rata sederhana tiga akurasi. Garis merah
adalah rata-rata akurasi netral enam model, bukan baseline uji statistik.

Tabel coverage di bawah grafik wajib dibaca bersama akurasinya: respons
abstain dan missing tidak masuk denominator akurasi. Perbedaan coverage dapat
mengubah komposisi respons yang dibandingkan. Sumbu vertikal tidak dimulai dari nol;
bandingkan nilai angkanya, bukan rasio tinggi batang.
""", id="five-category-accuracy-note")
    code = nbformat.v4.new_code_cell(
        "# Verify the restored assets before displaying them.\n"
        f"five_category_hashes = {hashes!r}\n"
        "for relative_path, expected in five_category_hashes.items():\n"
        "    assert sha256_file(NOTEBOOK_DIR / relative_path) == expected, relative_path\n"
        f"five_category_accuracy = pd.read_csv(TABLE_DIR / {Path(table_path).name!r})\n"
        "assert len(five_category_accuracy) == 30\n"
        f"display(Image(filename=str(FIGURE_DIR / '{STEM}.png'), width=1400))\n"
        "show_table(five_category_accuracy, ['model_label', 'category', *COVERAGE, 'accuracy_committed'])\n"
        "display(Markdown('Rentang commit_rate pada 30 batang: '"
        "+ f'{five_category_accuracy.commit_rate.min():.2%} hingga {five_category_accuracy.commit_rate.max():.2%}.'))\n",
        id="five-category-accuracy-display",
    )
    original_cells = {cell.id: cell.source for cell in notebook.cells}
    notebook.cells = [cell for cell in notebook.cells
                      if cell.id not in {markdown.id, code.id}]
    position = next(i for i, cell in enumerate(notebook.cells)
                    if cell.cell_type == "markdown" and cell.source.startswith("## 7."))
    notebook.cells[position:position] = [markdown, code]
    integration = next(cell for cell in notebook.cells
                       if cell.cell_type == "markdown" and cell.source.startswith("## 10."))
    integration.source = """## 10. INTEGRATION — Penggunaan dan batas klaim

Gunakan **Restart Kernel and Run All** untuk menampilkan ulang gambar, tabel,
dan pemeriksaan. Notebook membaca artefak lokal tanpa parsing atau API call.
Untuk memulihkan grafik lima kategori dari arsip dan menjalankan notebook yang sama:

```powershell
python src/benchmarks/ethics/restore_gender_accuracy_five_categories.py
```

Skrip ini mempertahankan bagian notebook lain dan membuat backup sebelum menyimpan.
Generator lama membangun ulang susunan notebook; jangan gunakan untuk sinkronisasi
versi ini karena tambahan yang lebih baru dapat hilang.

Akurasi per subset tetap menjadi rincian utama; grafik Commonsense + Justice
merupakan ringkasan tambahan, bukan satu skor untuk empat subset ETHICS.
Stance Deontology/Virtue menggambarkan jawaban terhadap prompt eksperimen dengan
keterbatasan pada SPEC. Perhatikan coverage, abstain, dan perbedaan narasi persona.
"""
    summary = notebook.cells[-1]
    if '"additional_accuracy_png"' not in summary.source:
        summary.source = summary.source.replace('"displayed_child_figures":',
            '"additional_accuracy_png": 1, "additional_accuracy_svg": 1,\n    "displayed_child_figures":')
    changed = [cell.id for cell in notebook.cells if cell.id in original_cells
               and cell.source != original_cells[cell.id]]
    assert set(changed) <= {integration.id, summary.id, markdown.id, code.id}
    print("[TESTING] Executing notebook, preserving all existing analyses", flush=True)
    manager = KernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Python 3", language="python")
    NotebookClient(notebook, km=manager, timeout=600,
                   resources={"metadata": {"path": str(BUNDLE)}}).execute()
    embedded = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
        for output in cell.get("outputs", []):
            assert output.output_type != "error"
            data = output.get("data", {}).get("image/png")
            if data:
                embedded.append(hashlib.sha256(base64.b64decode(data)).hexdigest())
    assert hashes[f"figures/{STEM}.png"] in embedded
    for path, expected in protected.items():
        assert digest(path) == expected, f"Existing asset changed: {path}"
    assert digest(TARGET) == previous_hash, "Notebook changed during execution"
    nbformat.validate(notebook)
    staged = backup / "validated.ipynb"
    nbformat.write(notebook, staged)
    assert nbformat.read(staged, as_version=4) == notebook
    os.replace(staged, TARGET)
    report = {"valid": True, "cells": len(notebook.cells),
              "executed_code_cells": sum(c.cell_type == "code" for c in notebook.cells),
              "embedded_png": len(embedded), "errors": 0,
              "protected_files_unchanged": len(protected), "asset_hashes": hashes,
              "before_sha256": previous_hash, "after_sha256": digest(TARGET),
              "source_rows": int(table.total.sum()), "category_rows": len(table),
              "scope": ["commonsense", "justice"], "api_calls": 0, "rescoring": False}
    with (backup / "validation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[INTEGRATION] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

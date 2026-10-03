"""Refresh installed Bab VI PNGs, retaining a verified recoverable backup."""
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "notebooks/gender/linguistic marker/ETHICS"
BUNDLE = ROOT / "result/benchmarks/ethics/reports/gender_lm_bab6_viz_20261002_r5"
NOTEBOOK = "ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh():
    fresh = pd.read_csv(BUNDLE / "figure_index.csv").set_index("figure_id")
    index = pd.read_csv(ACTIVE / "figure_index.csv").set_index("figure_id")
    manifest = json.loads((ACTIVE / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    assert not list((BUNDLE / "figures").glob("*.svg"))
    for relative, expected in manifest["source_hashes"].items():
        assert digest(ROOT / relative) == expected, relative
    for figure_id, row in fresh.iterrows():
        # The requested change is visual only: every source table must be identical.
        assert digest(ACTIVE / "figures" / f"{figure_id}.csv") == row.csv_sha256
        for ext in ("png", "csv"):
            assert digest(BUNDLE / "figures" / f"{figure_id}.{ext}") == row[f"{ext}_sha256"]
        for column, value in row.items():
            index.loc[figure_id, column] = value
    index = index.drop(columns=["svg_sha256"], errors="ignore")
    notebook = nbformat.read(ACTIVE / NOTEBOOK, as_version=4)
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "PNG untuk notebook; SVG untuk pembesaran/cetak.", "Gambar utama tersedia sebagai PNG 300 dpi; CSV memuat angka sumber."
    )
    notebook.cells[1].source = notebook.cells[1].source.replace("['png','svg','csv']", "['png','csv']")
    old_explanation = "Warna pekat menunjukkan p Holm < 0,05; pudar bukan berarti efek nol."
    new_explanation = "Semua titik memakai warna penuh agar terbaca. Segitiga menunjukkan p Holm < 0,05; lingkaran menunjukkan hasil lainnya, bukan berarti efek nol."
    assert old_explanation in notebook.cells[4].source or new_explanation in notebook.cells[4].source
    notebook.cells[4].source = notebook.cells[4].source.replace(old_explanation, new_explanation)
    backup = ACTIVE / "_arsip" / ("sebelum_kontras_png_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    originals = [NOTEBOOK, "README.md", "figure_index.csv", "gallery.html", "manifest.json", "presentation_source.json"]
    originals += [f"figures/{figure_id}.png" for figure_id in fresh.index]
    backup.mkdir(parents=True, exist_ok=False)
    for relative in originals:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ACTIVE / relative, target)
        assert digest(target) == digest(ACTIVE / relative)
    (backup / "backup_hashes.json").write_text(json.dumps({r: digest(backup / r) for r in originals}, indent=2), encoding="utf-8")
    print(f"Verified backup: {backup}", flush=True)
    try:
        for figure_id in fresh.index:
            shutil.copy2(BUNDLE / "figures" / f"{figure_id}.png", ACTIVE / "figures" / f"{figure_id}.png")
        index.reset_index().to_csv(ACTIVE / "figure_index.csv", index=False)
        manifest.update({k: v for k, v in new_manifest.items() if k not in {"source_hashes", "figure_count"}})
        manifest["previous_presentation"] = backup.relative_to(ACTIVE).as_posix()
        (ACTIVE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        provenance = json.loads((ACTIVE / "presentation_source.json").read_text(encoding="utf-8"))
        provenance.update(figure_bundle=BUNDLE.relative_to(ROOT).as_posix(), style_version="gender-ethics-bab6-clean-legend-png-v5", previous_active_artifacts=backup.relative_to(ACTIVE).as_posix())
        (ACTIVE / "presentation_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        sections = []
        for figure_id, row in index.iterrows():
            sections.append(f'<section><h2>{html.escape(figure_id)}</h2><img src="figures/{figure_id}.png"><p>{html.escape(row.caption)}</p>'
                            + " · ".join(f'<a href="figures/{figure_id}.{ext}">{ext.upper()}</a>' for ext in ("png", "csv")) + '</section>')
        (ACTIVE / "gallery.html").write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS — linguistic marker</title><style>body{font-family:Arial,sans-serif;margin:32px}img{max-width:100%}section{margin-bottom:48px}</style><h1>Gender ETHICS — linguistic marker</h1><p>Candidate. Grafik langsung: Man − Woman; segitiga = p Holm &lt; 0,05, lingkaran = lainnya. Gambar PNG; CSV untuk angka.</p>' + "\n".join(sections) + '</html>', encoding="utf-8")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        print("Executing notebook and checking all 13 embedded images...", flush=True)
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ACTIVE)}}).execute()
        images = sum("image/png" in output.get("data", {}) for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
        assert images == 13
        for relative, expected in manifest["source_hashes"].items():
            assert digest(ROOT / relative) == expected, relative
        nbformat.validate(notebook)
        nbformat.write(notebook, ACTIVE / NOTEBOOK)
    except Exception:
        for relative in originals:
            shutil.copy2(backup / relative, ACTIVE / relative)
        raise
    print("Updated 5 PNGs; 13 figures embedded; all source CSVs unchanged; no new SVGs.", flush=True)


if __name__ == "__main__":
    refresh()

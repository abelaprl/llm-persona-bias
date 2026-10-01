"""Generate the relocated notebook, changing presentation paths only."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[3]
NAME = "linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb"
PARENT = ROOT / "notebooks/gender/linguistic marker"


def relocate(source: Path, output: Path) -> None:
    original = source.read_bytes()
    if output.exists() and output.read_bytes() != original:
        raise FileExistsError(f"Refusing to overwrite a modified notebook: {output}")
    notebook = nbformat.read(source, as_version=4)
    replacements = {
        'FIG_DIR = BUNDLE_DIR / "images"': (
            'ACTIVE_DIR = REPO_ROOT / "notebooks/gender/linguistic marker/GSM8K"\n'
            'FIG_DIR = ACTIVE_DIR / "figures"'
        ),
        'pd.read_csv(BUNDLE_DIR / "figure_index.csv")':
            'pd.read_csv(ACTIVE_DIR / "figure_index.csv")',
        "f\"Dokumentasi bundle: `{(BUNDLE_DIR / 'README.md').relative_to(REPO_ROOT)}`\"":
            "f\"Panduan folder aktif: `{(ACTIVE_DIR / 'README.md').relative_to(REPO_ROOT)}`\"",
        '# Gunakan root proyek secara eksplisit agar notebook tidak bergantung pada\n'
        '# folder kerja saat Jupyter dijalankan.':
            '# Use an explicit project root independent of the Jupyter working directory.',
    }
    expected = [1, 2, 1, 1]
    for (before, after), count in zip(replacements.items(), expected, strict=True):
        actual = sum(cell.source.count(before) for cell in notebook.cells
                     if cell.cell_type == "code")
        if actual != count:
            raise ValueError(f"Expected {count} replacements, found {actual}: {before}")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(before, after)
    notebook.cells[0].source += (
        "\n\n**Penataan folder 4 Oktober 2026:** notebook aktif berada di `GSM8K/` "
        "dan membaca PNG dari `figures/` di sebelah notebook. Tabel tetap dibaca "
        "dari paket hasil beku; statistik tidak diubah. "
        "Lihat [panduan folder](README.md)."
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    notebook.metadata["presentation_relocation"] = {
        "date": "2026-10-04",
        "source_notebook_sha256": hashlib.sha256(original).hexdigest(),
        "source_notebook": str(source),
        "statistics_recomputed": False,
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, output)
    if source.read_bytes() != original:
        raise AssertionError("Source notebook changed")
    print(f"Generated relocated notebook: {output}")
    print("Only presentation paths and location notes changed; source is unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = PARENT / NAME
    if not default_source.exists():
        default_source = PARENT / "_arsip/sebelum_folder_gsm8k_20261004" / NAME
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=PARENT / "GSM8K" / NAME)
    args = parser.parse_args()
    relocate(args.source, args.output)

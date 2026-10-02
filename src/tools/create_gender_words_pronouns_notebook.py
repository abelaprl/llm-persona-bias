"""Create a focused executed-notebook source for explicit gender words."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


DEFAULT_OUTPUT = Path("notebooks/linguistic_marker_gender_words_pronouns.ipynb")


def build_notebook(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing notebook: {output}")

    cells = [
        new_markdown_cell(
            """# Kata dan Pronomina Gender pada Respons GSM8K

Notebook ini khusus menampilkan penyebutan gender dalam `visible_text` respons
model. Word cloud diletakkan di bagian awal agar mudah ditemukan.

Pemilihannya dilakukan dalam dua tahap:

1. respons disaring hanya pada **52 kata gender** yang sudah ditentukan, seperti
   `he`, `she`, `him`, `her`, `his`, `hers`, `man`, `woman`, `boy`, dan `girl`;
2. setiap panel menampilkan **16 kata gender yang paling sering muncul** pada
   persona tersebut.

Jadi, ini bukan kumpulan semua kata yang sering dipakai model. Kata harus lebih
dulu termasuk dalam leksikon gender. Ukuran kata menunjukkan kemunculan per
10.000 kata dan dirata-ratakan dengan bobot setara antar-model.
"""
        ),
        new_code_cell(
            """from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, Markdown, display

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name.casefold() == "notebooks":
    REPO_ROOT = REPO_ROOT.parent

bundles = sorted(
    (REPO_ROOT / "result" / "linguistic_markers").glob(
        "gender_gsm8k_bab4_visual_revision_v2_*"
    )
)
if not bundles:
    raise FileNotFoundError("Bundle visual linguistic marker gender tidak ditemukan.")
BUNDLE_DIR = bundles[-1]
IMAGE_DIR = BUNDLE_DIR / "images"

metadata = json.loads(
    (BUNDLE_DIR / "gendered_lexicon_metadata.json").read_text(encoding="utf-8")
)
terms = pd.read_csv(BUNDLE_DIR / "gendered_term_persona_aggregate.csv")
prevalence = pd.read_csv(BUNDLE_DIR / "gender_pronoun_class_prevalence.csv")
paired = pd.read_csv(BUNDLE_DIR / "paired_gender_pronoun_classes.csv")

display(Markdown(
    f"**Data:** {metadata['n_available_responses']:,} visible responses · "
    f"**leksikon:** {metadata['n_terms']} kata gender · "
    f"**hidden reasoning:** tidak digunakan"
))
"""
        ),
        new_markdown_cell(
            """## 1. Word cloud penyebutan gender

Warna biru menunjukkan kata maskulin dan magenta menunjukkan kata feminin.
Baris membandingkan neutral (N), explicit (E), serta tiga proxy implicit
(I1, I2, I3). Kolom kiri adalah persona Man dan kolom kanan persona Woman.
"""
        ),
        new_code_cell(
            """display(Image(
    filename=str(IMAGE_DIR / "11_gendered_terms_wordcloud_by_persona.png"),
    width=1450,
))
"""
        ),
        new_markdown_cell(
            """## 2. Angka di balik word cloud

Tabel berikut menampilkan 16 kata yang dipakai pada setiap panel, sehingga
besar kata pada word cloud dapat diperiksa dalam angka.
"""
        ),
        new_code_cell(
            """top_terms = (
    terms.sort_values(
        ["persona", "mean_model_occurrences_per_10k_words"],
        ascending=[True, False],
    )
    .groupby("persona", as_index=False)
    .head(16)
)
display(
    top_terms[[
        "persona", "term", "term_category",
        "mean_model_occurrences_per_10k_words",
        "mean_model_response_prevalence_pp",
        "total_occurrences",
    ]].style.format({
        "mean_model_occurrences_per_10k_words": "{:.2f}",
        "mean_model_response_prevalence_pp": "{:.2f}",
    })
)
"""
        ),
        new_markdown_cell(
            """## 3. Perbandingan pronomina per model dan persona

Grafik pertama menunjukkan persentase respons yang memuat sedikitnya satu
pronomina maskulin atau feminin. Grafik kedua menunjukkan perubahan I1/I2/I3
terhadap neutral pada pertanyaan yang sama.
"""
        ),
        new_code_cell(
            """display(Image(
    filename=str(IMAGE_DIR / "12_gender_pronoun_prevalence_by_model_persona.png"),
    width=1450,
))
display(Image(
    filename=str(IMAGE_DIR / "13_gender_pronoun_proxy_forest.png"),
    width=1650,
))
"""
        ),
        new_markdown_cell(
            """## 4. Perubahan pronomina terbesar

Nilai delta dilaporkan dalam percentage points. I1, I2, dan I3 tetap dipisah.
"""
        ),
        new_code_cell(
            """paired = paired.assign(
    delta_pp=100 * paired["prevalence_delta"],
    ci_low_pp=100 * paired["ci_low"],
    ci_high_pp=100 * paired["ci_high"],
)
display(
    paired.assign(abs_delta_pp=lambda frame: frame["delta_pp"].abs())
    .sort_values(["abs_delta_pp", "n_pairs"], ascending=[False, False])
    [[
        "model_id", "identity", "pronoun_class", "contrast_id", "n_pairs",
        "delta_pp", "ci_low_pp", "ci_high_pp", "p_holm",
    ]]
    .head(24)
    .style.format({
        "delta_pp": "{:+.2f}",
        "ci_low_pp": "{:+.2f}",
        "ci_high_pp": "{:+.2f}",
        "p_holm": "{:.3g}",
    })
)
"""
        ),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_notebook(args.output)
    print(f"[DONE] Notebook written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

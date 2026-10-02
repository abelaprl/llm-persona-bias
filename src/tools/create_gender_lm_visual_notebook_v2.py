"""Create the executable thesis-facing gender linguistic-marker v2 notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# Linguistic Marker Gender GSM8K — Visual Revision v2

Notebook ini menyajikan hasil **candidate** untuk Bab Evaluasi. Fokusnya adalah
pergeseran *within-model* terhadap pembanding yang sama, bukan perbandingan
nilai absolut antarmodel. Kondisi **I1, I2, dan I3 selalu dipisah** karena
ketiganya merepresentasikan konteks proxy yang berbeda.

Analisis utama tetap memakai seluruh respons linguistik yang tersedia. Analisis
`both-correct` di bawah hanya pemeriksaan sensitivitas dan diterapkan setelah
respons dipasangkan berdasarkan `model_id × question_id`.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import sys
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name.casefold() == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.analysis.gender_visuals.revision_v2 import build_revision_bundle

RUN_CANDIDATES = sorted(
    (REPO_ROOT / "result" / "linguistic_markers").glob(
        "gender_gsm8k_full_source_aligned_v1_2_1_*"
    ),
    key=lambda path: path.stat().st_mtime,
)
if not RUN_CANDIDATES:
    raise FileNotFoundError("Tidak menemukan full source-aligned gender LM run.")
RUN_DIR = RUN_CANDIDATES[-1]

BUNDLE_CANDIDATES = sorted(
    (REPO_ROOT / "result" / "linguistic_markers").glob(
        "gender_gsm8k_bab4_visual_revision_v2_*"
    ),
    key=lambda path: path.stat().st_mtime,
)
if not BUNDLE_CANDIDATES:
    raise FileNotFoundError(
        "Bundle v2 belum dibangun. Jalankan src/tools/build_gender_lm_visual_revision_v2.py."
    )
BUNDLE_DIR = BUNDLE_CANDIDATES[-1]
FIG_DIR = BUNDLE_DIR / "images"

source_manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
bundle_manifest = build_revision_bundle(RUN_DIR, BUNDLE_DIR, reuse_existing=True)

if source_manifest.get("pipeline_version") != "gender-lm-v1.2.1-candidate":
    warnings.warn(f"Versi pipeline berubah: {source_manifest.get('pipeline_version')}")
if source_manifest.get("response_records") != 71_215:
    warnings.warn(
        f"Jumlah respons sekarang {source_manifest.get('response_records'):,}; "
        "notebook melanjutkan dengan data yang ditemukan."
    )
if source_manifest.get("correctness_filter_applied") is not False:
    raise ValueError("Run utama harus menyatakan correctness_filter_applied=false.")

def show_figure(name: str, width: int = 1500):
    path = FIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=path, width=width))

display(Markdown(
    f"**Source:** `{RUN_DIR.relative_to(REPO_ROOT)}`  \n"
    f"**Bundle:** `{BUNDLE_DIR.relative_to(REPO_ROOT)}`  \n"
    f"**N:** {source_manifest.get('response_records', 0):,} respons · "
    f"{bundle_manifest.get('bootstrap_resamples', 0):,} bootstrap resample"
))
"""
        ),
        markdown(
            r"""
## 1. Provenance, visible surface, dan status audit

Hidden reasoning tidak pernah menjadi fallback teks. `reasoning_present`
hanyalah metadata sumber; marker dihitung pada visible final-answer surface
setelah `<think>...</think>` yang terlihat dipisahkan dan baris jawaban teknis
ditangani secara deterministik.
"""
        ),
        code(
            r"""
surface_audit = pd.read_csv(BUNDLE_DIR / "visible_surface_audit.csv")
manual_audit = pd.read_csv(BUNDLE_DIR / "manual_audit_status.csv")
display(surface_audit.assign(model_id=surface_audit.model_id.str.replace("_", " ")))
display(manual_audit)
if (manual_audit["n_annotated_rows"] == 0).any():
    display(Markdown(
        "⚠️ **Precision/recall manual audit belum dianotasi.** "
        "Semua hasil tetap berstatus candidate, bukan klaim final Bab Evaluasi."
    ))
"""
        ),
        markdown(
            r"""
## 2. Cakupan per model × 9 persona

Angka di dalam sel membuat gap aktual terlihat langsung. Missing,
technical-only, dan sumber yang tidak tersedia tidak diubah menjadi observasi
dengan marker nol.
"""
        ),
        code(
            r"""
show_figure("01_coverage_model_persona.png", width=1550)
coverage = pd.read_csv(BUNDLE_DIR / "response_counts_model_persona.csv")
display(coverage.sort_values("text_coverage_rate").head(12))
"""
        ),
        markdown(
            r"""
## 3. Verbosity dan dasar normalisasi

Semua density marker menggunakan satuan per 100 kata. Figur berikut sengaja
menampilkan panjang respons absolut agar perubahan marker tidak keliru dibaca
sebagai efek persona jika sebenarnya hanya berasal dari verbositas.
"""
        ),
        code('show_figure("02_verbosity_word_count_median_iqr.png", width=1450)'),
        markdown(
            r"""
## 4. Effect map numerik: magnitude lebih dahulu

Sumbu-x adalah rank-biserial dari selisih berpasangan; sumbu-y adalah
`−log10(p_holm)`. Titik yang tinggi tetapi dekat nol menunjukkan hasil yang
signifikan namun kecil secara praktis. Ukuran titik mengikuti jumlah pasangan
yang benar-benar berubah (`n_nonzero`), sehingga efek ekstrem dari marker sangat
sparse tidak tampak setara dengan efek yang ditopang banyak pasangan.
"""
        ),
        code(
            r"""
show_figure("03_numeric_effect_volcano_vs_neutral.png", width=1700)
show_figure("04_numeric_effect_volcano_implicit_vs_explicit.png", width=1600)
"""
        ),
        markdown(
            r"""
## 5. Forest plot ber-CI: I1, I2, dan I3 terpisah

Untuk marker yang sparse, *presence* sering lebih mudah dibaca daripada median
density yang dapat tetap nol. Titik menunjukkan perubahan proporsi respons yang
memuat marker; garis menunjukkan 95% paired bootstrap CI. Density numeriknya
tetap tersedia pada tabel utama dan volcano di atas.
"""
        ),
        code(
            r"""
show_figure("05_proxy_presence_forest_structure_logic.png", width=1700)
show_figure("06_proxy_presence_forest_stance_voice.png", width=1700)
"""
        ),
        markdown(
            r"""
## 6. Sensitivitas correctness

Subset dibuat dengan urutan yang aman: pasangkan kondisi pada soal yang sama,
kemudian pertahankan pasangan jika **keduanya benar**. Analisis utama tidak
diganti oleh subset ini. Tujuannya hanya memeriksa apakah arah efek linguistik
berubah setelah confounding correctness dikurangi.
"""
        ),
        code(
            r"""
show_figure("07_correctness_sensitivity_effect_concordance.png", width=1650)
sensitivity = pd.read_csv(BUNDLE_DIR / "correctness_sensitivity_summary.csv")
display(
    sensitivity.style.format({
        "median_pair_retention": "{:.1%}",
        "same_direction_rate_including_zero": "{:.1%}",
        "same_direction_rate_supported": "{:.1%}",
        "spearman_rank_biserial": "{:.3f}",
        "median_abs_rank_biserial_change": "{:.3f}",
    })
)
"""
        ),
        markdown(
            r"""
## 7. Opening sebagai komposisi multinomial

Setiap respons memiliki satu kategori pembukaan, sehingga seluruh segmen dalam
satu bar berjumlah 100%. Inferensi pasangan dilakukan sekali per kontras dengan
Stuart–Maxwell marginal homogeneity; sebelas kategori tidak diperlakukan sebagai
sebelas cerita independen.
"""
        ),
        code(
            r"""
show_figure("08_opening_composition_100pct.png", width=1500)
opening_joint = pd.read_csv(BUNDLE_DIR / "paired_opening_multinomial.csv")
display(
    opening_joint.sort_values("marginal_total_variation", ascending=False)
    [["model_id", "identity", "contrast_id", "n_pairs", "same_category_rate",
      "marginal_total_variation", "p_holm"]]
    .head(20)
    .style.format({
        "same_category_rate": "{:.1%}",
        "marginal_total_variation": "{:.1%}",
        "p_holm": "{:.3g}",
    })
)
"""
        ),
        markdown(
            r"""
## 8. Token distingtif umum: ranked lollipop

Bobot adalah selisih prevalensi token biner per respons, dipasangkan pada soal
yang sama dan dirata-ratakan dengan bobot setara per model. Nilai eksak tetap
terlihat; token hanya masuk jika arahnya konsisten pada sedikitnya empat dari
enam model. Sumbu menunjukkan selisih prevalensi token per respons.
"""
        ),
        code(
            r"""
show_figure("09_ranked_lexical_lollipop_vs_neutral.png", width=1500)
show_figure("10_ranked_lexical_lollipop_implicit_vs_explicit.png", width=1500)
lexical = pd.read_csv(BUNDLE_DIR / "lexical_contrast_token_weights.csv")
display(
    lexical.nlargest(20, "abs_mean_model_delta_pp")
    [["identity", "contrast_id", "token", "mean_model_delta_pp",
      "direction_support_models", "n_models"]]
    .style.format({"mean_model_delta_pp": "{:+.3f}"})
)
"""
        ),
        markdown(
            r"""
## 9. Kata gender eksplisit: he/she, him/her, dan gender nouns

Bagian ini berbeda dari token distingtif umum di atas. Analisis memakai closed
lexicon khusus `he/him/his/himself`, `she/her/hers/herself`, serta human nouns
seperti `man/woman`, `boy/girl`, dan `father/mother`. Seluruh 71.215 baris CSV
diperiksa; hitungan hanya dibuat dari 70.248 `visible_text` yang tersedia.

Word cloud menunjukkan frekuensi 52 kata gender yang sudah ditentukan sebelum
analisis. Dua grafik setelahnya menampilkan prevalensi per model dan paired CI.
Prompt, hidden reasoning, dan raw JSON tidak ikut dihitung.
"""
        ),
        code(
            r"""
gender_metadata = json.loads(
    (BUNDLE_DIR / "gendered_lexicon_metadata.json").read_text(encoding="utf-8")
)
if gender_metadata.get("hidden_reasoning_used") is not False:
    raise ValueError("Gender lexicon must use visible text only.")
display(Markdown(
    f"**Lexicon:** `{gender_metadata['lexicon_version']}` · "
    f"**visible responses scanned:** {gender_metadata['n_available_responses']:,} · "
    f"**terms:** {gender_metadata['n_terms']}"
))

show_figure("11_gendered_terms_wordcloud_by_persona.png", width=1450)
show_figure("12_gender_pronoun_prevalence_by_model_persona.png", width=1450)
show_figure("13_gender_pronoun_proxy_forest.png", width=1650)

gender_terms = pd.read_csv(BUNDLE_DIR / "gendered_term_persona_aggregate.csv")
gender_pairs = pd.read_csv(BUNDLE_DIR / "paired_gender_pronoun_classes.csv")
gender_pairs["delta_pp"] = 100 * gender_pairs["prevalence_delta"]
gender_pairs["ci_low_pp"] = 100 * gender_pairs["ci_low"]
gender_pairs["ci_high_pp"] = 100 * gender_pairs["ci_high"]

display(Markdown("**Kata gender dengan density tertinggi per persona:**"))
display(
    gender_terms.sort_values(
        ["persona", "mean_model_occurrences_per_10k_words"],
        ascending=[True, False],
    )
    .groupby("persona", as_index=False)
    .head(8)
    [["persona", "term", "term_category",
      "mean_model_occurrences_per_10k_words",
      "mean_model_response_prevalence_pp", "total_occurrences"]]
    .style.format({
        "mean_model_occurrences_per_10k_words": "{:.2f}",
        "mean_model_response_prevalence_pp": "{:.2f}",
    })
)

display(Markdown("**Paired pronoun shifts terbesar, diurutkan menurut besar perubahan:**"))
display(
    gender_pairs.assign(abs_delta_pp=lambda frame: frame["delta_pp"].abs())
    .sort_values(["abs_delta_pp", "n_pairs"], ascending=[False, False])
    [["model_id", "identity", "pronoun_class", "contrast_id", "n_pairs",
      "delta_pp", "ci_low_pp", "ci_high_pp", "p_holm"]]
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
        markdown(
            r"""
## 10. Artefak dan batas klaim

Seluruh figure mempunyai CSV sumber yang dapat ditelusuri. Hasil belum boleh
ditulis sebagai temuan final sebelum dua audit manual selesai dan agreement
antarpenilai dilaporkan.
"""
        ),
        code(
            r"""
figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(
    lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1)
)
display(figure_index)
display(Markdown(f"Dokumentasi bundle: `{(BUNDLE_DIR / 'README.md').relative_to(REPO_ROOT)}`"))
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite notebook: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    print(f"[DONE] Notebook written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

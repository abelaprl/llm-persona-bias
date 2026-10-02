"""Create the executable thesis-facing direct-gender v3 notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


DEFAULT_OUTPUT = Path("notebooks/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb")


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    """Build a v2-style visual narrative while retaining all v3 analyses."""
    cells = [
        markdown(
            r"""
# Linguistic Marker Gender GSM8K — Direct Man vs Woman v3

Notebook ini menyajikan hasil **candidate** untuk Bab Evaluasi dengan alur
visual yang ringkas. Figur pembanding Neutral/Explicit memberi konteks awal,
kemudian klaim perbedaan gender diuji secara **langsung** dengan memasangkan
respons persona Man dan Woman dari model, kondisi, dan soal GSM8K yang sama.

Empat kondisi selalu dipisah: **Explicit, I1, I2, dan I3**. Pada semua hasil
direct gender, arah effect size adalah **Man − Woman**: nilai positif berarti
marker lebih tinggi pada persona Man dan nilai negatif berarti lebih tinggi
pada persona Woman.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name.casefold() == "notebooks":
    REPO_ROOT = REPO_ROOT.parent

RESULT_ROOT = REPO_ROOT / "result" / "linguistic_markers"
candidates = []
for path in RESULT_ROOT.glob("gender_gsm8k_bab4_visual_revision_v3_direct_gender_*"):
    manifest_path = path / "manifest.json"
    required_table = path / "direct_gender_paired_numeric.csv"
    if not manifest_path.is_file() or not required_table.is_file():
        continue
    candidate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("bundle_version") == "gender-gsm8k-bab4-visual-v3-direct-gender":
        candidates.append((path.stat().st_mtime_ns, path, candidate_manifest))

if not candidates:
    raise FileNotFoundError("Bundle v3 direct gender yang lengkap tidak ditemukan.")

_, BUNDLE_DIR, manifest = max(candidates)
FIG_DIR = BUNDLE_DIR / "images"

if manifest.get("hidden_reasoning_used") is not False:
    raise ValueError("Notebook hanya boleh memakai visible final-answer surface.")
if manifest.get("primary_correctness_filter_applied") is not False:
    warnings.warn("Konfigurasi correctness analisis utama berubah dari desain awal.")

def show_figure(name: str, width: int = 1500):
    path = FIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))

display(Markdown(
    f"**Bundle:** `{BUNDLE_DIR.relative_to(REPO_ROOT)}`  \n"
    f"**N sumber:** {manifest.get('source_response_records', 0):,} respons · "
    f"**bootstrap:** {manifest.get('bootstrap_resamples', 0):,} resample · "
    "**pair key direct gender:** `model_id × question_id`"
))
"""
        ),
        markdown(
            r"""
## 1. Provenance, visible surface, dan status audit

Marker dihitung dari teks jawaban yang terlihat. Hidden reasoning tidak
digunakan sebagai fallback. `reasoning_present` hanya merupakan metadata
sumber, sedangkan baris tanpa teks linguistik yang dapat dianalisis dicatat
terpisah.
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
        "Status hasil masih candidate sampai audit selesai."
    ))
"""
        ),
        markdown(
            r"""
## 2. Cakupan per model × 9 persona

Angka di dalam sel menunjukkan respons dengan teks yang tersedia dibanding
target. Missing, technical-only, dan source gap tidak diubah menjadi observasi
dengan marker nol.
"""
        ),
        code(
            r"""
show_figure("01_coverage_model_persona.png", width=1550)
coverage = pd.read_csv(BUNDLE_DIR / "response_counts_model_persona.csv")
display(
    coverage.sort_values("text_coverage_rate")
    [["model_id", "persona", "target", "n_text_available",
      "n_missing_text", "n_technical_only", "n_source_gap", "text_coverage_rate"]]
    .head(12)
    .style.format({"text_coverage_rate": "{:.1%}"})
)
"""
        ),
        markdown(
            r"""
## 3. Differential availability Man vs Woman

Availability diperiksa sebelum marker linguistik ditafsirkan. Titik menunjukkan
selisih proporsi **Man − Woman**, sedangkan garis menunjukkan 95% paired
bootstrap CI. Exact McNemar digunakan untuk pasangan diskordan.
"""
        ),
        code(
            r"""
show_figure("16_direct_gender_availability_forest.png", width=1700)

availability = pd.read_csv(BUNDLE_DIR / "direct_gender_availability.csv").assign(
    rate_man_pp=lambda frame: 100 * frame["rate_left"],
    rate_woman_pp=lambda frame: 100 * frame["rate_right"],
    delta_pp=lambda frame: 100 * frame["rate_delta"],
    ci_low_pp=lambda frame: 100 * frame["ci_low"],
    ci_high_pp=lambda frame: 100 * frame["ci_high"],
)
display(
    availability.assign(abs_delta_pp=lambda frame: frame["delta_pp"].abs())
    .sort_values(["abs_delta_pp", "p_holm"], ascending=[False, True])
    [["model_id", "condition", "outcome_id", "n_union_questions",
      "rate_man_pp", "rate_woman_pp", "delta_pp", "ci_low_pp",
      "ci_high_pp", "p_holm"]]
    .head(12)
    .style.format({
        "rate_man_pp": "{:.2f}", "rate_woman_pp": "{:.2f}",
        "delta_pp": "{:+.2f}", "ci_low_pp": "{:+.2f}",
        "ci_high_pp": "{:+.2f}", "p_holm": "{:.3g}",
    })
)
"""
        ),
        markdown(
            r"""
## 4. Verbosity dan dasar normalisasi

Density marker menggunakan satuan per 100 kata. Panjang respons absolut tetap
ditampilkan agar perubahan marker tidak keliru dibaca sebagai efek persona
ketika sebenarnya mengikuti perubahan verbositas.
"""
        ),
        code('show_figure("02_verbosity_word_count_median_iqr.png", width=1450)'),
        markdown(
            r"""
## 5. Peta efek terhadap Neutral dan Explicit

Dua figur ini mempertahankan konteks dari v2. Sumbu-x adalah rank-biserial dari
selisih berpasangan dan sumbu-y adalah `−log10(p_holm)`. I1, I2, dan I3 tetap
berada pada panel terpisah sehingga heterogenitas proxy terlihat.
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
## 6. Hasil utama: direct Man vs Woman

Volcano menguji perbedaan Man dan Woman secara langsung pada soal yang sama.
Ukuran titik mengikuti jumlah pasangan yang benar-benar berubah
(`n_nonzero`). Figur konsistensi di bawahnya memperlihatkan apakah arah efek
yang sama muncul lintas enam model atau terutama didorong model tertentu.
Nilai `p_holm` asli tetap tersedia pada tabel ringkas.
"""
        ),
        code(
            r"""
show_figure("14_direct_gender_numeric_volcano.png", width=1800)
show_figure("15_direct_gender_effect_consistency.png", width=1800)

numeric = pd.read_csv(BUNDLE_DIR / "direct_gender_paired_numeric.csv")
primary = numeric[numeric["analysis_role"].eq("primary")].copy()
supported = primary[primary["n_nonzero"].ge(20)].copy()
display(Markdown(
    f"**Primary tests:** {len(primary):,} · "
    f"**p Holm < 0,05:** {primary['p_holm'].lt(0.05).sum():,} · "
    f"**supported (≥20 pasangan nonzero):** {len(supported):,}"
))

display(
    supported.assign(abs_rank_biserial=lambda frame: frame["rank_biserial"].abs())
    .sort_values(["p_holm", "abs_rank_biserial"], ascending=[True, False])
    [["model_id", "condition", "outcome_id", "n_pairs", "n_nonzero",
      "median_paired_delta", "rank_biserial", "p_holm", "test_used"]]
    .head(15)
    .style.format({
        "median_paired_delta": "{:+.3f}",
        "rank_biserial": "{:+.3f}",
        "p_holm": "{:.3g}",
    })
)
"""
        ),
        markdown(
            r"""
## 7. Forest plot marker presence per proxy

Untuk marker yang sparse, proporsi respons yang memuat marker sering lebih
mudah dibaca daripada median density. Titik menunjukkan perubahan prevalence
terhadap pembanding, garis menunjukkan 95% paired bootstrap CI, dan setiap
proxy implicit tetap ditampilkan terpisah.
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
## 8. Sensitivitas correctness

Pairing dilakukan lebih dahulu, lalu pasangan dipertahankan hanya jika kedua
respons benar. Figur menunjukkan pemeriksaan v2 terhadap pembanding; tabel
kedua merangkum stabilitas arah direct Man–Woman pada empat kondisi.
"""
        ),
        code(
            r"""
show_figure("07_correctness_sensitivity_effect_concordance.png", width=1650)

sensitivity = pd.read_csv(BUNDLE_DIR / "correctness_sensitivity_summary.csv")
display(Markdown("**Sensitivitas efek terhadap Neutral/Explicit:**"))
display(
    sensitivity.style.format({
        "median_pair_retention": "{:.1%}",
        "same_direction_rate_including_zero": "{:.1%}",
        "same_direction_rate_supported": "{:.1%}",
        "spearman_rank_biserial": "{:.3f}",
        "median_abs_rank_biserial_change": "{:.3f}",
    })
)

direct_sensitivity = pd.read_csv(BUNDLE_DIR / "direct_gender_correctness_summary.csv")
display(Markdown("**Sensitivitas direct Man–Woman:**"))
display(
    direct_sensitivity.style.format({
        "median_pair_retention": "{:.1%}",
        "same_direction_rate_supported": "{:.1%}",
    })
)

direct_comparison = pd.read_csv(BUNDLE_DIR / "direct_gender_correctness_comparison.csv")
reversals = direct_comparison[
    direct_comparison["supported_both"] & ~direct_comparison["same_direction"]
].copy()
display(Markdown(
    f"**Sign reversal dengan ≥20 perubahan pada kedua analisis:** {len(reversals)}"
))
if not reversals.empty:
    display(
        reversals[["model_id", "condition", "outcome_id", "n_pairs_all",
                   "n_pairs_both_correct", "rank_biserial_all",
                   "rank_biserial_both_correct", "p_holm_all",
                   "p_holm_both_correct"]]
        .head(10)
        .style.format({
            "rank_biserial_all": "{:+.3f}",
            "rank_biserial_both_correct": "{:+.3f}",
            "p_holm_all": "{:.3g}",
            "p_holm_both_correct": "{:.3g}",
        })
    )
"""
        ),
        markdown(
            r"""
## 9. Opening sebagai komposisi multinomial

Setiap respons memiliki satu kategori pembukaan, sehingga segmen dalam satu bar
berjumlah 100%. Figur pertama memberi konteks persona terhadap pembanding;
figur kedua menunjukkan perbedaan komposisi direct Man–Woman. Inferensi pasangan
dilakukan sekali per kondisi dengan Stuart–Maxwell marginal homogeneity
(Stuart, 1955; Maxwell, 1970), bukan sebagai sekumpulan outcome biner bebas.
"""
        ),
        code(
            r"""
show_figure("08_opening_composition_100pct.png", width=1500)
show_figure("17_direct_gender_opening_composition.png", width=1700)

opening = pd.read_csv(BUNDLE_DIR / "direct_gender_opening_multinomial.csv")
display(
    opening.sort_values("marginal_total_variation", ascending=False)
    [["model_id", "condition", "n_pairs", "same_category_rate",
      "marginal_total_variation", "statistic", "df", "p_holm"]]
    .head(12)
    .style.format({
        "same_category_rate": "{:.1%}",
        "marginal_total_variation": "{:.1%}",
        "statistic": "{:.3f}", "p_holm": "{:.3g}",
    })
)
"""
        ),
        markdown(
            r"""
## 10. Token distingtif umum: ranked lollipop

Bobot adalah selisih prevalence token biner per respons, dipasangkan pada soal
yang sama lalu dirata-ratakan dengan bobot setara per model. Nilai eksak dapat
dibaca pada sumbu dan tabel. Token ditampilkan jika arah perubahannya sama pada
sedikitnya empat dari enam model—ambang mayoritas ini digunakan agar ranking
tidak didominasi pola satu atau dua model.
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
## 11. Penyebutan gender: he/she, him/her, dan gender nouns

Bagian ini memakai closed lexicon kata gender pada seluruh `visible_text` yang
tersedia. Grafik pertama memperlihatkan prevalence kelas pronoun per model dan
persona dengan cabang Man dan Woman yang terpisah. Forest plot menguji direct
Man–Woman per kondisi dengan 95% paired bootstrap CI. Word cloud tidak
ditampilkan di notebook utama karena perbandingan kuantitatif lebih mudah
dibaca pada kedua grafik ini.
"""
        ),
        code(
            r"""
gender_metadata = json.loads(
    (BUNDLE_DIR / "gendered_lexicon_metadata.json").read_text(encoding="utf-8")
)
if gender_metadata.get("hidden_reasoning_used") is not False:
    raise ValueError("Gender lexicon harus memakai visible text saja.")
display(Markdown(
    f"**Lexicon:** `{gender_metadata['lexicon_version']}` · "
    f"**visible responses scanned:** {gender_metadata['n_available_responses']:,} · "
    f"**terms:** {gender_metadata['n_terms']}"
))

show_figure("12_gender_pronoun_prevalence_by_model_persona.png", width=1500)
show_figure("13_gender_pronoun_direct_man_vs_woman_forest.png", width=1700)

pronoun = pd.read_csv(BUNDLE_DIR / "paired_gender_pronoun_direct.csv").assign(
    delta_pp=lambda frame: 100 * frame["prevalence_delta"],
    ci_low_pp=lambda frame: 100 * frame["ci_low"],
    ci_high_pp=lambda frame: 100 * frame["ci_high"],
)
display(
    pronoun.assign(abs_delta_pp=lambda frame: frame["delta_pp"].abs())
    .sort_values(["p_holm", "abs_delta_pp"], ascending=[True, False])
    [["model_id", "condition", "pronoun_class", "n_pairs", "delta_pp",
      "ci_low_pp", "ci_high_pp", "p_holm"]]
    .head(12)
    .style.format({
        "delta_pp": "{:+.2f}", "ci_low_pp": "{:+.2f}",
        "ci_high_pp": "{:+.2f}", "p_holm": "{:.3g}",
    })
)
"""
        ),
        markdown(
            r"""
## 12. Artefak dan batas klaim

Seluruh figur memiliki tabel sumber yang dapat ditelusuri. Direct effect dibaca
per model; notebook tidak menghitung pooled causal effect lintas enam model.
Status candidate dipertahankan sampai audit manual precision/recall marker
selesai.
"""
        ),
        code(
            r"""
figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(
    lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1)
)
display(figure_index)

required_tables = [
    "direct_gender_paired_numeric.csv",
    "direct_gender_correctness_sensitivity.csv",
    "direct_gender_paired_binary.csv",
    "direct_gender_availability.csv",
    "direct_gender_opening_multinomial.csv",
    "paired_gender_pronoun_direct.csv",
]
missing = [name for name in required_tables if not (BUNDLE_DIR / name).is_file()]
if missing:
    raise FileNotFoundError(missing)

display(Markdown(
    f"**Figur dalam bundle:** {len(figure_index)} · "
    f"**tabel direct wajib:** {len(required_tables)} · "
    f"**file hilang:** {len(missing)}  \n"
    f"Dokumentasi bundle: `{(BUNDLE_DIR / 'README.md').relative_to(REPO_ROOT)}`"
))
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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

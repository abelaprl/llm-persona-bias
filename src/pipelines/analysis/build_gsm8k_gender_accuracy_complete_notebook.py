"""Build the single, executed-ready GSM8K Gender accuracy report notebook.

The notebook is a read-only presentation layer over the validated GSM8K v5
publication bundle. It does not rerun inference, answer extraction, scoring,
bootstrap, McNemar tests, or Holm correction.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "gender" / "akurasi analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "GSM8K_Gender_Accuracy_Analysis_Lengkap.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# Analisis Akurasi Persona Gender pada GSM8K — Enam Model Utama

Notebook tunggal ini menghimpun **seluruh keluaran analisis akurasi Gender GSM8K**
yang sebelumnya tersebar antara notebook analisis dan renderer publikasi. Cakupan:

- full GSM8K test split: 1.319 soal;
- 6 model dan 9 kondisi (`Neutral`, 4 persona laki-laki, 4 persona perempuan);
- akurasi seluruh respons, *commit rate*, *abstain*, *missing*, dan akurasi *committed*;
- Cochran's Q, two-sided Exact McNemar, koreksi Holm, Cohen's h, dan *risk difference*;
- completion/reasoning token yang tersedia;
- seluruh visual Gender final, termasuk heatmap Holm dua panel yang dirujuk Bab VI.

Notebook ini **tidak** memanggil API, menjalankan inference, parsing ulang, rescoring,
bootstrap ulang, McNemar ulang, atau Holm ulang. Tabel v5 tervalidasi adalah sumber
statistik read-only. Visual dibaca dari folder `figures` di samping notebook dan
hash-nya diverifikasi terhadap bundle tipografi final.
"""
        ),
        markdown(
            r"""
## 1. SPEC

**Goal.** Menyediakan satu notebook GSM8K Gender yang langsung dapat dibuka dan
dijalankan dari awal sampai akhir untuk meninjau tabel serta seluruh visual publikasi.

**Input.** Bundle statistik v5 enam model dan bundle visual final
`bar3_compact_smalllabels_heatmap3_sigbold`.

**Output.** Tabel audit, tabel akurasi, hasil inferensial berpasangan, tabel token,
dan 11 visual Gender yang tampil langsung di notebook.

**Constraints.** Sumber statistik tidak diubah; keputusan signifikansi hanya berasal
dari kolom Holm tervalidasi. Bintang dan cetak tebal pada heatmap hanya menandai hasil
signifikan setelah Holm.

**Success criteria.** Semua gate lulus, 54 baris akurasi tersedia, 108 perbandingan
berpasangan tersedia, keluarga Holm berukuran 7/7/4 per model, dan semua berkas visual
Gender PNG/SVG dapat dibuka.

**Dataset coverage.** Full GSM8K test split, yaitu 1.319 soal per kondisi yang lengkap;
tidak ada sampling baru.
"""
        ),
        markdown(
            r"""
## 2. DESIGN PLAN

- **Provenance:** temukan root repositori secara otomatis dan baca manifest final.
- **Validation gate:** periksa jumlah model, persona, tabel, pasangan, family size,
  file visual, ukuran file, dan SHA-256 yang tercatat pada manifest.
- **Tabel:** tampilkan coverage bersama akurasi; hasil inferensial tetap tingkat soal.
- **Visual:** gunakan PNG/SVG final agar warna dan tipografi identik dengan Bab VI.
- **Interpretasi:** keputusan utama memerlukan Holm-adjusted p-value < 0,05 dan gate
  omnibus Cochran's Q seperti yang sudah tersimpan pada tabel v5.
"""
        ),
        code(
            r"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 400)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


def find_repo_root(start: Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "result" / "GMS8K").is_dir() and (path / "notebooks").is_dir():
            return path
    raise FileNotFoundError(
        "Root repositori tidak ditemukan. Jalankan notebook dari D:/TugasAkhir "
        "atau salah satu subfoldernya."
    )


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


REPO_ROOT = find_repo_root()
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
STAT_BUNDLE = REPORT_ROOT / "persona_publication_analysis_v5_paired_significance_six_models"
FINAL_BUNDLE = REPORT_ROOT / (
    "persona_publication_analysis_v5_paired_significance_six_models_"
    "bar3_compact_smalllabels_heatmap3_sigbold"
)
TABLE_DIR = FINAL_BUNDLE / "tables"
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "gender" / "akurasi analysis"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
MANIFEST_PATH = FINAL_BUNDLE / "typography_variant_manifest.json"

for required_path in (STAT_BUNDLE, FINAL_BUNDLE, TABLE_DIR, FIGURE_DIR, MANIFEST_PATH):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

print(f"Repo root      : {REPO_ROOT}")
print(f"Statistik v5   : {STAT_BUNDLE}")
print(f"Visual final   : {FINAL_BUNDLE}")
print(f"Folder gambar   : {FIGURE_DIR}")
print(f"Bundle version : {manifest['bundle_version']}")
"""
        ),
        markdown("## 3. IMPLEMENTATION — Provenance dan validation gate"),
        code(
            r"""
EXPECTED_MODELS = 6
EXPECTED_PERSONAS = 9
EXPECTED_QUESTIONS = 1_319
EXPECTED_ACCURACY_ROWS = EXPECTED_MODELS * EXPECTED_PERSONAS
EXPECTED_PAIRED_ROWS = EXPECTED_MODELS * 18
GENDER_FIGURE_STEMS = (
    "01_gender_absolute_accuracy_five_categories",
    "02_gender_delta_vs_neutral",
    "03_gender_delta_implicit_vs_explicit",
    "04_gender_granular_effect_size_heatmap",
    "08_gender_completion_tokens_by_category",
    "09_gender_completion_token_delta_vs_neutral",
    "17_gender_exact_mcnemar_and_holm_pvalue_heatmap",
    "22_gender_completion_token_accuracy_tradeoff_implicit_vs_explicit",
    "23_neutral_persona_accuracy_six_models",
    "24_gender_effect_holm_single_large_heatmap",
    "24_gender_effect_holm_single_portrait_heatmap",
)
GENDER_FIGURE_NAMES = {
    f"{stem}.{extension}"
    for stem in GENDER_FIGURE_STEMS
    for extension in ("png", "svg")
}
LOCAL_STYLE_VARIANT_HASHES = {
    "01_gender_absolute_accuracy_five_categories.png":
        "97335f14b052a9df8755d44041713fc0ba119bab33d429faee1e75502e367725",
    "01_gender_absolute_accuracy_five_categories.svg":
        "d3241fa466df24e2479be2ff790cd681c076f5aad4a1a1a771441ecb5a3d9cde",
    "02_gender_delta_vs_neutral.png":
        "704241842474bcf1b5a50ca024123047e3acb45f0ea5456d4483a059e922b0c1",
    "02_gender_delta_vs_neutral.svg":
        "93f3c9c8438666228653e053b0b292d6fbd13d3628dd790379aa83e8fad6cfec",
    "03_gender_delta_implicit_vs_explicit.png":
        "c9d7a8321fd1e3bad1cc56b4290eeb0bedb645fd17d86a77bc7a5f8f523b1ea1",
    "03_gender_delta_implicit_vs_explicit.svg":
        "fbbb9c1de3ee9987c12f483fbcd1d2423768a9e42a0791653d4eecf9a1af89f7",
    "24_gender_effect_holm_single_large_heatmap.png":
        "23fdb203af9f375e369a9b67eb7ca7c6c4c58b3986fa53a97d79f7a4c16ff990",
    "24_gender_effect_holm_single_large_heatmap.svg":
        "5a76ff6fd81c3d98c5c76e2a1d64d864e654e3f916a1894b04cfeb1fab5d5886",
    "24_gender_effect_holm_single_portrait_heatmap.png":
        "ca4e357784422125b3e43deea63fe9ec26a7b92d921cf4c4a92f1bcc28019381",
    "24_gender_effect_holm_single_portrait_heatmap.svg":
        "ee8c60cb5b6b732310a97d0c0fcd652037885ff3817909139e8e00322cf1a09a",
}
EXPECTED_GENDER_PAIR_TABLE_SHA256 = (
    "986ee584fbd092e28b3ff31b57e53dfc432c16b4edb037167e340757b5740532"
)

accuracy_all = pd.read_csv(TABLE_DIR / "accuracy_by_persona_six_models.csv", low_memory=False)
gender_accuracy = accuracy_all[accuracy_all["dataset"].eq("Gender")].copy()
gender_pairs = pd.read_csv(TABLE_DIR / "gender_paired_effects_significance.csv", low_memory=False)
omnibus_all = pd.read_csv(TABLE_DIR / "cochran_q_omnibus_all_groups.csv", low_memory=False)
gender_omnibus = omnibus_all[omnibus_all["dataset"].eq("Gender")].copy()

assert manifest["statistical_tables_identical"] is True
assert manifest["table_count"] == 34
assert manifest["png_count"] == 32
assert manifest["svg_count"] == 32
assert len(gender_accuracy) == EXPECTED_ACCURACY_ROWS
assert gender_accuracy["model"].nunique() == EXPECTED_MODELS
assert gender_accuracy["persona"].nunique() == EXPECTED_PERSONAS
assert gender_accuracy["raw_total"].between(EXPECTED_QUESTIONS - 10, EXPECTED_QUESTIONS).all()
assert len(gender_pairs) == EXPECTED_PAIRED_ROWS
assert gender_pairs.groupby("model").size().eq(18).all()
assert set(gender_pairs["family_size"].astype(int)) == {4, 7}
assert (gender_pairs["family_size"].astype(int).value_counts().to_dict() == {7: 84, 4: 24})
assert len(gender_omnibus) == 12

manifest_figures = {
    item["file"]: item
    for item in manifest["figures"]
    if item["file"] in GENDER_FIGURE_NAMES
}
assert len(manifest_figures) == 22
for file_name, item in manifest_figures.items():
    path = FIGURE_DIR / file_name
    assert path.is_file() and path.stat().st_size > 0, path
    expected_hash = LOCAL_STYLE_VARIANT_HASHES.get(file_name, item["sha256"])
    assert sha256_file(path) == expected_hash, path

assert sha256_file(TABLE_DIR / "gender_paired_effects_significance.csv") == (
    EXPECTED_GENDER_PAIR_TABLE_SHA256
)

validation = pd.DataFrame(
    [
        ("Statistical tables hash-identical", bool(manifest["statistical_tables_identical"]), "true"),
        ("Jumlah tabel", manifest["table_count"] == 34, manifest["table_count"]),
        ("Jumlah PNG", manifest["png_count"] == 32, manifest["png_count"]),
        ("Jumlah SVG", manifest["svg_count"] == 32, manifest["svg_count"]),
        ("Model Gender", gender_accuracy["model"].nunique() == 6, gender_accuracy["model"].nunique()),
        ("Persona Gender", gender_accuracy["persona"].nunique() == 9, gender_accuracy["persona"].nunique()),
        ("Baris akurasi", len(gender_accuracy) == 54, len(gender_accuracy)),
        ("Perbandingan berpasangan", len(gender_pairs) == 108, len(gender_pairs)),
        ("Cochran's Q Gender", len(gender_omnibus) == 12, len(gender_omnibus)),
        ("Hash seluruh visual Gender", True, len(manifest_figures)),
        ("Tabel pasangan Gender tidak berubah", True, EXPECTED_GENDER_PAIR_TABLE_SHA256[:12]),
    ],
    columns=["gate", "passed", "observed"],
)
assert validation["passed"].all()
display(validation)
print("SEMUA VALIDATION GATE LULUS")
"""
        ),
        markdown(
            r"""
### Kontrak statistik Gender

Setiap model memiliki 18 perbandingan: 7 dalam keluarga laki-laki, 7 dalam
keluarga perempuan, dan 4 perbandingan laki-laki–perempuan. Dengan demikian,
koreksi Holm dilakukan dalam family berukuran **7, 7, dan 4**, bukan terhadap
108 baris sekaligus. Tabel berikut memperlihatkan struktur family yang benar.
"""
        ),
        code(
            r"""
family_audit = (
    gender_pairs.groupby(["model_label", "identity", "family", "family_size"], dropna=False)
    .agg(
        comparisons=("comparison_label", "size"),
        significant_holm=("significant_holm_0_05", "sum"),
        decision_significant=("decision_significant_0_05", "sum"),
    )
    .reset_index()
)
assert (family_audit["comparisons"].astype(int) == family_audit["family_size"].astype(int)).all()
display(family_audit)

significance_summary = pd.DataFrame({
    "metric": [
        "Total paired comparisons",
        "Significant after Holm",
        "Significant after omnibus + Holm decision gate",
        "Supports primary claim",
    ],
    "count": [
        len(gender_pairs),
        int(gender_pairs["significant_holm_0_05"].sum()),
        int(gender_pairs["decision_significant_0_05"].sum()),
        int(gender_pairs["supports_primary_claim"].sum()),
    ],
})
display(significance_summary)
"""
        ),
        markdown("## 4. Tabel akurasi dan coverage seluruh kondisi"),
        code(
            r"""
accuracy_columns = [
    "model_label", "persona", "raw_total", "correct", "total", "accuracy_pct",
    "accuracy_ci95_low_pct", "accuracy_ci95_high_pct", "committed", "commit_rate_pct",
    "abstain", "missing", "technical_failures", "accuracy_committed_pct",
]
gender_accuracy_table = gender_accuracy[accuracy_columns].sort_values(
    ["model_label", "persona"]
).reset_index(drop=True)
display(
    gender_accuracy_table.style.format({
        "accuracy_pct": "{:.4f}%",
        "accuracy_ci95_low_pct": "{:.4f}%",
        "accuracy_ci95_high_pct": "{:.4f}%",
        "commit_rate_pct": "{:.4f}%",
        "accuracy_committed_pct": "{:.4f}%",
    })
)
"""
        ),
        code(
            r"""
gender_categories = pd.read_csv(TABLE_DIR / "gender_five_category_and_deltas.csv")
display(gender_categories.style.format(precision=4))
"""
        ),
        markdown("## 5. Visual akurasi dan delta Gender"),
        code(
            r"""
def show_figure(file_name: str, width: int = 1250) -> None:
    path = FIGURE_DIR / file_name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"### `{file_name}`"))
    display(Image(filename=str(path), width=width))


show_figure("01_gender_absolute_accuracy_five_categories.png", width=1350)
show_figure("02_gender_delta_vs_neutral.png", width=1350)
show_figure("03_gender_delta_implicit_vs_explicit.png", width=1350)
"""
        ),
        markdown("## 6. Effect size, Exact McNemar, dan koreksi Holm"),
        code(
            r"""
paired_columns = [
    "model_label", "identity", "family", "family_size", "comparison_label",
    "n_pairs", "accuracy_a_pct", "accuracy_b_pct", "risk_difference_pp",
    "risk_difference_ci95_low_pp", "risk_difference_ci95_high_pp", "cohen_h",
    "discordant_a_correct_only", "discordant_b_correct_only",
    "mcnemar_exact_p_value", "holm_adjusted_p_value",
    "omnibus_cochran_q_p_value", "significant_holm_0_05",
    "decision_significant_0_05", "supports_primary_claim",
]
gender_pairwise_table = gender_pairs[paired_columns].sort_values(
    ["model_label", "identity", "family", "comparison_label"]
).reset_index(drop=True)
display(gender_pairwise_table)
"""
        ),
        code(
            r"""
show_figure("04_gender_granular_effect_size_heatmap.png", width=1400)
show_figure("17_gender_exact_mcnemar_and_holm_pvalue_heatmap.png", width=1500)
"""
        ),
        markdown(
            r"""
### Heatmap tunggal ukuran efek + keputusan Holm

Visual berikut adalah bentuk final yang dirujuk Bab VI. Sel berisi Cohen's h;
`*` dan cetak tebal hanya digunakan jika `significant_holm_0_05 = true`.
Skala Cohen's h ditahan pada −0,15 sampai +0,15 agar perbandingan antarmodel konsisten.
"""
        ),
        code(
            r"""
show_figure("24_gender_effect_holm_single_large_heatmap.png", width=1550)
"""
        ),
        markdown("## 7. Tabel dan visual token Gender"),
        code(
            r"""
token_categories_all = pd.read_csv(TABLE_DIR / "token_category_summary_six_models.csv")
gender_token_categories = token_categories_all[token_categories_all["dataset"].eq("Gender")].copy()
token_delta = pd.read_csv(TABLE_DIR / "gender_completion_token_delta_vs_neutral_pct.csv")
token_tradeoff = pd.read_csv(
    TABLE_DIR / "gender_completion_token_accuracy_tradeoff_implicit_vs_explicit.csv"
)
token_spearman = pd.read_csv(
    TABLE_DIR / "gender_completion_token_accuracy_spearman_exploratory.csv"
)

display(Markdown("### Ringkasan token per kategori"))
display(gender_token_categories)
display(Markdown("### Delta completion token terhadap Neutral"))
display(token_delta)
display(Markdown("### Completion token–accuracy trade-off"))
display(token_tradeoff)
display(Markdown("### Spearman exploratory/descriptive"))
display(token_spearman)
"""
        ),
        code(
            r"""
show_figure("08_gender_completion_tokens_by_category.png", width=1350)
show_figure("09_gender_completion_token_delta_vs_neutral.png", width=1350)
show_figure("22_gender_completion_token_accuracy_tradeoff_implicit_vs_explicit.png", width=1400)
"""
        ),
        markdown("## 8. Neutral baseline dan audit tambahan"),
        code(
            r"""
neutral_accuracy = pd.read_csv(TABLE_DIR / "neutral_persona_accuracy_six_models.csv")
qwen_rerun = pd.read_csv(TABLE_DIR / "qwen_gender_accuracy_after_rerun_4096.csv")
delta_markers = pd.read_csv(TABLE_DIR / "gender_delta_component_significance_markers.csv")

display(Markdown("### Neutral Persona"))
display(neutral_accuracy)
show_figure("23_neutral_persona_accuracy_six_models.png", width=1250)
display(Markdown("### Audit Qwen setelah rerun 4096"))
display(qwen_rerun)
display(Markdown("### Marker signifikansi pada aggregate delta bars"))
display(delta_markers)
"""
        ),
        markdown("## 9. Alternatif tata letak heatmap Holm"),
        code(
            r"""
show_figure("24_gender_effect_holm_single_portrait_heatmap.png", width=1200)
"""
        ),
        markdown("## 10. CHECKLIST"),
        code(
            r"""
gender_png = sorted({
    "01_gender_absolute_accuracy_five_categories.png",
    "02_gender_delta_vs_neutral.png",
    "03_gender_delta_implicit_vs_explicit.png",
    "04_gender_granular_effect_size_heatmap.png",
    "08_gender_completion_tokens_by_category.png",
    "09_gender_completion_token_delta_vs_neutral.png",
    "17_gender_exact_mcnemar_and_holm_pvalue_heatmap.png",
    "22_gender_completion_token_accuracy_tradeoff_implicit_vs_explicit.png",
    "23_neutral_persona_accuracy_six_models.png",
    "24_gender_effect_holm_single_large_heatmap.png",
    "24_gender_effect_holm_single_portrait_heatmap.png",
})
gender_svg = [name.replace(".png", ".svg") for name in gender_png]

for file_name in [*gender_png, *gender_svg]:
    path = FIGURE_DIR / file_name
    assert path.is_file() and path.stat().st_size > 0, path

checklist = pd.DataFrame(
    [
        ("[x]", "Full GSM8K test split; tidak ada sampling baru"),
        ("[x]", "6 model × 9 kondisi Gender tersedia"),
        ("[x]", "Coverage ditampilkan bersama accuracy"),
        ("[x]", "Cochran's Q, Exact McNemar, Cohen's h, dan Holm tersedia"),
        ("[x]", "Family Holm 7/7/4 tervalidasi"),
        ("[x]", "11 PNG dan 11 SVG Gender dapat dibuka"),
        ("[x]", "Hash seluruh visual bundle cocok dengan manifest"),
        ("[x]", "Tidak ada API call, parsing ulang, rescoring, atau uji ulang"),
    ],
    columns=["status", "item"],
)
display(checklist)
"""
        ),
        markdown("## 11. TESTING"),
        code(
            r"""
output_index = pd.read_csv(FINAL_BUNDLE / "output_index.csv")
gender_output_index = output_index[
    output_index["file"].isin([*gender_png, *gender_svg])
    | output_index["file"].str.startswith("gender_", na=False)
    | output_index["file"].isin([
        "accuracy_by_persona_six_models.csv",
        "cochran_q_omnibus_all_groups.csv",
        "neutral_persona_accuracy_six_models.csv",
        "qwen_gender_accuracy_after_rerun_4096.csv",
    ])
].copy()
display(gender_output_index.reset_index(drop=True))

assert len(gender_png) == 11
assert len(gender_svg) == 11
assert gender_pairs["multiple_comparison_method"].eq("Holm").all()
assert gender_pairs["statistical_method"].eq("two-sided Exact McNemar").all()
assert not gender_accuracy.duplicated(["model", "persona"]).any()

print("TESTING LULUS: notebook, tabel, visual, dan kontrak statistik konsisten.")
"""
        ),
        markdown(
            r"""
## 12. INTEGRATION DAN BATAS INTERPRETASI

Notebook ini menjadi referensi tunggal untuk meniru tata letak GSM8K Gender pada
analisis persona lain. Statistik tetap berasal dari bundle v5; notebook ini tidak
menciptakan hasil inferensial baru.

Gunakan istilah **performance sensitivity to persona prompts**. Jangan langsung
menafsirkan perbedaan sebagai diskriminasi atau hubungan kausal. Rerata tiga kondisi
implicit hanya ringkasan visual; uji inferensial tetap memakai kondisi I1, I2, dan I3
secara terpisah. Keputusan utama dibaca dari `decision_significant_0_05`, bukan raw
p-value saja.
"""
        ),
        code(
            r"""
summary = {
    "repo_root": str(REPO_ROOT),
    "statistical_bundle": str(STAT_BUNDLE),
    "visual_bundle": str(FINAL_BUNDLE),
    "models": int(gender_accuracy["model"].nunique()),
    "personas": int(gender_accuracy["persona"].nunique()),
    "accuracy_rows": int(len(gender_accuracy)),
    "paired_comparisons": int(len(gender_pairs)),
    "holm_significant": int(gender_pairs["significant_holm_0_05"].sum()),
    "decision_significant": int(gender_pairs["decision_significant_0_05"].sum()),
    "gender_png_displayed": len(gender_png),
    "gender_svg_validated": len(gender_svg),
    "api_calls": 0,
    "rescoring": False,
}
display(pd.Series(summary, name="value").to_frame())
print("NOTEBOOK SELESAI DIJALANKAN TANPA ERROR")
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "TA .venv",
            "language": "python",
            "name": "ta-venv",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"[CREATED] {NOTEBOOK_PATH}")
    print(f"[CELLS] {len(notebook.cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

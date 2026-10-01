"""Build the thesis-facing standardized ETHICS Religion accuracy notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "religion" / "ethics accuracy analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb"
BUNDLE_VERSION = "ethics-religion-accuracy-combined-standardized-v1"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# Analisis Akurasi Persona Religion pada ETHICS — Enam Model Utama

Notebook ini mengikuti struktur visual ETHICS Gender dan Regional: chart
akurasi besar, delta terhadap `Neutral`, status respons, alasan `abstain`,
effect size, exact McNemar, dan Holm-adjusted p-value.

Cakupan parent menggunakan 154.428 respons dari 6 model × 17 skenario × 4
subset. Status dipool dari seluruh subset, sedangkan akurasi parent hanya
memakai `commonsense` dan `justice`. Detail setiap subset ditempatkan di
`per_subset/`. Notebook ini tidak memanggil API, melakukan parsing ulang, atau
rescoring. Statistik per subset dibaca dari v3; statistik gabungan dihitung
oleh renderer dari output tervalidasi.
"""
        ),
        markdown(
            r"""
## 1. SPEC dan provenance

Sumber aktif ditelusuri dari `Bab VI - Evaluasi.tex` ke
`Bab VI - ETHICS Religion Akurasi.tex`, kemudian ke bundle v3 Religion yang
tervalidasi. Exact McNemar, koreksi Holm, dan Cohen's h per subset dibaca dari
tabel v3 secara read-only. Versi gabungan dihitung hanya pada Commonsense dan
Justice dengan pairing key `subset + question_id`. Jumlah pasangan observasi
tidak disajikan pada tabel ringkas atau visual notebook ini.
"""
        ),
        code(
            r"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 700)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


def find_repo_root(start: Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "result" / "ETHICS").is_dir() and (path / "notebooks").is_dir():
            return path
    raise FileNotFoundError("Root repositori tidak ditemukan.")


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


REPO_ROOT = find_repo_root()
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "religion" / "ethics accuracy analysis"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = NOTEBOOK_DIR / "tables"
CHILD_DIR = NOTEBOOK_DIR / "per_subset"
LOCAL_MANIFEST = json.loads((NOTEBOOK_DIR / "bundle_manifest.json").read_text(encoding="utf-8"))
SOURCE_BUNDLE = Path(LOCAL_MANIFEST["created_from"])
SOURCE_MANIFEST = json.loads((SOURCE_BUNDLE / "metadata" / "analysis_manifest.json").read_text(encoding="utf-8"))

condition_summary = pd.read_csv(TABLE_DIR / "ethics_condition_summary_source_derived.csv", low_memory=False)
accuracy = pd.read_csv(TABLE_DIR / "ethics_accuracy_by_model_persona_category.csv", low_memory=False)
status = pd.read_csv(TABLE_DIR / "ethics_status_by_model_persona_category.csv", low_memory=False)
reason_model = pd.read_csv(TABLE_DIR / "ethics_abstain_reasons_by_model.csv", low_memory=False)
paired_all = pd.read_csv(TABLE_DIR / "ethics_paired_effects_all.csv", low_memory=False)
paired_scenario = pd.read_csv(TABLE_DIR / "ethics_paired_effects_scenario.csv", low_memory=False)
paired_religion = pd.read_csv(TABLE_DIR / "ethics_paired_effects_religion.csv", low_memory=False)
accuracy_inference = pd.read_csv(TABLE_DIR / "ethics_accuracy_inference_construct_valid.csv", low_memory=False)
accuracy_inference_combined = pd.read_csv(TABLE_DIR / "ethics_accuracy_inference_construct_valid_combined.csv", low_memory=False)
output_index = pd.read_csv(NOTEBOOK_DIR / "output_index.csv")

print("Repo root      :", REPO_ROOT)
print("ETHICS source  :", SOURCE_BUNDLE)
print("Parent bundle  :", NOTEBOOK_DIR)
print("Source version :", SOURCE_MANIFEST["analysis_version"])
print("Local version  :", LOCAL_MANIFEST["bundle_version"])
"""
        ),
        markdown("## 2. DESIGN PLAN"),
        markdown(
            r"""
- Parent memiliki sembilan kategori: empat agama Explicit, empat rerata
  Implicit, dan Neutral.
- Akurasi committed parent hanya menggabungkan subset construct-valid.
- Status dan alasan abstain memakai seluruh empat subset.
- Statistik inferensial tersedia dalam dua bentuk: per subset dan gabungan
  `commonsense + justice` dengan pairing key `subset + question_id`.
- Detail semua subset tetap tersedia pada child `per_subset/`.
"""
        ),
        markdown("## 3. CHECKLIST — Validation gate parent"),
        code(
            r"""
EXPECTED_ROWS = 154_428
EXPECTED_PARENT_FIGURES = {
    "01_ethics_religion_absolute_accuracy_nine_categories",
    "02_ethics_religion_delta_vs_neutral",
    "03_ethics_religion_delta_implicit_vs_explicit",
    "04_ethics_religion_status_by_model_persona",
    "05_ethics_religion_abstain_reasons_by_model",
    "06_ethics_religion_effect_holm_heatmap",
    "07_ethics_religion_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "07_ethics_religion_exact_mcnemar_and_holm_pvalue_heatmap_part2",
    "08_ethics_religion_combined_effect_holm_heatmap",
    "09_ethics_religion_combined_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "09_ethics_religion_combined_exact_mcnemar_and_holm_pvalue_heatmap_part2",
}

assert SOURCE_MANIFEST["analysis_version"] == "ethics-religion-v3-accuracy-paired-holm-20260822"
assert SOURCE_MANIFEST["validation"]["valid"] is True
assert int(condition_summary["total"].sum()) == EXPECTED_ROWS
assert len(condition_summary) == 6 * 17 * 4
assert {name: int(condition_summary[name].sum()) for name in ("committed", "abstain", "missing")} == LOCAL_MANIFEST["status_counts"]
assert len(accuracy) == 54 and len(status) == 54
assert len(paired_all) == 1_248 and len(paired_scenario) == 672 and len(paired_religion) == 576
assert len(accuracy_inference) == 624
assert "n_pairs" not in accuracy_inference.columns
assert len(accuracy_inference_combined) == 312
assert "n_pairs" not in accuracy_inference_combined.columns
assert LOCAL_MANIFEST["accuracy_scope"] == "construct_valid_commonsense_justice_pooled"
assert set(LOCAL_MANIFEST["child_subset_dirs"]) == {"commonsense", "deontology", "justice", "virtue"}

parent_figures = {Path(name).stem for name in LOCAL_MANIFEST["parent_figure_files"] if name.endswith(".png")}
assert parent_figures == EXPECTED_PARENT_FIGURES

validation = pd.DataFrame(
    [
        ("Respons sumber", int(condition_summary["total"].sum()) == EXPECTED_ROWS, int(condition_summary["total"].sum())),
        ("Kategori persona parent", len(accuracy) == 54 and len(status) == 54, "6 × 9"),
        ("Statistik construct-valid", len(accuracy_inference) == 624, len(accuracy_inference)),
        ("n pasangan tidak ditampilkan", "n_pairs" not in accuracy_inference.columns, True),
        ("Parent figure", parent_figures == EXPECTED_PARENT_FIGURES, len(parent_figures)),
        ("Child subset", len(LOCAL_MANIFEST["child_subset_dirs"]) == 4, 4),
        ("Source read-only", LOCAL_MANIFEST["source_bundle_read_only"] is True, True),
    ],
    columns=["gate", "passed", "observed"],
)
display(validation)
assert validation["passed"].all()
print("SEMUA VALIDATION GATE ETHICS RELIGION LULUS")
"""
        ),
        markdown("## 4. Tabel akurasi, coverage, dan status"),
        code(
            r"""
accuracy_columns = [
    "model_label", "category", "total", "nonmissing", "committed", "commit_rate",
    "abstain", "abstain_rate", "missing", "missing_rate", "n_correct",
    "accuracy_committed", "effective_success_rate_nonmissing",
]
display(accuracy[accuracy_columns].sort_values(["model_label", "category"], kind="stable").style.format({
    "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}", "missing_rate": "{:.2%}",
    "accuracy_committed": "{:.2%}", "effective_success_rate_nonmissing": "{:.2%}",
}))

status_columns = [
    "model_label", "category", "total", "nonmissing", "committed", "commit_rate",
    "abstain", "abstain_rate", "missing", "missing_rate", "commit_rate_nonmissing",
]
display(status[status_columns].sort_values(["model_label", "category"], kind="stable").style.format({
    "commit_rate": "{:.2%}", "commit_rate_nonmissing": "{:.2%}",
    "abstain_rate": "{:.2%}", "missing_rate": "{:.2%}",
}))
"""
        ),
        markdown("## 5. Penyebab `abstain`"),
        code(
            r"""
display(reason_model[[
    "model_label", "reason_label", "reason_count", "abstain", "nonmissing",
    "pct_of_abstain", "pct_of_nonmissing",
]].sort_values(["model_label", "reason_count"], ascending=[True, False], kind="stable")
  .style.format({"pct_of_abstain": "{:.2%}", "pct_of_nonmissing": "{:.2%}"}))
"""
        ),
        markdown("## 6. Visual utama ETHICS Religion"),
        code(
            r"""
def show_figure(file_name: str, width: int = 1500) -> None:
    path = FIGURE_DIR / file_name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"### `{file_name}`"))
    display(Image(filename=str(path), width=width))


PARENT_FIGURE_STEMS = [
    "01_ethics_religion_absolute_accuracy_nine_categories",
    "02_ethics_religion_delta_vs_neutral",
    "03_ethics_religion_delta_implicit_vs_explicit",
    "04_ethics_religion_status_by_model_persona",
    "05_ethics_religion_abstain_reasons_by_model",
    "06_ethics_religion_effect_holm_heatmap",
    "07_ethics_religion_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "07_ethics_religion_exact_mcnemar_and_holm_pvalue_heatmap_part2",
    "08_ethics_religion_combined_effect_holm_heatmap",
    "09_ethics_religion_combined_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "09_ethics_religion_combined_exact_mcnemar_and_holm_pvalue_heatmap_part2",
]
for stem in PARENT_FIGURE_STEMS:
    show_figure(f"{stem}.png", width=1550)
"""
        ),
        markdown("## 7. Exact McNemar, koreksi Holm, dan effect size"),
        code(
            r"""
inference_columns = [
    "subset", "model_label", "comparison_short", "comparison_label",
    "delta_accuracy_pp", "effect_size", "exact_mcnemar_p_value",
    "holm_adjusted_p_value", "significant_holm_0_05",
]
display(Markdown("### Versi per subset: Commonsense dan Justice"))
display(accuracy_inference[inference_columns]
        .sort_values(["subset", "model_label", "comparison_short"], kind="stable")
        .style.format({
            "delta_accuracy_pp": "{:.4f}", "effect_size": "{:.4f}",
            "exact_mcnemar_p_value": "{:.3e}", "holm_adjusted_p_value": "{:.3e}",
        }))

combined_inference_columns = [
    "subset", "model_label", "comparison_short", "comparison_label",
    "risk_difference_pp", "effect_size", "exact_mcnemar_p_value",
    "holm_adjusted_p_value", "significant_holm_0_05",
]
display(Markdown("### Versi gabungan construct-valid: Commonsense + Justice"))
display(accuracy_inference_combined[combined_inference_columns]
        .sort_values(["model_label", "comparison_short"], kind="stable")
        .style.format({
            "risk_difference_pp": "{:.4f}", "effect_size": "{:.4f}",
            "exact_mcnemar_p_value": "{:.3e}", "holm_adjusted_p_value": "{:.3e}",
        }))
"""
        ),
        markdown("## 8. TESTING — Hash artefak dan struktur folder"),
        code(
            r"""
parent_png = [name for name in output_index.loc[output_index["kind"].eq("figure"), "file"] if name.startswith("figures/") and name.endswith(".png")]
parent_svg = [name for name in output_index.loc[output_index["kind"].eq("figure"), "file"] if name.startswith("figures/") and name.endswith(".svg")]
assert len(parent_png) == 11 and len(parent_svg) == 11
for relative_name in parent_png + parent_svg:
    path = NOTEBOOK_DIR / Path(relative_name)
    assert path.is_file() and path.stat().st_size > 0
    expected_hash = output_index.loc[output_index["file"].eq(relative_name), "sha256"].iloc[0]
    assert sha256_file(path) == expected_hash

assert status["total"].eq(status["committed"] + status["abstain"] + status["missing"]).all()
assert accuracy_inference["holm_adjusted_p_value"].between(0, 1).all()
assert accuracy_inference["exact_mcnemar_p_value"].between(0, 1).all()
assert "n_pairs" not in accuracy_inference.columns
assert "n_pairs" not in accuracy_inference_combined.columns

checklist = pd.DataFrame(
    [
        ("[x]", "Parent memakai format visual Gender/Regional/GSM8K"),
        ("[x]", "Akurasi dan delta memakai empat angka desimal"),
        ("[x]", "Status memakai layout 3 × 2 dan label berada dalam segmen"),
        ("[x]", "Exact McNemar, Holm-adjusted p-value, dan Cohen's h tersedia"),
        ("[x]", "Jumlah pasangan observasi tidak ditampilkan"),
        ("[x]", "Versi per subset dan versi gabungan Commonsense + Justice sama-sama tersedia"),
        ("[x]", "Akurasi parent hanya construct-valid pooled"),
        ("[x]", "Empat child subset tersedia"),
        ("[x]", "Tidak ada API call, reparsing, rescoring, atau perubahan result lama"),
    ],
    columns=["status", "item"],
)
display(checklist)
print("TESTING LULUS: parent ETHICS Religion siap dibaca.")
"""
        ),
        markdown(
            r"""
## 9. Batas klaim

Visual parent dapat dipakai untuk membahas sensitivitas akurasi keputusan pada
Commonsense/Justice, coverage, abstain, missing, dan perbedaan antarpersona.
Hasil ini tidak membuktikan perubahan proses moral reasoning. Deontology dan
Virtue tetap dibaca sebagai distribusi stance pada child, bukan akurasi moral
gabungan.
"""
        ),
        code(
            r"""
summary = {
    "source_rows": int(condition_summary["total"].sum()),
    "models": int(condition_summary["model"].nunique()),
    "personas": int(condition_summary["persona"].nunique()),
    "subsets": int(condition_summary["subset"].nunique()),
    "committed": int(condition_summary["committed"].sum()),
    "abstain": int(condition_summary["abstain"].sum()),
    "missing": int(condition_summary["missing"].sum()),
    "parent_figures_png": len(parent_png),
    "parent_figures_svg": len(parent_svg),
    "child_subset_dirs": len(LOCAL_MANIFEST["child_subset_dirs"]),
    "api_calls": 0,
    "reparsing": False,
    "rescoring": False,
    "statistics_recomputed": bool(LOCAL_MANIFEST["statistics_recomputed"]),
}
display(pd.Series(summary, name="value").to_frame())
print("NOTEBOOK ETHICS RELIGION SELESAI DIJALANKAN TANPA ERROR")
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {"display_name": "TA .venv", "language": "python", "name": "ta-venv"},
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    if NOTEBOOK_PATH.exists():
        manifest_path = NOTEBOOK_DIR / "bundle_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"Refusing to overwrite notebook without a recognized bundle manifest: {NOTEBOOK_PATH}"
            ) from exc
        if manifest.get("bundle_version") != BUNDLE_VERSION:
            raise FileExistsError(f"Refusing to overwrite unrelated notebook: {NOTEBOOK_PATH}")
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"[CREATED] {NOTEBOOK_PATH}")
    print(f"[CELLS] {len(notebook.cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

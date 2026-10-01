"""Build the thesis-facing ETHICS Regional-Cultural accuracy notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "regional" / "ethics accuracy analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb"
BUNDLE_VERSION = "ethics-regional-accuracy-combined-standardized-v1"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# Analisis Akurasi Persona Regional-Kultural pada ETHICS — Enam Model Utama

Notebook ini mengikuti format visual ETHICS/GSM8K Gender: bar chart besar,
anotasi angka yang mudah dibaca, delta terhadap `Neutral`, status respons, dan
penyebab `abstain`.

## Cakupan parent

- 154.428 respons dari 6 model × 17 kondisi persona × 4 subset ETHICS;
- status `committed`, `abstain`, dan `missing` dipool dari seluruh subset;
- akurasi parent hanya memakai pooled construct-valid ETHICS
  (`commonsense` dan `justice`);
- `deontology` dan `virtue` tersedia di `per_subset/` sebagai distribusi
  stance, bukan satu akurasi moral gabungan;
- empat region ditampilkan sebagai Explicit, rata-rata tiga kondisi Implicit,
  dan Neutral.

Notebook ini tidak memanggil API, mengubah parser, melakukan parsing ulang,
atau melakukan rescoring.
"""
        ),
        markdown(
            r"""
## 1. SPEC dan provenance

Sumber statistik adalah bundle v3 ETHICS Regional tervalidasi. CSV respons
sumber hanya dibaca untuk membentuk tabel status dan alasan abstain. Tabel
`accuracy_regional_overall.csv` dari sumber lama tidak dipakai sebagai satu
skor moral parent karena tabel tersebut menggabungkan empat subset dengan
konstruksi dan polaritas label yang berbeda.
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
pd.set_option("display.max_rows", 500)
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
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "regional" / "ethics accuracy analysis"
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
reason_categories = pd.read_csv(TABLE_DIR / "ethics_abstain_reasons_by_model_persona_category.csv", low_memory=False)
paired_all = pd.read_csv(TABLE_DIR / "ethics_paired_effects_all.csv", low_memory=False)
paired_scenario = pd.read_csv(TABLE_DIR / "ethics_paired_effects_scenario.csv", low_memory=False)
paired_regional = pd.read_csv(TABLE_DIR / "ethics_paired_effects_regional.csv", low_memory=False)
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
        markdown(
            r"""
## 2. DESIGN PLAN

- Parent visual memakai sembilan kategori: empat region Explicit, empat
  rata-rata region Implicit, dan Neutral.
- `accuracy_committed` pada parent dihitung hanya dari `commonsense` dan
  `justice`, serta dibaca bersama coverage/status.
- Status dan alasan abstain memakai seluruh empat subset.
- Hasil paired v3 disediakan sebagai tabel tanpa menghitung ulang uji statistik.
- Detail setiap subset berada di child folder dan bukan visual utama parent.
"""
        ),
        markdown("## 3. CHECKLIST — Validation gate parent"),
        code(
            r"""
EXPECTED_ROWS = 154_428
EXPECTED_STATUS = {"committed": 112_035, "abstain": 40_570, "missing": 1_823}
EXPECTED_PARENT_FIGURES = {
    "01_ethics_regional_absolute_accuracy_nine_categories",
    "02_ethics_regional_delta_vs_neutral",
    "03_ethics_regional_delta_implicit_vs_explicit",
    "04_ethics_regional_status_by_model_persona",
    "05_ethics_regional_abstain_reasons_by_model",
    "06_ethics_regional_effect_holm_heatmap",
    "07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part2",
    "08_ethics_regional_combined_effect_holm_heatmap",
    "09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part2",
}

assert SOURCE_MANIFEST["analysis_version"] == "ethics-regional-v3-accuracy-paired-holm-20260822"
assert SOURCE_MANIFEST["validation"]["valid"] is True
assert int(condition_summary["total"].sum()) == EXPECTED_ROWS
assert len(condition_summary) == 6 * 17 * 4
assert {name: int(condition_summary[name].sum()) for name in EXPECTED_STATUS} == EXPECTED_STATUS
assert len(accuracy) == 54 and len(status) == 54
assert LOCAL_MANIFEST["accuracy_scope"] == "construct_valid_commonsense_justice_pooled"
assert set(LOCAL_MANIFEST["child_subset_dirs"]) == {"commonsense", "deontology", "justice", "virtue"}
assert len(paired_all) == 1_248 and len(paired_scenario) == 672 and len(paired_regional) == 576
assert len(accuracy_inference) == 624
assert "n_pairs" not in accuracy_inference.columns
assert len(accuracy_inference_combined) == 312
assert "n_pairs" not in accuracy_inference_combined.columns

parent_figures = {Path(name).stem for name in LOCAL_MANIFEST["parent_figure_files"] if name.endswith(".png")}
assert parent_figures == EXPECTED_PARENT_FIGURES

validation = pd.DataFrame(
    [
        ("Respons sumber", int(condition_summary["total"].sum()) == EXPECTED_ROWS, int(condition_summary["total"].sum())),
        ("Committed", int(condition_summary["committed"].sum()) == EXPECTED_STATUS["committed"], int(condition_summary["committed"].sum())),
        ("Abstain", int(condition_summary["abstain"].sum()) == EXPECTED_STATUS["abstain"], int(condition_summary["abstain"].sum())),
        ("Missing", int(condition_summary["missing"].sum()) == EXPECTED_STATUS["missing"], int(condition_summary["missing"].sum())),
        ("Kategori persona parent", len(accuracy) == 54 and len(status) == 54, "6 × 9"),
        ("Parent figure", parent_figures == EXPECTED_PARENT_FIGURES, len(parent_figures)),
        ("Child subset", len(LOCAL_MANIFEST["child_subset_dirs"]) == 4, 4),
        ("Paired scenario", len(paired_scenario) == 672, len(paired_scenario)),
        ("Paired antarkawasan", len(paired_regional) == 576, len(paired_regional)),
        ("Source read-only", LOCAL_MANIFEST["source_bundle_read_only"] is True, True),
    ],
    columns=["gate", "passed", "observed"],
)
display(validation)
assert validation["passed"].all()
print("SEMUA VALIDATION GATE ETHICS REGIONAL LULUS")
"""
        ),
        markdown(
            r"""
## 4. Tabel akurasi, coverage, dan status

Akurasi parent adalah `accuracy_committed` pada dua subset construct-valid.
Coverage dan status ditampilkan berdampingan agar perubahan akurasi tidak
dibaca tanpa mempertimbangkan kesediaan model memberi keputusan.
"""
        ),
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
display(Markdown(
    "Denominator `pct_of_abstain` adalah seluruh abstain Regional; tabel ini "
    "menggabungkan empat subset dan enam model. Rincian kategori persona dan "
    "subset tersedia di tabel parent dan child."
))
display(reason_model[["model_label", "reason_label", "reason_count", "abstain", "nonmissing", "pct_of_abstain", "pct_of_nonmissing"]]
        .sort_values(["model_label", "reason_count"], ascending=[True, False], kind="stable")
        .style.format({"pct_of_abstain": "{:.2%}", "pct_of_nonmissing": "{:.2%}"}))
"""
        ),
        markdown("## 6. Visual utama ETHICS Regional-Kultural"),
        code(
            r"""
def show_figure(file_name: str, width: int = 1500) -> None:
    path = FIGURE_DIR / file_name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"### `{file_name}`"))
    display(Image(filename=str(path), width=width))


PARENT_FIGURE_STEMS = [
    "01_ethics_regional_absolute_accuracy_nine_categories",
    "02_ethics_regional_delta_vs_neutral",
    "03_ethics_regional_delta_implicit_vs_explicit",
    "04_ethics_regional_status_by_model_persona",
    "05_ethics_regional_abstain_reasons_by_model",
    "06_ethics_regional_effect_holm_heatmap",
    "07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part2",
    "08_ethics_regional_combined_effect_holm_heatmap",
    "09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part1",
    "09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part2",
]
for stem in PARENT_FIGURE_STEMS:
    show_figure(f"{stem}.png", width=1550 if stem.startswith(("04_", "05_")) else 1450)
"""
        ),
        markdown(
            r"""
## 7. Efek persona dan batas interpretasi

Versi per subset adalah salinan hasil v3. Versi gabungan dihitung oleh renderer
dengan menggabungkan strata `commonsense` dan `justice`, mempertahankan
`subset + question_id` sebagai pairing key, lalu menjalankan Exact McNemar dan
Holm untuk family gabungan. Notebook hanya membaca kedua hasil tersebut.
"""
        ),
        code(
            r"""
summary_effects = pd.DataFrame({
    "keluarga": ["Persona/skenario", "Antarkawasan"],
    "jumlah_perbandingan": [len(paired_scenario), len(paired_regional)],
    "signifikan_holm": [int(paired_scenario["significant_holm_0_05"].sum()), int(paired_regional["significant_holm_0_05"].sum())],
})
display(summary_effects)
display(Markdown("### Versi per subset: Commonsense dan Justice"))
display(accuracy_inference[[
    "subset", "model_label", "comparison_short", "comparison_label",
    "delta_accuracy_pp", "effect_size", "exact_mcnemar_p_value",
    "holm_adjusted_p_value", "significant_holm_0_05",
]].sort_values(["subset", "model_label", "comparison_short"], kind="stable")
  .style.format({
      "delta_accuracy_pp": "{:.4f}", "effect_size": "{:.4f}",
      "exact_mcnemar_p_value": "{:.3e}", "holm_adjusted_p_value": "{:.3e}",
  }))

display(Markdown("### Versi gabungan construct-valid: Commonsense + Justice"))
display(accuracy_inference_combined[[
    "subset", "model_label", "comparison_short", "comparison_label",
    "risk_difference_pp", "effect_size", "exact_mcnemar_p_value",
    "holm_adjusted_p_value", "significant_holm_0_05",
]].sort_values(["model_label", "comparison_short"], kind="stable")
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
assert set(LOCAL_MANIFEST["child_subset_dirs"]) == {"commonsense", "deontology", "justice", "virtue"}

checklist = pd.DataFrame(
    [
        ("[x]", "Parent memakai format bar chart besar seperti Gender/GSM8K"),
        ("[x]", "Judul parent tidak memakai kata overall"),
        ("[x]", "Anotasi angka akurasi dan delta berada di luar bar serta tidak bold"),
        ("[x]", "Status committed/abstain/missing tampil per model dan kategori persona"),
        ("[x]", "Penyebab abstain tampil sebagai chart khusus ETHICS"),
        ("[x]", "Exact McNemar, Holm-adjusted p-value, dan Cohen's h tampil tanpa n pasangan"),
        ("[x]", "Versi per subset dan versi gabungan Commonsense + Justice sama-sama tersedia"),
        ("[x]", "Akurasi parent hanya construct-valid pooled"),
        ("[x]", "Deontology dan virtue tersedia sebagai stance di child"),
        ("[x]", "PNG/SVG parent hash cocok dengan output_index"),
        ("[x]", "Tidak ada API call, reparsing, rescoring, atau perubahan result lama"),
    ],
    columns=["status", "item"],
)
display(checklist)
print("TESTING LULUS: parent ETHICS Regional-Kultural siap dibaca.")
"""
        ),
        markdown(
            r"""
## 9. Batas klaim

Visual parent dapat dipakai untuk membahas perbedaan persona regional pada
akurasi keputusan di Commonsense/Justice, coverage, status respons, dan profil
alasan abstain. Visual ini tidak membuktikan perubahan moral reasoning atau
bahwa satu region memiliki kemampuan moral yang lebih tinggi. Deontology dan
Virtue dibaca sebagai distribusi stance pada child, bukan akurasi moral yang
digabungkan.
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
}
display(pd.Series(summary, name="value").to_frame())
print("NOTEBOOK ETHICS REGIONAL SELESAI DIJALANKAN TANPA ERROR")
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

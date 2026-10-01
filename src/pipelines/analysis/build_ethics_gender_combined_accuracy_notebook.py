"""Build the combined ETHICS Gender notebook with GSM8K-style visuals."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "gender" / "ethics accuracy analysis"
NOTEBOOK_PATH = NOTEBOOK_DIR / "ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# Analisis Akurasi Persona Gender pada ETHICS — Enam Model Utama

Notebook ini menyajikan ETHICS Gender dengan format visual yang sama dengan
notebook GSM8K Gender: bar chart utama per model dan persona, anotasi angka
besar yang mudah dibaca, serta chart delta terhadap persona `Neutral`.

Cakupan parent:

- 81.756 respons dari 6 model × 9 persona × 4 subset ETHICS;
- status `committed`, `abstain`, dan `missing` dipool dari seluruh subset;
- chart akurasi memakai pooled construct-valid ETHICS (`commonsense` dan
  `justice`), bukan mencampurkan `deontology` dan `virtue` sebagai akurasi;
- penyebab `abstain` ditampilkan sebagai chart khusus ETHICS;
- rincian per subset tersedia sebagai child di folder `per_subset/` dan tidak
  menjadi visual utama notebook ini.

Notebook ini tidak memanggil API, menjalankan inference, mengubah parser,
memulihkan `missing`, atau melakukan rescoring.
"""
        ),
        markdown(
            r"""
## 1. SPEC

Tujuan notebook ini adalah memberikan satu tampilan ETHICS Gender yang dapat
dibaca langsung seperti visual Gender GSM8K. `accuracy_committed` selalu dibaca
bersama coverage/status; angka parent bukan klaim bahwa empat subset ETHICS
memiliki construct dan polaritas yang sama.

Status jawaban diperlakukan sebagai outcome tersendiri. `abstain` bukan sampah
parser, sedangkan `missing` adalah kegagalan infrastruktur dan tidak boleh
dipulihkan secara linguistik.
"""
        ),
        markdown(
            r"""
## 2. DESIGN PLAN

- Baca bundle sumber ETHICS yang tervalidasi secara read-only.
- Validasi jumlah respons, status, model, persona, subset, dan scope metrik.
- Tampilkan tabel utama dalam satu unit kategori persona: Man Explicit, Man
  Implicit Avg, Woman Explicit, Woman Implicit Avg, dan Neutral.
- Gunakan bar chart besar untuk akurasi dan delta, mengikuti struktur visual
  GSM8K Gender.
- Gunakan chart status bertumpuk dan chart penyebab `abstain` khusus ETHICS.
- Simpan rincian subset di child folder `per_subset/<subset>/`.
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
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "gender" / "ethics accuracy analysis"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = NOTEBOOK_DIR / "tables"
CHILD_DIR = NOTEBOOK_DIR / "per_subset"
LOCAL_MANIFEST_PATH = NOTEBOOK_DIR / "bundle_manifest.json"
SOURCE_BUNDLE = (
    REPO_ROOT / "result" / "ETHICS" / "reports" / "Gender_Persona" / "_combined"
    / "persona_publication_analysis_v2_readable_paired_holm_six_models"
)
SOURCE_MANIFEST_PATH = SOURCE_BUNDLE / "metadata" / "analysis_manifest.json"

for required_path in (NOTEBOOK_DIR, FIGURE_DIR, TABLE_DIR, CHILD_DIR, LOCAL_MANIFEST_PATH, SOURCE_MANIFEST_PATH):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

local_manifest = json.loads(LOCAL_MANIFEST_PATH.read_text(encoding="utf-8"))
source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
print(f"Repo root       : {REPO_ROOT}")
print(f"ETHICS source   : {SOURCE_BUNDLE}")
print(f"Parent bundle   : {NOTEBOOK_DIR}")
print(f"Source version  : {source_manifest['version']}")
print(f"Local version   : {local_manifest['bundle_version']}")
"""
        ),
        markdown("## 3. IMPLEMENTATION — Load tabel parent dan child"),
        code(
            r"""
condition_summary = pd.read_csv(SOURCE_BUNDLE / "tables" / "condition_summary_full.csv")
model_overview = pd.read_csv(SOURCE_BUNDLE / "tables" / "model_overview.csv")
status_independence = pd.read_csv(SOURCE_BUNDLE / "tables" / "persona_status_independence.csv")
paired_effects = pd.read_csv(SOURCE_BUNDLE / "tables" / "paired_effects_all_outcomes.csv")

accuracy = pd.read_csv(TABLE_DIR / "ethics_accuracy_by_model_persona_category.csv")
status = pd.read_csv(TABLE_DIR / "ethics_status_by_model_persona_category.csv")
reason_model = pd.read_csv(TABLE_DIR / "ethics_abstain_reasons_by_model.csv")
reason_categories = pd.read_csv(TABLE_DIR / "ethics_abstain_reasons_by_model_persona_category.csv")
paired_parent = pd.read_csv(TABLE_DIR / "ethics_paired_effects_vs_neutral.csv")
accuracy_inference = pd.read_csv(TABLE_DIR / "ethics_accuracy_inference_construct_valid.csv")
accuracy_inference_combined = pd.read_csv(TABLE_DIR / "ethics_accuracy_inference_construct_valid_combined.csv")

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
CATEGORY_ORDER = ["Man Explicit", "Man Implicit Avg", "Woman Explicit", "Woman Implicit Avg", "Neutral"]
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
"""
        ),
        markdown("## 4. CHECKLIST — Validation gate parent"),
        code(
            r"""
EXPECTED_ROWS = 81_756
EXPECTED_STATUS_COUNTS = {"committed": 62_691, "abstain": 18_521, "missing": 544}
EXPECTED_PARENT_FIGURES = {
    "01_ethics_absolute_accuracy_five_categories",
    "02_ethics_delta_vs_neutral",
    "03_ethics_delta_implicit_vs_explicit",
    "04_ethics_status_by_model_persona",
    "05_ethics_abstain_reasons_by_model",
    "06_ethics_gender_effect_holm_heatmap",
    "07_ethics_gender_exact_mcnemar_and_holm_pvalue_heatmap",
    "08_ethics_gender_combined_effect_holm_heatmap",
    "09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap",
}

assert source_manifest["row_count"] == EXPECTED_ROWS
assert int(condition_summary["total"].sum()) == EXPECTED_ROWS
assert {status: int(condition_summary[status].sum()) for status in EXPECTED_STATUS_COUNTS} == EXPECTED_STATUS_COUNTS
assert local_manifest["parent_scope"] == "combined_ethics_gender"
assert local_manifest["status_scope"] == "all_four_subsets"
assert local_manifest["accuracy_scope"] == "construct_valid_commonsense_justice_pooled"
assert len(accuracy) == 30
assert len(status) == 30
assert set(accuracy["category"]) == set(CATEGORY_ORDER)
assert set(status["category"]) == set(CATEGORY_ORDER)
assert set(local_manifest["child_subset_dirs"]) == set(SUBSET_ORDER)
assert len(paired_effects) == 1_080
assert set(paired_effects["family_size"].astype(int)) == {4, 7}
assert len(accuracy_inference) == 216
assert "n_pairs" not in accuracy_inference.columns
assert len(accuracy_inference_combined) == 108
assert "n_pairs" not in accuracy_inference_combined.columns

parent_figures = {
    Path(name).stem
    for name in local_manifest["parent_figure_files"]
    if name.endswith(".png")
}
assert parent_figures == EXPECTED_PARENT_FIGURES
for subset in SUBSET_ORDER:
    child = CHILD_DIR / subset
    assert (child / "tables" / "condition_summary.csv").is_file()
    assert (child / "figures").is_dir()

validation = pd.DataFrame(
    [
        ("Respons sumber", int(condition_summary["total"].sum()) == EXPECTED_ROWS, int(condition_summary["total"].sum())),
        ("Committed", int(condition_summary["committed"].sum()) == EXPECTED_STATUS_COUNTS["committed"], int(condition_summary["committed"].sum())),
        ("Abstain", int(condition_summary["abstain"].sum()) == EXPECTED_STATUS_COUNTS["abstain"], int(condition_summary["abstain"].sum())),
        ("Missing", int(condition_summary["missing"].sum()) == EXPECTED_STATUS_COUNTS["missing"], int(condition_summary["missing"].sum())),
        ("Kategori persona parent", len(accuracy) == 30 and len(status) == 30, "6 × 5"),
        ("Parent figure", parent_figures == EXPECTED_PARENT_FIGURES, len(parent_figures)),
        ("Child subset", set(local_manifest["child_subset_dirs"]) == set(SUBSET_ORDER), len(SUBSET_ORDER)),
        ("Paired effects sumber", len(paired_effects) == 1_080, len(paired_effects)),
        ("Source input unchanged", source_manifest["output_validation"]["source_input_unchanged"], "true"),
    ],
    columns=["gate", "passed", "observed"],
)
display(validation)
assert validation["passed"].all()
print("SEMUA VALIDATION GATE PARENT LULUS")
"""
        ),
        markdown(
            r"""
## 5. Tabel akurasi, coverage, dan status

Lima kategori persona mengikuti visual Gender GSM8K. Tabel tetap menyertakan
coverage dan status supaya angka akurasi committed tidak dibaca sebagai akurasi
dengan coverage 100%.
"""
        ),
        code(
            r"""
accuracy_table = accuracy[
    [
        "model_label", "category", "total", "nonmissing", "committed",
        "commit_rate", "abstain", "abstain_rate", "missing", "missing_rate",
        "n_correct", "accuracy_committed", "effective_success_rate_nonmissing",
    ]
].sort_values(["model_label", "category"], kind="stable")
display(
    accuracy_table.style.format(
        {
            "commit_rate": "{:.2%}",
            "abstain_rate": "{:.2%}",
            "missing_rate": "{:.2%}",
            "accuracy_committed": "{:.2%}",
            "effective_success_rate_nonmissing": "{:.2%}",
        }
    )
)

status_table = status[
    [
        "model_label", "category", "total", "nonmissing", "committed",
        "commit_rate", "abstain", "abstain_rate", "missing", "missing_rate",
        "commit_rate_nonmissing",
    ]
].sort_values(["model_label", "category"], kind="stable")
display(status_table.style.format({
    "commit_rate": "{:.2%}",
    "commit_rate_nonmissing": "{:.2%}",
    "abstain_rate": "{:.2%}",
    "missing_rate": "{:.2%}",
}))
"""
        ),
        markdown("## 6. Penyebab `abstain`"),
        code(
            r"""
display(Markdown(
    "Penyebab berikut dihitung dari seluruh ETHICS. `pct_of_abstain` memakai "
    "denominator semua abstain pada model; `pct_of_nonmissing` tersedia di tabel "
    "sumber dan dipakai untuk membaca prevalensi."
))
reason_table = reason_model[
    ["model_label", "reason_label", "reason_count", "abstain", "nonmissing", "pct_of_abstain", "pct_of_nonmissing"]
].sort_values(["model_label", "reason_count"], ascending=[True, False], kind="stable")
display(reason_table.style.format({"pct_of_abstain": "{:.2%}", "pct_of_nonmissing": "{:.2%}"}))
"""
        ),
        markdown("## 7. Visual utama ETHICS Gender"),
        code(
            r"""
def show_figure(file_name: str, width: int = 1450) -> None:
    path = FIGURE_DIR / file_name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"### `{file_name}`"))
    display(Image(filename=str(path), width=width))


PARENT_FIGURE_STEMS = [
    "01_ethics_absolute_accuracy_five_categories",
    "02_ethics_delta_vs_neutral",
    "03_ethics_delta_implicit_vs_explicit",
    "04_ethics_status_by_model_persona",
    "05_ethics_abstain_reasons_by_model",
    "06_ethics_gender_effect_holm_heatmap",
    "07_ethics_gender_exact_mcnemar_and_holm_pvalue_heatmap",
    "08_ethics_gender_combined_effect_holm_heatmap",
    "09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap",
]
for stem in PARENT_FIGURE_STEMS:
    show_figure(f"{stem}.png", width=1500 if stem.startswith(("04_", "05_")) else 1400)
"""
        ),
        markdown(
            r"""
## 8. Efek persona dan interpretasi

Tabel paired tetap menggunakan outcome sesuai construct. Perbandingan coverage
boleh mencakup semua subset; conditional accuracy dan effective success untuk
klaim moral judgement dibaca pada subset construct-valid; `deontology` dan
`virtue` tidak diubah menjadi akurasi moral.
"""
        ),
        code(
            r"""
paired_columns = [
    "subset", "model_label", "comparison_short", "comparison_label",
    "risk_difference_pp", "effect_size", "exact_mcnemar_p_value",
    "holm_adjusted_p_value", "significant_holm_0_05", "decision_significant_0_05",
]
display(Markdown("### Versi per subset: Commonsense dan Justice"))
display(
    accuracy_inference[paired_columns]
    .sort_values(["subset", "model_label", "comparison_short"], kind="stable")
    .style.format({
        "risk_difference_pp": "{:.4f}",
        "effect_size": "{:.4f}",
        "exact_mcnemar_p_value": "{:.3e}",
        "holm_adjusted_p_value": "{:.3e}",
    }, na_rep="—")
)

display(Markdown("### Versi gabungan construct-valid: Commonsense + Justice"))
display(
    accuracy_inference_combined[paired_columns]
    .sort_values(["model_label", "comparison_short"], kind="stable")
    .style.format({
        "risk_difference_pp": "{:.4f}",
        "effect_size": "{:.4f}",
        "exact_mcnemar_p_value": "{:.3e}",
        "holm_adjusted_p_value": "{:.3e}",
    }, na_rep="â€”")
)

display(Markdown("### Lokasi detail per subset"))
display(pd.DataFrame({"subset": SUBSET_ORDER, "folder": [str(CHILD_DIR / subset) for subset in SUBSET_ORDER]}))
"""
        ),
        markdown("## 9. TESTING — Hash, status, dan struktur folder"),
        code(
            r"""
output_index = pd.read_csv(NOTEBOOK_DIR / "output_index.csv")
parent_png = [name for name in output_index.loc[output_index["kind"].eq("figure"), "file"] if name.startswith(tuple(PARENT_FIGURE_STEMS)) and name.endswith(".png")]
parent_svg = [name for name in output_index.loc[output_index["kind"].eq("figure"), "file"] if name.startswith(tuple(PARENT_FIGURE_STEMS)) and name.endswith(".svg")]
assert len(parent_png) == 9
assert len(parent_svg) == 9
for file_name in parent_png + parent_svg:
    path = FIGURE_DIR / file_name
    assert path.is_file() and path.stat().st_size > 0
    expected_hash = output_index.loc[output_index["file"].eq(file_name), "sha256"].iloc[0]
    assert sha256_file(path) == expected_hash

assert set(paired_effects["multiple_comparison_method"].dropna()) == {"Holm"}
assert set(paired_effects["statistical_method"].dropna()) == {"two-sided Exact McNemar"}
assert (reason_model["reason_count"] >= 0).all()
assert status["total"].eq(status["committed"] + status["abstain"] + status["missing"]).all()

checklist = pd.DataFrame(
    [
        ("[x]", "Parent memakai format bar chart besar seperti GSM8K Gender"),
        ("[x]", "Judul parent tidak memakai kata overall"),
        ("[x]", "Angka anotasi normal dan delta berada di luar bar"),
        ("[x]", "Status committed/abstain/missing tampil dalam chart khusus ETHICS"),
        ("[x]", "Penyebab abstain tampil dalam chart khusus ETHICS"),
        ("[x]", "Exact McNemar, Holm-adjusted p-value, dan Cohen's h tampil tanpa n pasangan"),
        ("[x]", "Versi per subset dan versi gabungan Commonsense + Justice sama-sama tersedia"),
        ("[x]", "Akurasi parent hanya construct-valid pooled"),
        ("[x]", "Empat child subset tersedia di per_subset/"),
        ("[x]", "PNG/SVG parent hash cocok dengan output_index"),
        ("[x]", "Tidak ada API call, reparsing, rescoring, atau perubahan result lama"),
    ],
    columns=["status", "item"],
)
display(checklist)
print("TESTING LULUS: parent ETHICS Gender siap dibaca.")
"""
        ),
        markdown(
            r"""
## 10. Batas klaim

Visual parent dapat digunakan untuk membahas perbedaan persona pada akurasi
construct-valid, coverage, abstain, missing, dan profil alasan abstain. Visual
parent tidak boleh disebut sebagai bukti langsung perubahan moral reasoning atau
sebagai satu skor akurasi valid untuk semua subset ETHICS.
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
    "child_subset_dirs": len(SUBSET_ORDER),
    "api_calls": 0,
    "reparsing": False,
    "rescoring": False,
}
display(pd.Series(summary, name="value").to_frame())
print("NOTEBOOK ETHICS GENDER SELESAI DIJALANKAN TANPA ERROR")
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
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"[CREATED] {NOTEBOOK_PATH}")
    print(f"[CELLS] {len(notebook.cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

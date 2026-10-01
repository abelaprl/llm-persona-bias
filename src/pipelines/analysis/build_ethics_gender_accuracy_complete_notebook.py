"""Build the executed-ready ETHICS Gender accuracy analysis notebook.

The notebook is a read-only presentation and validation layer over the
validated ETHICS Gender publication bundle plus standardized local figures.
It does not call an API, reparse responses, rescore answers, or recompute the
inferential statistics.
"""

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

Notebook ini adalah pasangan ETHICS untuk notebook analisis akurasi Gender
GSM8K. Sumber statistiknya adalah bundle publikasi ETHICS Gender yang sudah
divalidasi; folder lokal ini hanya menyediakan presentasi, tabel turunan, dan
visual dengan aturan tipografi yang sama.

Cakupan:

- 1.514 soal × 9 kondisi persona × 6 model = 81.756 respons;
- status jawaban `committed`, `abstain`, dan `missing` dilaporkan bersama;
- `accuracy_committed` hanya untuk subset construct-valid (`commonsense` dan
  `justice`), selalu berdampingan dengan `commit_rate`;
- `deontology` dan `virtue` ditampilkan sebagai distribusi `stance wrong`,
  bukan sebagai akurasi penilaian moral;
- alasan `abstain` ditampilkan dengan denominator `% dari abstain` dan `% dari
  non-missing`;
- hasil berpasangan, ukuran efek, dan keputusan Holm dibaca dari tabel sumber.

Notebook ini **tidak** memanggil API, menjalankan inference, mengubah parser,
memulihkan `missing`, memaksakan `abstain` menjadi jawaban biner, atau menimpa
hasil lama.
"""
        ),
        markdown(
            r"""
## 1. SPEC

**Tujuan.** Menyediakan satu notebook ETHICS Gender yang dapat dibuka dan
dijalankan dari awal sampai akhir untuk meninjau status jawaban, coverage,
akurasi yang construct-valid, penyebab `abstain`, dan efek persona.

**Sumber.**

`result/ETHICS/reports/Gender_Persona/_combined/` 
`persona_publication_analysis_v2_readable_paired_holm_six_models`

**Kontrak interpretasi.** Tidak ada satu angka “akurasi ETHICS” yang melebur
empat subset. `commonsense` dan `justice` boleh diberi label akurasi karena
construct-nya valid pada pipeline ini. `deontology` dan `virtue` tetap penting
untuk analisis perbedaan jawaban, tetapi dilaporkan sebagai stance, bukan
sebagai bukti kemampuan penilaian moral.

**Kontrak status.** `missing` adalah kegagalan infrastruktur dan bukan
`abstain`. `abstain` adalah hasil parser yang tidak memuat verdict biner yang
dapat dipertanggungjawabkan. Alasan `abstain` tidak boleh dibaca tanpa
menyebutkan denominator-nya.
"""
        ),
        markdown(
            r"""
## 2. DESIGN PLAN

- Temukan root repositori dan baca `bundle_manifest.json` lokal serta manifest
  analisis ETHICS sumber.
- Validasi jumlah baris, model, persona, subset, status, duplikasi kunci, dan
  jumlah status terhadap bundle sumber.
- Tampilkan tabel per `(model, persona, subset)` dengan `total`, `committed`,
  `commit_rate`, `abstain`, `missing`, dan `accuracy_committed`.
- Tampilkan penyebab `abstain` pada dua denominator: komposisi di antara
  abstain dan prevalensi di antara respons non-missing.
- Tampilkan visual lokal PNG; semua pasangan PNG/SVG divalidasi melalui hash
  pada `output_index.csv`.
- Statistik inferensial tidak dihitung ulang pada notebook; tabel paired dan
  Holm dibaca sebagai keluaran analisis sumber yang sudah tervalidasi.
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
pd.set_option("display.max_rows", 500)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


def find_repo_root(start: Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "result" / "ETHICS").is_dir() and (path / "notebooks").is_dir():
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
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "gender" / "ethics accuracy analysis"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
TABLE_DIR = NOTEBOOK_DIR / "tables"
LOCAL_MANIFEST_PATH = NOTEBOOK_DIR / "bundle_manifest.json"
SOURCE_BUNDLE = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "_combined"
    / "persona_publication_analysis_v2_readable_paired_holm_six_models"
)
SOURCE_MANIFEST_PATH = SOURCE_BUNDLE / "metadata" / "analysis_manifest.json"

for required_path in (
    NOTEBOOK_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    LOCAL_MANIFEST_PATH,
    SOURCE_MANIFEST_PATH,
):
    if not required_path.exists():
        raise FileNotFoundError(required_path)

local_manifest = json.loads(LOCAL_MANIFEST_PATH.read_text(encoding="utf-8"))
source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

print(f"Repo root       : {REPO_ROOT}")
print(f"ETHICS source   : {SOURCE_BUNDLE}")
print(f"Notebook bundle : {NOTEBOOK_DIR}")
print(f"Source version  : {source_manifest['version']}")
print(f"Local version   : {local_manifest['bundle_version']}")
"""
        ),
        markdown("## 3. IMPLEMENTATION — Load table dan provenance"),
        code(
            r"""
condition_summary = pd.read_csv(SOURCE_BUNDLE / "tables" / "condition_summary_full.csv")
model_overview = pd.read_csv(SOURCE_BUNDLE / "tables" / "model_overview.csv")
status_independence = pd.read_csv(
    SOURCE_BUNDLE / "tables" / "persona_status_independence.csv"
)
paired_effects = pd.read_csv(
    SOURCE_BUNDLE / "tables" / "paired_effects_all_outcomes.csv"
)
omnibus = pd.read_csv(SOURCE_BUNDLE / "tables" / "cochran_q_omnibus.csv")

status_by_model_persona = pd.read_csv(TABLE_DIR / "status_by_model_persona.csv")
abstain_reason_by_model_persona = pd.read_csv(
    TABLE_DIR / "abstain_reason_by_model_persona.csv"
)
abstain_reason_by_persona = pd.read_csv(
    TABLE_DIR / "abstain_reason_by_persona.csv"
)
accuracy_valid = pd.read_csv(TABLE_DIR / "accuracy_construct_valid_subsets.csv")
stance_invalid = pd.read_csv(TABLE_DIR / "stance_construct_invalid_subsets.csv")
paired_primary = pd.read_csv(TABLE_DIR / "paired_primary_effects_vs_neutral.csv")

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
PERSONA_ORDER = [
    "man_explicit_1", "man_implicit_1", "man_implicit_2", "man_implicit_3",
    "woman_explicit_1", "woman_implicit_1", "woman_implicit_2", "woman_implicit_3",
    "neutral",
]
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
VALID_SUBSETS = ["commonsense", "justice"]
INVALID_STANCE_SUBSETS = ["deontology", "virtue"]
"""
        ),
        markdown("## 4. CHECKLIST — Validation gate data dan status"),
        code(
            r"""
EXPECTED_ROWS = 81_756
EXPECTED_QUESTIONS = 1_514
EXPECTED_MODELS = 6
EXPECTED_PERSONAS = 9
EXPECTED_SUBSETS = 4
EXPECTED_STATUS_COUNTS = {"committed": 62_691, "abstain": 18_521, "missing": 544}

assert source_manifest["version"] == (
    "ethics-gender-publication-paired-holm-v2-readable-20260821"
)
assert source_manifest["row_count"] == EXPECTED_ROWS
assert len(condition_summary) == EXPECTED_MODELS * EXPECTED_PERSONAS * EXPECTED_SUBSETS
assert condition_summary["model"].nunique() == EXPECTED_MODELS
assert condition_summary["persona"].nunique() == EXPECTED_PERSONAS
assert condition_summary["subset"].nunique() == EXPECTED_SUBSETS
assert not condition_summary.duplicated(["model", "persona", "subset"]).any()
assert set(condition_summary["subset"]) == set(SUBSET_ORDER)
assert set(condition_summary.loc[condition_summary["construct_valid"], "subset"]) == set(VALID_SUBSETS)
assert set(condition_summary.loc[~condition_summary["construct_valid"], "subset"]) == set(INVALID_STANCE_SUBSETS)

observed_status_counts = {
    status: int(condition_summary[status].sum())
    for status in ("committed", "abstain", "missing")
}
assert observed_status_counts == EXPECTED_STATUS_COUNTS
assert int(condition_summary["total"].sum()) == EXPECTED_ROWS
assert (
    condition_summary.groupby(["model", "persona"])["total"].sum()
    .eq(EXPECTED_QUESTIONS)
    .all()
)
assert len(model_overview) == EXPECTED_MODELS
assert len(status_independence) == EXPECTED_MODELS * 2
assert len(paired_effects) == 1_080
assert len(paired_primary) > 0
assert local_manifest["source_analysis_version"] == source_manifest["version"]
assert local_manifest["source_row_count"] == EXPECTED_ROWS
assert local_manifest["status_counts"] == EXPECTED_STATUS_COUNTS

validation = pd.DataFrame(
    [
        ("Jumlah respons sumber", int(condition_summary["total"].sum()) == EXPECTED_ROWS, int(condition_summary["total"].sum())),
        ("Model", condition_summary["model"].nunique() == EXPECTED_MODELS, condition_summary["model"].nunique()),
        ("Persona", condition_summary["persona"].nunique() == EXPECTED_PERSONAS, condition_summary["persona"].nunique()),
        ("Subset", condition_summary["subset"].nunique() == EXPECTED_SUBSETS, condition_summary["subset"].nunique()),
        ("Committed", observed_status_counts["committed"] == EXPECTED_STATUS_COUNTS["committed"], observed_status_counts["committed"]),
        ("Abstain", observed_status_counts["abstain"] == EXPECTED_STATUS_COUNTS["abstain"], observed_status_counts["abstain"]),
        ("Missing", observed_status_counts["missing"] == EXPECTED_STATUS_COUNTS["missing"], observed_status_counts["missing"]),
        ("Kunci respons unik", not condition_summary.duplicated(["model", "persona", "subset"]).any(), "true"),
        ("Paired effects", len(paired_effects) == 1_080, len(paired_effects)),
        ("Source input unchanged", source_manifest["output_validation"]["source_input_unchanged"], "true"),
    ],
    columns=["gate", "passed", "observed"],
)
display(validation)
assert validation["passed"].all()
print("SEMUA VALIDATION GATE SUMBER LULUS")
"""
        ),
        markdown(
            r"""
## 5. Tabel status jawaban dan akurasi

Tabel ini sengaja tidak menyembunyikan `abstain` atau `missing`. `accuracy_committed`
adalah akurasi bersyarat pada respons yang committed, sehingga tidak boleh dibaca
tanpa `commit_rate`. `effective_success_rate_nonmissing` memasukkan abstain sebagai
tidak berhasil tetapi mengecualikan missing; metrik ini dipakai sebagai sensitivitas,
bukan sebagai pengganti akurasi committed.
"""
        ),
        code(
            r"""
required_columns = [
    "model_label", "persona", "persona_label", "subset", "subset_label",
    "construct_valid", "total", "nonmissing", "committed", "commit_rate",
    "abstain", "abstain_rate", "missing", "missing_rate", "n_correct",
    "accuracy_committed", "effective_success_rate_nonmissing",
    "n_conflict", "accuracy_committed_excl_conflict",
]
condition_table = condition_summary[required_columns].copy()
condition_table = condition_table.sort_values(
    ["model_label", "persona", "subset"], kind="stable"
).reset_index(drop=True)
display(
    condition_table.style.format(
        {
            "commit_rate": "{:.2%}",
            "abstain_rate": "{:.2%}",
            "missing_rate": "{:.2%}",
            "accuracy_committed": "{:.2%}",
            "effective_success_rate_nonmissing": "{:.2%}",
            "accuracy_committed_excl_conflict": "{:.2%}",
        },
        na_rep="—",
    )
)
"""
        ),
        code(
            r"""
status_table = status_by_model_persona[
    [
        "model_label", "persona", "persona_label", "total", "nonmissing",
        "committed", "commit_rate", "abstain", "abstain_rate", "missing",
        "missing_rate", "commit_rate_nonmissing",
    ]
].sort_values(["model_label", "persona"], kind="stable")
display(
    status_table.style.format(
        {
            "commit_rate": "{:.2%}",
            "commit_rate_nonmissing": "{:.2%}",
            "abstain_rate": "{:.2%}",
            "missing_rate": "{:.2%}",
        }
    )
)
"""
        ),
        markdown(
            r"""
## 6. Penyebab `abstain`

`% dari abstain` menjawab: “dari semua respons yang abstain, apa komposisi
penyebabnya?”

`% dari non-missing` menjawab: “seberapa sering penyebab itu muncul di seluruh
respons yang tersedia?” Kedua angka diperlukan karena model dengan tingkat
abstain tinggi dan rendah tidak boleh dibandingkan hanya dari komposisi internal
abstain.
"""
        ),
        code(
            r"""
reason_columns = [
    "persona", "reason_label", "abstain_reason", "reason_count",
    "total_abstain", "total_nonmissing", "pct_of_abstain", "pct_of_nonmissing",
]
reason_persona_table = abstain_reason_by_persona[reason_columns].sort_values(
    ["persona", "reason_count"], ascending=[True, False], kind="stable"
)
display(
    reason_persona_table.style.format(
        {"pct_of_abstain": "{:.2%}", "pct_of_nonmissing": "{:.2%}"},
        na_rep="—",
    )
)

top_reason_by_model = (
    abstain_reason_by_model_persona
    .sort_values(["model", "persona", "reason_count"], ascending=[True, True, False])
    .groupby(["model", "persona"], as_index=False, sort=False)
    .head(3)
)
display(Markdown("### Tiga penyebab dengan hitungan terbesar per model × persona"))
display(
    top_reason_by_model[
        ["model", "persona", "reason_label", "reason_count", "pct_of_abstain", "pct_of_nonmissing"]
    ].style.format({"pct_of_abstain": "{:.2%}", "pct_of_nonmissing": "{:.2%}"})
)
"""
        ),
        markdown("## 7. Validitas construct dan hasil inferensial"),
        code(
            r"""
display(Markdown("### Subset construct-valid: accuracy committed dan effective success"))
display(
    accuracy_valid[
        [
            "model_label", "persona", "subset", "total", "committed", "abstain", "missing",
            "commit_rate", "accuracy_committed", "effective_success_rate_nonmissing",
        ]
    ].style.format(
        {
            "commit_rate": "{:.2%}",
            "accuracy_committed": "{:.2%}",
            "effective_success_rate_nonmissing": "{:.2%}",
        }
    )
)

display(Markdown("### Subset construct-invalid: distribusi stance wrong"))
display(
    stance_invalid[
        [
            "model_label", "persona", "subset", "total", "committed", "abstain", "missing",
            "commit_rate", "pct_answer_wrong", "pct_answer_acceptable",
        ]
    ].style.format(
        {
            "commit_rate": "{:.2%}",
            "pct_answer_wrong": "{:.2%}",
            "pct_answer_acceptable": "{:.2%}",
        }
    )
)
"""
        ),
        code(
            r"""
display(Markdown("### Asosiasi persona × status jawaban"))
display(
    status_independence[
        ["scope", "model_label", "n", "chi2", "dof", "p_value", "cramer_v", "significant_0_05"]
    ].style.format({"p_value": "{:.3e}", "cramer_v": "{:.4f}"})
)

paired_columns = [
    "outcome", "subset", "model_label", "comparison_label", "n_pairs",
    "pair_coverage_pct", "excluded_missing_either", "excluded_not_both_committed",
    "risk_difference_pp", "risk_difference_ci95_low_pp", "risk_difference_ci95_high_pp",
    "cohen_h", "holm_adjusted_p_value", "family_size",
    "omnibus_cochran_q_p_value", "significant_holm_0_05", "decision_significant_0_05",
]
display(Markdown("### Paired effects vs Neutral (tabel sumber; Holm per family)"))
display(
    paired_primary[paired_columns]
    .sort_values(["outcome", "subset", "model_label", "comparison_label"], kind="stable")
    .style.format(
        {
            "pair_coverage_pct": "{:.2f}%",
            "risk_difference_pp": "{:.3f}",
            "risk_difference_ci95_low_pp": "{:.3f}",
            "risk_difference_ci95_high_pp": "{:.3f}",
            "cohen_h": "{:.4f}",
            "holm_adjusted_p_value": "{:.3e}",
            "omnibus_cochran_q_p_value": "{:.3e}",
        },
        na_rep="—",
    )
)
"""
        ),
        markdown(
            r"""
## 8. Visual ETHICS Gender

Aturan visual mengikuti notebook Gender GSM8K: latar `#FCFBF8`, judul dan judul
panel **bold**, angka anotasi normal, model/persona tetap pada urutan yang sama,
dan tidak ada catatan kaki signifikansi yang ditempelkan di bawah chart. Tanda
`*` pada heatmap efek adalah nilai yang `decision_significant_0_05`-nya true;
detail family, `n_pairs`, dan p-value tetap dibaca dari tabel.
"""
        ),
        code(
            r"""
def show_figure(file_name: str, width: int = 1350) -> None:
    path = FIGURE_DIR / file_name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"### `{file_name}`"))
    display(Image(filename=str(path), width=width))


FIGURE_STEMS = [
    "01_ethics_status_composition_by_model",
    "02_ethics_status_by_model_persona_heatmap",
    "03_ethics_commit_rate_by_persona_subset_heatmap",
    "04_ethics_abstain_reason_share_by_model_persona_heatmap",
    "05_ethics_abstain_reason_prevalence_by_model_persona_heatmap",
    "06_ethics_accuracy_committed_valid_subsets_heatmap",
    "07_ethics_effective_success_valid_subsets_heatmap",
    "08_ethics_stance_wrong_construct_invalid_heatmap",
    "09_ethics_status_persona_cramers_v",
    "10_ethics_coverage_effect_holm_heatmap",
    "11_ethics_conditional_accuracy_effect_holm_heatmap",
    "12_ethics_effective_success_effect_holm_heatmap",
    "13_ethics_stance_wrong_effect_holm_heatmap",
]

for stem in FIGURE_STEMS:
    show_figure(f"{stem}.png", width=1450 if stem.startswith(("04_", "05_")) else 1350)
"""
        ),
        markdown("## 9. TESTING — Hash visual, tabel, dan kontrak statistik"),
        code(
            r"""
output_index = pd.read_csv(NOTEBOOK_DIR / "output_index.csv")
figure_files = output_index.loc[output_index["kind"].eq("figure"), "file"].tolist()
table_files = output_index.loc[output_index["kind"].eq("table"), "file"].tolist()

assert len(figure_files) == local_manifest["figure_count"]
assert len(figure_files) == 26
assert len(table_files) == 6
assert len({Path(name).stem for name in figure_files}) == 13
for file_name in figure_files + table_files:
    path = NOTEBOOK_DIR / ("figures" if file_name.endswith((".png", ".svg")) else "tables") / file_name
    assert path.is_file() and path.stat().st_size > 0, path
    expected_hash = output_index.loc[output_index["file"].eq(file_name), "sha256"].iloc[0]
    assert sha256_file(path) == expected_hash

assert set(paired_effects["multiple_comparison_method"].dropna()) == {"Holm"}
assert set(paired_effects["statistical_method"].dropna()) == {"two-sided Exact McNemar"}
assert set(paired_effects["family_size"].astype(int)) == {4, 7}
assert not status_by_model_persona.duplicated(["model", "persona"]).any()
assert (status_by_model_persona["total"] == EXPECTED_QUESTIONS).all()
assert (abstain_reason_by_model_persona["reason_count"] >= 0).all()

checklist = pd.DataFrame(
    [
        ("[x]", "81.756 respons sumber tervalidasi"),
        ("[x]", "Status committed, abstain, missing dilaporkan bersama"),
        ("[x]", "Accuracy hanya pada commonsense dan justice"),
        ("[x]", "Deontology dan virtue dilaporkan sebagai stance wrong"),
        ("[x]", "Penyebab abstain memakai dua denominator"),
        ("[x]", "Persona × status diuji dengan chi-square dan Cramer's V"),
        ("[x]", "Paired effects memakai tabel sumber dan koreksi Holm"),
        ("[x]", "13 visual × PNG/SVG tersedia dan hash cocok"),
        ("[x]", "Tidak ada API call, reparsing, rescoring, atau overwrite result lama"),
    ],
    columns=["status", "item"],
)
display(checklist)
print("TESTING LULUS: tabel, visual, hash, dan kontrak interpretasi konsisten.")
"""
        ),
        markdown(
            r"""
## 10. INTEGRATION DAN BATAS KLAIM

Hasil ETHICS ini boleh dipakai untuk menyatakan bahwa kondisi persona berkaitan
dengan perubahan pada:

- kecenderungan model memberikan verdict (`commit_rate`);
- kecenderungan abstain dan profil penyebab abstain;
- akurasi bersyarat pada respons committed di `commonsense` dan `justice`;
- effective success yang secara eksplisit memperhitungkan abstain;
- distribusi `stance wrong` pada `deontology` dan `virtue`;
- outcome berpasangan yang telah disimpan dengan ukuran efek dan koreksi Holm.

Hasil ini belum boleh dipakai untuk menyatakan bahwa:

- terdapat satu “akurasi ETHICS” yang valid untuk keempat subset;
- `deontology` dan `virtue` membuktikan kemampuan moral judgement;
- `accuracy_committed` bebas dari bias seleksi akibat perbedaan commit rate;
- perubahan lexical atau jawaban otomatis membuktikan perubahan moral reasoning;
- hubungan persona–output adalah efek kausal tanpa membahas prompt, model,
  status, dan interaksi persona × model.
"""
        ),
        code(
            r"""
summary = {
    "repo_root": str(REPO_ROOT),
    "source_bundle": str(SOURCE_BUNDLE),
    "local_bundle": str(NOTEBOOK_DIR),
    "source_rows": int(condition_summary["total"].sum()),
    "models": int(condition_summary["model"].nunique()),
    "personas": int(condition_summary["persona"].nunique()),
    "subsets": int(condition_summary["subset"].nunique()),
    "committed": int(condition_summary["committed"].sum()),
    "abstain": int(condition_summary["abstain"].sum()),
    "missing": int(condition_summary["missing"].sum()),
    "paired_effect_rows": int(len(paired_effects)),
    "visual_stems": int(len(FIGURE_STEMS)),
    "visual_files_png_svg": int(len(figure_files)),
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

"""Update and execute the existing Gender ETHICS notebook in place."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / 'notebooks/gender/ethics accuracy analysis/ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb'


def main():
    notebook = nbformat.read(TARGET, as_version=4)
    cells = []

    def markdown(source):
        cells.append(nbformat.v4.new_markdown_cell(source))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(source))

    markdown('''# Analisis Persona Gender pada ETHICS — Keseluruhan dan Empat Subset

Notebook ini menampilkan hasil keseluruhan ETHICS serta rincian Commonsense,
Deontology, Justice, dan Virtue. Seluruh 81.756 respons dari enam model dan
sembilan persona pada bundle sumber digunakan tanpa sampling tambahan.

- Bagian utama tanpa pemisahan subset: distribusi wrong/acceptable/abstain/missing
  gabungan empat subset, delta terhadap netral, dan delta implicit–explicit.
- Keseluruhan: status, coverage, dan alasan abstain untuk empat subset.
- Commonsense/Justice: akurasi per subset, bersama commit_rate.
- Deontology/Virtue: distribusi wrong/acceptable, bersama commit_rate.

Figure utama dibaca langsung dari `figures/`. Bar chart detail
empat subset dibaca dari `per_subset/`. Semua ditampilkan langsung di notebook ini.''')
    markdown('''## 1. SPEC

Tujuan: menyajikan versi keseluruhan dan versi per subset dalam notebook yang
sama. Kriteria selesai: empat subset tampil pada tabel dan figure, akurasi
disertai coverage, hash artefak cocok, dan semua sel berhasil dijalankan.

Sumber dan hasil parsing tetap digunakan apa adanya. Tidak ada API call,
parsing ulang, atau rescoring. Deontology memiliki masalah excuse yang tidak
terkirim; prompt Virtue tidak sesuai tugas kecocokan sifat. Keterbatasan tersebut
mengikuti `docs/plan_parsing_ethics.md`. Stance keduanya menggambarkan jawaban
terhadap prompt eksperimen.''')
    markdown('''## 2. DESIGN PLAN

- Baca tabel sumber dan bundle revisi yang sudah ada.
- Periksa cakupan model, persona, subset, denominator, dan hash artefak.
- Tampilkan status keseluruhan serta akurasi/stance per subset bersama coverage.
- Tampilkan 3 chart gabungan, 13 chart pendukung, dan 8 bar chart detail.
- Seluruh teks pada figure selain judul diperbesar 2 pt dari ukuran renderer asal.
- Sajikan efek berpasangan, effect size, dan koreksi Holm dari sumber.

Tabel mempertahankan masing-masing persona I1, I2, dan I3. Bar chart detail
merangkum ketiganya dalam kategori implicit mengikuti generator yang tersedia.''')
    markdown('## 3. IMPLEMENTATION — Baca sumber dan bundle revisi')
    code('''import hashlib
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

def find_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "result/ETHICS").is_dir() and (path / "notebooks").is_dir():
            return path
    raise FileNotFoundError("Root repositori tidak ditemukan")

def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

ROOT = find_root()
NOTEBOOK_DIR = ROOT / "notebooks/gender/ethics accuracy analysis"
REVISION_DIR = NOTEBOOK_DIR
FIGURE_DIR = REVISION_DIR / "figures"
TABLE_DIR = REVISION_DIR / "tables"
CHILD_DIR = NOTEBOOK_DIR / "per_subset"
SOURCE = ROOT / "result/ETHICS/reports/Gender_Persona/_combined/persona_publication_analysis_v2_readable_paired_holm_six_models"
manifest = json.loads((REVISION_DIR / "bundle_manifest.json").read_text(encoding="utf-8"))
source_manifest_path = SOURCE / "metadata/analysis_manifest.json"
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
condition_summary = pd.read_csv(SOURCE / "tables/condition_summary_full.csv")
independence = pd.read_csv(SOURCE / "tables/persona_status_independence.csv")
status = pd.read_csv(TABLE_DIR / "status_by_model_persona.csv")
accuracy = pd.read_csv(TABLE_DIR / "accuracy_construct_valid_subsets.csv")
stance = pd.read_csv(TABLE_DIR / "stance_construct_invalid_subsets.csv")
reasons = pd.read_csv(TABLE_DIR / "abstain_reason_by_model_persona.csv")
paired = pd.read_csv(TABLE_DIR / "paired_primary_effects_vs_neutral.csv")
combined = pd.read_csv(TABLE_DIR / "ethics_combined_four_subsets.csv")
combined_deltas = pd.read_csv(TABLE_DIR / "ethics_combined_four_subsets_deltas.csv")
SUBSETS = ["commonsense", "deontology", "justice", "virtue"]
VALID = ["commonsense", "justice"]
STANCE = ["deontology", "virtue"]
per_subset = pd.concat([accuracy, stance], ignore_index=True)
COVERAGE = ["total", "committed", "commit_rate", "abstain", "abstain_rate", "missing", "missing_rate"]
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 500)
print("Sumber:", SOURCE)
print("Figure revisi:", FIGURE_DIR)
display(condition_summary.groupby("subset")["total"].agg(["count", "sum"]))''')
    markdown('## 4. CHECKLIST — Kelengkapan data dan konsistensi tabel')
    code('''assert manifest["source_row_count"] == source_manifest["row_count"] == 81756
assert int(status["total"].sum()) == int(condition_summary["total"].sum()) == 81756
assert set(per_subset["subset"]) == set(SUBSETS)
assert set(accuracy["subset"]) == set(VALID)
assert set(stance["subset"]) == set(STANCE)
assert len(accuracy) == 108 and len(stance) == 108 and len(status) == 54
assert per_subset.groupby("subset").size().eq(54).all()
assert set(per_subset["model"]) == set(manifest["models"])
assert set(per_subset["persona"]) == set(manifest["personas"])
keys = ["model", "persona", "subset"]
assert not per_subset.duplicated(keys).any()
pd.testing.assert_frame_equal(
    per_subset.sort_values(keys).reset_index(drop=True)[condition_summary.columns],
    condition_summary.sort_values(keys).reset_index(drop=True),
    check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
)
for frame in (status, per_subset):
    assert frame["total"].eq(frame["committed"] + frame["abstain"] + frame["missing"]).all()
    for count, rate in [("committed", "commit_rate"), ("abstain", "abstain_rate"), ("missing", "missing_rate")]:
        assert (frame[rate] - frame[count] / frame["total"]).abs().max() < 1e-12
for name, expected in {"committed": 62691, "abstain": 18521, "missing": 544}.items():
    assert int(status[name].sum()) == int(per_subset[name].sum()) == expected
assert manifest["api_calls"] == 0 and manifest["rescore"] is False
assert len(combined) == 30 and int(combined["total"].sum()) == 81756
assert combined["subset_count"].eq(4).all()
assert set(combined["scope"]) == {"all_four_subsets"}
assert combined["n_wrong"].add(combined["n_acceptable"]).eq(combined["committed"]).all()
assert combined["total"].eq(combined["committed"] + combined["abstain"] + combined["missing"]).all()
print("[x] Empat subset lengkap; tabel revisi sama dengan sumber; denominator konsisten.")''')
    markdown('''## 5. Tabel keseluruhan dan per subset

Distribusi gabungan menjumlahkan hitungan seluruh empat subset untuk setiap
model dan kategori persona. Grafik komposisi memakai seluruh respons sebagai
denominator, sehingga wrong, acceptable, abstain, dan missing berjumlah 100%.
Delta memakai proporsi wrong pada respons committed. Nilai positif berarti
proporsi jawaban wrong lebih tinggi; nilai ini tidak menunjukkan perbaikan akurasi.
Tiga varian implicit digabung berdasarkan jumlah respons, bukan rata-rata p-value
atau rata-rata persentase. Subset dengan jumlah respons lebih besar berbobot lebih
besar dalam agregat; perubahan coverage juga dapat mengubah komposisi committed.

Status keseluruhan menggabungkan empat subset. Akurasi tetap disajikan per
subset untuk Commonsense/Justice; Deontology/Virtue memakai distribusi stance.
Proporsi stance memakai respons committed sebagai denominator. Commit rate,
abstain rate, dan missing rate memakai seluruh respons. Perbedaan coverage
perlu diperhatikan karena respons committed merupakan sampel terseleksi.''')
    code('''def show_table(frame, columns):
    percent_columns = [c for c in columns if "rate" in c or c.startswith("pct_") or c == "accuracy_committed"]
    display(frame[columns].style.format({c: "{:.2%}" for c in percent_columns}))

display(Markdown("### Keseluruhan ETHICS — empat subset"))
show_table(combined, ["model_label", "category", *COVERAGE, "n_wrong", "n_acceptable", "pct_answer_wrong", "pct_answer_acceptable"])
display(Markdown("### Delta gabungan — deskriptif"))
display(combined_deltas.style.format({"delta_wrong": "{:.4f}"}))
display(Markdown("### Status keseluruhan per persona"))
show_table(status, ["model_label", "persona_label", *COVERAGE])
for subset in SUBSETS:
    metrics = ["accuracy_committed"] if subset in VALID else ["pct_answer_wrong", "pct_answer_acceptable"]
    label = "akurasi committed" if subset in VALID else "distribusi stance"
    display(Markdown(f"### {subset.title()} — {label}"))
    show_table(per_subset[per_subset["subset"].eq(subset)], ["model_label", "persona_label", "subset", *COVERAGE, *metrics])''')
    markdown('''## 6. Penyebab abstain — keseluruhan empat subset

Tabel mempertahankan setiap model dan persona. `pct_of_abstain` memakai seluruh
abstain pada model × persona; `pct_of_nonmissing` memakai seluruh respons
non-missing pada model × persona.''')
    code('''show_table(reasons, ["model", "persona", "reason_label", "reason_count", "abstain", "nonmissing", "pct_of_abstain", "pct_of_nonmissing"])''')
    markdown('''## 7. Figure keseluruhan dan empat subset

Semua figure ditampilkan dari file yang sudah tersedia. Coverage untuk keempat
subset, akurasi Commonsense/Justice, serta stance Deontology/Virtue dibaca
bersama tabel bagian 5. Bar chart detail setiap subset juga ditampilkan.

Effect size dan p-value disajikan pada bagian 8; p-value antar-subset tidak
dirata-ratakan. Inferensi yang digunakan berasal dari tabel sumber.''')
    code('''def show_figure(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    display(Markdown(f"#### {path.stem}"))
    display(Image(filename=str(path), width=1400))

GROUPS = {
    "UTAMA — gabungan empat subset tanpa pemisahan subset": [
        "00_ethics_combined_four_subsets_distribution",
        "00_ethics_combined_four_subsets_delta_vs_neutral",
        "00_ethics_combined_four_subsets_delta_implicit_explicit",
        "08_ethics_gender_combined_effect_holm_heatmap",
        "09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap",
    ],
    "Keseluruhan ETHICS — empat subset": [
        "01_ethics_status_composition_by_model",
        "02_ethics_status_by_model_persona_heatmap",
        "04_ethics_abstain_reason_share_by_model_persona_heatmap",
        "05_ethics_abstain_reason_prevalence_by_model_persona_heatmap",
        "09_ethics_status_persona_cramers_v",
    ],
    "Per subset — coverage keempat subset": [
        "03_ethics_commit_rate_by_persona_subset_heatmap",
        "10_ethics_coverage_effect_holm_heatmap",
    ],
    "Per subset — akurasi Commonsense dan Justice": [
        "06_ethics_accuracy_committed_valid_subsets_heatmap",
        "07_ethics_effective_success_valid_subsets_heatmap",
        "11_ethics_conditional_accuracy_effect_holm_heatmap",
        "12_ethics_effective_success_effect_holm_heatmap",
    ],
    "Per subset — stance Deontology dan Virtue": [
        "08_ethics_stance_wrong_construct_invalid_heatmap",
        "13_ethics_stance_wrong_effect_holm_heatmap",
    ],
}
for title, stems in GROUPS.items():
    display(Markdown(f"### {title}"))
    for stem in stems:
        show_figure(FIGURE_DIR / f"{stem}.png")

child_paths = []
for subset in SUBSETS:
    label = "akurasi committed" if subset in VALID else "stance wrong"
    display(Markdown(f"### Detail {subset.title()} — {label} dan coverage"))
    for stem in [f"01_{subset}_performance_by_persona", f"02_{subset}_status_by_persona"]:
        path = CHILD_DIR / subset / "figures" / f"{stem}.png"
        child_paths.append(path)
        show_figure(path)
print("5 chart utama, 13 chart pendukung, dan 8 chart detail ditampilkan.")''')
    markdown('''## 8. Efek persona dan independensi status

Coverage memakai empat subset; conditional accuracy memakai Commonsense/Justice;
stance wrong memakai Deontology/Virtue. Tabel mempertahankan eligibility sumber
serta menyajikan delta, Cohen's h, Exact McNemar, dan p-value setelah Holm.

Uji status × persona menggambarkan perbedaan status teramati. Signifikansinya
sendiri belum memisahkan perilaku abstain model dari kegagalan parser; hasilnya
dibaca bersama audit ekstraksi.''')
    code('''columns = ["subset", "model_label", "comparison_label", "outcome", "eligibility_definition", "risk_difference_pp", "cohen_h", "mcnemar_exact_p_value", "holm_adjusted_p_value", "significant_holm_0_05"]
for outcome, subsets in [("coverage", SUBSETS), ("conditional_accuracy", VALID), ("stance_wrong", STANCE)]:
    frame = paired[paired["outcome"].eq(outcome)]
    assert set(frame["subset"]) == set(subsets)
    display(Markdown(f"### {outcome} — efek terhadap persona netral"))
    display(frame[columns].style.format({"risk_difference_pp": "{:.4f}", "cohen_h": "{:.4f}", "mcnemar_exact_p_value": "{:.3e}", "holm_adjusted_p_value": "{:.3e}"}))
display(Markdown("### Status × persona — uji sumber"))
display(independence)''')
    markdown('## 9. TESTING — Hash, denominator, dan kelengkapan figure')
    code('''index = pd.read_csv(REVISION_DIR / "output_index.csv")
stems = [stem for group in GROUPS.values() for stem in group]
png = [FIGURE_DIR / f"{stem}.png" for stem in stems]
svg = [FIGURE_DIR / f"{stem}.svg" for stem in stems]
assert len(png) == len(svg) >= 16
assert {p.stem for p in png} == {p.stem for p in svg} == set(stems)
assert {p.name for p in png + svg}.issuperset(set(manifest["active_figure_files"]))
for row in index.itertuples(index=False):
    path = NOTEBOOK_DIR / row.relative_path
    assert path.is_file() and sha256_file(path) == row.sha256, str(path)
assert sha256_file(source_manifest_path) == manifest["source_manifest_sha256"]
old_index = pd.read_csv(NOTEBOOK_DIR / "output_index.csv")
for path in child_paths:
    matches = old_index[old_index["file"].eq(path.name)]
    assert len(matches) == 1 and sha256_file(path) == matches["sha256"].iloc[0], str(path)
assert set(paired["statistical_method"].dropna()) == {"two-sided Exact McNemar"}
assert set(paired["multiple_comparison_method"].dropna()) == {"Holm"}
assert (reasons["reason_count"] >= 0).all()
assert (per_subset["pct_answer_wrong"] + per_subset["pct_answer_acceptable"] - 1).abs().max() < 1e-10
assert (accuracy["accuracy_committed"] - accuracy["n_correct"] / accuracy["committed"]).abs().max() < 1e-10
font_audit = pd.read_csv(TABLE_DIR / "font_size_audit.csv")
increment = font_audit["after_pt"] - font_audit["before_pt"]
assert increment[font_audit["is_title"]].eq(0).all()
assert increment[~font_audit["is_title"]].eq(2).all()
assert (combined["pct_answer_wrong"] + combined["pct_answer_acceptable"] - 1).abs().max() < 1e-10
print(f"TESTING LULUS: {len(index)} hash artefak revisi, 8 hash figure detail, empat subset.")''')
    markdown('''## 10. INTEGRATION — Penggunaan dan batas klaim

Gunakan **Restart Kernel and Run All** untuk menampilkan ulang seluruh hasil
dan pemeriksaan. Notebook ini membaca figure/tabel yang ada. Untuk membangun
ulang isi dan output notebook yang sama dari root repositori:

```powershell
python src/benchmarks/ethics/render_gender_combined_existing.py
python src/benchmarks/ethics/update_gender_notebook_existing.py
```

Generator `src/pipelines/analysis/build_ethics_gender_combined_accuracy_notebook.py` masih
menghasilkan susunan lama; gunakan perintah di atas untuk notebook revisi ini.

Akurasi tetap per subset. Stance Deontology/Virtue menggambarkan jawaban terhadap
prompt eksperimen dengan keterbatasan pada SPEC. Perbedaan hasil belum membuktikan
perubahan proses reasoning atau mekanisme kausal persona. Interpretasi perlu
memperhatikan coverage, abstain, dan perbedaan narasi antarvarian persona.''')
    code('''display(pd.Series({
    "source_rows": int(condition_summary["total"].sum()),
    "models": int(condition_summary["model"].nunique()),
    "personas": int(condition_summary["persona"].nunique()),
    "subsets": int(condition_summary["subset"].nunique()),
    "revision_png": len(png), "revision_svg": len(svg),
    "displayed_child_figures": len(child_paths),
    "api_calls": 0, "reparsing": False, "rescoring": False,
}, name="value").to_frame())
print("NOTEBOOK ETHICS GENDER SELESAI DIJALANKAN TANPA ERROR")''')
    notebook.cells = cells
    nbformat.validate(notebook)
    print(f'Executing {len(cells)} cells: {TARGET}', flush=True)
    client = NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(TARGET.parent)}})
    client.execute()
    nbformat.validate(notebook)
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in notebook.cells if c.cell_type == 'code' for o in c.outputs)
    nbformat.write(notebook, TARGET)
    print(f'Updated existing notebook: {TARGET}', flush=True)
    print('All code cells executed without errors.', flush=True)


if __name__ == '__main__':
    main()

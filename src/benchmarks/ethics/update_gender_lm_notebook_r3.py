"""Refresh the authorized existing notebook from the frozen r3 report bundle."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TARGET = ROOT / "notebooks/gender/linguistic marker/ETHICS/ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"
BUNDLE = ROOT / "result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3"


def figure_bundle():
    """Resolve the selected presentation separately from the frozen statistics."""
    config = TARGET.parent / "presentation_source.json"
    if not config.exists():
        return BUNDLE
    selected = (ROOT / json.loads(config.read_text(encoding="utf-8"))["figure_bundle"]).resolve()
    if not selected.is_relative_to((ROOT / "result/ETHICS/reports").resolve()):
        raise ValueError("Figure bundle must remain within ETHICS reports")
    return selected


def copy_local_figures():
    """Copy frozen images and their index without overwriting different files."""
    figure_source = figure_bundle()
    index = figure_source / "figure_index.csv"
    with index.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = ["figure_index.csv"]
    names.extend(row[column] for row in rows for column in ("png_path", "svg_path"))
    copies = []
    for name in names:
        source = (figure_source / name).resolve()
        target = (TARGET.parent / name).resolve()
        if not source.is_relative_to(figure_source.resolve()) or not target.is_relative_to(TARGET.parent.resolve()):
            raise ValueError(f"Artifact path escapes its directory: {name}")
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise FileExistsError(f"Local artifact differs from frozen source: {target}")
        copies.append((source, target, expected))
    for source, target, expected in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, target
    print(f"Verified {len(rows)} local PNG/SVG pairs and figure index", flush=True)
    from src.benchmarks.ethics.integrate_gender_lm_diagnostics import selected_bundle, install_assets
    diagnostic_source = selected_bundle(ROOT, TARGET.parent)
    if diagnostic_source is not None:
        install_assets(diagnostic_source, TARGET.parent)


def snapshot():
    def digest(path):
        with path.open("rb") as handle:
            return path.relative_to(BUNDLE).as_posix(), hashlib.file_digest(handle, "sha256").hexdigest()

    paths = [path for path in sorted(BUNDLE.rglob("*")) if path.is_file()]
    hashes = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (name, value) in enumerate(pool.map(digest, paths), 1):
            hashes[name] = value
            if index % 100 == 0 or index == len(paths):
                print(f"Hashed {index}/{len(paths)} source files", flush=True)
    return hashes


def execute_notebook(notebook):
    """Execute rich displays without requiring kernel connection files or sockets."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(TARGET.parent)
        shell.reset(new_session=False)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if not result.success:
                raise RuntimeError(f"Cell {count} failed:\n{captured.stdout}\n{captured.stderr}")
            cell.execution_count = count
            cell.outputs = []
            for name in ("stdout", "stderr"):
                if text := getattr(captured, name):
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
            print(f"Executed code cell {count}", flush=True)
    finally:
        os.chdir(previous_cwd)
        InteractiveShell.clear_instance()


def build_notebook():
    notebook = nbformat.read(TARGET, as_version=4)
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    md("""# Linguistic marker ETHICS gender — v2.0.1 candidate, revisi r3

**Hasil 16 September 2026, sesuai rencana 15 September.** Isi menggunakan bundle r3.
Notebook berada di folder `ETHICS`; seluruh PNG dan SVG tersedia di subfolder
`figures/`. Tabel statistik tetap dibaca dari bundle sumber r3.
Pengaturan `presentation_source.json`, jika tersedia, menunjuk bundle gambar
yang digunakan; penyelarasan gaya visual tidak menghitung ulang statistik.

Seluruh **81.756 respons**, enam model, sembilan persona, dan empat subset
masuk accounting. Ada **65.678 penjelasan tersedia (80,3342%)**, 15.531 respons
verdict saja, 544 missing, dan tiga teknis saja. Sumber angka:
`status_by_model.csv` dan `extraction_manifest.json` dalam bundle r3.

**Status masih candidate: audit manusia selesai 0; precision/recall 99% belum
terbukti.** Uji implementasi bukan audit kebenaran semantik marker. Notebook
memuat seluruh 41 marker dan 88 figur utama; model tetap terpisah dan tampilan
utama memakai keseluruhan corpus. Jika `diagnostic_source.json` tersedia,
halaman tambahan konteks, dukungan pasangan, dan sensitivitas dipasang pada
bagian 9–11. Tabel empat subset tetap tersedia pada bagian sensitivitas.

## SPEC dan desain penyajian

Notebook membaca hasil beku r3, memeriksa accounting serta hash gambar/tabel,
kemudian menampilkan coverage, atlas, forest dan bukti. Statistik tidak dihitung
ulang. Satu respons adalah satu observasi; pairing memakai model, subset, dan
ID soal. Semua penjelasan termasuk refusal/abstain diperhitungkan tanpa filter
committed. Hidden reasoning tidak masuk pengukuran marker. Respons tanpa
penjelasan memakai NA untuk marker; availability bukan recall atau commit rate.

Rujukan metodologi dan pekerjaan terbuka: [validasi r3](../../../../docs/validasi_ethics_gender_lm_v2_r3_20260916.md).
""")
    code('''from pathlib import Path
import hashlib
import html
import json
import os
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src/benchmarks/ethics").is_dir())
NOTEBOOK_DIR = ROOT / "notebooks/gender/linguistic marker/ETHICS"
BUNDLE = ROOT / "result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3"
PRESENTATION_CONFIG = NOTEBOOK_DIR / "presentation_source.json"
FIGURE_BUNDLE = (ROOT / json.loads(PRESENTATION_CONFIG.read_text(encoding="utf-8"))["figure_bundle"]
                 if PRESENTATION_CONFIG.exists() else BUNDLE)

def load_json(name):
    return json.loads((BUNDLE / name).read_text(encoding="utf-8"))

def artifact_path(name):
    path = Path(name)
    if name == "figure_index.csv" or (path.parts[0] == "figures" and path.suffix in {".png", ".svg"}):
        return NOTEBOOK_DIR / path
    return BUNDLE / path

def artifact_link(name, label=None):
    target = artifact_path(name)
    assert target.is_file(), target
    relative = os.path.relpath(target, NOTEBOOK_DIR).replace("\\\\", "/")
    return '<a href="' + html.escape(relative, quote=True) + '">' + html.escape(label or name) + '</a>'

def show_table(frame, source):
    display(HTML(artifact_link(source, "Berkas sumber lengkap: " + source)))
    display(frame)

manifest = load_json("manifest.json")
extraction = load_json("extraction_manifest.json")
analysis = load_json("analysis_manifest.json")
audit_manifest = load_json("audit/manifest.json")
registry = pd.DataFrame(load_json("registry.json"))
figures = pd.read_csv(NOTEBOOK_DIR / "figure_index.csv").fillna("")
coverage = pd.read_csv(BUNDLE / "report_status_coverage.csv")
status = pd.read_csv(BUNDLE / "status_by_model.csv")
summary = pd.read_csv(BUNDLE / "marker_summary.csv")
comparisons = pd.read_csv(BUNDLE / "comparisons.csv")
interactions = pd.read_csv(BUNDLE / "interaction_comparisons.csv")
triplets = pd.read_csv(BUNDLE / "neutral_triplet_comparisons.csv")
availability = pd.read_csv(BUNDLE / "availability_tests.csv")
lengths = pd.read_csv(BUNDLE / "response_length_summary.csv")
audit = pd.read_csv(BUNDLE / "audit/marker_audit_summary.csv")
pd.set_option("display.max_columns", 24)
pd.set_option("display.max_rows", 60)
pd.set_option("display.precision", 6)
shown_figure_ids = []

def show_figures(kind):
    selected = figures.loc[figures.figure_kind.eq(kind)]
    assert not selected.empty, kind
    for row in selected.itertuples():
        display(Markdown("### " + row.title))
        display(Image(filename=str(artifact_path(row.png_path)), width=1700))
        display(Markdown(f"**Candidate · {row.scope}.** {row.denominator}. "
                         f"Estimand: `{row.estimand}`; measurement: `{row.measurement or 'status'}`."))
        display(HTML(" · ".join(artifact_link(path, label) for path, label in
                               [(row.png_path, "PNG"), (row.svg_path, "SVG"), (row.source_csv, "Tabel angka") ])))
        shown_figure_ids.append(row.figure_id)

display(HTML(" · ".join([artifact_link("report.html", "Laporan HTML r3"),
                         artifact_link("evidence_explorer.html", "Penjelajah respons penuh"),
                         artifact_link("figure_index.csv", "Indeks seluruh figur")])))
print("Bundle:", BUNDLE)
print("Sumber gambar:", FIGURE_BUNDLE)
''')
    md("## Pemeriksaan sumber dan denominator")
    code('''assert extraction["rows"] == 81756
assert extraction["markers"] == len(registry) == 41
assert manifest["precision_recall_99_demonstrated"] is False
assert manifest["status"] == "candidate_pending_independent_human_audit"
assert audit_manifest["human_annotations_completed"] == 0
assert audit.target_status.eq("not_demonstrated").all()
assert len(figures) == 88 and figures.figure_id.is_unique
assert figures.scope.eq("overall").all() and figures.audit_status.eq("candidate").all()
assert set(summary.scope) == {"overall", "commonsense", "deontology", "justice", "virtue"}
assert set(summary.marker_id) == set(registry.marker_id)
assert summary.model_id.nunique() == 6 and summary.persona.nunique() == 9
assert (NOTEBOOK_DIR / "figure_index.csv").read_bytes() == (FIGURE_BUNDLE / "figure_index.csv").read_bytes()
for row in figures.itertuples():
    for name, expected in [(row.png_path, row.png_sha256), (row.source_csv, row.source_sha256)]:
        assert hashlib.sha256(artifact_path(name).read_bytes()).hexdigest() == expected, name
    assert artifact_path(row.svg_path).read_bytes() == (FIGURE_BUNDLE / row.svg_path).read_bytes()
for frame, name in [(comparisons, "comparisons"), (interactions, "interaction_comparisons"),
                    (triplets, "neutral_triplet_comparisons"), (availability, "availability_tests")]:
    assert len(frame) == analysis["tables"][name]
assert not comparisons.duplicated(["model_id", "scope", "marker_id", "measurement", "contrast_id"]).any()
assert np.allclose(triplets.d_woman_neutral - triplets.d_man_neutral, triplets.d_woman_man, equal_nan=True)
for frame in [comparisons, interactions, availability]:
    assert frame.adjusted_p.dropna().between(0, 1).all()

status_columns = ["substantive_explanation", "verdict_only", "missing", "technical_only"]
status["total"] = status[status_columns].sum(axis=1)
assert status.total.eq(13626).all() and status.total.sum() == 81756
for name in status_columns:
    assert status[name].sum() == extraction["text_status"][name]
    status[name + "_rate"] = status[name] / status.total
overall_coverage = coverage.loc[coverage.scope.eq("overall")]
assert overall_coverage["count"].sum() == 81756
assert overall_coverage.groupby(["model_id", "persona"])["count"].sum().eq(1514).all()
print("Accounting, 41 marker, 88 aset figur, tabel statistik dan identitas common-triplet sesuai r3.")
''')
    md("""## 1. Coverage dan panjang respons

Rate pada tabel berikut memakai seluruh respons model. `substantive_explanation`
adalah kategori teknis ketersediaan teks, bukan jaminan argumentasi lengkap.
**GPT-OSS hanya memiliki 1.179 penjelasan dari 13.626 respons.**
Perbandingan marker bersifat kondisional pada penjelasan yang tersedia.

Distribusi panjang memakai penjelasan tersedia **serta verdict-only bernilai
panjang penjelasan nol**, sesuai denominator `response_length_summary.csv`.
Denominator tersebut berbeda dari marker. Availability diuji dengan permutasi
persona di dalam blok soal, disertai Cramér's V dan koreksi Holm.
""")
    code('''show_table(status, "status_by_model.csv")
show_table(availability.loc[availability.scope.eq("overall"),
    ["model_id", "outcome", "n_responses", "n_question_blocks", "availability_rate_range_pp",
     "cramers_v", "raw_p", "adjusted_p"]], "availability_tests.csv")
show_figures("coverage")
show_figures("response_length")
''')
    md("""## 2. Atlas seluruh marker

Atlas merangkum mean count, density per 100 kata, dan presence. Jumlah mentah
tersedia pada `raw_occurrences` di `marker_summary.csv`. Nol berbeda dari NA.
Selisih dua mean atlas belum tentu sama dengan efek forest karena pasangan
eligible dapat berbeda. Gunakan CSV untuk nilai kecil yang dibulatkan pada gambar.
""")
    code('''show_figures("marker_atlas")
show_table(registry[["marker_id", "display_family", "surface", "theoretical_source", "custom_adaptation",
                     "operational_definition", "exclusion_rule", "rule_status"]], "registry.json")
''')
    md("""## 3. Woman−Man pada kondisi sepadan

E = explicit; I1/I2/I3 = tiga kondisi implicit. Effect positif berarti outcome
lebih tinggi pada woman. Count memakai occurrence, density memakai per 100 kata,
presence memakai poin persentase. CI nominal 95% mengacu pada **selisih rerata
berpasangan**, bukan CI rank-biserial atau CI simultan.

Wilcoxon digunakan untuk count/density, McNemar untuk presence. Effect size,
raw p dan p Holm tersedia pada tabel. `n_changed < 10` ditandai low-support;
tidak signifikan bukan bukti kesetaraan. Semua inferensi tetap eksploratif.
""")
    code('show_figures("forest_woman_man")')
    md("""## 4. Woman dan Man terhadap Neutral

Delapan contrast terhadap neutral serta empat Woman−Man membentuk 12 contrast
utama. Family Holm mengikuti bundle r3: marker dan measurement dalam
model/scope/display-family/role/status yang sama. Neutral sumber tidak digandakan
sebagai observasi independen. Denominator kiri, kanan, dan pasangan lengkap
tersedia pada tabel bagian 7.
""")
    code('show_figures("forest_neutral")')
    md("""## 5. Kondisi explicit/implicit dan interaction

Contrast sekunder membandingkan implicit−explicit dan pasangan I1/I2/I3
dalam masing-masing gender. Interaction menguji perubahan Woman−Man
antarjenis implicit dengan common-quadruplet dan permutasi berpasangan.
Perbedaan dua p-value bukan bukti interaction. Narasi prompt turut berbeda;
efek tidak boleh disebut sifat intrinsik atau efek gender murni.
""")
    code('''show_figures("forest_implicit")
show_figures("forest_interaction")
show_figures("condition_profile")''')
    md("""## 6. Common-triplet terhadap Neutral

Sumbu X = Man−Neutral, Y = Woman−Neutral. Semua anggota memakai soal yang
sama, sehingga `(Woman−Neutral) − (Man−Neutral) = Woman−Man` untuk selisih
rerata/prevalence. Denominator common-triplet berbeda dari pairwise-complete.
Titik dekat nol dapat bertumpuk; koordinat lengkap tersedia di CSV figur.
""")
    code('show_figures("neutral_triplet_map")')
    md("""## 7. Tabel rinci dan sensitivitas per subset

Ubah pilihan di sel berikut lalu jalankan ulang sel tersebut untuk menampilkan
marker/model/measurement lain. `MODEL = None` menampilkan seluruh model.
Semua contrast ditampilkan, tanpa menyaring signifikansi. Empat subset selalu
ikut tabel sensitivitas; keseluruhan ETHICS di sini adalah gaya bahasa, bukan akurasi.
""")
    code('''MARKER = "stance_hedge"
MEASUREMENT = "density"  # count, density, or presence
MODEL = None  # Use a model_id string to select one model.
assert MARKER in set(registry.marker_id)
assert MEASUREMENT in {"count", "density", "presence"}
assert MODEL is None or MODEL in set(summary.model_id)

def select(frame, scope=None):
    mask = frame.marker_id.eq(MARKER) & frame.measurement.eq(MEASUREMENT)
    if MODEL is not None:
        mask &= frame.model_id.eq(MODEL)
    if scope is not None:
        mask &= frame.scope.eq(scope)
    return frame.loc[mask]

with pd.option_context("display.max_rows", None):
    show_table(select(summary), "marker_summary.csv")
    columns = ["model_id", "scope", "contrast_id", "measurement", "n_left_total", "n_right_total",
        "n_left_available", "n_right_available", "n_pairs", "n_changed", "effect", "ci_low", "ci_high",
        "matched_rank_biserial", "raw_p", "adjusted_p", "family_size", "low_support", "sample_scope"]
    show_table(select(comparisons, "overall")[columns], "comparisons.csv")
    show_table(select(interactions, "overall")[columns], "interaction_comparisons.csv")
    show_table(select(triplets, "overall")[["model_id", "condition", "n_triplets", "d_woman_neutral",
        "d_man_neutral", "d_woman_man", "sample_scope"]], "neutral_triplet_comparisons.csv")
    subset_direct = select(comparisons)
    subset_direct = subset_direct.loc[~subset_direct.scope.eq("overall")
        & subset_direct.left.str.startswith("woman_") & subset_direct.right.str.startswith("man_")]
    show_table(subset_direct[columns], "comparisons.csv")
''')
    md("""## 8. Audit, perubahan r2→r3, dan bukti respons

Koreksi 12 wrapper jawaban mengubah statusnya menjadi verdict-only; 81.744
respons lain tetap identik. Total span candidate tetap 928.710. Penghapusan
wrapper dari denominator mengubah sebagian hasil statistik; perbandingan
lengkap tersedia pada tabel delta. Pergeseran p permutasi juga dapat mencakup
variasi Monte Carlo. Tidak ada perubahan keputusan Holm pada contrast
langsung Woman−Man menurut analisis sensitivitas r3.

Seluruh 41 target marker masih `not_demonstrated`. Audit manusia, evaluasi
held-out, tujuh kategori tambahan, audit prefix inline pada 1.922 respons,
dan ablation lengkap masih terbuka. Angka 1.922 adalah lingkup audit,
bukan jumlah kesalahan tervalidasi. Validasi parser jawaban merupakan
prasyarat terpisah untuk menghubungkan marker dengan akurasi/committed.

Penjelajah menampilkan respons utuh neutral/woman/man dan span aturan.
**Pembacaan respons held-out untuk memperbaiki aturan harus dicatat sebagai
paparan**; respons tersebut tidak boleh tetap diklaim held-out independen.
""")
    code('''show_figures("audit")
show_table(audit[["marker_id", "n_reviewed_precision", "n_reviewed_recall", "false_positive",
                  "precision", "recall", "target_status"]], "audit/marker_audit_summary.csv")
show_table(pd.read_csv(BUNDLE / "r2_r3_statistical_delta_summary.csv"), "r2_r3_statistical_delta_summary.csv")
display(HTML("<br>".join(artifact_link(name) for name in [
    "evidence_explorer.html", "evidence_index.csv", "audit/manifest.json", "baseline_delta.csv",
    "r2_r3_pairwise_statistical_changes.csv", "r2_r3_interaction_statistical_changes.csv",
    "verification_report.json", "independent_statistics_validation.json"])))
''')
    from src.benchmarks.ethics.integrate_gender_lm_diagnostics import diagnostic_cells, selected_bundle
    extra_cells, extra_images = diagnostic_cells(ROOT, TARGET.parent)
    cells.extend(extra_cells)
    md("## Checklist akhir eksekusi notebook")
    code('''assert len(shown_figure_ids) == len(set(shown_figure_ids)) == 88
assert set(shown_figure_ids) == set(figures.figure_id)
assert len(summary) == 33210
assert len(comparisons) == 88560 and len(interactions) == 11070 and len(triplets) == 14760
assert load_json("verification_report.json")["status"] == "pass"
print("Seluruh 88 figur utama ditampilkan; accounting dan tabel r3 sesuai. Status tetap candidate.")
print("Audit manusia:", audit_manifest["human_annotations_completed"], "; klaim precision/recall 99%: belum terbukti.")
''')
    notebook.cells = cells
    notebook.metadata.update({"result_status": "candidate", "ethics_aggregation": "overall",
                              "source_bundle": BUNDLE.relative_to(ROOT).as_posix(),
                              "pipeline_version": "gender-lm-v2.0.1-candidate",
                              "figure_bundle": figure_bundle().relative_to(ROOT).as_posix()})
    notebook.metadata["diagnostic_default_pngs"] = extra_images
    diagnostic_source = selected_bundle(ROOT, TARGET.parent)
    notebook.metadata["diagnostic_bundle"] = diagnostic_source.relative_to(ROOT).as_posix() if diagnostic_source else None
    notebook.metadata["diagnostic_figure_count"] = 0
    if diagnostic_source:
        notebook.metadata["diagnostic_figure_count"] = json.loads((diagnostic_source / "diagnostic_manifest.json").read_text(encoding="utf-8"))["figure_count"]
    nbformat.validate(notebook)
    return notebook


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "result/ETHICS/reports" / ("gender_lm_v2_notebook_integration_" + stamp)
    output.mkdir(exist_ok=False)
    original = TARGET.read_bytes()
    (output / "notebook_before.ipynb").write_bytes(original)
    before = snapshot()
    copy_local_figures()
    notebook = build_notebook()
    print("Executing notebook from r3:", len(notebook.cells), "cells", flush=True)
    execute_notebook(notebook)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    errors = [out for cell in code_cells for out in cell.outputs if out.output_type == "error"]
    assert not errors, errors
    images = sum("image/png" in out.get("data", {}) for cell in code_cells for out in cell.outputs)
    assert images == 88 + notebook.metadata["diagnostic_default_pngs"], images
    assert before == snapshot(), "Frozen r3 bundle changed during notebook execution"
    assert TARGET.read_bytes() == original, "Notebook changed concurrently"
    prepared = output / "notebook_after.ipynb"
    nbformat.write(notebook, prepared)
    verification = {"status": "pass", "notebook": TARGET.relative_to(ROOT).as_posix(),
        "figure_bundle": figure_bundle().relative_to(ROOT).as_posix(),
        "source_bundle": BUNDLE.relative_to(ROOT).as_posix(), "cells": len(notebook.cells),
        "executed_code_cells": len(code_cells), "error_outputs": len(errors), "embedded_pngs": images,
        "diagnostic_bundle": notebook.metadata["diagnostic_bundle"],
        "diagnostic_figures_available": notebook.metadata["diagnostic_figure_count"],
        "execution_engine": "IPython InteractiveShell; in-memory history; all cells",
        "bundle_files_unchanged": len(before), "bundle_sha256": before,
        "notebook_before_sha256": hashlib.sha256(original).hexdigest(),
        "notebook_after_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "result_status": "candidate", "human_annotations_completed": 0,
        "precision_recall_99_demonstrated": False}
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    temporary = TARGET.with_name(TARGET.stem + "." + stamp + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(prepared.read_bytes())
    assert TARGET.read_bytes() == original, "Notebook changed concurrently before installation"
    os.replace(temporary, TARGET)
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == verification["notebook_after_sha256"]
    print(json.dumps({key: value for key, value in verification.items() if key != "bundle_sha256"}, indent=2))
    print("Verification:", output / "verification.json")


if __name__ == "__main__":
    main()

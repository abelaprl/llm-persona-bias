"""Create and optionally execute the ETHICS Gender LM v1.4 candidate notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executed-output", type=Path)
    return parser


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def _cells():
    return [
        _markdown(
            """
# ETHICS Gender Persona — Linguistic Markers v1.4

Notebook ini menyajikan analisis berpasangan gender ETHICS dari bundle
`gender-ethics-linguistic-markers-v1.4.0-candidate`.

**Status ilmiah: candidate.** Angka, interval kepercayaan, nilai p, koreksi
Holm, dan gambar belum boleh menjadi klaim final tesis sampai audit *precision*
dan *recall* oleh dua reviewer manusia, *inter-rater agreement*, serta
adjudikasi disagreement selesai.

Analisis utama memakai seluruh *visible text* yang tersedia, termasuk respons
`abstain`. Analisis sensitivitas memakai pasangan yang kedua responsnya
`committed`. `<think>` dan *hidden reasoning* tidak digunakan.
"""
        ),
        _code(
            """
from pathlib import Path
import json

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display


def locate_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").is_dir() and (candidate / "result").is_dir():
            return candidate
    raise FileNotFoundError("Repo root D:/TugasAkhir tidak ditemukan.")


REPO_ROOT = locate_repo_root()
BUNDLE_DIR = (
    REPO_ROOT
    / "result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis"
    / "gender_ethics_linguistic_markers_v1_4_candidate_r4_20260826"
)
FIG_DIR = BUNDLE_DIR / "images"
manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
final_gate = pd.read_csv(BUNDLE_DIR / "final_result_gate.csv")

assert manifest["bundle_version"] == "gender-ethics-linguistic-markers-v1.4.0-candidate"
assert manifest["result_status"] == "candidate"
assert manifest["source_input_sha256"] == "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"
assert manifest["total_responses"] == 81_756
assert manifest["responses_with_linguistically_available_text"] == 80_840
assert manifest["hidden_reasoning_used"] is False
assert manifest["marker_extraction_rerun"] is False
assert manifest["validations"]["all_marker_density_rows_have_raw_counts"] is True
assert not bool(final_gate.loc[0, "final_result_available"])
assert all((FIG_DIR / name).is_file() for name in manifest["generated_figures"])

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {item: item.title() for item in SUBSET_ORDER}


def show_figure(filename: str, width: int = 1800) -> None:
    path = FIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))


def candidate_top_effects(frame: pd.DataFrame, per_subset: int = 5) -> pd.DataFrame:
    ranked = frame.assign(abs_rank_biserial=frame["rank_biserial"].abs()).sort_values(
        ["subset", "abs_rank_biserial"], ascending=[True, False]
    )
    selected = ranked.groupby("subset", sort=False, group_keys=False).head(per_subset).copy()
    selected["model"] = selected["model_id"].map(MODEL_LABELS)
    columns = [
        "subset", "model", "contrast_id", "outcome_id", "n_pairs", "n_nonzero",
        "median_paired_delta", "rank_biserial", "ci_low", "ci_high", "p_holm",
        "n_raw_count_pairs", "raw_count_sum_left", "raw_count_sum_right",
        "median_paired_raw_count_delta",
    ]
    return selected[[column for column in columns if column in selected.columns]]


pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
print(f"Bundle     : {manifest['bundle_version']}")
print(f"Input SHA  : {manifest['source_input_sha256']}")
print(f"Respons    : {manifest['total_responses']:,}")
print(f"Visible    : {manifest['responses_with_linguistically_available_text']:,}")
print(f"Final gate : {bool(final_gate.loc[0, 'final_result_available'])}")
"""
        ),
        _markdown(
            """
## 1. Provenance, status respons, dan cakupan teks

`missing` dan `technical_only` tidak diubah menjadi nol marker. Respons
`abstain` tetap dianalisis bila *visible text* tersedia.
"""
        ),
        _code(
            """
provenance = pd.DataFrame([
    ("Input kanonik", manifest["source_input_csv"]),
    ("SHA-256 input", manifest["source_input_sha256"]),
    ("Pipeline marker", manifest["source_marker_pipeline_version"]),
    ("Total respons", f"{manifest['total_responses']:,}"),
    ("Teks available", f"{manifest['responses_with_linguistically_available_text']:,}"),
    ("Teks technical-only", f"{manifest['linguistic_text_status_counts']['technical_only']:,}"),
    ("Missing", f"{manifest['response_status_counts']['missing']:,}"),
    ("Committed available", f"{manifest['available_answer_status_counts']['committed']:,}"),
    ("Abstain available", f"{manifest['available_answer_status_counts']['abstain']:,}"),
    ("Model x persona x pertanyaan", f"{manifest['models']} x {manifest['personas']} x {manifest['questions']:,}"),
    ("Bootstrap", f"{manifest['bootstrap_resamples']:,}"),
    ("Hidden reasoning", str(manifest["hidden_reasoning_used"])),
], columns=["Komponen", "Nilai"])
display(provenance)

status = pd.read_csv(BUNDLE_DIR / "response_status_counts.csv")
total = status.groupby(["model_id", "persona", "subset"], as_index=False)["n_responses"].sum()
answer = (
    status.groupby(["model_id", "persona", "subset", "answer_status"], as_index=False)["n_responses"].sum()
    .pivot(index=["model_id", "persona", "subset"], columns="answer_status", values="n_responses")
    .fillna(0).reset_index()
)
coverage = total.merge(answer, on=["model_id", "persona", "subset"], validate="one_to_one")
for column in ["committed", "abstain", "missing"]:
    if column not in coverage:
        coverage[column] = 0
coverage["commit_rate"] = coverage["committed"] / coverage["n_responses"]
coverage["abstain_rate"] = coverage["abstain"] / coverage["n_responses"]
coverage["missing_rate"] = coverage["missing"] / coverage["n_responses"]
available = (
    status[status["text_status"].eq("available")]
    .groupby(["model_id", "persona", "subset"], as_index=False)["n_responses"].sum()
    .rename(columns={"n_responses": "text_available"})
)
coverage = coverage.merge(available, on=["model_id", "persona", "subset"], how="left")
coverage["visible_text_rate"] = coverage["text_available"].fillna(0) / coverage["n_responses"]
coverage["model"] = coverage["model_id"].map(MODEL_LABELS)
display(
    coverage[[
        "model", "persona", "subset", "n_responses", "committed", "commit_rate",
        "abstain", "abstain_rate", "missing", "missing_rate", "text_available",
        "visible_text_rate",
    ]].style.format({
        "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}",
        "missing_rate": "{:.2%}", "visible_text_rate": "{:.2%}",
    })
)
show_figure("01_visible_text_coverage.png", width=1800)
"""
        ),
        _markdown(
            """
## 2. Inventory marker dan dua bentuk pengukuran

Tiga outcome permukaan dihitung otomatis. Delapan belas marker terdaftar masih
berstatus `candidate_pending_manual_audit`. Untuk marker, tabel menyimpan
jumlah mentah dan densitas per 100 kata secara berdampingan.
"""
        ),
        _code(
            """
inventory = pd.read_csv(BUNDLE_DIR / "marker_inventory_21_numeric.csv")
assert len(inventory) == 21
assert inventory["outcome_id"].nunique() == 21
assert (inventory.loc[inventory["outcome_type"].eq("registered_marker"), "validation_status"] == "candidate_pending_manual_audit").all()
display(inventory)

direct_all = pd.read_csv(BUNDLE_DIR / "paired_numeric_direct_man_woman_all_text.csv")
marker_measurements = direct_all[direct_all["n_raw_count_pairs"].notna()].copy()
assert marker_measurements["n_raw_count_pairs"].eq(marker_measurements["n_pairs"]).all()
display(
    marker_measurements.sort_values("raw_count_sum_left", ascending=False).head(24)[[
        "model_id", "subset", "condition", "outcome_id", "n_pairs",
        "median_paired_delta", "n_raw_count_pairs", "raw_count_sum_left",
        "raw_count_sum_right", "median_paired_raw_count_delta",
    ]].assign(model=lambda d: d["model_id"].map(MODEL_LABELS)).drop(columns="model_id")
)
"""
        ),
        _markdown(
            """
## 3. Persona gender dibandingkan dengan Neutral

Arah kontras adalah E/I1/I2/I3 dikurangi Neutral, terpisah untuk Man dan Woman.
Tabel di bawah hanya mengurutkan *effect size* kandidat untuk inspeksi; ini
bukan daftar temuan final.
"""
        ),
        _code(
            """
vs_neutral = pd.read_csv(BUNDLE_DIR / "paired_numeric_vs_neutral_all_text.csv")
assert len(vs_neutral) == 4_032
for number, subset in enumerate(SUBSET_ORDER, start=2):
    display(Markdown(f"### {SUBSET_LABELS[subset]}"))
    show_figure(f"{number:02d}_effects_vs_neutral_{subset}.png", width=1900)
display(
    candidate_top_effects(vs_neutral).style.format({
        "median_paired_delta": "{:+.3f}", "rank_biserial": "{:+.3f}",
        "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "p_holm": "{:.4g}",
    })
)
"""
        ),
        _markdown(
            """
## 4. Persona implicit dibandingkan dengan Explicit

Arah kontras adalah I1/I2/I3 dikurangi E, terpisah untuk Man dan Woman serta
untuk setiap subset ETHICS.
"""
        ),
        _code(
            """
vs_explicit = pd.read_csv(BUNDLE_DIR / "paired_numeric_vs_explicit_all_text.csv")
assert len(vs_explicit) == 3_024
for number, subset in enumerate(SUBSET_ORDER, start=6):
    display(Markdown(f"### {SUBSET_LABELS[subset]}"))
    show_figure(f"{number:02d}_effects_vs_explicit_{subset}.png", width=1900)
display(
    candidate_top_effects(vs_explicit).style.format({
        "median_paired_delta": "{:+.3f}", "rank_biserial": "{:+.3f}",
        "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "p_holm": "{:.4g}",
    })
)
"""
        ),
        _markdown(
            """
## 5. Perbandingan langsung Man dan Woman

Estimand adalah **Man − Woman** pada soal, model, subset, dan kondisi yang sama.
Nilai positif berarti outcome lebih tinggi pada Man; nilai negatif berarti
lebih tinggi pada Woman.
"""
        ),
        _code(
            """
assert len(direct_all) == 2_016
for number, subset in enumerate(SUBSET_ORDER, start=10):
    display(Markdown(f"### {SUBSET_LABELS[subset]}"))
    show_figure(f"{number:02d}_effects_direct_man_woman_{subset}.png", width=1900)
display(
    candidate_top_effects(direct_all).style.format({
        "median_paired_delta": "{:+.3f}", "rank_biserial": "{:+.3f}",
        "ci_low": "{:+.3f}", "ci_high": "{:+.3f}", "p_holm": "{:.4g}",
    })
)
"""
        ),
        _markdown(
            """
## 6. Sensitivitas pada pasangan committed

Analisis utama tidak membuang `abstain`. Bagian ini membandingkan *effect size*
utama dengan pasangan yang kedua responsnya `committed`, sehingga perubahan
karena seleksi coverage terlihat eksplisit.
"""
        ),
        _code(
            """
show_figure("14_committed_sensitivity_concordance.png", width=1900)
sensitivity = pd.read_csv(BUNDLE_DIR / "paired_numeric_committed_sensitivity_summary.csv")
assert len(sensitivity) == 12
display(
    sensitivity.style.format({
        "median_pair_retention": "{:.2%}", "min_pair_retention": "{:.2%}",
        "same_direction_rate_supported": "{:.2%}", "spearman_rank_biserial": "{:.3f}",
        "median_abs_rank_biserial_change": "{:.3f}",
    })
)
"""
        ),
        _markdown(
            """
## 7. Komposisi opening

Marginal total variation tidak memiliki arah. Nol berarti distribusi kategori
opening identik; nilai yang lebih besar berarti komposisinya lebih berbeda.
"""
        ),
        _code(
            """
show_figure("15_opening_composition_within_identity.png", width=1850)
show_figure("16_opening_composition_direct_man_woman.png", width=1850)

opening_within = pd.read_csv(BUNDLE_DIR / "paired_opening_within_identity_all_text.csv")
opening_direct = pd.read_csv(BUNDLE_DIR / "paired_opening_direct_man_woman_all_text.csv")
assert len(opening_within) == 336
assert len(opening_direct) == 96
display(
    opening_direct.sort_values("marginal_total_variation", ascending=False).head(24)
    .assign(model=lambda d: d["model_id"].map(MODEL_LABELS))[[
        "model", "subset", "condition", "n_pairs", "same_category_rate",
        "marginal_total_variation", "p_holm",
    ]].style.format({
        "same_category_rate": "{:.2%}", "marginal_total_variation": "{:.2%}",
        "p_holm": "{:.4g}",
    })
)
"""
        ),
        _markdown(
            """
## 8. Audit manusia dan final gate

Notebook tidak mengubah status kandidat. Dua reviewer harus bekerja pada
salinan lembar audit yang berbeda dan tidak melihat ground truth atau lembar
reviewer lain. Satu *false positive* atau *recall miss* menutup gate dan
mengharuskan revisi aturan serta sampling ulang dari nol.
"""
        ),
        _code(
            """
audit_status = pd.read_csv(BUNDLE_DIR / "manual_audit_status.csv")
display(audit_status)
display(final_gate)

assert audit_status["completed_items"].sum() == 0
assert manifest["human_precision_audit"]["completed_judgments"] == 0
assert manifest["human_recall_audit"]["completed_judgments"] == 0
assert manifest["inter_rater_agreement_complete"] is False
assert manifest["disagreement_adjudication_complete"] is False
assert not bool(final_gate.loc[0, "final_result_available"])

gate_summary = pd.DataFrame([
    ("Precision", f"{manifest['human_precision_audit']['completed_judgments']}/{manifest['human_precision_audit']['required_judgments']} judgment"),
    ("Recall", f"{manifest['human_recall_audit']['completed_judgments']}/{manifest['human_recall_audit']['required_judgments']} judgment"),
    ("Inter-rater agreement", str(manifest["inter_rater_agreement_complete"])),
    ("Adjudikasi disagreement", str(manifest["disagreement_adjudication_complete"])),
    ("Final result available", str(bool(final_gate.loc[0, "final_result_available"]))),
], columns=["Gate", "Status"])
display(gate_summary)
"""
        ),
        _markdown(
            """
## 9. Ringkasan penggunaan

Bundle ini layak untuk QA struktur, inspeksi pola, dan persiapan audit. Bundle
ini belum layak sebagai bukti final bahwa persona Man dan Woman menyebabkan
perbedaan marker tertentu. Empat subset tidak boleh dilebur menjadi satu angka
ETHICS, dan hasil kandidat tidak boleh dipilih hanya karena `p_holm < 0.05`.
"""
        ),
        _code(
            """
figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
pair_counts = pd.read_csv(BUNDLE_DIR / "pair_count_summary.csv")
display(pair_counts)
display(figure_index)

summary = pd.DataFrame([
    ("Analisis utama", "Semua visible text yang tersedia, termasuk abstain"),
    ("Sensitivity", "Kedua respons committed dan visible text tersedia"),
    ("Perbandingan", "vs Neutral; vs Explicit; direct Man − Woman"),
    ("Outcome", "3 surface + 18 registered markers"),
    ("Raw marker count", "Disimpan bersama density per 100 kata"),
    ("Subset", "Commonsense, Deontology, Justice, Virtue tetap terpisah"),
    ("Status", "Candidate; menunggu audit dua reviewer"),
], columns=["Aspek", "Keputusan"])
display(summary)
print(f"Bundle statistik : {BUNDLE_DIR}")
print(f"Manifest         : {BUNDLE_DIR / 'manifest.json'}")
print(f"Final gate       : {BUNDLE_DIR / 'final_result_gate.csv'}")
"""
        ),
    ]


def build_notebook(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite notebook: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=_cells(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    with output.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    return output


def execute_notebook(source: Path, output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite executed notebook: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as handle:
        notebook = nbf.read(handle, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()
    with output.open("w", encoding="utf-8") as handle:
        nbf.write(executed, handle)
    return output


def main() -> int:
    args = _parser().parse_args()
    source = build_notebook(args.output)
    print(f"[DONE] Source notebook: {source}")
    if args.executed_output:
        executed = execute_notebook(source, args.executed_output)
        print(f"[DONE] Executed notebook: {executed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

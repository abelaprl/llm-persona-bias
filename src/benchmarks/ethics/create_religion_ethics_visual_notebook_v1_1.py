"""Create the compact ETHICS Religion Persona thesis-facing notebook v1.1."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite notebook: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        _markdown(
            """
# Linguistic Marker Religion Persona pada ETHICS — Visualisasi v1.1

Notebook ini menyatukan visual dasar ETHICS dengan visual tambahan yang mengikuti pola presentasi notebook Religion GSM8K: term ranking, *paired effect map* per model, perbandingan langsung antaridentitas, sensitivity terhadap correctness, komposisi opening, dan differential availability.

Seluruh **154.428 respons** dari enam model, 17 skenario, dan empat subset tetap berada dalam inventaris. Marker dihitung hanya dari jawaban yang terlihat; hidden reasoning tidak digunakan; respons *abstain* dengan teks tetap disertakan; `missing` tidak diubah menjadi marker nol. Karena registry masih `candidate`, seluruh effect marker di notebook ini **deskriptif**: tidak ada p-value atau confidence interval marker baru. Inferensi yang ditampilkan hanya inferensi akurasi dari bundle terpisah yang sudah tervalidasi.
"""
        ),
        _code(
            """
from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, Markdown, display


def locate_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").is_dir() and (candidate / "result").is_dir():
            return candidate
    raise FileNotFoundError("Repo root D:/TugasAkhir tidak ditemukan.")


REPO_ROOT = locate_repo_root()
REPORT_ROOT = REPO_ROOT / "result/ETHICS/reports/Religion_Persona"
MARKER_DIR = REPORT_ROOT / "linguistic_markers_religion_candidate_20260824_v1_1_10"
AUDIT_R2_DIR = REPORT_ROOT / "linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r2_adjudicated_v3"
BASE_ASSET_DIR = REPO_ROOT / "notebooks/religion/img/ethics"
SUPPLEMENT_DIR = REPORT_ROOT / "linguistic_markers/visual/religion_ethics_bab4_visual_supplement_v1_0_0_20260824_r3"
SUPPLEMENT_FIG_DIR = SUPPLEMENT_DIR / "images"
ACCURACY_DIR = REPORT_ROOT / "_combined/persona_publication_analysis_v3_accuracy_paired_holm_six_models"
ACCURACY_FIG_DIR = ACCURACY_DIR / "figures"
ACCURACY_TABLE_DIR = ACCURACY_DIR / "tables"

required = [
    MARKER_DIR / "manifest.json",
    AUDIT_R2_DIR / "adjudication_manifest.json",
    BASE_ASSET_DIR / "manifest.json",
    SUPPLEMENT_DIR / "manifest.json",
    ACCURACY_DIR / "metadata/analysis_manifest.json",
    ACCURACY_DIR / "metadata/validation.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Artefak wajib tidak ditemukan:\\n" + "\\n".join(missing))

marker_manifest = json.loads((MARKER_DIR / "manifest.json").read_text(encoding="utf-8"))
audit_manifest = json.loads((AUDIT_R2_DIR / "adjudication_manifest.json").read_text(encoding="utf-8"))
base_manifest = json.loads((BASE_ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
supplement_manifest = json.loads((SUPPLEMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
accuracy_manifest = json.loads((ACCURACY_DIR / "metadata/analysis_manifest.json").read_text(encoding="utf-8"))
accuracy_validation = json.loads((ACCURACY_DIR / "metadata/validation.json").read_text(encoding="utf-8"))

EXPECTED_SHA = "c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a"
assert marker_manifest["pipeline_version"] == "ethics-religion-lm-v1.1.10-candidate"
assert marker_manifest["input_sha256"] == EXPECTED_SHA
assert supplement_manifest["source_input_sha256"] == EXPECTED_SHA
assert supplement_manifest["rows"] == marker_manifest["rows_input"] == 154_428
assert supplement_manifest["marker_inference_run"] is False
assert supplement_manifest["p_values_computed"] is False
assert supplement_manifest["confidence_intervals_computed"] is False
assert accuracy_validation["valid"] is True

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
IDENTITY_LABELS = {"islam": "Islam", "christian": "Christian", "jewish": "Jewish", "atheist": "Atheist"}
CONDITION_LABELS = {"explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}
SUBSET_LABELS = {"commonsense": "Commonsense", "deontology": "Deontology", "justice": "Justice", "virtue": "Virtue"}


def show_base(filename: str, width: int = 1550) -> None:
    display(Image(filename=str(BASE_ASSET_DIR / filename), width=width))


def show_supplement(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(SUPPLEMENT_FIG_DIR / filename), width=width))


def show_accuracy(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(ACCURACY_FIG_DIR / filename), width=width))


pd.set_option("display.max_colwidth", 120)
print(f"Marker source : {marker_manifest['pipeline_version']}")
print(f"Visual bundle: {supplement_manifest['bundle_version']}")
print(f"Respons      : {marker_manifest['rows_input']:,}")
print(f"Status       : {marker_manifest['registry_status']}")
"""
        ),
        _markdown(
            """
## 1. Provenance, coverage, dan validation gate

Setiap angka pada notebook mengacu ke sumber kanonik dengan SHA-256 yang sama. Coverage adalah ketersediaan teks yang dapat diukur; ia bukan commit rate. Audit AI lengkap tidak menggantikan dua reviewer manusia independen.
"""
        ),
        _code(
            """
provenance = pd.DataFrame([
    ("Versi marker", marker_manifest["pipeline_version"]),
    ("Versi visual tambahan", supplement_manifest["bundle_version"]),
    ("SHA-256 sumber", marker_manifest["input_sha256"]),
    ("Respons unik", f"{marker_manifest['rows_input']:,}"),
    ("Surface terukur", f"{marker_manifest['responses_text_available']:,}"),
    ("Abstain disertakan", f"{marker_manifest['responses_abstain']:,}"),
    ("Missing menjadi NA", f"{marker_manifest['responses_missing']:,}"),
    ("Inferensi marker", supplement_manifest["marker_inference_run"]),
    ("P-value marker baru", supplement_manifest["p_values_computed"]),
    ("Validasi manusia", audit_manifest["human_review_status"]),
], columns=["Komponen", "Nilai"])
display(provenance)
show_base("01_coverage_model_persona.png", width=1750)
coverage = pd.read_csv(BASE_ASSET_DIR / "coverage_model_persona.csv")
display(coverage.sort_values("coverage_pct").head(15).style.format({"coverage_pct": "{:.2f}%"}))
"""
        ),
        _markdown(
            """
## 2. Status respons dan tabel pelaporan wajib

`accuracy_committed` tidak dibaca tanpa `commit_rate`. Tabel berikut memuat `total`, `committed`, `abstain`, `missing`, seluruh rate, serta accuracy pada baris committed. Commonsense dan Justice adalah subset construct-valid; Deontology dan Virtue dilaporkan melalui distribusi jawaban.
"""
        ),
        _code(
            """
show_accuracy("04_response_status_distribution_by_model.png", width=1350)
scoring = pd.read_csv(BASE_ASSET_DIR / "scoring_summary_by_model_persona_subset.csv")
required_columns = [
    "total", "committed", "commit_rate", "abstain", "abstain_rate",
    "missing", "missing_rate", "accuracy_committed",
]
assert set(required_columns).issubset(scoring.columns)
display(
    scoring.sort_values(["commit_rate", "model_id", "persona", "subset"])
    [["model_id", "persona", "subset", *required_columns, "construct_valid"]]
    .head(24)
    .style.format({
        "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}",
        "missing_rate": "{:.2%}", "accuracy_committed": "{:.2%}",
    })
)
"""
        ),
        _markdown(
            """
## 3. Verbosity sebagai dasar normalisasi

Median kata berbeda kuat antarmodel dan skenario. Marker frekuensi tetap tersedia sebagai raw count serta density per 100 kata; gambar ini hanya memberi konteks panjang respons.
"""
        ),
        _code('show_base("02_verbosity_model_persona.png", width=1750)'),
        _markdown(
            """
## 4. Penyebutan identitas dan akomodasi Explicit

Penyebutan identitas, adopsi perspektif, dan framing agama dipertahankan sebagai outcome terpisah. Nilai tinggi tidak otomatis membuktikan kausalitas persona karena overlap skenario tetap mungkin.
"""
        ),
        _code(
            """
show_base("03_aligned_identity_mentions_explicit.png", width=1150)
show_base("04_religious_accommodation_explicit.png", width=1650)
"""
        ),
        _markdown(
            """
## 5. Term agama yang benar-benar muncul

Lollipop menggunakan closed lexicon dan hanya term yang ditangkap marker relevan serta tidak muncul pada skenario soal yang sama. Sumbu adalah rerata prevalence respons dengan bobot model setara; raw jumlah respons tetap tersedia pada tabel sumber.
"""
        ),
        _code(
            """
show_supplement("05_religious_terms_ranked_explicit_response_prevalence.png", width=1700)
terms = pd.read_csv(SUPPLEMENT_DIR / "religion_terms_explicit_response_prevalence.csv")
term_summary = (
    terms[terms["condition"].eq("explicit")]
    .groupby(["identity", "term"], as_index=False)
    .agg(mean_response_prevalence_pct=("response_prevalence_pct", "mean"), raw_response_hits=("n_responses", "sum"))
)
display(
    term_summary.sort_values(["identity", "mean_response_prevalence_pct"], ascending=[True, False])
    .groupby("identity", sort=False).head(8)
    .style.format({"mean_response_prevalence_pct": "{:.2f}%"})
)
"""
        ),
        _markdown(
            """
## 6. Profil marker pada E, I1, I2, dan I3

Neutral, Explicit, I1, I2, dan I3 tidak digabung lebih awal. Heatmap adalah inventaris deskriptif lintas enam model; nilai per model ditampilkan pada effect map berikutnya.
"""
        ),
        _code('show_base("05_marker_profile_all_scenarios.png", width=1700)'),
        _markdown(
            """
## 7. Marker agama terhadap Neutral

Setiap titik memasangkan model, subset, dan ID soal yang sama. Titik berwarna adalah model; berlian hitam adalah median enam model. Tidak ada CI atau penanda signifikansi karena registry masih candidate. Tabel sumber juga menyimpan hasil terpisah per subset.
"""
        ),
        _code(
            """
for identity in ("islam", "christian", "jewish", "atheist"):
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"07_{identity}_religion_marker_forest_vs_neutral_descriptive.png", width=1800)

within_religion = pd.read_csv(SUPPLEMENT_DIR / "religion_marker_vs_neutral_descriptive.csv")
display(
    within_religion[within_religion["subset"].eq("all")]
    .assign(abs_delta=lambda d: d["prevalence_delta_pp"].abs())
    .nlargest(24, "abs_delta")
    .drop(columns="abs_delta")
    .style.format({"prevalence_delta_pp": "{:+.2f}"})
)
"""
        ),
        _markdown(
            """
## 8. Marker linguistik umum terhadap Neutral

Outcome panjang, organisasi, stance, connective, self-reference, engagement, dan empathy memakai paired rank-biserial agar skala density yang berbeda dapat dibandingkan. Nilai ini deskriptif dan bukan hasil uji hipotesis.
"""
        ),
        _code(
            """
for identity in ("islam", "christian", "jewish", "atheist"):
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"08_{identity}_general_effects_vs_neutral_descriptive.png", width=1800)
"""
        ),
        _markdown(
            """
## 9. Perbandingan langsung antaridentitas

Enam kontras antaragama dihitung langsung pada condition, model, subset, dan ID soal yang sama. Klaim antaragama tidak diturunkan dari selisih dua grafik terhadap Neutral.
"""
        ),
        _code(
            """
for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3"):
    display(Markdown(f"### Kondisi {CONDITION_LABELS[condition]}"))
    show_supplement(f"09_direct_identity_religion_forest_{condition}_descriptive.png", width=1650)

direct = pd.read_csv(SUPPLEMENT_DIR / "religion_marker_direct_identity_descriptive.csv")
display(
    direct[direct["subset"].eq("all")]
    .assign(abs_delta=lambda d: d["prevalence_delta_pp"].abs())
    .nlargest(24, "abs_delta")
    .drop(columns="abs_delta")
    .style.format({"prevalence_delta_pp": "{:+.2f}"})
)
"""
        ),
        _markdown(
            """
## 10. Heterogenitas antar-subset ETHICS

Perbedaan profil subset dapat berasal dari isi dan format tugas, bukan hanya persona. Figur utama yang mempool empat subset selalu mempunyai tabel pendamping per subset.
"""
        ),
        _code('show_base("07_marker_profile_by_subset_explicit.png", width=1650)'),
        _markdown(
            """
## 11. Sensitivitas terhadap correctness

Sensitivity hanya memakai Commonsense dan Justice. Pasangan difilter setelah pairing dan hanya dipertahankan bila kedua respons benar. Deontology dan Virtue tidak masuk karena keterbatasan konstruk eksperimen v9.
"""
        ),
        _code(
            """
show_supplement("10_correctness_sensitivity_construct_valid_descriptive.png", width=1500)
sensitivity_summary = pd.read_csv(SUPPLEMENT_DIR / "correctness_sensitivity_summary.csv")
display(sensitivity_summary.style.format({
    "median_pair_retention": "{:.1%}",
    "same_direction_rate_supported": "{:.1%}",
    "median_abs_effect_change": "{:.3f}",
}))
"""
        ),
        _markdown(
            """
## 12. Opening, penolakan, dan perbandingan antaridentitas

Opening adalah kategori tunggal per respons tersedia. Komposisi memakai bobot model setara. Perbandingan langsung menggunakan marginal total variation secara deskriptif; refusal tetap menjadi kategori hasil, bukan data yang dibuang.
"""
        ),
        _code(
            """
show_supplement("11_opening_composition_100pct.png", width=1600)
show_supplement("12_direct_identity_opening_descriptive.png", width=1750)
show_accuracy("05_abstain_reason_by_persona_heatmap.png", width=1700)
opening_direct = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_opening_mtv_descriptive.csv")
display(
    opening_direct.nlargest(20, "marginal_total_variation")
    .style.format({"marginal_total_variation": "{:.2%}"})
)
"""
        ),
        _markdown(
            """
## 13. Differential availability

Grafik ini memeriksa apakah teks tersedia berbeda pada pasangan identitas, model, condition, subset, dan ID soal yang sama. Missingness tidak diasumsikan MCAR.
"""
        ),
        _code(
            """
show_supplement("13_direct_identity_availability_descriptive.png", width=1750)
availability = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_availability_descriptive.csv")
display(
    availability.reindex(availability["availability_delta_pp"].abs().sort_values(ascending=False).index)
    .head(24)
    .style.format({"availability_delta_pp": "{:+.2f}"})
)
"""
        ),
        _markdown(
            """
## 14. Accuracy committed dan distribusi jawaban

Commonsense dan Justice menampilkan accuracy committed berdampingan dengan commit rate. Deontology dan Virtue menampilkan distribusi jawaban serta commit rate; angka accuracy keduanya tidak dipakai sebagai kemampuan moral.
"""
        ),
        _code(
            """
for subset in ("commonsense", "justice"):
    show_base(f"09a_{subset}_accuracy_committed_and_commit_rate.png", width=1800)
for subset in ("deontology", "virtue"):
    show_base(f"09b_{subset}_answer_distribution_and_commit_rate.png", width=1800)
"""
        ),
        _markdown(
            """
## 15. Sensitivitas inferensial accuracy penyebut penuh

Bagian ini memakai uji berpasangan accuracy dari bundle tervalidasi, bukan inferensi marker candidate. Interpretasi utama tetap memakai accuracy committed + commit rate; hasil penyebut penuh adalah sensitivity analysis untuk differential commitment.
"""
        ),
        _code(
            """
show_base("11_full_denominator_accuracy_significance_construct_valid.png", width=1250)
for subset in ("commonsense", "justice"):
    display(Markdown(f"### {SUBSET_LABELS[subset]}"))
    show_accuracy(f"03_effect_size_{subset}_holm_heatmap.png", width=1500)
    show_accuracy(f"03b_effect_size_religion_pairwise_{subset}_holm_heatmap.png", width=1800)
"""
        ),
        _markdown(
            """
## 16. Audit marker dan validation gate

Audit AI menyaring seluruh occurrence dan respons, tetapi kolom dua reviewer manusia masih kosong. Karena itu hasil marker tetap kandidat dan deskriptif.
"""
        ),
        _code(
            """
show_base("10_audit_marker_response_coverage.png", width=1250)
audit_summary = pd.DataFrame([
    ("Occurrence screened", audit_manifest["occurrences_reviewed_by_second_pass_screen"]),
    ("Context cluster", audit_manifest["occurrence_clusters_reviewed"]),
    ("Response screened", audit_manifest["responses_reviewed_by_open_anchor_screen"]),
    ("Potential false positive", audit_manifest["potential_false_positive_occurrences"]),
    ("Human review status", audit_manifest["human_review_status"]),
], columns=["Ukuran", "Nilai"])
display(audit_summary)
"""
        ),
        _markdown(
            """
## 17. Indeks figur dan batas interpretasi

1. Tidak ada satu skor agama atau satu angka accuracy ETHICS gabungan.
2. I1, I2, dan I3 tetap terpisah karena membawa warmup berbeda.
3. Marker candidate hanya boleh disebut deskriptif sampai dua reviewer manusia dan inter-rater agreement selesai.
4. Black diamond adalah median enam model, bukan pooled effect.
5. Perbandingan accuracy selalu menyebut coverage; Deontology dan Virtue memakai distribusi jawaban.
6. Correctness sensitivity marker hanya berlaku untuk Commonsense dan Justice.
7. Ground truth hanya dipakai sebagai filter sensitivity setelah marker selesai diekstrak; ia tidak pernah menentukan marker.
"""
        ),
        _code(
            """
base_index = pd.read_csv(BASE_ASSET_DIR / "figure_index.csv").assign(source="ETHICS base v1")
supplement_index = pd.read_csv(SUPPLEMENT_DIR / "figure_index.csv").assign(source="GSM8K-aligned supplement v1.0.0")
display(pd.concat([base_index, supplement_index], ignore_index=True, sort=False))
display(Markdown(
    f"**Notebook source:** `{REPO_ROOT / 'notebooks/religion/linguistic_marker_religion_ethics_bab4_visual_v1_1.ipynb'}`  \\n"
    f"**Visual supplement:** `{SUPPLEMENT_DIR}`  \\n"
    f"**Marker status:** `{marker_manifest['registry_status']}`"
))
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    with output.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"[DONE] Notebook created: {output}")
    return output


def main() -> int:
    args = _parser().parse_args()
    build_notebook(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

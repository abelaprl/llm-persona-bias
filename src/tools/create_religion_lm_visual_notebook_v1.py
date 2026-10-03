"""Create the executed-thesis-facing religion linguistic-marker notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> Path:
    cells = [
        markdown(
            """
# Linguistic Marker Religion Persona pada GSM8K

Notebook ini menyajikan analisis **Islam, Christian, Jewish, dan atheist** pada enam model. Kondisi Explicit (E), Implicit 1 (I1), Implicit 2 (I2), dan Implicit 3 (I3) selalu dipertahankan sebagai konstruk terpisah. Neutral (N) menjadi baseline bersama.

Unit analisis inferensial adalah pasangan respons dari model dan soal yang sama. Teks yang dianalisis hanya jawaban yang terlihat; hidden reasoning tidak digunakan. Nilai antar-model ditampilkan satu per satu, sedangkan berlian hitam merangkum median enam model.
"""
        ),
        code(
            """
from pathlib import Path
import json
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display


def locate_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").is_dir() and (candidate / "result").is_dir():
            return candidate
    raise FileNotFoundError("Repo root D:/TugasAkhir tidak ditemukan dari working directory.")


REPO_ROOT = locate_repo_root()
RUN_DIR = REPO_ROOT / "result/benchmarks/gsm8k/reports/Religion_Persona/linguistic_markers/religion_gsm8k_full_source_aligned_v1_3_20260818"
VISUAL_ROOT = REPO_ROOT / "result/benchmarks/gsm8k/reports/Religion_Persona/linguistic_markers/visual"
EXPECTED_BUNDLE = "religion_gsm8k_bab4_visual_v1_3_0_20260818"
BUNDLE_DIR = VISUAL_ROOT / EXPECTED_BUNDLE
FIG_DIR = BUNDLE_DIR / "images"
AUDIT_DIR = RUN_DIR / "audit_codex_contextual_review_20260818"

if not BUNDLE_DIR.is_dir():
    raise FileNotFoundError(f"Visual bundle tidak ditemukan: {BUNDLE_DIR}")
if not RUN_DIR.is_dir():
    raise FileNotFoundError(f"Source run tidak ditemukan: {RUN_DIR}")
if not AUDIT_DIR.is_dir():
    raise FileNotFoundError(f"Audit bundle tidak ditemukan: {AUDIT_DIR}")

bundle_manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
run_manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
audit_manifest = json.loads((AUDIT_DIR / "audit_manifest.json").read_text(encoding="utf-8"))
if bundle_manifest.get("bundle_version") != "religion-gsm8k-bab4-visual-v1.3.0":
    raise ValueError(f"Bundle version tidak sesuai: {bundle_manifest.get('bundle_version')}")
if bundle_manifest.get("hidden_reasoning_used") is not False:
    raise ValueError("Notebook mensyaratkan visible response surface saja.")

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
IDENTITY_LABELS = {"islam": "Islam", "christian": "Christian", "jewish": "Jewish", "atheist": "Atheist"}
CONDITION_LABELS = {"neutral": "N", "explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}
MARKER_LABELS = {
    "religion_identity_islam": "Identitas Islam",
    "religion_identity_christian": "Identitas Christian",
    "religion_identity_jewish": "Identitas Jewish",
    "religion_identity_atheist": "Identitas atheist",
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Tuhan/deity",
    "religion_scripture_authority": "Kitab/otoritas",
    "religion_practice_community": "Praktik/komunitas",
    "religion_formulaic_expression": "Sapaan/formula agama",
    "religion_identity_irrelevance": "Identitas tidak relevan",
    "religion_blessing_invocation": "Doa/blessing",
    "religion_respect_acknowledgement": "Pengakuan/respect",
}
ALIGNED = {
    "islam": "religion_identity_islam",
    "christian": "religion_identity_christian",
    "jewish": "religion_identity_jewish",
    "atheist": "religion_identity_atheist",
}


def show_figure(filename: str, width: int = 1550) -> None:
    path = FIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))


pd.set_option("display.max_colwidth", 125)
print(f"Source run : {RUN_DIR.name}")
print(f"Visual bundle: {BUNDLE_DIR.name}")
print(f"Respons aktual: {run_manifest['response_records']:,}; gap sumber: {run_manifest['source_gap_records']:,}")
print(f"Audit: {audit_manifest['audit_status']}")
"""
        ),
        markdown(
            """
## 1. Provenance dan cakupan 17 kondisi persona

Setiap sel menampilkan teks tersedia dibanding target 1.319 soal. Data religion berisi 16 kondisi identitas, sedangkan Neutral diambil dari sumber gabungan yang sama dengan analisis gender. Missing source dan respons tanpa teks tetap dilaporkan sebagai gap, bukan diubah menjadi marker bernilai nol.
"""
        ),
        code(
            """
provenance = pd.DataFrame([
    ("Versi pipeline", run_manifest["pipeline_version"]),
    ("Versi visual", bundle_manifest["bundle_version"]),
    ("Jumlah model", len(MODEL_LABELS)),
    ("Respons aktual", f"{run_manifest['response_records']:,}"),
    ("Gap sumber", f"{run_manifest['source_gap_records']:,}"),
    ("Bootstrap", f"{run_manifest['bootstrap_resamples']:,}"),
    ("Precision audit", f"{audit_manifest['precision_sample_hits']:,} hit; {audit_manifest['precision_false_positives']} false positive"),
    ("Recall audit", f"{audit_manifest['manual_positive_recall_responses']}/{audit_manifest['manual_positive_recall_responses']} respons positif tertangkap"),
    ("Hidden reasoning digunakan", bundle_manifest["hidden_reasoning_used"]),
], columns=["Komponen", "Nilai"])
display(provenance)
show_figure("01_coverage_model_persona.png", width=1650)

coverage = pd.read_csv(BUNDLE_DIR / "response_counts_model_persona.csv")
coverage["model"] = coverage["model_id"].map(MODEL_LABELS)
display(
    coverage.sort_values(["text_coverage_rate", "model", "persona"])
    [["model", "persona", "target", "n_text_available", "n_source_gap", "n_missing_text", "text_coverage_rate"]]
    .head(14)
    .style.format({"text_coverage_rate": "{:.2%}"})
)
"""
        ),
        markdown(
            """
## 2. Verbosity sebagai dasar normalisasi

Titik berwarna adalah median jumlah kata tiap model dan garis menunjukkan IQR. Marker frekuensi dalam pipeline juga tersedia sebagai density per 100 kata, sehingga respons panjang tidak otomatis dianggap memiliki kecenderungan linguistik lebih kuat.
"""
        ),
        code('show_figure("02_verbosity_by_identity_condition.png", width=1500)'),
        markdown(
            """
## 3. Apakah model benar-benar menyebut identitas agama?

Ya, tetapi terutama pada kondisi Explicit dan sangat bergantung pada model. Figur pertama hanya menghitung marker identitas yang selaras dengan persona. Ini memperlihatkan bahwa klaim lama “bahasa agama tidak muncul” tidak berlaku pada sumber gabungan enam model saat ini.
"""
        ),
        code(
            """
show_figure("03_aligned_identity_mentions.png", width=1500)

prevalence = pd.read_csv(BUNDLE_DIR / "religion_marker_prevalence_model_persona.csv")
aligned_rows = []
for identity, marker in ALIGNED.items():
    selected = prevalence[
        prevalence["identity"].eq(identity)
        & prevalence["marker_id"].eq(marker)
        & prevalence["condition"].eq("explicit")
    ].copy()
    selected["identity"] = IDENTITY_LABELS[identity]
    selected["model"] = selected["model_id"].map(MODEL_LABELS)
    aligned_rows.append(selected)
aligned_explicit = pd.concat(aligned_rows, ignore_index=True)
display(
    aligned_explicit.sort_values("prevalence_pp", ascending=False)
    [["identity", "model", "n_available", "n_present", "prevalence_pp"]]
    .style.format({"prevalence_pp": "{:.2f}%"})
)
"""
        ),
        markdown(
            """
## 4. Bentuk akomodasi bahasa agama pada Explicit

Selain label identitas, respons dapat memuat penyebutan Tuhan, kitab, praktik, doa atau blessing, pengakuan terhadap persona, dan pernyataan bahwa identitas tidak relevan bagi soal. Panel mempertahankan nilai tiap model agar satu model yang sangat responsif tidak tertutup oleh rerata keseluruhan.
"""
        ),
        code('show_figure("04_religious_accommodation_explicit.png", width=1600)'),
        markdown(
            """
## 5. Kata agama yang benar-benar muncul

Lollipop adalah tampilan kuantitatif: sumbu menunjukkan jumlah kemunculan per 10.000 kata yang dirata-ratakan dengan bobot model setara. Word cloud memakai bobot yang sama, tetapi hanya sebagai ringkasan visual. Keduanya memakai closed lexicon istilah agama; kata seperti `Christian` yang juga merupakan nama tokoh GSM8K hanya dihitung dalam konteks identitas yang dibatasi pola.
"""
        ),
        code(
            """
show_figure("05_religious_terms_ranked_explicit.png", width=1500)
show_figure("06_religious_terms_cloud_explicit.png", width=1500)

terms = pd.read_csv(BUNDLE_DIR / "religion_term_prevalence_model_persona.csv")
term_summary = (
    terms[terms["condition"].eq("explicit")]
    .groupby(["identity", "term"], as_index=False)
    .agg(
        occurrence_per_10k=("occurrences_per_10k_words", "mean"),
        response_prevalence_pp=("response_prevalence_pp", "mean"),
        total_occurrences=("occurrences", "sum"),
    )
)
term_summary = term_summary[term_summary["occurrence_per_10k"].gt(0)]
term_summary["identity"] = term_summary["identity"].map(IDENTITY_LABELS)
display(
    term_summary.sort_values(["identity", "occurrence_per_10k"], ascending=[True, False])
    .groupby("identity", sort=False).head(8)
    .style.format({"occurrence_per_10k": "{:.3f}", "response_prevalence_pp": "{:.3f}%"})
)
"""
        ),
        markdown(
            """
## 6. Efek marker agama terhadap Neutral

Titik adalah perubahan prevalence dalam poin persentase pada pasangan model×soal; garis adalah bootstrap CI 95%. E, I1, I2, dan I3 memakai skala-x bersama di dalam setiap figur sehingga efek Explicit yang besar dan efek implicit yang kecil dapat dibandingkan secara jujur. Transparansi rendah menandai hasil yang tidak lolos Holm.
"""
        ),
        code(
            """
for identity in ("islam", "christian", "jewish", "atheist"):
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_figure(f"07_{identity}_religion_marker_forest_vs_neutral.png", width=1750)

within = pd.read_csv(BUNDLE_DIR / "religion_marker_within_identity_tests.csv")
within["delta_pp"] = 100 * within["prevalence_delta"]
within["ci_low_pp"] = 100 * within["ci_low"]
within["ci_high_pp"] = 100 * within["ci_high"]
within["model"] = within["model_id"].map(MODEL_LABELS)
within["marker"] = within["outcome_id"].map(MARKER_LABELS)
within["identity_label"] = within["identity"].map(IDENTITY_LABELS)
top_within = (
    within[within["contrast_id"].str.endswith("_vs_neutral")]
    .sort_values(["p_holm", "abs_delta_pp"], ascending=[True, False])
    .groupby(["identity", "contrast_id"], sort=False).head(3)
)
display(
    top_within[["identity_label", "contrast_id", "model", "marker", "delta_pp", "ci_low_pp", "ci_high_pp", "p_holm", "n_pairs"]]
    .style.format({"delta_pp": "{:+.2f}", "ci_low_pp": "{:+.2f}", "ci_high_pp": "{:+.2f}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 7. Marker linguistik umum terhadap Neutral

Forest effect map berikut mencakup panjang dan keragaman teks, organisasi langkah, stance, konektif, self-reference, engagement, dan register matematika. Sumbu adalah rank-biserial berpasangan; garis putus-putus pada ±0,3 membantu membedakan efek yang lebih substantif dari hasil yang hanya besar secara statistik.
"""
        ),
        code(
            """
for identity in ("islam", "christian", "jewish", "atheist"):
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_figure(f"08_{identity}_general_effects_vs_neutral.png", width=1750)
"""
        ),
        markdown(
            """
## 8. Perbandingan langsung antaridentitas

Perbedaan Islam–Christian, Islam–Jewish, Islam–atheist, Christian–Jewish, Christian–atheist, dan Jewish–atheist diuji langsung pada kondisi yang sama. Karena itu, klaim perbedaan antaragama tidak diturunkan hanya dari dua grafik identity-vs-Neutral. Koreksi Holm diterapkan sebagai satu family berisi 24 perbandingan per model dan marker.
"""
        ),
        code(
            """
for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3"):
    display(Markdown(f"### Kondisi {CONDITION_LABELS[condition]}"))
    show_figure(f"09_direct_identity_religion_forest_{condition}.png", width=1550)

direct = pd.read_csv(BUNDLE_DIR / "religion_marker_direct_identity_tests.csv")
direct["delta_pp"] = 100 * direct["prevalence_delta"]
direct["ci_low_pp"] = 100 * direct["ci_low"]
direct["ci_high_pp"] = 100 * direct["ci_high"]
direct["model"] = direct["model_id"].map(MODEL_LABELS)
direct["marker"] = direct["outcome_id"].map(MARKER_LABELS)
top_direct = (
    direct.sort_values(["p_holm", "abs_delta_pp"], ascending=[True, False])
    .groupby(["condition", "identity_contrast"], sort=False).head(2)
)
display(
    top_direct[["condition", "identity_contrast", "model", "marker", "delta_pp", "ci_low_pp", "ci_high_pp", "p_holm", "n_pairs"]]
    .style.format({"delta_pp": "{:+.2f}", "ci_low_pp": "{:+.2f}", "ci_high_pp": "{:+.2f}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 9. Sensitivitas terhadap correctness

Subset sensitivitas dibuat setelah pemasangan: hanya pasangan yang kedua responsnya benar yang dipertahankan. Kedekatan titik dengan diagonal menunjukkan seberapa stabil arah dan magnitude effect size ketika respons salah dikeluarkan.
"""
        ),
        code(
            """
show_figure("10_correctness_sensitivity.png", width=1450)
within_sensitivity = pd.read_csv(BUNDLE_DIR / "correctness_sensitivity_within_summary.csv")
direct_sensitivity = pd.read_csv(BUNDLE_DIR / "correctness_sensitivity_direct_summary.csv")
display(Markdown("**Persona vs pembanding**"))
display(within_sensitivity.style.format({
    "median_pair_retention": "{:.1%}",
    "same_direction_rate_supported": "{:.1%}",
    "median_abs_effect_change": "{:.3f}",
}))
display(Markdown("**Perbandingan langsung antaridentitas**"))
display(direct_sensitivity.style.format({
    "median_pair_retention": "{:.1%}",
    "same_direction_rate_supported": "{:.1%}",
    "median_abs_effect_change": "{:.3f}",
}))
"""
        ),
        markdown(
            """
## 10. Opening sebagai komposisi

Setiap respons mempunyai tepat satu opening category, sehingga segmen pada satu batang berjumlah 100%. Neutral yang sama direplikasi secara visual di semua panel identitas. Inferensi pasangan memakai Stuart–Maxwell marginal homogeneity, bukan serangkaian McNemar independen per kategori.

Kategori yang tidak muncul pada GSM8K, termasuk `refusal`, tidak ditampilkan di legenda.
"""
        ),
        code(
            """
show_figure("11_opening_composition_100pct.png", width=1500)
show_figure("12_direct_identity_opening.png", width=1600)

opening_within = pd.read_csv(RUN_DIR / "paired_opening_multinomial.csv")
opening_direct = pd.read_csv(RUN_DIR / "direct_identity_opening_multinomial.csv")
display(Markdown("**Perubahan komposisi terbesar terhadap Neutral**"))
display(
    opening_within[opening_within["contrast_id"].str.endswith("_vs_neutral")]
    .sort_values("marginal_total_variation", ascending=False)
    [["model_id", "identity", "contrast_id", "n_pairs", "marginal_total_variation", "p_holm"]]
    .head(16)
    .style.format({"marginal_total_variation": "{:.2%}", "p_holm": "{:.3g}"})
)
display(Markdown("**Perbedaan komposisi terbesar antaridentitas**"))
display(
    opening_direct.sort_values("marginal_total_variation", ascending=False)
    [["model_id", "identity_contrast", "condition", "n_pairs", "marginal_total_variation", "p_holm"]]
    .head(16)
    .style.format({"marginal_total_variation": "{:.2%}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 11. Differential availability

Grafik ini memeriksa apakah ketersediaan teks berbeda antaridentitas pada soal dan model yang sama. Efek yang terkonsentrasi pada GPT-OSS perlu disebutkan saat membahas missingness; data yang hilang tidak langsung diasumsikan *missing completely at random*.
"""
        ),
        code(
            """
show_figure("13_direct_identity_availability.png", width=1600)
availability = pd.read_csv(RUN_DIR / "direct_identity_availability.csv")
availability = availability[availability["outcome_id"].eq("text_available")].copy()
availability["delta_pp"] = 100 * availability["rate_delta"]
availability["model"] = availability["model_id"].map(MODEL_LABELS)
display(
    availability.reindex(availability["delta_pp"].abs().sort_values(ascending=False).index)
    [["model", "identity_contrast", "condition", "delta_pp", "p_holm", "n_pairs"]]
    .head(20)
    .style.format({"delta_pp": "{:+.2f}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 12. Audit istilah dan contoh konteks

Registry di bawah mendokumentasikan definisi operasional serta aturan eksklusi. Contoh konteks berasal dari sampel precision yang telah diperiksa. Beberapa keputusan penting: `Christian` tidak dihitung sebagai identitas ketika hanya menjadi nama tokoh soal; `church`, `Christmas`, dan `fast` tidak dipakai sebagai sinyal tunggal; `Lent` hanya cocok dalam konteks observance; sapaan/formula agama dibatasi pada bentuk yang benar-benar muncul di korpus; dan nama kitab yang ambigu hanya dihitung jika disertai pola pasal:ayat.
"""
        ),
        code(
            """
registry = pd.read_csv(RUN_DIR / "marker_registry.csv")
religion_registry = registry[registry["marker_id"].str.startswith("religion_")].copy()
religion_registry["marker"] = religion_registry["marker_id"].map(MARKER_LABELS)
display(religion_registry[["marker", "operational_definition", "exclusion_rule", "version", "validation_status"]])

audit_hits = pd.read_csv(AUDIT_DIR / "precision_review.csv")
religion_hits = audit_hits[audit_hits["marker_id"].str.startswith("religion_")].copy()
religion_hits["marker"] = religion_hits["marker_id"].map(MARKER_LABELS)
religion_hits["model"] = religion_hits["model_id"].map(MODEL_LABELS)
examples = (
    religion_hits.sort_values(["marker_id", "model_id", "persona", "question_id"])
    .groupby("marker_id", sort=False).head(2)
    [["marker", "model", "persona", "question_id", "matched_text", "match_context"]]
)
display(examples)

precision_metrics = pd.read_csv(AUDIT_DIR / "precision_metrics.csv")
recall_metrics = pd.read_csv(AUDIT_DIR / "recall_metrics.csv")
display(
    precision_metrics.tail(1).style.format({
        "precision": "{:.1%}",
        "ci95_low_clopper_pearson": "{:.1%}",
        "ci95_high_clopper_pearson": "{:.1%}",
    })
)
display(
    recall_metrics.style.format({
        "recall": "{:.1%}",
        "ci95_low_clopper_pearson": "{:.1%}",
        "ci95_high_clopper_pearson": "{:.1%}",
    })
)
"""
        ),
        markdown(
            """
## 13. Implikasi untuk Bab IV

Bagian metode Bab IV sebaiknya menambahkan hal berikut:

1. empat identitas dan ketiga proxy implicit dianalisis terpisah;
2. perbandingan utama memakai pasangan model×question_id;
3. klaim antaridentitas memakai enam kontras langsung pada setiap kondisi, dengan Holm atas 24 perbandingan;
4. marker frekuensi dilaporkan sebagai raw count dan density per 100 kata, sedangkan marker sparse juga dianalisis sebagai presence;
5. opening diperlakukan sebagai outcome multinomial dan diuji dengan Stuart–Maxwell;
6. analisis utama memakai semua respons, disertai sensitivitas pasangan yang keduanya benar;
7. missingness dan differential availability dilaporkan eksplisit;
8. closed lexicon agama memakai aturan konteks dan audit precision/recall terdokumentasi;
9. sapaan/formula agama dan sitasi nama-kitab+pasal:ayat dimasukkan karena keduanya ditemukan lewat audit recall pada respons nyata.

Untuk Bab Hasil, temuan utama perlu ditulis sebagai **heterogenitas antar-model**: penyebutan identitas dan akomodasi agama kuat pada beberapa model di kondisi Explicit, sementara I1/I2/I3 jauh lebih kecil dan tidak boleh digabung menjadi satu efek implicit.
"""
        ),
        code(
            """
figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1))
display(figure_index)
display(Markdown(f"**Source run:** `{RUN_DIR}`  \\n**Visual bundle:** `{BUNDLE_DIR}`  \\n**Audit bundle:** `{AUDIT_DIR}`  \\n**Notebook:** `{Path.cwd() / 'notebooks' / 'linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb'}`"))
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"[DONE] Notebook created: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    build_notebook()

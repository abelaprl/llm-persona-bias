"""Create the executed thesis-facing regional-cultural marker notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPO_ROOT
    / "notebooks"
    / "linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1.ipynb"
)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> Path:
    cells = [
        markdown(
            """
# Linguistic Marker Regional/Cultural Persona pada GSM8K

Notebook ini menyajikan analisis persona **Amerika Utara, Asia Timur, Asia Selatan, dan Afrika Sub-Sahara** pada enam model. Kondisi Explicit (E), Implicit 1 (I1: nama dan nilai sosial), Implicit 2 (I2: perayaan), dan Implicit 3 (I3: tradisi pernikahan) selalu dipertahankan sebagai konstruk terpisah. Neutral (N) menjadi baseline bersama.

Unit inferensial adalah pasangan respons dari model dan soal yang sama. Analisis hanya memakai jawaban yang terlihat, bukan hidden reasoning. Titik berwarna menunjukkan tiap model dan berlian hitam menunjukkan median enam model.
"""
        ),
        code(
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
RUN_DIR = REPO_ROOT / "result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/regional_gsm8k_full_source_aligned_v1_1_20260818"
BUNDLE_DIR = REPO_ROOT / "result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/visual/regional_gsm8k_bab4_visual_v1_1_0_20260818"
FIG_DIR = BUNDLE_DIR / "images"
AUDIT_DIR = RUN_DIR / "audit_codex_contextual_review_20260818"

for required in (RUN_DIR, BUNDLE_DIR, FIG_DIR, AUDIT_DIR):
    if not required.is_dir():
        raise FileNotFoundError(required)

run_manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
bundle_manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
audit_manifest = json.loads((AUDIT_DIR / "audit_manifest.json").read_text(encoding="utf-8"))
if bundle_manifest.get("bundle_version") != "regional-gsm8k-bab4-visual-v1.1.0":
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
IDENTITIES = ("north_american", "east_asian", "south_asian", "sub_saharan_african")
IDENTITY_LABELS = {
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
CONDITION_LABELS = {
    "neutral": "N", "explicit": "E", "implicit_1": "I1",
    "implicit_2": "I2", "implicit_3": "I3",
}
MARKER_LABELS = {
    "culture_region_north_american": "Label Amerika Utara",
    "culture_region_east_asian": "Label Asia Timur",
    "culture_region_south_asian": "Label Asia Selatan",
    "culture_region_sub_saharan_african": "Label Afrika Sub-Sahara",
    "culture_general_reference": "Rujukan budaya umum",
    "culture_regional_address": "Sapaan kepada asal kawasan",
    "culture_identity_mirroring": "Mirroring identitas kawasan",
    "culture_geographic_greeting": "Sapaan geografis umum",
    "culture_formulaic_greeting": "Sapaan budaya (Namaste)",
    "culture_named_celebration": "Perayaan bernama",
    "culture_named_tradition": "Tradisi/objek bernama",
    "culture_warmup_name_echo": "Nama warmup terbawa",
    "culture_meta_framing": "Framing budaya eksplisit",
    "culture_identity_irrelevance": "Identitas tidak relevan",
}
ALIGNED = {
    "north_american": "culture_region_north_american",
    "east_asian": "culture_region_east_asian",
    "south_asian": "culture_region_south_asian",
    "sub_saharan_african": "culture_region_sub_saharan_african",
}


def show_figure(filename: str, width: int = 1550) -> None:
    path = FIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))


pd.set_option("display.max_colwidth", 125)
print(f"Source run   : {RUN_DIR.name}")
print(f"Visual bundle: {BUNDLE_DIR.name}")
print(f"Respons aktual: {run_manifest['response_records']:,}; gap sumber: {run_manifest['source_gap_records']:,}")
print(f"Audit: {audit_manifest['audit_status']}")
"""
        ),
        markdown(
            """
## 1. Provenance dan cakupan 17 kondisi persona

Regional/Cultural menyediakan 16 kondisi persona dan satu Neutral bersama. Setiap sel menunjukkan teks yang tersedia dibanding target 1.319 soal. Missing source dan respons tanpa teks tetap dicatat sebagai gap, bukan diubah menjadi marker bernilai nol.
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
    ("Marker terdaftar", len(pd.read_csv(RUN_DIR / "marker_registry.csv"))),
    ("Precision audit", f"{audit_manifest['precision_sample_hits']:,} hit; {audit_manifest['precision_false_positives']} false positive"),
    ("Blind recall", f"{audit_manifest['manual_positive_recall_responses']} respons positif; {audit_manifest['missed_positive_recall_responses']} terlewat"),
    ("Hidden reasoning digunakan", bundle_manifest["hidden_reasoning_used"]),
], columns=["Komponen", "Nilai"])
display(provenance)
show_figure("01_coverage_model_persona.png", width=1700)

coverage = pd.read_csv(BUNDLE_DIR / "response_counts_model_persona.csv")
coverage["model"] = coverage["model_id"].map(MODEL_LABELS)
display(
    coverage.sort_values(["text_coverage_rate", "model", "persona"])
    [["model", "persona", "target", "n_text_available", "n_source_gap", "n_missing_text", "text_coverage_rate"]]
    .head(18).style.format({"text_coverage_rate": "{:.2%}"})
)
"""
        ),
        markdown(
            """
## 2. Verbosity dan normalisasi panjang respons

Titik menunjukkan median jumlah kata tiap model dan garis menunjukkan IQR. Marker frekuensi juga tersedia sebagai density per 100 kata, sehingga respons yang lebih panjang tidak otomatis dianggap lebih kuat secara linguistik.
"""
        ),
        code('show_figure("02_verbosity_by_identity_condition.png", width=1550)'),
        markdown(
            """
## 3. Penyebutan identitas kawasan yang selaras

Panel ini menghitung label lengkap yang selaras dengan persona, misalnya `East Asia` untuk Asia Timur. Label lengkap dipilih agar kata seperti `American`, `Chinese`, `Indian`, atau `African` dalam isi soal GSM8K tidak menjadi false positive.
"""
        ),
        code(
            """
show_figure("03_aligned_identity_mentions.png", width=1500)

prevalence = pd.read_csv(BUNDLE_DIR / "regional_marker_prevalence_model_persona.csv")
aligned_rows = []
for identity, marker in ALIGNED.items():
    selected = prevalence[
        prevalence["identity"].eq(identity)
        & prevalence["marker_id"].eq(marker)
        & prevalence["condition"].eq("explicit")
    ].copy()
    selected["identity_label"] = IDENTITY_LABELS[identity]
    selected["model"] = selected["model_id"].map(MODEL_LABELS)
    aligned_rows.append(selected)
aligned_explicit = pd.concat(aligned_rows, ignore_index=True)
display(
    aligned_explicit.sort_values("prevalence_pp", ascending=False)
    [["identity_label", "model", "n_available", "n_present", "prevalence_pp"]]
    .style.format({"prevalence_pp": "{:.2f}%"})
)
"""
        ),
        markdown(
            """
## 4. Bentuk akomodasi bahasa regional-budaya pada Explicit

Selain label kawasan, respons dapat menyapa asal pengguna, melakukan mirroring, memakai sapaan geografis umum, menyebut budaya/tradisi, membawa nama atau istilah dari warmup, atau memberi framing budaya pada soal. Nilai tiap model dipertahankan agar responsivitas satu model tidak tertutup rerata agregat.
"""
        ),
        code('show_figure("04_regional_accommodation_explicit.png", width=1650)'),
        markdown(
            """
## 5. Istilah regional-budaya yang benar-benar muncul

Lollipop adalah tampilan kuantitatif per 10.000 kata dengan bobot model setara. Word cloud memakai bobot yang sama hanya sebagai ringkasan visual. Keduanya **bukan kumpulan semua kata yang sering dipakai**: hanya closed lexicon regional-budaya pada respons E, I1, I2, dan I3, termasuk label kawasan, `Namaste`, `Diwali`, `hongbao`, `mehendi`, `diyas`, serta sapaan geografis yang ditemukan saat audit.
"""
        ),
        code(
            """
show_figure("05_regional_terms_ranked.png", width=1550)
show_figure("06_regional_terms_cloud.png", width=1550)

terms = pd.read_csv(BUNDLE_DIR / "regional_term_prevalence_model_persona.csv")
term_summary = (
    terms[~terms["condition"].eq("neutral")]
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
    .groupby("identity", sort=False).head(10)
    .style.format({"occurrence_per_10k": "{:.3f}", "response_prevalence_pp": "{:.3f}%"})
)
"""
        ),
        markdown(
            """
## 6. Efek marker budaya terhadap Neutral

Titik adalah perubahan prevalence pada pasangan model×soal; garis adalah bootstrap CI 95%. E, I1, I2, dan I3 memakai skala yang sama di dalam setiap figur. Transparansi rendah menandai hasil yang tidak lolos koreksi Holm.
"""
        ),
        code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_figure(f"07_{identity}_regional_marker_forest_vs_neutral.png", width=1750)

within = pd.read_csv(BUNDLE_DIR / "regional_marker_within_identity_tests.csv")
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

Forest effect map mencakup panjang dan keragaman teks, organisasi langkah, stance, konektif, self-reference, engagement, dan register matematika. Sumbu adalah rank-biserial berpasangan; garis referensi ±0,3 membantu membaca magnitude, bukan hanya signifikansi pada N besar.
"""
        ),
        code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_figure(f"08_{identity}_general_effects_vs_neutral.png", width=1750)
"""
        ),
        markdown(
            """
## 8. Perbandingan langsung antarkawasan

Enam pasangan kawasan diuji langsung pada kondisi yang sama. Karena itu, klaim perbedaan regional tidak diturunkan hanya dari dua grafik kawasan-vs-Neutral. Koreksi Holm diterapkan sebagai satu family berisi 24 perbandingan per model dan marker.
"""
        ),
        code(
            """
for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3"):
    display(Markdown(f"### Kondisi {CONDITION_LABELS[condition]}"))
    show_figure(f"09_direct_identity_regional_forest_{condition}.png", width=1600)

direct = pd.read_csv(BUNDLE_DIR / "regional_marker_direct_identity_tests.csv")
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

Subset sensitivitas dibentuk setelah pemasangan: hanya pasangan yang kedua responsnya benar yang dipertahankan. Kedekatan titik dengan diagonal menunjukkan stabilitas arah dan magnitude ketika respons salah dikeluarkan.
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
display(Markdown("**Perbandingan langsung antarkawasan**"))
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

Setiap respons mempunyai tepat satu opening category, sehingga segmen pada satu batang berjumlah 100%. Neutral yang sama direplikasi di setiap panel. Inferensi pasangan memakai Stuart–Maxwell marginal homogeneity, bukan McNemar independen per kategori.
"""
        ),
        code(
            """
show_figure("11_opening_composition_100pct.png", width=1550)
show_figure("12_direct_identity_opening.png", width=1600)

opening_within = pd.read_csv(RUN_DIR / "paired_opening_multinomial.csv")
opening_direct = pd.read_csv(RUN_DIR / "direct_identity_opening_multinomial.csv")
display(Markdown("**Perubahan komposisi terbesar terhadap Neutral**"))
display(
    opening_within[opening_within["contrast_id"].str.endswith("_vs_neutral")]
    .sort_values("marginal_total_variation", ascending=False)
    [["model_id", "identity", "contrast_id", "n_pairs", "marginal_total_variation", "p_holm"]]
    .head(16).style.format({"marginal_total_variation": "{:.2%}", "p_holm": "{:.3g}"})
)
display(Markdown("**Perbedaan komposisi terbesar antarkawasan**"))
display(
    opening_direct.sort_values("marginal_total_variation", ascending=False)
    [["model_id", "identity_contrast", "condition", "n_pairs", "marginal_total_variation", "p_holm"]]
    .head(16).style.format({"marginal_total_variation": "{:.2%}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 11. Differential availability

Grafik memeriksa apakah ketersediaan teks berbeda antarkawasan pada soal dan model yang sama. Gap terutama terkonsentrasi pada GPT-OSS dan kondisi implicit, sehingga missingness tidak langsung diasumsikan *missing completely at random*.
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
    .head(20).style.format({"delta_pp": "{:+.2f}", "p_holm": "{:.3g}"})
)
"""
        ),
        markdown(
            """
## 12. Audit marker dan contoh konteks

Registry mendokumentasikan definisi operasional dan aturan eksklusi. Audit precision meninjau sampel hit dalam konteks. Blind recall memakai screen yang lebih lebar—termasuk `family`, `values`, `collective`, dan `customs`—kemudian menolak pemakaian matematis atau isi soal. Audit pertama menemukan sapaan geografis yang terlewat; marker itu ditambahkan dan seluruh pipeline dijalankan ulang sebagai v1.1.
"""
        ),
        code(
            """
registry = pd.read_csv(RUN_DIR / "marker_registry.csv")
regional_registry = registry[registry["marker_id"].str.startswith("culture_")].copy()
regional_registry["marker"] = regional_registry["marker_id"].map(MARKER_LABELS)
display(regional_registry[["marker", "operational_definition", "exclusion_rule", "version", "validation_status"]])

audit_hits = pd.read_csv(AUDIT_DIR / "precision_review.csv")
audit_hits["marker"] = audit_hits["marker_id"].map(MARKER_LABELS)
audit_hits["model"] = audit_hits["model_id"].map(MODEL_LABELS)
examples = (
    audit_hits.sort_values(["marker_id", "model_id", "persona", "question_id"])
    .groupby("marker_id", sort=False).head(2)
    [["marker", "model", "persona", "question_id", "matched_text", "match_context"]]
)
display(examples)

precision_metrics = pd.read_csv(AUDIT_DIR / "precision_metrics.csv")
recall_metrics = pd.read_csv(AUDIT_DIR / "recall_metrics.csv")
iteration_history = pd.read_csv(AUDIT_DIR / "audit_iteration_history.csv")
display(precision_metrics.tail(1).style.format({
    "precision": "{:.1%}", "ci95_low_clopper_pearson": "{:.1%}",
    "ci95_high_clopper_pearson": "{:.1%}",
}))
display(recall_metrics.style.format({
    "recall": "{:.1%}", "ci95_low_clopper_pearson": "{:.1%}",
    "ci95_high_clopper_pearson": "{:.1%}",
}))
display(iteration_history)
"""
        ),
        markdown(
            """
## 13. Implikasi untuk Bab IV dan Bab Hasil

Bagian metode Bab IV sebaiknya menegaskan:

1. empat kawasan serta I1, I2, dan I3 dianalisis terpisah;
2. pasangan utama dibentuk dari model×`question_id` yang sama;
3. klaim antarkawasan memakai enam kontras langsung per kondisi dengan Holm atas 24 perbandingan;
4. label kawasan harus lengkap; demonym dan country term yang bertabrakan dengan isi GSM8K dikeluarkan;
5. marker frekuensi tersedia sebagai count dan density per 100 kata, sedangkan marker sparse diuji sebagai presence;
6. opening diperlakukan sebagai outcome multinomial dengan Stuart–Maxwell;
7. analisis utama memakai semua respons dan dilengkapi sensitivity subset pasangan yang keduanya benar;
8. differential availability dilaporkan eksplisit;
9. closed lexicon dan riwayat perbaikan blind-recall didokumentasikan.

Untuk Bab Hasil, temuan perlu ditulis sebagai heterogenitas antar-model. Label kawasan dan sapaan budaya sangat kuat pada model tertentu di Explicit, sedangkan I1/I2/I3 jauh lebih jarang dan tidak boleh digabung menjadi satu efek implicit.
"""
        ),
        code(
            """
figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(
    lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1)
)
display(figure_index)
display(Markdown(
    f"**Source run:** `{RUN_DIR}`<br>"
    f"**Visual bundle:** `{BUNDLE_DIR}`<br>"
    f"**Audit bundle:** `{AUDIT_DIR}`<br>"
    f"**Notebook:** `{REPO_ROOT / 'notebooks' / 'linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1.ipynb'}`"
))
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"[DONE] Notebook created: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    build_notebook()

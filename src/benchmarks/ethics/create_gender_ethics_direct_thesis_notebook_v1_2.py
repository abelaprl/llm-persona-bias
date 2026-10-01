"""Create the thesis-facing ETHICS Gender direct-comparison notebook v1.2."""

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
# ETHICS Gender Persona — Perbandingan Langsung Man dan Woman v1.2

Notebook ini menjawab pertanyaan utama pada level **soal yang sama**: apakah persona Man dan Woman menghasilkan perbedaan cara menjawab ETHICS, seberapa besar arahnya, dan apakah pola itu konsisten pada enam model.

Arah seluruh kontras adalah **Man − Woman**. Nilai positif berarti lebih tinggi pada Man dan nilai negatif berarti lebih tinggi pada Woman. Commonsense, Deontology, Justice, dan Virtue tidak digabung; E, I1, I2, dan I3 juga tidak dirata-ratakan.
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
REPORT_ROOT = REPO_ROOT / "result/ETHICS/reports/Gender_Persona"
BUNDLE_DIR = REPORT_ROOT / "linguistic_markers/visual/gender_ethics_direct_thesis_visuals_v1_2_0_r4_20260825"
FIG_DIR = BUNDLE_DIR / "images"
MARKER_DIR = REPORT_ROOT / "linguistic_markers_gender_candidate_20260821_v1_6_0"

manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
marker_manifest = json.loads((MARKER_DIR / "manifest.json").read_text(encoding="utf-8"))

assert manifest["bundle_version"] == "gender-ethics-direct-thesis-visuals-v1.2.0"
assert manifest["source_input_sha256"] == "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"
assert manifest["rows"] == marker_manifest["rows_input"] == 81_756
assert manifest["visible_rows"] == marker_manifest["responses_text_available"] == 80_840
assert manifest["pair_direction"] == "man_minus_woman"
assert manifest["marker_inference_run"] is False
assert manifest["hidden_reasoning_used"] is False
assert manifest["validations"]["term_reconstruction_all_match"] is True
assert manifest["validations"]["implicit_proxies_aggregated"] is False

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
CONDITION_LABELS = {"explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}
SUBSET_LABELS = {"commonsense": "Commonsense", "deontology": "Deontology", "justice": "Justice", "virtue": "Virtue"}


def show_figure(filename: str, width: int = 1750) -> None:
    path = FIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))


pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 120)
print(f"Bundle     : {manifest['bundle_version']}")
print(f"Respons    : {manifest['rows']:,}")
print(f"Visible    : {manifest['visible_rows']:,}")
print(f"Pair key   : {' × '.join(manifest['pair_key'])}")
print(f"Arah efek  : {manifest['pair_direction']}")
"""
        ),
        _markdown(
            """
## 1. Apakah notebook ini sudah menjawab kebutuhan analisis?

Unit analisis adalah satu respons dan unit pembanding adalah pasangan Man–Woman untuk `model_id × subset × question_id`. Missing atau technical-only tidak diubah menjadi nol marker. Respons abstain yang masih memiliki penjelasan tetap masuk analisis linguistic marker.
"""
        ),
        _code(
            """
readiness = pd.DataFrame([
    ("Man dibandingkan langsung dengan Woman", "Ya", "Empat kontras: E, I1, I2, I3"),
    ("Pasangan berasal dari soal yang sama", "Ya", "model_id × subset × question_id"),
    ("Enam model tetap terlihat", "Ya", f"{manifest['models']} model"),
    ("Empat subset ETHICS tetap terpisah", "Ya", ", ".join(manifest["subsets"])),
    ("I1/I2/I3 tidak diagregasi", "Ya", str(not manifest["validations"]["implicit_proxies_aggregated"])),
    ("Kata gender dapat dilacak ke Man/Woman", "Ya", "count dan prevalence kedua sisi tersimpan"),
    ("Term dari skenario dipisahkan", "Ya", "response-only = term tidak ada pada scenario"),
    ("Hidden reasoning digunakan", "Tidak", str(manifest["hidden_reasoning_used"])),
    ("Inferensi marker sudah boleh diklaim", "Belum", "registry masih candidate"),
], columns=["Kebutuhan", "Status", "Bukti"])
display(readiness)

validation = pd.DataFrame(
    [(key, value) for key, value in manifest["validations"].items()],
    columns=["Validation gate", "Nilai"],
)
display(validation)
"""
        ),
        _markdown(
            """
## 2. Provenance dan cakupan data

Closed gender lexicon direkonstruksi pada visible response dan dicocokkan kembali dengan coverage v1.6. Equality diperiksa per term untuk jumlah occurrence dan jumlah respons, bukan hanya pada total korpus.
"""
        ),
        _code(
            """
term_validation = pd.read_csv(BUNDLE_DIR / "gender_term_surface_reconstruction_validation.csv")
provenance = pd.DataFrame([
    ("Pipeline marker", manifest["source_pipeline_version"]),
    ("SHA-256 sumber", manifest["source_input_sha256"]),
    ("Respons unik", f"{manifest['rows']:,}"),
    ("Visible response", f"{manifest['visible_rows']:,}"),
    ("Abstain disertakan", f"{marker_manifest['responses_abstain']:,}"),
    ("Missing", f"{marker_manifest['responses_missing']:,}"),
    ("Pertanyaan unik", f"{manifest['questions']:,}"),
    ("Term cocok per occurrence", int(term_validation["occurrences_match"].sum())),
    ("Term cocok per response", int(term_validation["responses_match"].sum())),
], columns=["Komponen", "Nilai"])
display(provenance)
assert term_validation[["occurrences_match", "responses_match"]].all().all()
"""
        ),
        _markdown(
            """
## 3. Verbosity: Man dibandingkan langsung dengan Woman

Gambar lama menunjukkan nilai absolut sembilan persona, tetapi belum menjadikan Man–Woman sebagai estimand. Tiga gambar berikut memakai median selisih berpasangan pada soal yang sama dan memisahkan model, subset, serta E/I1/I2/I3.
"""
        ),
        _code(
            """
for filename in [
    "01_direct_verbosity_word_count.png",
    "02_direct_verbosity_avg_sentence_length.png",
    "03_direct_verbosity_ttr_300.png",
]:
    show_figure(filename, width=1800)

direct_general = pd.read_csv(BUNDLE_DIR / "general_marker_direct_man_woman_descriptive.csv")
surface = direct_general[
    direct_general["outcome_family"].eq("surface") & ~direct_general["subset"].eq("all")
].copy()
surface["abs_median_delta"] = surface["median_delta"].abs()
display(
    surface.nlargest(20, "abs_median_delta")[[
        "model_id", "condition", "subset", "outcome_id", "n_pairs", "median_delta", "rank_biserial"
    ]].assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
        subset=lambda d: d["subset"].map(SUBSET_LABELS),
    )[["model", "condition", "subset", "outcome_id", "n_pairs", "median_delta", "rank_biserial"]]
    .style.format({"median_delta": "{:+.3f}", "rank_biserial": "{:+.3f}"})
)
"""
        ),
        _markdown(
            """
## 4. Kata gender: berasal dari jawaban Man atau Woman?

Titik biru menunjukkan jawaban Man dan titik magenta menunjukkan jawaban Woman. Term **response-only** berarti exact term ditemukan pada visible response tetapi tidak ditemukan pada skenario soal. Ini membuat asal sisi Man/Woman eksplisit sekaligus mengurangi salinan langsung dari skenario.

Gambar pertama menunjukkan prevalensi absolut kedua sisi. Empat heatmap berikut menunjukkan selisih berpasangan Man − Woman per model dan subset.
"""
        ),
        _code(
            """
show_figure("04_gender_words_absolute_man_woman.png", width=1700)
for number, condition in zip(range(5, 9), ["explicit", "implicit_1", "implicit_2", "implicit_3"]):
    display(Markdown(f"### {CONDITION_LABELS[condition]}"))
    show_figure(f"{number:02d}_direct_gender_words_{condition}.png", width=1800)

direct_terms = pd.read_csv(BUNDLE_DIR / "gender_term_direct_man_woman_response_only.csv")
selected_terms = manifest["selected_gender_terms"]
term_focus = direct_terms[
    direct_terms["term"].isin(selected_terms) & ~direct_terms["subset"].eq("all")
].copy()
term_focus["abs_delta"] = term_focus["prevalence_delta_pp"].abs()
display(
    term_focus.nlargest(30, "abs_delta")[[
        "model_id", "condition", "subset", "term", "n_pairs",
        "man_responses_with_response_only_term", "woman_responses_with_response_only_term",
        "man_prevalence_pct", "woman_prevalence_pct", "prevalence_delta_pp",
        "man_raw_occurrences", "woman_raw_occurrences",
    ]].assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
        subset=lambda d: d["subset"].map(SUBSET_LABELS),
    ).drop(columns="model_id")
    .style.format({
        "man_prevalence_pct": "{:.2f}%",
        "woman_prevalence_pct": "{:.2f}%",
        "prevalence_delta_pp": "{:+.2f}",
    })
)
"""
        ),
        _markdown(
            """
## 5. Marker gender: direct Man–Woman per subset

Marker di bagian ini lebih sempit daripada daftar kata: sapaan persona, acknowledgment, rujukan identitas, adopsi perspektif, irrelevance, neutralization, discourse, role framing, dan evaluasi normatif. Titik berwarna adalah model; wajik hitam adalah median enam model.
"""
        ),
        _code(
            """
for number, condition in zip(range(13, 17), ["explicit", "implicit_1", "implicit_2", "implicit_3"]):
    display(Markdown(f"### {CONDITION_LABELS[condition]}"))
    show_figure(f"{number:02d}_direct_gender_markers_{condition}.png", width=1850)

direct_gender = pd.read_csv(BUNDLE_DIR / "gender_marker_direct_man_woman_by_subset_descriptive.csv")
gender_rank = direct_gender[~direct_gender["subset"].eq("all")].copy()
gender_rank["abs_delta"] = gender_rank["prevalence_delta_pp"].abs()
display(
    gender_rank.nlargest(24, "abs_delta")[[
        "model_id", "condition", "subset", "outcome_id", "n_pairs",
        "man_prevalence_pct", "woman_prevalence_pct", "prevalence_delta_pp",
        "man_raw_positive_responses", "woman_raw_positive_responses",
    ]].assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
        subset=lambda d: d["subset"].map(SUBSET_LABELS),
    ).drop(columns="model_id")
    .style.format({
        "man_prevalence_pct": "{:.2f}%", "woman_prevalence_pct": "{:.2f}%",
        "prevalence_delta_pp": "{:+.2f}",
    })
)
"""
        ),
        _markdown(
            """
## 6. Marker linguistik umum: direct Man–Woman per subset

Outcome adalah density per 100 kata. Effect size deskriptif adalah paired rank-biserial pada soal yang sama; marker candidate belum digunakan untuk klaim inferensial.
"""
        ),
        _code(
            """
for number, condition in zip(range(9, 13), ["explicit", "implicit_1", "implicit_2", "implicit_3"]):
    display(Markdown(f"### {CONDITION_LABELS[condition]}"))
    show_figure(f"{number:02d}_direct_general_markers_{condition}.png", width=1900)

marker_effects = direct_general[
    direct_general["outcome_family"].eq("general_marker") & ~direct_general["subset"].eq("all")
].copy()
marker_effects["abs_effect"] = marker_effects["rank_biserial"].abs()
display(
    marker_effects.nlargest(30, "abs_effect")[[
        "model_id", "condition", "subset", "outcome_id", "n_pairs", "n_nonzero",
        "mean_delta", "median_delta", "rank_biserial",
    ]].assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
        subset=lambda d: d["subset"].map(SUBSET_LABELS),
    ).drop(columns="model_id")
    .style.format({"mean_delta": "{:+.3f}", "median_delta": "{:+.3f}", "rank_biserial": "{:+.3f}"})
)
"""
        ),
        _markdown(
            """
## 7. Commit rate, accuracy committed, dan distribusi jawaban

Commit rate ditampilkan langsung sebagai Man − Woman. Untuk Commonsense dan Justice, akurasi dihitung pada pasangan yang kedua responsnya committed dan selalu diletakkan di samping paired coverage. Untuk Deontology dan Virtue, pembanding utama adalah distribusi jawaban wrong serta commit rate, bukan akurasi.
"""
        ),
        _code(
            """
show_figure("17_direct_commit_rate_man_woman.png", width=1750)
show_figure("18_direct_accuracy_committed_with_coverage.png", width=1750)
show_figure("19_direct_answer_distribution_with_coverage.png", width=1750)

outcomes = pd.read_csv(BUNDLE_DIR / "response_outcomes_direct_man_woman_by_subset.csv")

required_rows = []
for row in outcomes.itertuples(index=False):
    for identity in ("man", "woman"):
        required_rows.append({
            "model_id": row.model_id,
            "condition": row.condition,
            "subset": row.subset,
            "identity": identity.title(),
            "total": row.total_pairs,
            "committed": getattr(row, f"{identity}_committed"),
            "commit_rate": getattr(row, f"{identity}_commit_rate"),
            "abstain": getattr(row, f"{identity}_abstain"),
            "abstain_rate": getattr(row, f"{identity}_abstain_rate"),
            "missing": getattr(row, f"{identity}_missing"),
            "missing_rate": getattr(row, f"{identity}_missing_rate"),
            "accuracy_committed": getattr(row, f"{identity}_accuracy_committed", np.nan),
            "pct_answer_wrong": getattr(row, f"{identity}_pct_answer_wrong", np.nan),
        })
required_report = pd.DataFrame(required_rows)
assert {"total", "committed", "commit_rate", "abstain", "abstain_rate", "missing", "missing_rate", "accuracy_committed"}.issubset(required_report)

largest_commit = outcomes.reindex(outcomes["commit_rate_delta_pp"].abs().sort_values(ascending=False).index).head(16)
display(
    largest_commit[[
        "model_id", "condition", "subset", "total_pairs",
        "man_committed", "man_commit_rate", "woman_committed", "woman_commit_rate",
        "commit_rate_delta_pp", "man_abstain_rate", "woman_abstain_rate",
        "man_missing_rate", "woman_missing_rate",
    ]].assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
        subset=lambda d: d["subset"].map(SUBSET_LABELS),
    ).drop(columns="model_id")
    .style.format({
        "man_commit_rate": "{:.2%}", "woman_commit_rate": "{:.2%}",
        "commit_rate_delta_pp": "{:+.2f}", "man_abstain_rate": "{:.2%}",
        "woman_abstain_rate": "{:.2%}", "man_missing_rate": "{:.2%}",
        "woman_missing_rate": "{:.2%}",
    })
)
"""
        ),
        _markdown(
            """
## 8. Komposisi opening

Marginal total variation mengukur besar perbedaan komposisi kategori opening antara Man dan Woman. Nilainya tidak memiliki arah Man/Woman; nol berarti komposisi identik dan nilai lebih besar berarti komposisi makin berbeda.
"""
        ),
        _code(
            """
show_figure("20_direct_opening_composition_man_woman.png", width=1300)
opening = pd.read_csv(BUNDLE_DIR / "opening_direct_man_woman_descriptive.csv")
display(
    opening.sort_values("marginal_total_variation", ascending=False).head(18)
    .assign(
        model=lambda d: d["model_id"].map(MODEL_LABELS),
        condition=lambda d: d["condition"].map(CONDITION_LABELS),
    )[["model", "condition", "n_man", "n_woman", "marginal_total_variation"]]
    .style.format({"marginal_total_variation": "{:.2%}"})
)
"""
        ),
        _markdown(
            """
## 9. Sintesis untuk penulisan tugas akhir

Notebook ini sudah dapat menjawab **arah dan besar perbedaan deskriptif Man–Woman** per model, subset, dan kondisi pada soal yang sama. Pola tidak boleh diringkas menjadi satu angka “efek gender” karena arah dan besar efek berbeda menurut model, subset, serta proxy.

Klaim inferensial linguistic marker masih ditahan karena registry v1.6 berstatus candidate sampai audit manusia independen selesai. Tabel exact tetap dapat digunakan untuk memilih pola yang akan dibahas dan contoh respons yang akan diaudit.
"""
        ),
        _code(
            """
summary = pd.DataFrame([
    ("Efek utama", "Direct Man − Woman pada soal yang sama"),
    ("Heterogenitas", "Dipertahankan per model, subset, dan E/I1/I2/I3"),
    ("Verbosity", "3 outcome surface; median delta + rank-biserial"),
    ("Kata gender", f"{len(manifest['selected_gender_terms'])} term utama di visual; seluruh closed lexicon di tabel"),
    ("Marker gender", "Prevalence Man, Woman, delta pp, dan raw positive responses"),
    ("Marker umum", "Raw density per 100 kata; paired rank-biserial"),
    ("Accuracy", "Commonsense/Justice pada both-committed pairs + coverage"),
    ("Deontology/Virtue", "Distribusi jawaban + commit rate"),
    ("Status ilmiah", "Deskriptif; inferensi marker menunggu audit manusia"),
], columns=["Aspek", "Keputusan"])
display(summary)

figure_index = pd.read_csv(BUNDLE_DIR / "figure_index.csv")
display(figure_index)
print(f"Bundle visual : {BUNDLE_DIR}")
print(f"Tabel utama   : {BUNDLE_DIR / 'general_marker_direct_man_woman_descriptive.csv'}")
print(f"Term detail   : {BUNDLE_DIR / 'gender_term_direct_man_woman_response_only.csv'}")
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "TA .venv", "language": "python", "name": "ta-venv"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
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

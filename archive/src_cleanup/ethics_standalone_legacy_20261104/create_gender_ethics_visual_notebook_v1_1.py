"""Create the GSM8K-aligned ETHICS Gender visual notebook v1.1."""

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
# Linguistic Marker Gender Persona pada ETHICS — Visualisasi v1.1

Notebook ini memperluas analisis Gender ETHICS v1 dengan bentuk visual Gender GSM8K: effect map terhadap Neutral, direct Man–Woman per proxy, correctness sensitivity, opening multinomial, differential availability, serta ranked gender words dan pronouns.

Seluruh **81.756 respons** dari enam model, sembilan persona, 1.514 pertanyaan, dan empat subset tetap berada dalam inventaris. Marker dihitung dari visible response, termasuk abstain yang masih memiliki teks. Missing dan technical-only tidak dijadikan marker nol. Registry masih candidate, sehingga seluruh efek linguistic marker bersifat deskriptif tanpa p-value atau confidence interval baru.
"""
        ),
        _code(
            """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display


def locate_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").is_dir() and (candidate / "result").is_dir():
            return candidate
    raise FileNotFoundError("Repo root D:/TugasAkhir tidak ditemukan.")


REPO_ROOT = locate_repo_root()
REPORT_ROOT = REPO_ROOT / "result/benchmarks/ethics/reports/Gender_Persona"
MARKER_DIR = REPORT_ROOT / "linguistic_markers_gender_candidate_20260821_v1_6_0"
AUDIT_DIR = MARKER_DIR / "audit_codex_r1_20260822_final"
LEGACY_DIR = REPORT_ROOT / "notebook_ethics_gender_lm_v1_assets"
SUPPLEMENT_DIR = REPORT_ROOT / "linguistic_markers/visual/gender_ethics_bab4_visual_supplement_v1_0_0_20260825"
FIG_DIR = SUPPLEMENT_DIR / "images"
ACCURACY_DIR = REPORT_ROOT / "_combined/persona_publication_analysis_v3_accuracy_paired_holm_six_models"
ACCURACY_FIG_DIR = ACCURACY_DIR / "figures"

required = [
    MARKER_DIR / "manifest.json",
    AUDIT_DIR / "manifest.json",
    LEGACY_DIR / "artifact_index.csv",
    SUPPLEMENT_DIR / "manifest.json",
    ACCURACY_DIR / "metadata/validation.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Artefak wajib tidak ditemukan:\\n" + "\\n".join(missing))

marker_manifest = json.loads((MARKER_DIR / "manifest.json").read_text(encoding="utf-8"))
audit_manifest = json.loads((AUDIT_DIR / "manifest.json").read_text(encoding="utf-8"))
supplement_manifest = json.loads((SUPPLEMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
accuracy_validation = json.loads((ACCURACY_DIR / "metadata/validation.json").read_text(encoding="utf-8"))

EXPECTED_SHA = "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"
assert marker_manifest["pipeline_version"] == "ethics-gender-lm-v1.6.0-candidate"
assert marker_manifest["input_sha256"] == EXPECTED_SHA
assert supplement_manifest["source_input_sha256"] == EXPECTED_SHA
assert supplement_manifest["rows"] == marker_manifest["rows_input"] == 81_756
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
IDENTITIES = ("man", "woman")
IDENTITY_LABELS = {"man": "Man", "woman": "Woman"}
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = {"explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}


def show_legacy(filename: str, width: int = 1650) -> None:
    display(Image(filename=str(LEGACY_DIR / filename), width=width))


def show_supplement(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(FIG_DIR / filename), width=width))


def show_accuracy(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(ACCURACY_FIG_DIR / filename), width=width))


sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_colwidth", 120)
print(f"Marker source : {marker_manifest['pipeline_version']}")
print(f"Visual bundle: {supplement_manifest['bundle_version']}")
print(f"Respons      : {marker_manifest['rows_input']:,}")
print(f"Audit status : {audit_manifest['status']}")
"""
        ),
        _markdown(
            """
## 1. Provenance, coverage, dan validation gate

Audit Codex adalah first-pass, bukan independent human audit. Gender words dan pronouns diperlakukan sebagai audit anchor, bukan otomatis persona uptake.
"""
        ),
        _code(
            """
provenance = pd.DataFrame([
    ("Versi marker", marker_manifest["pipeline_version"]),
    ("Versi visual", supplement_manifest["bundle_version"]),
    ("SHA-256 sumber", marker_manifest["input_sha256"]),
    ("Respons unik", f"{marker_manifest['rows_input']:,}"),
    ("Surface terukur", f"{marker_manifest['responses_text_available']:,}"),
    ("Abstain disertakan", f"{marker_manifest['responses_abstain']:,}"),
    ("Missing menjadi NA", f"{marker_manifest['responses_missing']:,}"),
    ("Marker terdaftar", marker_manifest["marker_count"]),
    ("Gender marker hits", f"{marker_manifest['gender_marker_hits']:,}"),
    ("Inferensi marker", supplement_manifest["marker_inference_run"]),
    ("Audit", audit_manifest["status"]),
], columns=["Komponen", "Nilai"])
display(provenance)

status = pd.read_csv(MARKER_DIR / "data_status_summary.csv")
coverage = (
    status.assign(available=lambda d: d["n_responses"].where(d["text_status"].eq("available"), 0))
    .groupby(["model_id", "persona"], as_index=False)
    .agg(total=("n_responses", "sum"), text_available=("available", "sum"))
)
coverage["coverage_pct"] = 100 * coverage["text_available"] / coverage["total"]
panel = coverage.pivot(index="model_id", columns="persona", values="coverage_pct").reindex(index=MODEL_LABELS)
panel.index = [MODEL_LABELS[item] for item in panel.index]
fig, ax = plt.subplots(figsize=(17, 6))
sns.heatmap(panel, annot=True, fmt=".1f", cmap="Blues", vmin=90, vmax=100, ax=ax)
ax.set_title("Coverage visible response per model dan persona")
ax.set_xlabel("Persona")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
display(coverage.sort_values("coverage_pct").head(14).style.format({"coverage_pct": "{:.2f}%"}))
"""
        ),
        _markdown(
            """
## 2. Status respons dan tabel pelaporan wajib

Accuracy committed selalu ditampilkan bersama commit rate. Commonsense dan Justice adalah subset construct-valid.
"""
        ),
        _code(
            """
show_accuracy("04_response_status_distribution_by_model.png", width=1350)
scoring = pd.read_csv(SUPPLEMENT_DIR / "scoring_summary_by_model_persona_subset.csv")
required_columns = [
    "total", "committed", "commit_rate", "abstain", "abstain_rate",
    "missing", "missing_rate", "accuracy_committed",
    "pct_answer_wrong", "pct_answer_acceptable", "construct_valid",
]
assert set(required_columns).issubset(scoring.columns)
display(
    scoring.sort_values(["commit_rate", "model_id", "persona", "subset"])
    [["model_id", "persona", "subset", *required_columns]]
    .head(24)
    .style.format({
        "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}",
        "missing_rate": "{:.2%}", "accuracy_committed": "{:.2%}",
        "pct_answer_wrong": "{:.2%}", "pct_answer_acceptable": "{:.2%}",
    })
)
"""
        ),
        _markdown("## 3. Verbosity dan profil marker dasar"),
        _code(
            """
show_legacy("02_response_length_by_subset.png", width=1700)
show_legacy("03_overall_marker_prevalence.png", width=1500)
show_legacy("06_abstain_vs_committed_marker_profile.png", width=1500)
"""
        ),
        _markdown(
            """
## 4. Gender words, pronouns, dan nouns

Grafik ini adalah surface inventory. Kemunculan he, she, man, woman, dan bentuk serupa tidak otomatis membuktikan persona uptake karena kata yang sama dapat berasal dari skenario.
"""
        ),
        _code(
            """
show_supplement("05_gender_terms_ranked_response_prevalence.png", width=1650)
terms = pd.read_csv(SUPPLEMENT_DIR / "gender_anchor_terms_response_prevalence.csv")
display(
    terms.sort_values("overall_prevalence_pct", ascending=False)
    [["term", "term_category", "interpretation", "responses_with_term", "overall_prevalence_pct", "explicit_prevalence_pct", "implicit_prevalence_pct", "neutral_prevalence_pct", "responses_same_term_in_scenario"]]
    .head(30)
    .style.format({
        "overall_prevalence_pct": "{:.2f}%", "explicit_prevalence_pct": "{:.2f}%",
        "implicit_prevalence_pct": "{:.2f}%", "neutral_prevalence_pct": "{:.2f}%",
    })
)
"""
        ),
        _markdown(
            """
## 5. Effect marker gender terhadap Neutral

Titik berwarna mempertahankan enam model dan wajik hitam adalah median antarmodel. Tidak ada significance claim baru.
"""
        ),
        _code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"07_{identity}_gender_marker_forest_vs_neutral_descriptive.png", width=1850)
"""
        ),
        _markdown("## 6. Marker linguistik umum terhadap Neutral"),
        _code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"08_{identity}_general_effects_vs_neutral_descriptive.png", width=1850)
"""
        ),
        _markdown("## 7. Direct Man vs Woman per kondisi"),
        _code(
            """
for condition in CONDITIONS:
    display(Markdown(f"### Kondisi {CONDITION_LABELS[condition]}"))
    show_supplement(f"09_direct_identity_gender_forest_{condition}_descriptive.png", width=1650)
"""
        ),
        _markdown(
            """
## 8. Profil per subset dan correctness sensitivity

Ground truth hanya digunakan sebagai filter pasca-ekstraksi pada Commonsense dan Justice. Ia tidak masuk ke detektor marker.
"""
        ),
        _code(
            """
direct = pd.read_csv(SUPPLEMENT_DIR / "gender_marker_direct_man_woman_descriptive.csv")
subset_effects = (
    direct[~direct["subset"].eq("all")]
    .groupby(["condition", "subset", "outcome_id"], as_index=False)
    .agg(median_delta_pp=("prevalence_delta_pp", "median"), min_pairs=("n_pairs", "min"))
)
subset_effects["abs_delta"] = subset_effects["median_delta_pp"].abs()
display(
    subset_effects.sort_values("abs_delta", ascending=False)
    .groupby("subset", sort=False).head(5)
    [["subset", "condition", "outcome_id", "median_delta_pp", "min_pairs"]]
    .style.format({"median_delta_pp": "{:+.2f}"})
)
show_supplement("10_correctness_sensitivity_construct_valid_descriptive.png", width=1600)
sensitivity = pd.read_csv(SUPPLEMENT_DIR / "correctness_sensitivity_summary.csv")
display(sensitivity.style.format({
    "median_pair_retention": "{:.1%}",
    "same_direction_rate_supported": "{:.1%}",
    "median_abs_effect_change": "{:.3f}",
}))
"""
        ),
        _markdown("## 9. Opening dan differential availability"),
        _code(
            """
show_supplement("11_opening_composition_100pct.png", width=1650)
show_supplement("12_direct_identity_opening_descriptive.png", width=1550)
show_supplement("13_direct_identity_availability_descriptive.png", width=1550)
opening = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_opening_mtv_descriptive.csv")
availability = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_availability_descriptive.csv")
display(opening.sort_values("marginal_total_variation", ascending=False).head(16).style.format({"marginal_total_variation": "{:.2%}"}))
display(availability.reindex(availability["availability_delta_pp"].abs().sort_values(ascending=False).index).head(16).style.format({"availability_delta_pp": "{:+.2f}"}))
"""
        ),
        _markdown("## 10. Gender uptake dan scenario-overlap audit"),
        _code(
            """
show_legacy("09_gender_uptake_by_subset.png", width=1700)
hit_summary = pd.read_csv(LEGACY_DIR / "gender_marker_hit_audit_summary.csv")
display(hit_summary.sort_values("hit_rows", ascending=False).style.format({"scenario_overlap_pct": "{:.2f}%"}))
"""
        ),
        _markdown(
            """
## 11. Accuracy construct-valid dan distribusi jawaban Deontology/Virtue

Inferensi accuracy berasal dari bundle terpisah yang lolos validation gate. Deontology dan Virtue tidak digunakan sebagai pembanding accuracy antarpersona.
"""
        ),
        _code(
            """
show_accuracy("03_effect_size_commonsense_holm_heatmap.png", width=1650)
show_accuracy("03_effect_size_justice_holm_heatmap.png", width=1650)
construct_valid = scoring[scoring["subset"].isin(["commonsense", "justice"])]
display(
    construct_valid.sort_values("accuracy_committed", ascending=False)
    [["model_id", "persona", "subset", "total", "committed", "commit_rate", "abstain", "abstain_rate", "missing", "missing_rate", "accuracy_committed"]]
    .head(30)
    .style.format({
        "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}",
        "missing_rate": "{:.2%}", "accuracy_committed": "{:.2%}",
    })
)

distribution = scoring[scoring["subset"].isin(["deontology", "virtue"])].copy()
fig, axes = plt.subplots(2, 1, figsize=(17, 11), constrained_layout=True)
for ax, subset in zip(axes, ["deontology", "virtue"]):
    panel = distribution[distribution["subset"].eq(subset)].pivot(
        index="model_id", columns="persona", values="pct_answer_wrong"
    ).reindex(index=MODEL_LABELS)
    panel.index = [MODEL_LABELS[item] for item in panel.index]
    sns.heatmap(panel * 100, annot=True, fmt=".1f", cmap="RdYlBu_r", vmin=0, vmax=100, ax=ax)
    ax.set_title(f"{subset.title()}: jawaban wrong pada committed responses")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=45)
plt.show()
display(
    distribution[["model_id", "persona", "subset", "committed", "commit_rate", "pct_answer_wrong", "pct_answer_acceptable"]]
    .head(30)
    .style.format({
        "commit_rate": "{:.2%}", "pct_answer_wrong": "{:.2%}",
        "pct_answer_acceptable": "{:.2%}",
    })
)
"""
        ),
        _markdown("## 12. Audit, artefak, dan batas interpretasi"),
        _code(
            """
registry = pd.read_csv(MARKER_DIR / "marker_registry.csv")
gender_registry = registry[registry["marker_id"].str.startswith("gender")]
display(pd.DataFrame([
    ("Audit status", audit_manifest["status"]),
    ("Workbook rows", sum(audit_manifest["sheet_rows"].values())),
    ("Gender marker terdaftar", len(gender_registry)),
    ("Marker inference", supplement_manifest["marker_inference_run"]),
    ("Inferential claims allowed", False),
], columns=["Komponen", "Nilai"]))
display(
    gender_registry.groupby(["family", "analysis_role", "validation_status"], observed=True)
    .size().rename("n_markers").reset_index()
)
figure_index = pd.read_csv(SUPPLEMENT_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1))
display(figure_index)
print(f"Notebook source : {REPO_ROOT / 'notebooks/gender/linguistic_marker_gender_ethics_bab4_visual_v1_1.ipynb'}")
print(f"Visual bundle  : {SUPPLEMENT_DIR}")
print(f"Marker status  : {audit_manifest['status']}")
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

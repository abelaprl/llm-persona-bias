"""Create the thesis-facing ETHICS Regional-Cultural visual notebook v1."""

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
# Linguistic Marker Regional–Cultural Persona pada ETHICS — Visualisasi v1

Notebook ini mengikuti pola visual Regional/Cultural GSM8K dengan batas konstruk ETHICS. Seluruh **154.428 respons** dari enam model, 17 persona, 1.514 pertanyaan, dan empat subset tetap berada dalam inventaris.

Marker dihitung hanya dari jawaban yang terlihat. Hidden reasoning tidak dipakai, respons abstain dengan teks tetap disertakan, dan missing tidak dijadikan marker nol. Karena registry masih candidate, seluruh effect marker bersifat **deskriptif** tanpa p-value atau confidence interval baru. Correctness sensitivity dibatasi pada Commonsense dan Justice.
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
REPORT_ROOT = REPO_ROOT / "result/ETHICS/reports/Regional_Cultural_Persona"
MARKER_DIR = REPORT_ROOT / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16"
AUDIT_DIR = REPORT_ROOT / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_audit_r1"
SUPPLEMENT_DIR = REPORT_ROOT / "linguistic_markers/visual/regional_cultural_ethics_bab4_visual_supplement_v1_0_0_20260824"
FIG_DIR = SUPPLEMENT_DIR / "images"
ACCURACY_DIR = REPORT_ROOT / "_combined/persona_publication_analysis_v3_accuracy_paired_holm_six_models"
ACCURACY_FIG_DIR = ACCURACY_DIR / "figures"

required = [
    MARKER_DIR / "manifest.json",
    AUDIT_DIR / "audit_manifest.json",
    SUPPLEMENT_DIR / "manifest.json",
    ACCURACY_DIR / "metadata/validation.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Artefak wajib tidak ditemukan:\\n" + "\\n".join(missing))

marker_manifest = json.loads((MARKER_DIR / "manifest.json").read_text(encoding="utf-8"))
audit_manifest = json.loads((AUDIT_DIR / "audit_manifest.json").read_text(encoding="utf-8"))
supplement_manifest = json.loads((SUPPLEMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
accuracy_validation = json.loads((ACCURACY_DIR / "metadata/validation.json").read_text(encoding="utf-8"))

EXPECTED_SHA = "09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5"
assert marker_manifest["pipeline_version"] == "ethics-regional-cultural-lm-v1.0.16-candidate"
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
IDENTITIES = ("north_american", "east_asian", "south_asian", "sub_saharan_african")
IDENTITY_LABELS = {
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
CONDITION_LABELS = {"explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}
ALIGNED_MARKERS = {
    "north_american": "culture_region_north_american",
    "east_asian": "culture_region_east_asian",
    "south_asian": "culture_region_south_asian",
    "sub_saharan_african": "culture_region_sub_saharan_african",
}


def show_supplement(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(FIG_DIR / filename), width=width))


def show_accuracy(filename: str, width: int = 1750) -> None:
    display(Image(filename=str(ACCURACY_FIG_DIR / filename), width=width))


sns.set_theme(style="whitegrid", context="notebook")
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

Coverage adalah ketersediaan surface jawaban, bukan commit rate. Audit Codex adalah AI first-pass dan tidak menggantikan dua reviewer manusia independen.
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
    ("Marker regional–budaya", marker_manifest["marker_count_regional_cultural"]),
    ("Inferensi marker", supplement_manifest["marker_inference_run"]),
    ("Validasi manusia", audit_manifest["human_review_status"]),
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
fig, ax = plt.subplots(figsize=(22, 6))
sns.heatmap(panel, annot=True, fmt=".1f", cmap="Blues", vmin=90, vmax=100, ax=ax)
ax.set_title("Coverage surface jawaban per model dan persona")
ax.set_xlabel("Persona")
ax.set_ylabel("Model")
ax.tick_params(axis="x", rotation=55)
plt.tight_layout()
plt.show()
display(coverage.sort_values("coverage_pct").head(16).style.format({"coverage_pct": "{:.2f}%"}))
"""
        ),
        _markdown(
            """
## 2. Status respons, accuracy, dan coverage wajib

Accuracy committed selalu dibaca bersama commit rate. Commonsense dan Justice adalah subset construct-valid; Deontology dan Virtue dibaca melalui distribusi jawaban.
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
        _markdown("## 3. Verbosity dan normalisasi panjang respons"),
        _code(
            """
surface = pd.read_csv(MARKER_DIR / "surface_summary.csv")
surface = surface.groupby(["model_id", "persona"], as_index=False).agg(
    median_word_count=("median_word_count", "mean"),
    mean_ttr_300=("mean_ttr_300", "mean"),
)
fig, axes = plt.subplots(2, 1, figsize=(22, 11), constrained_layout=True)
for ax, value, title, cmap, fmt in [
    (axes[0], "median_word_count", "Rerata median jumlah kata per subset", "YlGnBu", ".1f"),
    (axes[1], "mean_ttr_300", "Rerata TTR-300 per subset", "PuBuGn", ".3f"),
]:
    panel = surface.pivot(index="model_id", columns="persona", values=value).reindex(index=MODEL_LABELS)
    panel.index = [MODEL_LABELS[item] for item in panel.index]
    sns.heatmap(panel, annot=True, fmt=fmt, cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=55)
plt.show()
"""
        ),
        _markdown("## 4. Penyebutan identitas dan akomodasi regional–budaya"),
        _code(
            """
prevalence = pd.read_csv(MARKER_DIR / "marker_prevalence.csv")
aligned_rows = []
for identity, marker_id in ALIGNED_MARKERS.items():
    selected = prevalence[
        prevalence["persona"].eq(f"{identity}_explicit")
        & prevalence["marker_id"].eq(marker_id)
    ]
    grouped = selected.groupby("model_id", as_index=False).agg(
        n_available=("n_available", "sum"),
        responses_present=("responses_present", "sum"),
    )
    grouped["prevalence_pct"] = 100 * grouped["responses_present"] / grouped["n_available"]
    grouped["identity"] = IDENTITY_LABELS[identity]
    aligned_rows.append(grouped)
aligned = pd.concat(aligned_rows, ignore_index=True)
aligned["model"] = aligned["model_id"].map(MODEL_LABELS)
fig, ax = plt.subplots(figsize=(15, 7))
sns.barplot(data=aligned, x="identity", y="prevalence_pct", hue="model", ax=ax)
ax.set_title("Penyebutan identitas kawasan selaras pada persona Explicit")
ax.set_xlabel("Persona")
ax.set_ylabel("Respons dengan marker (%)")
ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

accommodation_markers = [
    "culture_meta_framing", "culture_relativist_framing",
    "regional_persona_direct_address", "regional_perspective_adoption",
    "culture_named_concept_reference", "culture_named_practice_reference",
    "culture_identity_irrelevance",
]
persona_identity = {f"{identity}_explicit": identity for identity in IDENTITIES}
selected = prevalence[
    prevalence["persona"].isin(persona_identity)
    & prevalence["marker_id"].isin(accommodation_markers)
].copy()
selected["identity"] = selected["persona"].map(persona_identity)
accommodation = selected.groupby(["identity", "marker_id"], as_index=False).agg(
    n_available=("n_available", "sum"),
    responses_present=("responses_present", "sum"),
)
accommodation["prevalence_pct"] = 100 * accommodation["responses_present"] / accommodation["n_available"]
panel = accommodation.pivot(index="marker_id", columns="identity", values="prevalence_pct").reindex(columns=IDENTITIES)
panel.columns = [IDENTITY_LABELS[item] for item in panel.columns]
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(panel, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
ax.set_title("Akomodasi regional–budaya pada persona Explicit")
ax.set_xlabel("Identitas")
ax.set_ylabel("Marker")
plt.tight_layout()
plt.show()
"""
        ),
        _markdown("## 5. Istilah regional–budaya yang benar-benar muncul"),
        _code(
            """
show_supplement("05_regional_cultural_terms_ranked_explicit_response_prevalence.png", width=1750)
terms = pd.read_csv(SUPPLEMENT_DIR / "regional_cultural_terms_explicit_response_prevalence.csv")
display(
    terms[terms["condition"].eq("explicit")]
    .sort_values(["identity", "response_prevalence_pct"], ascending=[True, False])
    .groupby("identity", sort=False).head(12)
    .style.format({"response_prevalence_pct": "{:.3f}%"})
)
"""
        ),
        _markdown("## 6. Marker regional–budaya terhadap Neutral"),
        _code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"07_{identity}_regional_cultural_marker_forest_vs_neutral_descriptive.png", width=1850)
"""
        ),
        _markdown("## 7. Marker linguistik umum terhadap Neutral"),
        _code(
            """
for identity in IDENTITIES:
    display(Markdown(f"### {IDENTITY_LABELS[identity]}"))
    show_supplement(f"08_{identity}_general_effects_vs_neutral_descriptive.png", width=1850)
"""
        ),
        _markdown("## 8. Perbandingan langsung antarkawasan"),
        _code(
            """
for condition in CONDITION_LABELS:
    display(Markdown(f"### Kondisi {CONDITION_LABELS[condition]}"))
    show_supplement(f"09_direct_identity_regional_cultural_forest_{condition}_descriptive.png", width=1800)
"""
        ),
        _markdown(
            """
## 9. Profil per subset dan correctness sensitivity

ETHICS tidak dilebur menjadi satu accuracy. Correctness hanya dipakai sebagai filter pasca-ekstraksi pada Commonsense dan Justice.
"""
        ),
        _code(
            """
within = pd.read_csv(SUPPLEMENT_DIR / "regional_cultural_marker_vs_neutral_descriptive.csv")
subset_effects = (
    within[~within["subset"].eq("all")]
    .groupby(["identity", "condition", "subset", "outcome_id"], as_index=False)
    .agg(median_delta_pp=("prevalence_delta_pp", "median"), min_pairs=("n_pairs", "min"))
)
subset_effects["abs_delta"] = subset_effects["median_delta_pp"].abs()
display(
    subset_effects.sort_values("abs_delta", ascending=False)
    .groupby(["identity", "subset"], sort=False).head(3)
    [["identity", "subset", "condition", "outcome_id", "median_delta_pp", "min_pairs"]]
    .style.format({"median_delta_pp": "{:+.2f}"})
)
show_supplement("10_correctness_sensitivity_construct_valid_descriptive.png", width=1600)
summary = pd.read_csv(SUPPLEMENT_DIR / "correctness_sensitivity_summary.csv")
display(summary.style.format({
    "median_pair_retention": "{:.1%}",
    "same_direction_rate_supported": "{:.1%}",
    "median_abs_effect_change": "{:.3f}",
}))
"""
        ),
        _markdown("## 10. Opening dan differential availability"),
        _code(
            """
show_supplement("11_opening_composition_100pct.png", width=1750)
show_supplement("12_direct_identity_opening_descriptive.png", width=1750)
show_supplement("13_direct_identity_availability_descriptive.png", width=1750)
opening = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_opening_mtv_descriptive.csv")
availability = pd.read_csv(SUPPLEMENT_DIR / "direct_identity_availability_descriptive.csv")
display(opening.sort_values("marginal_total_variation", ascending=False).head(16).style.format({"marginal_total_variation": "{:.2%}"}))
display(availability.reindex(availability["availability_delta_pp"].abs().sort_values(ascending=False).index).head(16).style.format({"availability_delta_pp": "{:+.2f}"}))
"""
        ),
        _markdown("## 11. Accuracy construct-valid dan distribusi jawaban Deontology/Virtue"),
        _code(
            """
show_accuracy("03_effect_size_commonsense_holm_heatmap.png", width=1750)
show_accuracy("03_effect_size_justice_holm_heatmap.png", width=1750)
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
fig, axes = plt.subplots(2, 1, figsize=(22, 12), constrained_layout=True)
for ax, subset in zip(axes, ["deontology", "virtue"]):
    panel = distribution[distribution["subset"].eq(subset)].pivot(
        index="model_id", columns="persona", values="pct_answer_wrong"
    ).reindex(index=MODEL_LABELS)
    panel.index = [MODEL_LABELS[item] for item in panel.index]
    sns.heatmap(panel * 100, annot=True, fmt=".1f", cmap="RdYlBu_r", vmin=0, vmax=100, ax=ax)
    ax.set_title(f"{subset.title()}: jawaban wrong pada committed responses")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=55)
plt.show()
display(
    distribution[["model_id", "persona", "subset", "committed", "commit_rate", "pct_answer_wrong", "pct_answer_acceptable"]]
    .head(32)
    .style.format({
        "commit_rate": "{:.2%}", "pct_answer_wrong": "{:.2%}",
        "pct_answer_acceptable": "{:.2%}",
    })
)
"""
        ),
        _markdown("## 12. Audit marker, artefak, dan batas interpretasi"),
        _code(
            """
audit_summary = pd.read_csv(AUDIT_DIR / "audit_codex_summary.csv")
registry = pd.read_csv(MARKER_DIR / "marker_registry.csv")
regional_registry = registry[registry["registry_scope"].eq("regional_cultural")]
display(pd.DataFrame([
    ("Jenis audit", audit_manifest["audit_type"]),
    ("Reviewer", audit_manifest["reviewer"]),
    ("Human review", audit_manifest["human_review_status"]),
    ("Marker regional–budaya", len(regional_registry)),
    ("Inferential claims allowed", False),
], columns=["Komponen", "Nilai"]))
display(audit_summary)
display(
    regional_registry.groupby(["family", "analysis_role", "validation_status"], observed=True)
    .size().rename("n_markers").reset_index()
)

figure_index = pd.read_csv(SUPPLEMENT_DIR / "figure_index.csv")
figure_index["size_kb"] = figure_index["figure"].map(lambda name: round((FIG_DIR / name).stat().st_size / 1024, 1))
display(figure_index)
print(f"Notebook source : {REPO_ROOT / 'notebooks/regional/linguistic_marker_regional_cultural_ethics_bab4_visual_v1.ipynb'}")
print(f"Visual bundle  : {SUPPLEMENT_DIR}")
print(f"Marker status  : {marker_manifest['registry_status']}")
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

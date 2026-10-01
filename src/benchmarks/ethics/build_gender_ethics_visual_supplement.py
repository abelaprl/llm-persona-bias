"""Build GSM8K-aligned descriptive visuals for ETHICS Gender Persona.

The shared paired-effect engine is configured for the two gender identities.
Gender words and pronouns remain audit anchors and are never reinterpreted as
automatic persona uptake. Ground truth is used only for the explicitly
labelled correctness-sensitivity filter after marker extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_religion_ethics_visual_supplement as base


base.BUNDLE_VERSION = "gender-ethics-bab4-visual-supplement-v1.0.0"
base.EXPECTED_PIPELINE_VERSION = "ethics-gender-lm-v1.6.0-candidate"
base.EXPECTED_INPUT_SHA256 = "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"
base.DOMAIN_KEY = "gender"
base.DOMAIN_LABEL = "gender"
base.DOMAIN_TITLE = "Gender"
base.LEXICON_FILENAME = "gender_lexicon_coverage.csv"
base.TERM_TABLE_FILENAME = "gender_anchor_terms_response_prevalence.csv"
base.WITHIN_TABLE_FILENAME = "gender_marker_vs_neutral_descriptive.csv"
base.DIRECT_TABLE_FILENAME = "gender_marker_direct_man_woman_descriptive.csv"
base.TERM_FIGURE_FILENAME = "05_gender_terms_ranked_response_prevalence.png"
base.WITHIN_FIGURE_TOKEN = "gender_marker"
base.DIRECT_FIGURE_TOKEN = "gender"

base.IDENTITIES = ("man", "woman")
base.DIRECT_IDENTITY_PAIRS = (("man", "woman"),)
base.IDENTITY_LABELS = {"man": "Man", "woman": "Woman"}
base.IDENTITY_COLORS = {"man": "#2563EB", "woman": "#D946EF"}
base.PAIR_LABELS = {"man_vs_woman": "Man − Woman"}

base.RELIGION_MARKERS = (
    "gender_persona_direct_address",
    "gender_persona_acknowledgment",
    "gender_second_person_identity_reference",
    "gender_hypothetical_identity_reference",
    "gender_first_person_identity_reference",
    "gender_perspective_adoption",
    "gender_identity_relevance_denial",
    "gender_identity_neutralization",
    "gender_identity_discourse",
    "gender_role_framing",
    "gendered_normative_evaluation",
)
base.RELIGION_MARKER_LABELS = {
    "gender_persona_direct_address": "Sapaan persona langsung",
    "gender_persona_acknowledgment": "Acknowledgment persona",
    "gender_second_person_identity_reference": "Identitas orang kedua",
    "gender_hypothetical_identity_reference": "Identitas hipotetis",
    "gender_first_person_identity_reference": "Identitas orang pertama",
    "gender_perspective_adoption": "Adopsi perspektif",
    "gender_identity_relevance_denial": "Gender dinyatakan tak relevan",
    "gender_identity_neutralization": "Netralisasi identitas",
    "gender_identity_discourse": "Diskursus identitas",
    "gender_role_framing": "Framing peran gender",
    "gendered_normative_evaluation": "Evaluasi normatif gender",
}


def _load_sources(
    marker_dir: Path,
    canonical_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads((marker_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != base.EXPECTED_PIPELINE_VERSION:
        raise ValueError(f"Unexpected pipeline version: {manifest.get('pipeline_version')}")
    if manifest.get("input_sha256") != base.EXPECTED_INPUT_SHA256:
        raise ValueError("Canonical input SHA-256 does not match the validated ETHICS Gender source.")
    if manifest.get("inference_run") is not False:
        raise ValueError("The candidate marker bundle must not contain marker inference.")

    marker_columns = [f"{marker}__present" for marker in base.RELIGION_MARKERS]
    marker_columns += [f"{marker}__density" for marker in base.GENERAL_MARKERS]
    usecols = [
        "model_id",
        "persona",
        "condition",
        "question_id",
        "subset",
        "answer_status",
        "text_available",
        "word_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        *marker_columns,
    ]
    response_path = marker_dir / "response_level_markers.csv.gz"
    print(f"[1/7] Reading response-level marker surface: {response_path}")
    frame = pd.read_csv(response_path, usecols=usecols, low_memory=False)
    frame["identity"] = frame["persona"].str.extract(r"^(man|woman)_", expand=False)
    frame.loc[frame["persona"].eq("neutral"), "identity"] = "neutral"
    frame["text_available"] = base._bool_mask(frame["text_available"])
    for column in marker_columns + ["word_count", "avg_sentence_length", "ttr_300"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    print(f"[2/7] Reading correctness keys only: {canonical_csv}")
    correctness = pd.read_csv(
        canonical_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "is_correct",
            "construct_valid",
            "extracted_stance",
        ],
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    correctness["is_correct"] = base._nullable_bool(correctness["is_correct"])
    correctness["construct_valid"] = base._nullable_bool(correctness["construct_valid"])

    key = ["model_id", "persona", "question_id"]
    if frame.duplicated(key).any() or correctness.duplicated(key).any():
        raise ValueError("Duplicate model/persona/question key found in a canonical source.")
    frame = frame.merge(correctness, on=key, how="left", validate="one_to_one")
    if len(frame) != manifest["rows_input"]:
        raise ValueError("Merged response-level row count does not match the manifest.")
    if set(frame["model_id"].dropna().unique()) != set(base.MODEL_ORDER):
        raise ValueError("Unexpected model set in response-level source.")
    if set(frame["subset"].dropna().unique()) != set(base.SUBSETS):
        raise ValueError("Unexpected ETHICS subset set.")

    lexicon = pd.read_csv(marker_dir / base.LEXICON_FILENAME, low_memory=False)
    registry = pd.read_csv(marker_dir / "marker_registry.csv", low_memory=False)
    return frame, lexicon, manifest, registry


def _term_table(frame: pd.DataFrame, lexicon: pd.DataFrame) -> pd.DataFrame:
    available = frame["text_available"]
    denominators = {
        "overall": int(available.sum()),
        "explicit": int((available & frame["condition"].eq("explicit")).sum()),
        "implicit": int((available & frame["condition"].str.startswith("implicit")).sum()),
        "neutral": int((available & frame["condition"].eq("neutral")).sum()),
    }
    result = lexicon[lexicon["observed_in_response_corpus"].eq(True)].copy()
    count_columns = {
        "overall": "responses_with_term",
        "explicit": "explicit_responses_with_term",
        "implicit": "implicit_responses_with_term",
        "neutral": "neutral_responses_with_term",
    }
    for scope, count_column in count_columns.items():
        result[f"{scope}_prevalence_pct"] = 100 * result[count_column] / denominators[scope]
        result[f"{scope}_denominator"] = denominators[scope]
    return result


def _plot_term_lollipop(data: pd.DataFrame, image_dir: Path) -> Path:
    selected = data.nlargest(22, "overall_prevalence_pct").sort_values("overall_prevalence_pct")
    y = np.arange(len(selected))
    colors = {
        "overall": "#111827",
        "explicit": "#2563EB",
        "implicit": "#7C3AED",
        "neutral": "#059669",
    }
    offsets = {"overall": -0.24, "explicit": -0.08, "implicit": 0.08, "neutral": 0.24}
    fig, ax = base.plt.subplots(figsize=(15, 12))
    for scope in ("overall", "explicit", "implicit", "neutral"):
        values = selected[f"{scope}_prevalence_pct"].to_numpy()
        ax.scatter(values, y + offsets[scope], s=75, color=colors[scope], label=scope.title())
    ax.set_yticks(y, selected["term"])
    ax.set_xlabel("Respons dengan audit-anchor term (%)")
    ax.set_ylabel("Gender word/pronoun")
    ax.set_title(
        "Gender words dan pronouns pada visible response ETHICS\n"
        "(audit anchor; bukan otomatis persona uptake)",
        fontweight="bold",
    )
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    ax.legend(ncol=4, loc="lower right")
    fig.tight_layout()
    return base._save(fig, image_dir, base.TERM_FIGURE_FILENAME)


def _plot_direct_gender(data: pd.DataFrame, image_dir: Path, condition: str) -> Path:
    selected = data[data["condition"].eq(condition) & data["subset"].eq("all")]
    y_lookup = {
        marker: len(base.RELIGION_MARKERS) - 1 - index
        for index, marker in enumerate(base.RELIGION_MARKERS)
    }
    limit = max(1.0, float(np.nanquantile(np.abs(selected["prevalence_delta_pp"]), 0.99)) * 1.08)
    fig, ax = base.plt.subplots(figsize=(14, 9.5))
    for _, point in selected.iterrows():
        ax.scatter(
            point["prevalence_delta_pp"],
            y_lookup[point["outcome_id"]],
            s=75,
            color=base.MODEL_COLORS[point["model_id"]],
            alpha=0.9,
        )
    medians = selected.groupby("outcome_id")["prevalence_delta_pp"].median()
    for marker, value in medians.items():
        ax.scatter(value, y_lookup[marker], marker="D", s=85, color="#111827", zorder=5)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(
        [y_lookup[marker] for marker in base.RELIGION_MARKERS],
        [base.RELIGION_MARKER_LABELS[marker] for marker in base.RELIGION_MARKERS],
    )
    ax.set_xlabel("Man − Woman: Δ prevalensi (poin persentase; deskriptif)")
    ax.set_title(f"Direct Man vs Woman pada kondisi {base.CONDITION_LABELS[condition]} — ETHICS")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.legend(handles=base._model_legend(), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return base._save(
        fig,
        image_dir,
        f"09_direct_identity_{base.DIRECT_FIGURE_TOKEN}_forest_{condition}_descriptive.png",
    )


base._load_sources = _load_sources
base._term_table = _term_table
base._plot_term_lollipop = _plot_term_lollipop
base._plot_direct_religion = _plot_direct_gender


if __name__ == "__main__":
    raise SystemExit(base.main())

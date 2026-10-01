"""Build the candidate ETHICS Gender linguistic-marker analysis v1.4.

The analysis mirrors the three Gender GSM8K comparison families while keeping
all ETHICS subsets separate:

1. E/I1/I2/I3 minus Neutral, within identity.
2. I1/I2/I3 minus Explicit, within identity.
3. Man minus Woman within E/I1/I2/I3.

Primary statistics use every response with linguistically available visible
text. A second analysis retains only pairs where both responses are committed.
Human audit fields are never filled by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.linguistic_markers.config import (
    IDENTITY_CONDITIONS,
    MAIN_MODELS,
)
from src.core.linguistic_markers.gender_registry import ETHICS_GENDER_MARKERS
from src.core.linguistic_markers.registry import MarkerSpec
from src.core.linguistic_markers.statistics import (
    build_paired_gender_numeric,
    build_paired_gender_opening_multinomial,
    build_paired_numeric,
    build_paired_opening_multinomial,
)


BUNDLE_VERSION = "gender-ethics-linguistic-markers-v1.4.0-candidate"
EXPECTED_SOURCE_SHA256 = "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"
EXPECTED_ROWS = 81_756
EXPECTED_VISIBLE = 80_840
BOOTSTRAP_RESAMPLES = 2_000

SUBSETS = ("commonsense", "deontology", "justice", "virtue")
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
NEUTRAL_CONTRASTS = (
    "explicit_vs_neutral",
    "implicit_1_vs_neutral",
    "implicit_2_vs_neutral",
    "implicit_3_vs_neutral",
)
EXPLICIT_CONTRASTS = (
    "implicit_1_vs_explicit",
    "implicit_2_vs_explicit",
    "implicit_3_vs_explicit",
)

SURFACE_OUTCOMES = ("word_count", "avg_sentence_length", "ttr_300")
SHARED_MARKERS = (
    "organization_explicit_step",
    "organization_bullet",
    "organization_section_header",
    "stance_hedge",
    "stance_booster",
    "stance_assumption",
    "stance_self_correction",
    "connective_causal",
    "connective_contrastive",
    "connective_conditional",
    "self_reference_first_person",
    "self_reference_inclusive_we",
    "engagement_reader_directive",
)
ETHICS_ADDITIONAL_MARKERS = (
    "stance_contextual_hedge",
    "stance_distancing",
    "connective_additive",
    "connective_concessive",
    "affect_empathy_validation",
)
MARKER_IDS = SHARED_MARKERS + ETHICS_ADDITIONAL_MARKERS
OUTCOME_IDS = SURFACE_OUTCOMES + MARKER_IDS

MODEL_ORDER = tuple(MAIN_MODELS)
MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_COLORS = dict(
    zip(MODEL_ORDER, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
)
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
CONDITION_LABELS = {
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
CONTRAST_LABELS = {
    "explicit_vs_neutral": "E − N",
    "implicit_1_vs_neutral": "I1 − N",
    "implicit_2_vs_neutral": "I2 − N",
    "implicit_3_vs_neutral": "I3 − N",
    "implicit_1_vs_explicit": "I1 − E",
    "implicit_2_vs_explicit": "I2 − E",
    "implicit_3_vs_explicit": "I3 − E",
}
OUTCOME_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Rata-rata panjang kalimat",
    "ttr_300": "TTR-300",
    "organization_explicit_step": "Langkah eksplisit",
    "organization_bullet": "Bullet",
    "organization_section_header": "Section header",
    "stance_hedge": "Hedge",
    "stance_booster": "Booster",
    "stance_assumption": "Asumsi eksplisit",
    "stance_self_correction": "Self-correction",
    "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras",
    "connective_conditional": "Konektif kondisional",
    "self_reference_first_person": "Rujukan orang pertama",
    "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca",
    "stance_contextual_hedge": "Hedge kontekstual",
    "stance_distancing": "Distancing",
    "connective_additive": "Konektif aditif",
    "connective_concessive": "Konektif konsesif",
    "affect_empathy_validation": "Empati/validasi",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.casefold().isin(("true", "1"))


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "figure.titlesize": 20,
            "legend.fontsize": 16,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, image_dir: Path, filename: str) -> Path:
    path = image_dir / filename
    svg = path.with_suffix(".svg")
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if path.stat().st_size < 10_000 or svg.stat().st_size < 2_000:
        raise ValueError(f"Rendered figure is unexpectedly small: {path}")
    return path


def _selected_markers() -> tuple[MarkerSpec, ...]:
    selected = tuple(
        marker for marker in ETHICS_GENDER_MARKERS if marker.marker_id in MARKER_IDS
    )
    selected_ids = tuple(marker.marker_id for marker in selected)
    if set(selected_ids) != set(MARKER_IDS) or len(selected) != 18:
        raise ValueError(f"Expected 18 registered markers, found {selected_ids}")
    if any(marker.measurement != "density" for marker in selected):
        raise ValueError("All selected v1.4 markers must use density per 100 words.")
    return selected


def _load_and_validate(
    marker_dir: Path,
    canonical_csv: Path,
    *,
    markers: Iterable[MarkerSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, object]]:
    marker_manifest_path = marker_dir / "manifest.json"
    response_path = marker_dir / "response_level_markers.csv.gz"
    registry_path = marker_dir / "marker_registry.csv"
    if not marker_manifest_path.is_file() or not response_path.is_file() or not registry_path.is_file():
        raise FileNotFoundError("The candidate ETHICS marker bundle is incomplete.")
    if not canonical_csv.is_file():
        raise FileNotFoundError(canonical_csv)

    source_sha = _sha256(canonical_csv)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Canonical ETHICS source SHA changed: {source_sha}")
    marker_manifest = json.loads(marker_manifest_path.read_text(encoding="utf-8"))
    if marker_manifest.get("input_sha256") != source_sha:
        raise ValueError("Marker bundle and canonical accuracy source have different SHA-256 values.")
    if marker_manifest.get("pipeline_version") != "ethics-gender-lm-v1.6.0-candidate":
        raise ValueError("Unexpected marker pipeline version.")

    marker_tuple = tuple(markers) if markers is not None else _selected_markers()
    marker_columns = [
        column
        for marker in marker_tuple
        for column in (
            f"{marker.marker_id}__count",
            f"{marker.marker_id}__density",
            f"{marker.marker_id}__present",
        )
    ]
    response_columns = [
        "model_id",
        "persona",
        "subset",
        "question_id",
        "answer_status",
        "text_status",
        "text_available",
        "opening_category",
        *SURFACE_OUTCOMES,
        *marker_columns,
    ]
    print("[1/8] Reading response-level marker table")
    frame = pd.read_csv(
        response_path,
        usecols=response_columns,
        low_memory=False,
    )
    print("[2/8] Reading canonical parser-refinement source")
    canonical = pd.read_csv(
        canonical_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "subset",
            "answer_status",
            "response_content_full",
        ],
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})

    key = ["model_id", "persona", "subset", "question_id"]
    if len(frame) != EXPECTED_ROWS or len(canonical) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows in both sources.")
    if frame.duplicated(key).any() or canonical.duplicated(key).any():
        raise ValueError("Duplicate response keys detected.")

    status_check = frame[key + ["answer_status"]].merge(
        canonical[key + ["answer_status"]],
        on=key,
        how="outer",
        suffixes=("_marker", "_canonical"),
        indicator=True,
        validate="one_to_one",
    )
    key_match = status_check["_merge"].eq("both").all()
    status_match = status_check["answer_status_marker"].eq(
        status_check["answer_status_canonical"]
    ).all()
    if not key_match or not status_match:
        mismatches = status_check[
            ~status_check["_merge"].eq("both")
            | ~status_check["answer_status_marker"].eq(status_check["answer_status_canonical"])
        ]
        raise ValueError(
            "Response status differs from the final accuracy source; "
            f"first mismatch: {mismatches.head(1).to_dict('records')}"
        )

    frame["text_available"] = _bool_series(frame["text_available"])
    frame["benchmark"] = "ethics"
    canonical_nonempty = canonical["response_content_full"].fillna("").astype(str).str.strip().ne("")
    status_counts = canonical["answer_status"].value_counts().to_dict()
    text_counts = frame["text_status"].value_counts().to_dict()
    validations = {
        "source_sha_matches_expected": source_sha == EXPECTED_SOURCE_SHA256,
        "row_count_matches": len(frame) == len(canonical) == EXPECTED_ROWS,
        "response_keys_match_one_to_one": bool(key_match),
        "response_status_matches_accuracy_source": bool(status_match),
        "canonical_nonempty_rows": int(canonical_nonempty.sum()),
        "visible_rows_match_marker_manifest": int(frame["text_available"].sum())
        == marker_manifest.get("responses_text_available")
        == EXPECTED_VISIBLE,
        "four_subsets_present": set(frame["subset"]) == set(SUBSETS),
        "six_models_present": set(frame["model_id"]) == set(MODEL_ORDER),
        "nine_personas_present": set(frame["persona"]) == set(
            persona for values in IDENTITY_CONDITIONS.values() for persona in values.values()
        ),
        "status_committed": int(status_counts.get("committed", 0)),
        "status_abstain": int(status_counts.get("abstain", 0)),
        "status_missing": int(status_counts.get("missing", 0)),
        "text_available": int(text_counts.get("available", 0)),
        "text_missing": int(text_counts.get("missing", 0)),
        "text_technical_only": int(text_counts.get("technical_only", 0)),
    }
    return frame, canonical, marker_manifest, validations


def _committed_frame(frame: pd.DataFrame, markers: Iterable[MarkerSpec]) -> pd.DataFrame:
    keep = frame["answer_status"].eq("committed") & frame["text_available"]
    value_columns = [*SURFACE_OUTCOMES]
    for marker in markers:
        value_columns.extend(
            (
                f"{marker.marker_id}__count",
                f"{marker.marker_id}__density",
                f"{marker.marker_id}__present",
            )
        )
    metadata_columns = [
        "model_id",
        "benchmark",
        "subset",
        "persona",
        "question_id",
    ]
    committed = frame.loc[:, metadata_columns].copy()
    committed["text_available"] = keep.to_numpy(bool)
    committed["opening_category"] = frame["opening_category"].where(keep, pd.NA)
    # Assign one value column at a time.  The source bundle contains many more
    # markers than this analysis needs, so a whole-frame deep copy wastes
    # hundreds of MB across the core and integrated builds.
    for column in value_columns:
        committed[column] = pd.to_numeric(frame[column], errors="coerce").where(
            keep, np.nan
        )
    return committed


def _tag_table(
    frame: pd.DataFrame,
    *,
    analysis_family: str,
    pair_filter: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "result_status", "candidate")
    result.insert(1, "analysis_family", analysis_family)
    result.insert(2, "pair_filter", pair_filter)
    return result


def _attach_raw_count_summaries(
    table: pd.DataFrame,
    source: pd.DataFrame,
    markers: Iterable[MarkerSpec],
) -> pd.DataFrame:
    """Attach paired raw-count summaries to density-based marker outcomes."""

    marker_tuple = tuple(markers)
    marker_ids = {marker.marker_id for marker in marker_tuple}
    count_columns = [f"{marker.marker_id}__count" for marker in marker_tuple]
    comparison_keys = [
        "model_id",
        "subset",
        "contrast_id",
        "left_persona",
        "right_persona",
    ]
    comparison_rows = table[comparison_keys].drop_duplicates()
    source_groups = {
        key: group[["persona", "question_id", *count_columns]]
        for key, group in source.groupby(["model_id", "subset"], sort=False)
    }
    summaries: list[dict[str, object]] = []
    for comparison in comparison_rows.itertuples(index=False):
        group = source_groups[(comparison.model_id, comparison.subset)]
        left = group.loc[
            group["persona"].eq(comparison.left_persona),
            ["question_id", *count_columns],
        ]
        right = group.loc[
            group["persona"].eq(comparison.right_persona),
            ["question_id", *count_columns],
        ]
        paired = left.merge(
            right,
            on="question_id",
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for marker in marker_tuple:
            base = f"{marker.marker_id}__count"
            left_values = pd.to_numeric(paired[f"{base}_left"], errors="coerce")
            right_values = pd.to_numeric(paired[f"{base}_right"], errors="coerce")
            valid = left_values.notna() & right_values.notna()
            left_valid = left_values[valid]
            right_valid = right_values[valid]
            deltas = left_valid - right_valid
            summaries.append(
                {
                    **{key: getattr(comparison, key) for key in comparison_keys},
                    "outcome_id": marker.marker_id,
                    "n_raw_count_pairs": int(valid.sum()),
                    "raw_count_sum_left": int(left_valid.sum()),
                    "raw_count_sum_right": int(right_valid.sum()),
                    "median_raw_count_left": float(left_valid.median()),
                    "median_raw_count_right": float(right_valid.median()),
                    "median_paired_raw_count_delta": float(deltas.median()),
                }
            )

    summary = pd.DataFrame(summaries)
    result = table.merge(
        summary,
        on=[*comparison_keys, "outcome_id"],
        how="left",
        validate="one_to_one",
    )
    marker_rows = result["outcome_id"].isin(marker_ids)
    if not result.loc[marker_rows, "n_raw_count_pairs"].eq(
        result.loc[marker_rows, "n_pairs"]
    ).all():
        raise ValueError("Raw-count pair support differs from density pair support.")
    return result


def _raw_count_summaries_complete(table: pd.DataFrame) -> bool:
    marker_rows = table[table["outcome_id"].isin(MARKER_IDS)]
    required = marker_rows[
        ["n_raw_count_pairs", "raw_count_sum_left", "raw_count_sum_right"]
    ].notna().all().all()
    supported = marker_rows["n_raw_count_pairs"].gt(0)
    supported_medians = marker_rows.loc[
        supported,
        [
            "median_raw_count_left",
            "median_raw_count_right",
            "median_paired_raw_count_delta",
        ],
    ].notna().all().all()
    return bool(required and supported_medians)


def _sensitivity_table(
    all_text: pd.DataFrame,
    committed: pd.DataFrame,
    keys: list[str],
    analysis_family: str,
) -> pd.DataFrame:
    columns = keys + ["n_pairs", "n_nonzero", "rank_biserial", "p_holm"]
    left = all_text[columns].rename(
        columns={
            "n_pairs": "n_pairs_all_text",
            "n_nonzero": "n_nonzero_all_text",
            "rank_biserial": "rank_biserial_all_text",
            "p_holm": "p_holm_all_text",
        }
    )
    right = committed[columns].rename(
        columns={
            "n_pairs": "n_pairs_committed",
            "n_nonzero": "n_nonzero_committed",
            "rank_biserial": "rank_biserial_committed",
            "p_holm": "p_holm_committed",
        }
    )
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged.insert(0, "result_status", "candidate")
    merged.insert(1, "analysis_family", analysis_family)
    merged["pair_retention"] = (
        merged["n_pairs_committed"] / merged["n_pairs_all_text"].replace(0, np.nan)
    )
    merged["same_direction"] = np.sign(merged["rank_biserial_all_text"]).eq(
        np.sign(merged["rank_biserial_committed"])
    )
    merged["supported_both"] = merged["n_nonzero_all_text"].ge(20) & merged[
        "n_nonzero_committed"
    ].ge(20)
    merged["abs_rank_biserial_change"] = (
        merged["rank_biserial_committed"] - merged["rank_biserial_all_text"]
    ).abs()
    return merged


def _sensitivity_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, subset), group in comparison.groupby(["analysis_family", "subset"], sort=False):
        valid = group.dropna(subset=["rank_biserial_all_text", "rank_biserial_committed"])
        supported = valid[valid["supported_both"]]
        rows.append(
            {
                "result_status": "candidate",
                "analysis_family": family,
                "subset": subset,
                "n_tests": len(group),
                "median_pair_retention": float(group["pair_retention"].median()),
                "min_pair_retention": float(group["pair_retention"].min()),
                "n_supported_both": len(supported),
                "same_direction_rate_supported": (
                    float(supported["same_direction"].mean()) if len(supported) else math.nan
                ),
                "spearman_rank_biserial": (
                    float(valid["rank_biserial_all_text"].corr(
                        valid["rank_biserial_committed"], method="spearman"
                    ))
                    if len(valid) > 1
                    else math.nan
                ),
                "median_abs_rank_biserial_change": float(
                    valid["abs_rank_biserial_change"].median()
                ) if len(valid) else math.nan,
                "n_holm_significant_all_text": int(group["p_holm_all_text"].lt(0.05).sum()),
                "n_holm_significant_committed": int(group["p_holm_committed"].lt(0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def _model_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]


def _plot_coverage(frame: pd.DataFrame, image_dir: Path) -> Path:
    coverage = (
        frame.groupby(["model_id", "persona"], as_index=False)
        .agg(total=("question_id", "size"), text_available=("text_available", "sum"))
    )
    coverage["coverage_pct"] = 100 * coverage["text_available"] / coverage["total"]
    persona_order = ["neutral", *IDENTITY_CONDITIONS["man"].values(), *IDENTITY_CONDITIONS["woman"].values()]
    persona_order = list(dict.fromkeys(persona_order))
    matrix = coverage.pivot(index="model_id", columns="persona", values="coverage_pct").reindex(
        index=MODEL_ORDER, columns=persona_order
    )
    matrix.index = [MODEL_LABELS[item] for item in matrix.index]
    fig, ax = plt.subplots(figsize=(22, 8.5))
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=max(0, float(matrix.min().min()) - 2),
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 12.5, "fontweight": "bold"},
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Visible text tersedia (%)"},
        ax=ax,
    )
    ax.set_title("Cakupan visible text per model dan persona", fontweight="bold")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, image_dir, "01_visible_text_coverage.png")


def _plot_within_effects(
    table: pd.DataFrame,
    subset: str,
    contrast_ids: tuple[str, ...],
    image_dir: Path,
    filename: str,
    title: str,
) -> Path:
    data = table[table["subset"].eq(subset) & table["contrast_id"].isin(contrast_ids)]
    y_lookup = {outcome: len(OUTCOME_IDS) - 1 - i for i, outcome in enumerate(OUTCOME_IDS)}
    fig, axes = plt.subplots(1, len(contrast_ids), figsize=(6.0 * len(contrast_ids), 16.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, contrast in zip(axes, contrast_ids):
        panel = data[data["contrast_id"].eq(contrast)]
        for _, point in panel.iterrows():
            y = y_lookup[point["outcome_id"]]
            marker = "o" if point["identity"] == "man" else "^"
            ax.scatter(
                point["rank_biserial"],
                y,
                s=55,
                marker=marker,
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.45,
                edgecolors="white",
                linewidths=0.6,
            )
        medians = panel.groupby("outcome_id")["rank_biserial"].median()
        for outcome, value in medians.items():
            ax.scatter(value, y_lookup[outcome], marker="D", s=62, color="#111827", zorder=4)
        ax.axvline(0, color="#334155", linewidth=0.9)
        ax.set_xlim(-1.03, 1.03)
        ax.set_title(CONTRAST_LABELS[contrast], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks(
        [y_lookup[item] for item in OUTCOME_IDS],
        [OUTCOME_LABELS[item] for item in OUTCOME_IDS],
        fontsize=14.5,
    )
    handles = _model_legend() + [
        Line2D([0], [0], marker="o", linestyle="", color="#475569", label="Man"),
        Line2D([0], [0], marker="^", linestyle="", color="#475569", label="Woman"),
        Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("Paired rank-biserial persona − pembanding", y=0.065)
    fig.suptitle(f"{SUBSET_LABELS[subset]} — {title}", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.96), w_pad=1.0)
    return _save(fig, image_dir, filename)


def _plot_direct_effects(table: pd.DataFrame, subset: str, image_dir: Path, filename: str) -> Path:
    data = table[table["subset"].eq(subset)]
    y_lookup = {outcome: len(OUTCOME_IDS) - 1 - i for i, outcome in enumerate(OUTCOME_IDS)}
    fig, axes = plt.subplots(1, 4, figsize=(24, 16), sharex=True, sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(
                point["rank_biserial"],
                y_lookup[point["outcome_id"]],
                s=58,
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.95 if point["p_holm"] < 0.05 else 0.45,
                edgecolors="white",
                linewidths=0.6,
            )
        medians = panel.groupby("outcome_id")["rank_biserial"].median()
        for outcome, value in medians.items():
            ax.scatter(value, y_lookup[outcome], marker="D", s=65, color="#111827", zorder=4)
        ax.axvline(0, color="#334155", linewidth=0.9)
        ax.set_xlim(-1.03, 1.03)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks(
        [y_lookup[item] for item in OUTCOME_IDS],
        [OUTCOME_LABELS[item] for item in OUTCOME_IDS],
        fontsize=14.5,
    )
    handles = _model_legend() + [
        Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("Paired rank-biserial Man − Woman", y=0.065)
    fig.suptitle(
        f"{SUBSET_LABELS[subset]} — perbandingan langsung Man dan Woman",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.96), w_pad=1.0)
    return _save(fig, image_dir, filename)


def _plot_sensitivity(comparison: pd.DataFrame, image_dir: Path) -> Path:
    families = ("vs_neutral", "vs_explicit", "man_vs_woman")
    labels = {"vs_neutral": "Terhadap N", "vs_explicit": "Terhadap E", "man_vs_woman": "Man − Woman"}
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharex=True, sharey=True)
    for ax, family in zip(axes, families):
        panel = comparison[comparison["analysis_family"].eq(family)]
        for subset, color in zip(SUBSETS, ("#2563EB", "#D97706", "#059669", "#7C3AED")):
            points = panel[panel["subset"].eq(subset)]
            ax.scatter(
                points["rank_biserial_all_text"],
                points["rank_biserial_committed"],
                s=18,
                color=color,
                alpha=0.26,
                label=SUBSET_LABELS[subset],
            )
        ax.plot((-1, 1), (-1, 1), color="#111827", linewidth=1.0, linestyle="--")
        ax.axhline(0, color="#94A3B8", linewidth=0.7)
        ax.axvline(0, color="#94A3B8", linewidth=0.7)
        ax.set_xlim(-1.03, 1.03)
        ax.set_ylim(-1.03, 1.03)
        ax.set_title(labels[family], fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=0.5)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    fig.supxlabel("Rank-biserial seluruh visible text", y=0.10)
    fig.supylabel("Rank-biserial pasangan committed", x=0.01)
    fig.suptitle("Perbandingan effect size seluruh respons dan respons committed", fontweight="bold")
    fig.tight_layout(rect=(0.04, 0.16, 0.99, 0.93), w_pad=1.5)
    return _save(fig, image_dir, "14_committed_sensitivity_concordance.png")


def _opening_heatmap(
    table: pd.DataFrame,
    image_dir: Path,
    filename: str,
    *,
    direct: bool,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSETS):
        panel = table[table["subset"].eq(subset)].copy()
        if direct:
            panel["row"] = panel["model_id"].map(MODEL_LABELS)
            columns = [f"man_vs_woman_{condition}" for condition in CONDITIONS]
            column_labels = [CONDITION_LABELS[item] for item in CONDITIONS]
        else:
            panel["row"] = panel["model_id"].map(MODEL_LABELS) + " | " + panel["identity"].str.title()
            columns = [*NEUTRAL_CONTRASTS, *EXPLICIT_CONTRASTS]
            column_labels = [CONTRAST_LABELS[item] for item in columns]
        matrix = (100 * panel.pivot(index="row", columns="contrast_id", values="marginal_total_variation")).reindex(columns=columns)
        sns.heatmap(
            matrix,
            cmap="YlOrBr",
            vmin=0,
            vmax=max(5.0, float(matrix.max().max())),
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 11.5, "fontweight": "bold"},
            linewidths=0.7,
            linecolor="white",
            cbar_kws={"label": "Marginal total variation (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset], fontweight="bold")
        ax.set_xlabel("Skenario")
        ax.set_ylabel("Model" if direct else "Model dan persona")
        ax.set_xticklabels(column_labels, rotation=30, ha="right")
        ax.tick_params(axis="y", rotation=0, labelsize=12.5)
    title = "Komposisi opening: direct Man–Woman" if direct else "Komposisi opening: perbandingan antarskenario"
    fig.suptitle(title, fontweight="bold")
    return _save(fig, image_dir, filename)


def _build_audit_sheets(
    marker_dir: Path,
    output_dir: Path,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    precision_source = pd.read_csv(marker_dir / "audit_marker_hits.csv", low_memory=False)
    recall_source = pd.read_csv(marker_dir / "audit_recall_responses.csv", low_memory=False)
    precision = precision_source[precision_source["marker_id"].isin(MARKER_IDS)].copy()
    recall = recall_source.copy()
    if "subset" not in precision.columns:
        precision["subset"] = precision["question_id"].str.extract(
            r"^ethics_([^_]+)_", expand=False
        )

    response_lookup = canonical[
        ["model_id", "persona", "subset", "question_id", "response_content_full"]
    ].rename(columns={"response_content_full": "visible_response_full"})
    precision = precision.merge(
        response_lookup,
        on=["model_id", "persona", "subset", "question_id"],
        how="left",
        validate="many_to_one",
    )
    if precision["visible_response_full"].isna().any():
        raise ValueError("Precision audit contains rows without a full visible response.")
    precision.insert(0, "audit_item_id", [f"precision_{index:05d}" for index in range(1, len(precision) + 1)])
    recall.insert(0, "audit_item_id", [f"recall_{index:05d}" for index in range(1, len(recall) + 1)])

    status_rows: list[dict[str, object]] = []
    audit_files: list[tuple[str, pd.DataFrame]] = []
    for audit_type, source in (("precision", precision), ("recall", recall)):
        for reviewer_slot in (1, 2):
            template = source.copy()
            for column in (
                "human_review_complete",
                "reviewer_id",
                "reviewer",
                "is_true_positive",
                "manual_marker_ids",
                "missed_span_notes",
                "audit_note",
            ):
                if column in template.columns:
                    template = template.drop(columns=column)
            template.insert(
                1,
                "human_review_complete",
                pd.Series(pd.NA, index=template.index, dtype="boolean"),
            )
            template.insert(2, "reviewer_slot", reviewer_slot)
            template.insert(3, "reviewer_id", pd.NA)
            if audit_type == "precision":
                template["is_true_positive"] = pd.NA
                template["audit_note"] = pd.NA
            else:
                template["manual_marker_ids"] = pd.NA
                template["missed_span_notes"] = pd.NA
                template["audit_note"] = pd.NA
            filename = f"human_{audit_type}_audit_reviewer_{reviewer_slot}.csv"
            audit_files.append((filename, template))
            completed = int(_bool_series(template["human_review_complete"]).sum())
            status_rows.append(
                {
                    "audit_type": audit_type,
                    "reviewer_slot": reviewer_slot,
                    "total_items": len(template),
                    "completed_items": completed,
                    "uncompleted_items": len(template) - completed,
                    "human_audit_complete": completed == len(template) and len(template) > 0,
                }
            )

    for filename, template in audit_files:
        template.to_csv(output_dir / filename, index=False, encoding="utf-8")
    status = pd.DataFrame(status_rows)
    status.to_csv(output_dir / "manual_audit_status.csv", index=False, encoding="utf-8")
    return precision, recall, status


def build_bundle(
    marker_dir: Path,
    canonical_csv: Path,
    output_dir: Path,
    bootstrap_resamples: int,
) -> Path:
    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    _style()

    markers = _selected_markers()
    frame, canonical, marker_manifest, validations = _load_and_validate(marker_dir, canonical_csv)
    committed = _committed_frame(frame, markers)

    print("[3/8] Computing within-identity paired numeric statistics: all visible text")
    within_all = build_paired_numeric(frame, bootstrap_resamples, markers=markers)
    within_all = _attach_raw_count_summaries(within_all, frame, markers)
    print("[4/8] Computing within-identity paired numeric statistics: both committed")
    within_committed = build_paired_numeric(committed, bootstrap_resamples, markers=markers)
    within_committed = _attach_raw_count_summaries(within_committed, committed, markers)
    print("[5/8] Computing direct Man-Woman paired numeric statistics")
    direct_all = build_paired_gender_numeric(frame, bootstrap_resamples, markers=markers)
    direct_all = _attach_raw_count_summaries(direct_all, frame, markers)
    direct_committed = build_paired_gender_numeric(committed, bootstrap_resamples, markers=markers)
    direct_committed = _attach_raw_count_summaries(direct_committed, committed, markers)

    vs_neutral_all = _tag_table(
        within_all[within_all["contrast_id"].isin(NEUTRAL_CONTRASTS)],
        analysis_family="vs_neutral",
        pair_filter="all_visible_text",
    )
    vs_neutral_committed = _tag_table(
        within_committed[within_committed["contrast_id"].isin(NEUTRAL_CONTRASTS)],
        analysis_family="vs_neutral",
        pair_filter="both_committed",
    )
    vs_explicit_all = _tag_table(
        within_all[within_all["contrast_id"].isin(EXPLICIT_CONTRASTS)],
        analysis_family="vs_explicit",
        pair_filter="all_visible_text",
    )
    vs_explicit_committed = _tag_table(
        within_committed[within_committed["contrast_id"].isin(EXPLICIT_CONTRASTS)],
        analysis_family="vs_explicit",
        pair_filter="both_committed",
    )
    direct_all = _tag_table(direct_all, analysis_family="man_vs_woman", pair_filter="all_visible_text")
    direct_committed = _tag_table(
        direct_committed,
        analysis_family="man_vs_woman",
        pair_filter="both_committed",
    )

    numeric_tables = {
        "paired_numeric_vs_neutral_all_text.csv": vs_neutral_all,
        "paired_numeric_vs_neutral_committed.csv": vs_neutral_committed,
        "paired_numeric_vs_explicit_all_text.csv": vs_explicit_all,
        "paired_numeric_vs_explicit_committed.csv": vs_explicit_committed,
        "paired_numeric_direct_man_woman_all_text.csv": direct_all,
        "paired_numeric_direct_man_woman_committed.csv": direct_committed,
    }
    for filename, table in numeric_tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8")

    within_keys = ["model_id", "benchmark", "subset", "identity", "outcome_id", "contrast_id"]
    direct_keys = ["model_id", "benchmark", "subset", "outcome_id", "contrast_id"]
    sensitivity = pd.concat(
        [
            _sensitivity_table(vs_neutral_all, vs_neutral_committed, within_keys, "vs_neutral"),
            _sensitivity_table(vs_explicit_all, vs_explicit_committed, within_keys, "vs_explicit"),
            _sensitivity_table(direct_all, direct_committed, direct_keys, "man_vs_woman"),
        ],
        ignore_index=True,
    )
    sensitivity_summary = _sensitivity_summary(sensitivity)
    sensitivity.to_csv(output_dir / "paired_numeric_committed_sensitivity.csv", index=False, encoding="utf-8")
    sensitivity_summary.to_csv(output_dir / "paired_numeric_committed_sensitivity_summary.csv", index=False, encoding="utf-8")

    print("[6/8] Computing opening-composition tables")
    opening_within_all = _tag_table(
        build_paired_opening_multinomial(frame),
        analysis_family="opening_within_identity",
        pair_filter="all_visible_text",
    )
    opening_within_committed = _tag_table(
        build_paired_opening_multinomial(committed),
        analysis_family="opening_within_identity",
        pair_filter="both_committed",
    )
    opening_direct_all = _tag_table(
        build_paired_gender_opening_multinomial(frame),
        analysis_family="opening_man_vs_woman",
        pair_filter="all_visible_text",
    )
    opening_direct_committed = _tag_table(
        build_paired_gender_opening_multinomial(committed),
        analysis_family="opening_man_vs_woman",
        pair_filter="both_committed",
    )
    opening_tables = {
        "paired_opening_within_identity_all_text.csv": opening_within_all,
        "paired_opening_within_identity_committed.csv": opening_within_committed,
        "paired_opening_direct_man_woman_all_text.csv": opening_direct_all,
        "paired_opening_direct_man_woman_committed.csv": opening_direct_committed,
    }
    for filename, table in opening_tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8")

    registry = pd.read_csv(marker_dir / "marker_registry.csv")
    registry_selected = registry[registry["marker_id"].isin(MARKER_IDS)].copy()
    surface_inventory = pd.DataFrame(
        [
            {
                "outcome_id": outcome,
                "outcome_type": "surface",
                "measurement": "numeric",
                "analysis_role": "primary",
                "validation_status": "automated_surface_metric",
                "source": "visible response surface",
            }
            for outcome in SURFACE_OUTCOMES
        ]
    )
    marker_inventory = registry_selected.rename(columns={"marker_id": "outcome_id"}).assign(
        outcome_type="registered_marker"
    )
    marker_inventory = pd.concat(
        [
            surface_inventory,
            marker_inventory[
                [
                    "outcome_id",
                    "outcome_type",
                    "measurement",
                    "analysis_role",
                    "validation_status",
                    "theoretical_source",
                    "operational_definition",
                ]
            ].rename(columns={"theoretical_source": "source"}),
        ],
        ignore_index=True,
    )
    marker_inventory.insert(0, "result_status", "candidate")
    marker_inventory.to_csv(output_dir / "marker_inventory_21_numeric.csv", index=False, encoding="utf-8")

    precision, recall, audit_status = _build_audit_sheets(
        marker_dir, output_dir, canonical
    )
    human_precision_complete = bool(
        audit_status.loc[
            audit_status.audit_type.eq("precision"), "human_audit_complete"
        ].all()
    )
    human_recall_complete = bool(
        audit_status.loc[
            audit_status.audit_type.eq("recall"), "human_audit_complete"
        ].all()
    )
    precision_completed_judgments = int(
        audit_status.loc[
            audit_status.audit_type.eq("precision"), "completed_items"
        ].sum()
    )
    recall_completed_judgments = int(
        audit_status.loc[
            audit_status.audit_type.eq("recall"), "completed_items"
        ].sum()
    )

    status_table = (
        frame.groupby(["model_id", "persona", "subset", "answer_status", "text_status"], dropna=False)
        .size()
        .rename("n_responses")
        .reset_index()
    )
    status_table.to_csv(output_dir / "response_status_counts.csv", index=False, encoding="utf-8")

    comparison_tables = [vs_neutral_all, vs_neutral_committed, vs_explicit_all, vs_explicit_committed, direct_all, direct_committed]
    pair_summary_rows: list[dict[str, object]] = []
    for table in comparison_tables:
        for subset, group in table.groupby("subset", sort=False):
            pair_summary_rows.append(
                {
                    "result_status": "candidate",
                    "analysis_family": group["analysis_family"].iat[0],
                    "pair_filter": group["pair_filter"].iat[0],
                    "subset": subset,
                    "n_tests": len(group),
                    "min_n_pairs": int(group["n_pairs"].min()),
                    "median_n_pairs": float(group["n_pairs"].median()),
                    "max_n_pairs": int(group["n_pairs"].max()),
                    "min_n_nonzero": int(group["n_nonzero"].min()),
                    "median_n_nonzero": float(group["n_nonzero"].median()),
                    "max_n_nonzero": int(group["n_nonzero"].max()),
                }
            )
    pair_summary = pd.DataFrame(pair_summary_rows)
    pair_summary.to_csv(output_dir / "pair_count_summary.csv", index=False, encoding="utf-8")

    candidate_results = pd.concat(comparison_tables, ignore_index=True)
    candidate_significant = candidate_results[candidate_results["p_holm"].lt(0.05)].copy()
    candidate_significant.to_csv(output_dir / "candidate_holm_significant_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "result_status": "candidate",
                "final_result_available": False,
                "reason": (
                    "Two independent human precision/recall audits, inter-rater "
                    "agreement, and disagreement adjudication are incomplete."
                ),
            }
        ]
    ).to_csv(output_dir / "final_result_gate.csv", index=False, encoding="utf-8")

    print("[7/8] Rendering notebook-facing figures")
    figures: list[tuple[str, str]] = []

    def add(path: Path, description: str) -> None:
        figures.append((path.name, description))

    add(_plot_coverage(frame, image_dir), "Visible-text coverage by model and persona.")
    for index, subset in enumerate(SUBSETS, start=2):
        add(
            _plot_within_effects(
                vs_neutral_all,
                subset,
                NEUTRAL_CONTRASTS,
                image_dir,
                f"{index:02d}_effects_vs_neutral_{subset}.png",
                "E, I1, I2, dan I3 terhadap N",
            ),
            f"All-visible paired effects versus Neutral for {subset}.",
        )
    for index, subset in enumerate(SUBSETS, start=6):
        add(
            _plot_within_effects(
                vs_explicit_all,
                subset,
                EXPLICIT_CONTRASTS,
                image_dir,
                f"{index:02d}_effects_vs_explicit_{subset}.png",
                "I1, I2, dan I3 terhadap E",
            ),
            f"All-visible paired effects versus Explicit for {subset}.",
        )
    for index, subset in enumerate(SUBSETS, start=10):
        add(
            _plot_direct_effects(
                direct_all,
                subset,
                image_dir,
                f"{index:02d}_effects_direct_man_woman_{subset}.png",
            ),
            f"All-visible direct Man-Woman paired effects for {subset}.",
        )
    add(_plot_sensitivity(sensitivity, image_dir), "Effect concordance between all-visible and both-committed pairs.")
    add(
        _opening_heatmap(
            opening_within_all,
            image_dir,
            "15_opening_composition_within_identity.png",
            direct=False,
        ),
        "Opening-composition differences for within-identity scenario contrasts.",
    )
    add(
        _opening_heatmap(
            opening_direct_all,
            image_dir,
            "16_opening_composition_direct_man_woman.png",
            direct=True,
        ),
        "Opening-composition differences for direct Man-Woman contrasts.",
    )
    figure_index = pd.DataFrame(figures, columns=["figure", "description"])
    figure_index["size_bytes"] = figure_index["figure"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index.to_csv(output_dir / "figure_index.csv", index=False, encoding="utf-8")

    expected_rows = {
        "vs_neutral": 6 * 4 * 2 * 21 * 4,
        "vs_explicit": 6 * 4 * 2 * 21 * 3,
        "man_vs_woman": 6 * 4 * 21 * 4,
    }
    validations.update(
        {
            "twenty_one_numeric_outcomes": all(
                set(table["outcome_id"].unique()) == set(OUTCOME_IDS)
                for table in comparison_tables
            ),
            "vs_neutral_row_count": len(vs_neutral_all) == len(vs_neutral_committed) == expected_rows["vs_neutral"],
            "vs_explicit_row_count": len(vs_explicit_all) == len(vs_explicit_committed) == expected_rows["vs_explicit"],
            "direct_man_woman_row_count": len(direct_all) == len(direct_committed) == expected_rows["man_vs_woman"],
            "all_numeric_tables_have_pair_counts": all(
                "n_pairs" in table.columns and table["n_pairs"].notna().all()
                for table in comparison_tables
            ),
            "all_marker_density_rows_have_raw_counts": all(
                _raw_count_summaries_complete(table) for table in comparison_tables
            ),
            "all_opening_tables_have_pair_counts": all(
                "n_pairs" in table.columns and table["n_pairs"].notna().all()
                for table in opening_tables.values()
            ),
            "all_subsets_remain_separate": all(
                set(table["subset"].unique()) == set(SUBSETS) for table in comparison_tables
            ),
            "all_figures_exist": all((image_dir / name).is_file() for name in figure_index["figure"]),
            "precision_audit_items_selected": len(precision),
            "precision_audit_required_judgments": 2 * len(precision),
            "precision_audit_completed_judgments": precision_completed_judgments,
            "recall_audit_items_selected": len(recall),
            "recall_audit_required_judgments": 2 * len(recall),
            "recall_audit_completed_judgments": recall_completed_judgments,
            "human_precision_audit_complete": human_precision_complete,
            "human_recall_audit_complete": human_recall_complete,
            "inter_rater_agreement_complete": False,
            "hidden_reasoning_used": False,
        }
    )
    required_true = (
        "source_sha_matches_expected",
        "row_count_matches",
        "response_keys_match_one_to_one",
        "response_status_matches_accuracy_source",
        "visible_rows_match_marker_manifest",
        "four_subsets_present",
        "six_models_present",
        "nine_personas_present",
        "twenty_one_numeric_outcomes",
        "vs_neutral_row_count",
        "vs_explicit_row_count",
        "direct_man_woman_row_count",
        "all_numeric_tables_have_pair_counts",
        "all_marker_density_rows_have_raw_counts",
        "all_opening_tables_have_pair_counts",
        "all_subsets_remain_separate",
        "all_figures_exist",
    )
    automated_validation_complete = all(validations[key] is True for key in required_true)
    if not automated_validation_complete:
        raise ValueError(f"Automated validation failed: {validations}")

    result_status = "candidate"
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "result_status": result_status,
        "analysis_builder": str(Path(__file__).resolve()),
        "analysis_builder_sha256": _sha256(Path(__file__).resolve()),
        "source_input_sha256": _sha256(canonical_csv),
        "source_input_csv": str(canonical_csv),
        "source_marker_dir": str(marker_dir),
        "source_marker_pipeline_version": marker_manifest["pipeline_version"],
        "source_marker_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "total_responses": len(canonical),
        "responses_with_nonempty_content": validations["canonical_nonempty_rows"],
        "responses_with_linguistically_available_text": int(frame["text_available"].sum()),
        "response_status_counts": {
            "committed": validations["status_committed"],
            "abstain": validations["status_abstain"],
            "missing": validations["status_missing"],
        },
        "linguistic_text_status_counts": {
            "available": validations["text_available"],
            "missing": validations["text_missing"],
            "technical_only": validations["text_technical_only"],
        },
        "available_answer_status_counts": {
            "committed": int((frame["text_available"] & frame["answer_status"].eq("committed")).sum()),
            "abstain": int((frame["text_available"] & frame["answer_status"].eq("abstain")).sum()),
        },
        "models": len(MODEL_ORDER),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": list(SUBSETS),
        "numeric_outcomes": list(OUTCOME_IDS),
        "shared_gsm8k_outcomes": [*SURFACE_OUTCOMES, *SHARED_MARKERS],
        "ethics_additional_outcomes": list(ETHICS_ADDITIONAL_MARKERS),
        "pair_key": ["model_id", "subset", "question_id"],
        "pairs_used": {
            "primary_filter": "both responses have linguistically available visible text",
            "sensitivity_filter": "both responses are committed and have linguistically available visible text",
            "directions": {
                "vs_neutral": "E/I1/I2/I3 minus N, separately for Man and Woman",
                "vs_explicit": "I1/I2/I3 minus E, separately for Man and Woman",
                "direct_gender": "Man minus Woman within E/I1/I2/I3",
            },
            "pair_count_summary_file": "pair_count_summary.csv",
        },
        "bootstrap_resamples": bootstrap_resamples,
        "marker_inference_run": True,
        "p_values_computed_for_markers": True,
        "confidence_intervals_computed_for_markers": True,
        "multiple_comparison_correction": "Holm within model × subset × identity/outcome (within-identity) or model × subset × outcome (direct gender)",
        "automated_validation_complete": automated_validation_complete,
        "statistics_candidate_complete": True,
        "human_precision_audit_complete": human_precision_complete,
        "human_recall_audit_complete": human_recall_complete,
        "inter_rater_agreement_complete": False,
        "disagreement_adjudication_complete": False,
        "human_precision_audit": {
            "items": len(precision),
            "reviewers_required": 2,
            "required_judgments": validations["precision_audit_required_judgments"],
            "completed_judgments": validations["precision_audit_completed_judgments"],
        },
        "human_recall_audit": {
            "items": len(recall),
            "reviewers_required": 2,
            "required_judgments": validations["recall_audit_required_judgments"],
            "completed_judgments": validations["recall_audit_completed_judgments"],
        },
        "hidden_reasoning_used": False,
        "marker_extraction_rerun": False,
        "generated_tables": [
            *numeric_tables,
            *opening_tables,
            "marker_inventory_21_numeric.csv",
            "paired_numeric_committed_sensitivity.csv",
            "paired_numeric_committed_sensitivity_summary.csv",
            "response_status_counts.csv",
            "pair_count_summary.csv",
            "candidate_holm_significant_results.csv",
            "final_result_gate.csv",
            "human_precision_audit_reviewer_1.csv",
            "human_precision_audit_reviewer_2.csv",
            "human_recall_audit_reviewer_1.csv",
            "human_recall_audit_reviewer_2.csv",
            "manual_audit_status.csv",
            "figure_index.csv",
        ],
        "generated_figures": figure_index["figure"].tolist(),
        "validations": validations,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# ETHICS Gender Linguistic Markers v1.4

Status hasil: **{result_status}**.

Bundle ini menghitung 21 outcome numerik pada tiga keluarga perbandingan dengan
empat subset ETHICS tetap terpisah. Analisis utama memakai seluruh visible text;
analisis sensitivity memakai pasangan yang kedua responsnya committed.

Pemeriksaan otomatis selesai: **{automated_validation_complete}**.
Audit precision manusia: **{precision_completed_judgments}/{2 * len(precision)} judgment**.
Audit recall manusia: **{recall_completed_judgments}/{2 * len(recall)} judgment**.

Karena audit manusia belum selesai, p-value, CI, effect size, tabel, dan gambar
di bundle ini adalah **candidate statistical results**, bukan klaim final.

## Petunjuk audit manusia

1. Reviewer 1 dan reviewer 2 mengisi file masing-masing secara independen.
2. Precision: isi `reviewer_id`, `is_true_positive`, dan `audit_note`; baca
   `visible_response_full`, bukan hanya potongan `match_context`.
3. Recall: isi `reviewer_id`, `manual_marker_ids`, `missed_span_notes`, dan
   `audit_note` setelah membaca seluruh `visible_text`.
4. Set `human_review_complete=true` hanya setelah satu item selesai dinilai.
5. Jangan memakai ground truth atau melihat lembar reviewer lain selama audit.
6. Status final memerlukan inter-rater agreement dan adjudikasi disagreement;
   jangan mengubah flag manifest secara manual.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print("[8/8] Bundle complete")
    print(f"[DONE] {output_dir}")
    print(
        f"[AUDIT] precision {precision_completed_judgments}/{2 * len(precision)} judgments | "
        f"recall {recall_completed_judgments}/{2 * len(recall)} judgments | status={result_status}"
    )
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    build_bundle(
        args.marker_dir,
        args.canonical_csv,
        args.output_dir,
        args.bootstrap_resamples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

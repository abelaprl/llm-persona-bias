"""Build the thesis-facing v2 visual bundle for GSM8K gender markers.

The source run remains the all-response primary analysis.  This builder adds:

* model x persona response coverage;
* within-model verbosity profiles;
* effect-first numeric volcano plots;
* proxy-separated presence forests with bootstrap intervals;
* a paired both-correct sensitivity analysis;
* a joint paired Stuart-Maxwell analysis of opening composition; and
* ranked lexical lollipops instead of word clouds.

The output directory is deliberately write-once.  Existing candidate results
are never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .contrastive_wordclouds import (  # noqa: E402
    compute_lexical_contrasts,
    load_available_responses,
)
from .lexicon_analysis import (  # noqa: E402
    build_gendered_lexicon_artifacts,
)
from src.core.linguistic_markers.config import (  # noqa: E402
    GENDER_PERSONAS,
    IDENTITY_CONDITIONS,
    OPENING_CATEGORIES,
)
from src.core.linguistic_markers.statistics import (  # noqa: E402
    build_paired_numeric_correctness_sensitivity,
    build_paired_opening_multinomial,
)


EXPECTED_PIPELINE = "gender-lm-v1.2.1-candidate"
IDENTITIES = ("man", "woman")
CONDITION_ORDER = ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")
NEUTRAL_CONTRASTS = (
    "explicit_vs_neutral",
    "implicit_1_vs_neutral",
    "implicit_2_vs_neutral",
    "implicit_3_vs_neutral",
)
PROXY_CONTRASTS = (
    "implicit_1_vs_neutral",
    "implicit_2_vs_neutral",
    "implicit_3_vs_neutral",
)
EXPLICIT_COMPARATOR_CONTRASTS = (
    "implicit_1_vs_explicit",
    "implicit_2_vs_explicit",
    "implicit_3_vs_explicit",
)

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_CODES = {
    model: code
    for model, code in zip(MODEL_LABELS, ("L31", "L33", "MIN", "DSR1", "GPT", "QWN"))
}
MODEL_ORDER = tuple(MODEL_LABELS)
PERSONA_LABELS = {
    "neutral": "N",
    "man_explicit_1": "M-E",
    "man_implicit_1": "M-I1",
    "man_implicit_2": "M-I2",
    "man_implicit_3": "M-I3",
    "woman_explicit_1": "W-E",
    "woman_implicit_1": "W-I1",
    "woman_implicit_2": "W-I2",
    "woman_implicit_3": "W-I3",
}
CONDITION_LABELS = {
    "neutral": "N",
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
IDENTITY_COLORS = {"man": "#2563EB", "woman": "#D946A1"}
CONDITION_COLORS = {
    "neutral": "#64748B",
    "explicit": "#7C3AED",
    "implicit_1": "#0F9D8A",
    "implicit_2": "#E07A1F",
    "implicit_3": "#2C7FB8",
}
FAMILY_COLORS = {
    "text_length": "#264653",
    "lexical_diversity": "#2A9D8F",
    "reasoning_organization": "#E9C46A",
    "epistemic_stance": "#E76F51",
    "logical_connective": "#6A4C93",
    "self_reference": "#1982C4",
    "engagement": "#8AC926",
    "textual_elaboration": "#FF924C",
    "mathematical_presentation": "#C9184A",
}
FAMILY_LABELS = {
    "text_length": "Panjang teks",
    "lexical_diversity": "Keragaman leksikal",
    "reasoning_organization": "Organisasi",
    "epistemic_stance": "Sikap epistemik",
    "logical_connective": "Konektif logis",
    "self_reference": "Rujukan diri",
    "engagement": "Engagement",
    "textual_elaboration": "Elaborasi",
    "mathematical_presentation": "Presentasi matematis",
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
    "math_latex_display": "LaTeX display",
    "math_inline_arithmetic": "Aritmetika inline",
}
OPENING_COLORS = {
    category: color
    for category, color in zip(
        OPENING_CATEGORIES,
        (
            "#4E79A7",
            "#E15759",
            "#59A14F",
            "#B07AA1",
            "#F28E2B",
            "#76B7B2",
            "#EDC948",
            "#9C755F",
            "#FF9DA7",
            "#BAB0AC",
            "#1B9E77",
        ),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.casefold().eq("true")


def _save(fig: plt.Figure, figure_dir: Path, name: str) -> Path:
    path = figure_dir / name
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            # Readable after the wide figures are scaled inside a notebook or
            # thesis page, without letting labels dominate the data marks.
            # Calibrated for figures that are reduced from 15--20 inches to
            # roughly A4 text width. These values print at about 8--11 pt.
            "font.size": 14.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titlesize": 20,
            "legend.fontsize": 18,
            "legend.markerscale": 1.25,
            "legend.handlelength": 2.1,
            "legend.handletextpad": 0.7,
            "legend.columnspacing": 1.45,
            "legend.frameon": False,
        }
    )


def _persona(identity: str, condition: str) -> str:
    return IDENTITY_CONDITIONS[identity][condition]


def _discover_numeric_columns(run_dir: Path) -> list[str]:
    registry = pd.read_csv(run_dir / "marker_registry.csv")
    selected = registry[
        registry["benchmarks"].astype(str).str.contains("gsm8k", regex=False)
        & registry["analysis_role"].ne("quality_control")
        & registry["measurement"].ne("binary")
    ]
    return [f"{marker_id}__density" for marker_id in selected["marker_id"]]


def _discover_presence_columns(run_dir: Path) -> list[str]:
    registry = pd.read_csv(run_dir / "marker_registry.csv")
    selected = registry[
        registry["benchmarks"].astype(str).str.contains("gsm8k", regex=False)
        & registry["analysis_role"].ne("quality_control")
    ]
    return [f"{marker_id}__present" for marker_id in selected["marker_id"]]


def _load_analysis_frame(run_dir: Path) -> pd.DataFrame:
    usecols = [
        "model_id",
        "benchmark",
        "subset",
        "persona",
        "question_id",
        "is_correct",
        "text_status",
        "text_available",
        "reasoning_present",
        "think_block_stripped",
        "technical_lines_stripped",
        "word_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        *_discover_numeric_columns(run_dir),
        *_discover_presence_columns(run_dir),
    ]
    return pd.read_csv(run_dir / "response_records.csv", usecols=usecols, low_memory=False)


def _build_coverage(qc: pd.DataFrame) -> pd.DataFrame:
    result = qc.copy()
    result["target"] = pd.to_numeric(result["n_expected_union"], errors="coerce")
    result["n_source_gap"] = (result["target"] - result["n_records"]).clip(lower=0)
    result["n_not_linguistically_available"] = (
        result["n_records"] - result["n_text_available"]
    )
    result["text_coverage_rate"] = result["n_text_available"] / result["target"]
    return result[
        [
            "model_id",
            "persona",
            "target",
            "n_records",
            "n_response_success",
            "n_text_available",
            "n_missing_text",
            "n_technical_only",
            "n_unseparable_think",
            "n_source_gap",
            "n_not_linguistically_available",
            "text_coverage_rate",
        ]
    ].sort_values(["model_id", "persona"], kind="stable")


def _build_surface_audit(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.assign(
        reasoning_present_flag=_bool_mask(frame["reasoning_present"]),
        think_stripped_flag=_bool_mask(frame["think_block_stripped"]),
        text_available_flag=_bool_mask(frame["text_available"]),
    )
    rows = []
    for model_id, group in work.groupby("model_id", sort=True):
        statuses = group["text_status"].fillna("")
        rows.append(
            {
                "model_id": model_id,
                "n_records": len(group),
                "n_visible_text_available": int(group["text_available_flag"].sum()),
                "n_reasoning_metadata_present": int(group["reasoning_present_flag"].sum()),
                "n_visible_think_blocks_stripped": int(group["think_stripped_flag"].sum()),
                "n_unseparable_think": int((statuses == "unseparable_think").sum()),
                "n_technical_only": int((statuses == "technical_only").sum()),
                "n_missing": int((statuses == "missing").sum()),
                "n_technical_lines_stripped": int(
                    pd.to_numeric(group["technical_lines_stripped"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "hidden_reasoning_used_as_marker_text": False,
            }
        )
    return pd.DataFrame(rows)


def _build_opening_composition(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[_bool_mask(frame["text_available"])].copy()
    rows: list[dict[str, object]] = []
    for model_id in MODEL_ORDER:
        model = available[available["model_id"].eq(model_id)]
        for identity in IDENTITIES:
            for condition in CONDITION_ORDER:
                persona = _persona(identity, condition)
                selected = model[model["persona"].eq(persona)]
                counts = selected["opening_category"].value_counts()
                denominator = int(counts.reindex(OPENING_CATEGORIES, fill_value=0).sum())
                for category in OPENING_CATEGORIES:
                    count = int(counts.get(category, 0))
                    rows.append(
                        {
                            "model_id": model_id,
                            "identity": identity,
                            "condition": condition,
                            "persona": persona,
                            "opening_category": category,
                            "n": count,
                            "n_available": denominator,
                            "proportion": count / denominator if denominator else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _sensitivity_comparison(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_id", "benchmark", "subset", "identity", "outcome_id", "contrast_id"]
    primary_columns = keys + [
        "outcome_family",
        "analysis_role",
        "n_pairs",
        "n_nonzero",
        "median_paired_delta",
        "ci_low",
        "ci_high",
        "rank_biserial",
        "p_holm",
    ]
    sensitivity_columns = keys + [
        "n_pairs",
        "n_nonzero",
        "median_paired_delta",
        "ci_low",
        "ci_high",
        "rank_biserial",
        "p_holm",
    ]
    comparison = primary[primary_columns].merge(
        sensitivity[sensitivity_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_all_responses", "_both_correct"),
    )
    comparison["pair_retention"] = (
        comparison["n_pairs_both_correct"] / comparison["n_pairs_all_responses"]
    )
    tolerance = 1e-12
    main_sign = np.sign(
        comparison["rank_biserial_all_responses"].where(
            comparison["rank_biserial_all_responses"].abs() > tolerance, 0
        )
    )
    sensitivity_sign = np.sign(
        comparison["rank_biserial_both_correct"].where(
            comparison["rank_biserial_both_correct"].abs() > tolerance, 0
        )
    )
    comparison["same_direction_including_zero"] = main_sign.eq(sensitivity_sign)
    comparison["strict_sign_reversal"] = (main_sign * sensitivity_sign) < 0
    comparison["adequate_nonzero_support"] = (
        comparison["n_nonzero_all_responses"].ge(20)
        & comparison["n_nonzero_both_correct"].ge(20)
    )
    comparison["material_sign_reversal"] = (
        comparison["strict_sign_reversal"]
        & comparison["adequate_nonzero_support"]
        & comparison["rank_biserial_all_responses"].abs().ge(0.10)
        & comparison["rank_biserial_both_correct"].abs().ge(0.10)
    )
    comparison["abs_rank_biserial_change"] = (
        comparison["rank_biserial_both_correct"]
        - comparison["rank_biserial_all_responses"]
    ).abs()

    rows: list[dict[str, object]] = []
    primary_only = comparison[comparison["analysis_role"].eq("primary")]
    for (identity, contrast_id), group in primary_only.groupby(
        ["identity", "contrast_id"], sort=True
    ):
        supported = group[group["adequate_nonzero_support"]]
        correlation = spearmanr(
            group["rank_biserial_all_responses"],
            group["rank_biserial_both_correct"],
            nan_policy="omit",
        )
        rows.append(
            {
                "identity": identity,
                "contrast_id": contrast_id,
                "n_primary_tests": len(group),
                "median_pairs_all_responses": float(group["n_pairs_all_responses"].median()),
                "median_pairs_both_correct": float(group["n_pairs_both_correct"].median()),
                "median_pair_retention": float(group["pair_retention"].median()),
                "same_direction_rate_including_zero": float(
                    group["same_direction_including_zero"].mean()
                ),
                "n_tests_with_at_least_20_nonzero_pairs_in_both": len(supported),
                "same_direction_rate_supported": (
                    float(supported["same_direction_including_zero"].mean())
                    if len(supported)
                    else math.nan
                ),
                "n_strict_sign_reversals": int(group["strict_sign_reversal"].sum()),
                "n_material_sign_reversals": int(group["material_sign_reversal"].sum()),
                "spearman_rank_biserial": float(correlation.statistic),
                "median_abs_rank_biserial_change": float(
                    group["abs_rank_biserial_change"].median()
                ),
            }
        )
    return comparison, pd.DataFrame(rows)


def _plot_coverage(coverage: pd.DataFrame, figure_dir: Path) -> Path:
    persona_order = list(GENDER_PERSONAS)
    values = (
        coverage.pivot(index="model_id", columns="persona", values="text_coverage_rate")
        .reindex(index=MODEL_ORDER, columns=persona_order)
        .to_numpy(float)
    )
    available = (
        coverage.pivot(index="model_id", columns="persona", values="n_text_available")
        .reindex(index=MODEL_ORDER, columns=persona_order)
        .to_numpy(float)
    )
    targets = (
        coverage.pivot(index="model_id", columns="persona", values="target")
        .reindex(index=MODEL_ORDER, columns=persona_order)
        .to_numpy(float)
    )
    nonlinguistic = (
        coverage.pivot(
            index="model_id", columns="persona", values="n_not_linguistically_available"
        )
        .reindex(index=MODEL_ORDER, columns=persona_order)
        .to_numpy(float)
    )
    source_gap = (
        coverage.pivot(index="model_id", columns="persona", values="n_source_gap")
        .reindex(index=MODEL_ORDER, columns=persona_order)
        .to_numpy(float)
    )

    cmap = LinearSegmentedColormap.from_list(
        "coverage", ("#FEE2E2", "#FEF3C7", "#DBEAFE", "#1D4ED8")
    )
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    image = ax.imshow(values, cmap=cmap, vmin=0.94, vmax=1.0, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if not np.isfinite(values[row, column]):
                label = "tidak ada"
            else:
                label = f"{int(available[row, column]):,}/{int(targets[row, column]):,}"
                gap = int(source_gap[row, column])
                unavailable = int(nonlinguistic[row, column])
                if gap or unavailable:
                    label += f"\n−{gap + unavailable:,}"
            color = "white" if values[row, column] >= 0.985 else "#172033"
            ax.text(column, row, label, ha="center", va="center", fontsize=14, color=color)
    ax.set_xticks(range(len(persona_order)), [PERSONA_LABELS[p] for p in persona_order])
    ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlabel("Persona (N = neutral; E = explicit; I1/I2/I3 = proxy implicit terpisah)")
    ax.set_ylabel("")
    ax.set_title(
        "Cakupan respons linguistik per model × 9 persona\n"
        "angka = teks tersedia / target; baris kedua = total gap sumber + teks tidak layak"
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.018)
    colorbar.set_label("Proporsi target dengan teks yang dapat dianalisis")
    fig.text(
        0.5,
        0.015,
        "Sel berlabel gap menunjukkan respons missing, technical-only, atau think block yang tidak dapat dipisahkan.",
        ha="center",
        color="#475569",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, figure_dir, "01_coverage_model_persona.png")


def _plot_word_count(descriptives: pd.DataFrame, figure_dir: Path) -> Path:
    data = descriptives[descriptives["outcome_id"].eq("word_count")].copy()
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13), constrained_layout=False)
    x = np.arange(len(CONDITION_ORDER))
    for ax, model_id in zip(axes.flat, MODEL_ORDER):
        model = data[data["model_id"].eq(model_id)]
        for identity, offset in (("man", -0.09), ("woman", 0.09)):
            selected = (
                model[model["identity"].eq(identity)]
                .set_index("condition")
                .reindex(CONDITION_ORDER)
            )
            median = selected["median"].to_numpy(float)
            q1 = selected["q1"].to_numpy(float)
            q3 = selected["q3"].to_numpy(float)
            ax.errorbar(
                x + offset,
                median,
                yerr=np.vstack((median - q1, q3 - median)),
                fmt="o-",
                markersize=8,
                linewidth=1.8,
                capsize=4,
                color=IDENTITY_COLORS[identity],
                label=identity.title(),
            )
        ax.set_title(MODEL_LABELS[model_id], loc="left", fontweight="bold")
        ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
        ax.set_ylabel("Jumlah kata")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(
        "Verbosity dalam setiap model: median dan IQR jumlah kata",
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Density marker dinormalisasi per 100 kata; panel ini menunjukkan apakah persona juga mengubah panjang respons absolut.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.925), h_pad=2.0, w_pad=1.5)
    return _save(fig, figure_dir, "02_verbosity_word_count_median_iqr.png")


def _volcano_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    y_cap: float = 50.0,
) -> None:
    p_values = pd.to_numeric(data["p_holm"], errors="coerce").clip(
        lower=np.nextafter(0.0, 1.0)
    )
    data = data.assign(y=(-np.log10(p_values)).clip(upper=y_cap))
    for family, family_data in data.groupby("outcome_family", sort=True):
        support_size = 42 + 22 * np.log10(
            pd.to_numeric(family_data["n_nonzero"], errors="coerce").fillna(0) + 1
        )
        ax.scatter(
            family_data["rank_biserial"],
            family_data["y"],
            s=support_size,
            alpha=np.where(family_data["p_holm"].lt(0.05), 0.95, 0.55),
            color=FAMILY_COLORS.get(family, "#64748B"),
            edgecolors="white",
            linewidths=0.75,
        )
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.axhline(-math.log10(0.05), color="#94A3B8", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.2, y_cap + 1)
    ax.set_title(title, fontweight="bold")
    ax.grid(color="#E2E8F0", linewidth=0.55, alpha=0.8)

def _plot_volcano_group(
    numeric: pd.DataFrame,
    contrast_ids: tuple[str, ...],
    figure_dir: Path,
    filename: str,
    title: str,
) -> Path:
    data = numeric[
        numeric["analysis_role"].eq("primary")
        & numeric["contrast_id"].isin(contrast_ids)
    ].copy()
    fig, axes = plt.subplots(2, len(contrast_ids), figsize=(5.0 * len(contrast_ids), 9.2))
    axes = np.atleast_2d(axes)
    for row, identity in enumerate(IDENTITIES):
        for column, contrast_id in enumerate(contrast_ids):
            selected = data[
                data["identity"].eq(identity) & data["contrast_id"].eq(contrast_id)
            ]
            _volcano_panel(
                axes[row, column],
                selected,
                title=f"{identity.title()} · {CONTRAST_LABELS[contrast_id]}",
            )
    fig.supylabel("−log₁₀(p Holm), dipotong pada 50", x=0.006)
    fig.supxlabel(
        "Rank-biserial (arah: kondisi kiri − pembanding)",
        y=0.075,
    )
    legend_families = [
        family for family in FAMILY_COLORS if family in set(data["outcome_family"])
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS.get(family, family),
            markersize=12,
        )
        for family in legend_families
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.suptitle(
        title
        + "\nEfek besar tampak jauh dari nol; ukuran titik mengikuti jumlah pasangan yang berubah (nonzero)",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.035, 0.15, 0.99, 0.94), h_pad=2.0, w_pad=1.4)
    return _save(fig, figure_dir, filename)


def _forest_row_order() -> list[tuple[str, str]]:
    return [(model_id, identity) for model_id in MODEL_ORDER for identity in IDENTITIES]


def _plot_proxy_presence_forest(
    presence: pd.DataFrame,
    outcomes: tuple[str, ...],
    figure_dir: Path,
    filename: str,
    title: str,
) -> Path:
    selected = presence[
        presence["outcome_id"].isin(outcomes)
        & presence["contrast_id"].isin(PROXY_CONTRASTS)
    ].copy()
    selected["delta_pp"] = 100 * selected["prevalence_delta"]
    selected["ci_low_pp"] = 100 * selected["ci_low"]
    selected["ci_high_pp"] = 100 * selected["ci_high"]
    row_order = _forest_row_order()
    y_lookup = {key: len(row_order) - 1 - index for index, key in enumerate(row_order)}
    fig, axes = plt.subplots(
        len(outcomes),
        len(PROXY_CONTRASTS),
        figsize=(19, 5.0 * len(outcomes)),
        squeeze=False,
    )
    for outcome_row, outcome_id in enumerate(outcomes):
        outcome_data = selected[selected["outcome_id"].eq(outcome_id)]
        bound = float(
            np.nanmax(
                np.abs(outcome_data[["ci_low_pp", "ci_high_pp"]].to_numpy(float))
            )
        )
        bound = max(1.0, bound * 1.12)
        for contrast_column, contrast_id in enumerate(PROXY_CONTRASTS):
            ax = axes[outcome_row, contrast_column]
            panel = outcome_data[outcome_data["contrast_id"].eq(contrast_id)]
            for _, row in panel.iterrows():
                key = (row["model_id"], row["identity"])
                if key not in y_lookup:
                    continue
                y = y_lookup[key]
                effect = row["delta_pp"]
                lower = effect - row["ci_low_pp"]
                upper = row["ci_high_pp"] - effect
                ax.errorbar(
                    effect,
                    y,
                    xerr=np.array([[max(lower, 0)], [max(upper, 0)]]),
                    fmt="o",
                    markersize=8,
                    capsize=3.5,
                    elinewidth=1.45,
                    color=IDENTITY_COLORS[row["identity"]],
                    alpha=0.9,
                )
            ax.axvline(0, color="#334155", linewidth=0.8)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-0.8, len(row_order) - 0.2)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
            if outcome_row == 0:
                ax.set_title(CONTRAST_LABELS[contrast_id], fontweight="bold")
            if contrast_column == 0:
                ax.set_yticks(
                    [y_lookup[key] for key in row_order],
                    [f"{MODEL_LABELS[m]} · {i.title()}" for m, i in row_order],
                    fontsize=14.5,
                )
                ax.set_ylabel(OUTCOME_LABELS.get(outcome_id, outcome_id), fontweight="bold")
            else:
                ax.set_yticks([])
    fig.supxlabel(
        "Δ respons yang memuat marker (pp), 95% bootstrap CI",
        y=0.052,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=IDENTITY_COLORS[identity],
            label=identity.title(),
        )
        for identity in IDENTITIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.003),
    )
    fig.suptitle(
        title + "\nSetiap kolom adalah proxy yang berbeda; sumbu sama hanya di dalam satu baris marker",
        y=0.998,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.955), h_pad=2.0, w_pad=1.1)
    return _save(fig, figure_dir, filename)


def _plot_correctness_sensitivity(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    data = comparison[
        comparison["analysis_role"].eq("primary")
        & comparison["contrast_id"].isin(NEUTRAL_CONTRASTS)
    ].copy()
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), sharex=True, sharey=True)
    for row_index, identity in enumerate(IDENTITIES):
        for column_index, contrast_id in enumerate(NEUTRAL_CONTRASTS):
            ax = axes[row_index, column_index]
            panel = data[
                data["identity"].eq(identity) & data["contrast_id"].eq(contrast_id)
            ]
            for family, family_data in panel.groupby("outcome_family", sort=True):
                ax.scatter(
                    family_data["rank_biserial_all_responses"],
                    family_data["rank_biserial_both_correct"],
                    s=64,
                    color=FAMILY_COLORS.get(family, "#64748B"),
                    alpha=0.88,
                    edgecolors="white",
                    linewidths=0.7,
                )
            ax.plot((-1, 1), (-1, 1), color="#475569", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="#CBD5E1", linewidth=0.7)
            ax.axvline(0, color="#CBD5E1", linewidth=0.7)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.grid(color="#E2E8F0", linewidth=0.5)
            ax.set_title(f"{identity.title()} · {CONTRAST_LABELS[contrast_id]}", fontweight="bold")
            metric = summary[
                summary["identity"].eq(identity)
                & summary["contrast_id"].eq(contrast_id)
            ]
            if len(metric):
                metric = metric.iloc[0]
                ax.text(
                    0.04,
                    0.96,
                    (
                        f"arah sama* {metric['same_direction_rate_supported']:.0%}\n"
                        f"median pair retained {metric['median_pair_retention']:.0%}"
                    ),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=13.5,
                    bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#CBD5E1"},
                )
            reversals = panel[panel["material_sign_reversal"]]
            for _, point in reversals.iterrows():
                label = f"{MODEL_CODES.get(point['model_id'], '?')}·{OUTCOME_LABELS.get(point['outcome_id'], point['outcome_id'])}"
                ax.annotate(
                    label,
                    (
                        point["rank_biserial_all_responses"],
                        point["rank_biserial_both_correct"],
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=12.5,
                    color="#B91C1C",
                )
    fig.supxlabel("Rank-biserial · semua respons (analisis utama)", y=0.072)
    fig.supylabel("Rank-biserial · pasangan keduanya benar", x=0.006)
    fig.suptitle(
        "Analisis sensitivitas correctness: apakah arah efek bertahan?\n"
        "Filter diterapkan setelah pairing; hanya soal yang benar pada kedua kondisi dipertahankan",
        y=0.995,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Titik pada garis diagonal = efek stabil. *Arah dihitung untuk outcome dengan ≥20 pasangan nonzero di kedua analisis; utama tetap seluruh respons.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.13, 0.99, 0.94), h_pad=1.7, w_pad=1.2)
    return _save(fig, figure_dir, "07_correctness_sensitivity_effect_concordance.png")


def _plot_opening_composition(composition: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(6, 2, figsize=(17, 24), sharex=True)
    y = np.arange(len(CONDITION_ORDER))
    active_categories = tuple(
        category
        for category in OPENING_CATEGORIES
        if composition.loc[
            composition["opening_category"].eq(category), "n"
        ].sum()
        > 0
    )
    for model_row, model_id in enumerate(MODEL_ORDER):
        for identity_column, identity in enumerate(IDENTITIES):
            ax = axes[model_row, identity_column]
            panel = composition[
                composition["model_id"].eq(model_id)
                & composition["identity"].eq(identity)
            ]
            cumulative = np.zeros(len(CONDITION_ORDER), dtype=float)
            for category in active_categories:
                values = (
                    panel[panel["opening_category"].eq(category)]
                    .set_index("condition")
                    .reindex(CONDITION_ORDER)["proportion"]
                    .fillna(0)
                    .to_numpy(float)
                )
                ax.barh(
                    y,
                    values,
                    left=cumulative,
                    height=0.68,
                    color=OPENING_COLORS[category],
                    label=category.replace("_", " "),
                )
                cumulative += values
            ax.set_xlim(0, 1)
            ax.set_yticks(y, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.55)
            ax.set_title(
                f"{MODEL_LABELS[model_id]} · {identity.title()}",
                loc="left",
                fontweight="bold",
            )
            if identity_column == 0:
                ax.set_ylabel("Kondisi")
            if model_row == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Proporsi opening (jumlah = 100%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(
        "Komposisi kategori pembukaan respons\n"
        "Setiap bar menunjukkan distribusi seluruh kategori yang berjumlah 100%",
        y=0.995,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.052,
        "Perbedaan komposisi antarkondisi dirangkum dengan marginal total variation pada paired_opening_multinomial.csv.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0.01, 0.075, 0.99, 0.955), h_pad=1.65, w_pad=1.1)
    return _save(fig, figure_dir, "08_opening_composition_100pct.png")


def _top_diverging_terms(group: pd.DataFrame, n_each: int = 5) -> pd.DataFrame:
    positive = group[group["mean_model_delta_pp"] > 0].nlargest(
        n_each, "mean_model_delta_pp"
    )
    negative = group[group["mean_model_delta_pp"] < 0].nsmallest(
        n_each, "mean_model_delta_pp"
    )
    return pd.concat((negative, positive), ignore_index=True).sort_values(
        "mean_model_delta_pp", kind="stable"
    )


def _plot_lexical_group(
    weights: pd.DataFrame,
    contrast_ids: tuple[str, ...],
    figure_dir: Path,
    filename: str,
    title: str,
) -> Path:
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    max_effect = 0.0
    for contrast_id in contrast_ids:
        for identity in IDENTITIES:
            group = weights[
                weights["contrast_id"].eq(contrast_id)
                & weights["identity"].eq(identity)
            ]
            top = _top_diverging_terms(group)
            panels[(contrast_id, identity)] = top
            if len(top):
                max_effect = max(max_effect, float(top["mean_model_delta_pp"].abs().max()))
    max_effect = max(max_effect * 1.32, 0.5)

    fig, axes = plt.subplots(
        len(contrast_ids),
        2,
        figsize=(17, 4.4 * len(contrast_ids)),
        squeeze=False,
        sharex=True,
    )
    for row, contrast_id in enumerate(contrast_ids):
        for column, identity in enumerate(IDENTITIES):
            ax = axes[row, column]
            panel = panels[(contrast_id, identity)].reset_index(drop=True)
            positions = np.arange(len(panel))
            comparator = str(panel["right_condition"].iloc[0]) if len(panel) else "neutral"
            colors = np.where(
                panel["mean_model_delta_pp"].to_numpy(float) > 0,
                IDENTITY_COLORS[identity],
                CONDITION_COLORS.get(comparator, "#64748B"),
            )
            ax.hlines(
                positions,
                0,
                panel["mean_model_delta_pp"],
                color=colors,
                linewidth=2.2,
                alpha=0.92,
            )
            ax.scatter(
                panel["mean_model_delta_pp"],
                positions,
                color=colors,
                s=85,
                zorder=3,
                edgecolors="white",
                linewidths=0.85,
            )
            for position, value in enumerate(panel["mean_model_delta_pp"]):
                horizontal = "left" if value >= 0 else "right"
                offset = 4 if value >= 0 else -4
                ax.annotate(
                    f"{value:+.2f}",
                    (value, position),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=horizontal,
                    va="center",
                    fontsize=13,
                    color="#334155",
                )
            ax.axvline(0, color="#334155", linewidth=0.8)
            ax.set_yticks(positions, panel["token"], fontsize=15)
            ax.set_xlim(-max_effect, max_effect)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
            ax.set_title(
                f"{identity.title()} · {CONTRAST_LABELS[contrast_id]}",
                loc="left",
                fontweight="bold",
            )
    fig.supxlabel(
        "Rata-rata setara per model: Δ prevalensi token per respons (pp)",
        y=0.055,
    )
    fig.suptitle(
        title + "\nTop-5 tiap arah; paired by question; token hadir/tidak hadir per respons; arah stabil pada ≥4/6 model",
        y=0.998,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Sumbu menunjukkan selisih prevalensi token per respons: kiri lebih lazim pada pembanding, kanan lebih lazim pada kondisi persona.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.945), h_pad=2.0, w_pad=2.2)
    return _save(fig, figure_dir, filename)


def _audit_status(run_dir: Path) -> pd.DataFrame:
    hits = pd.read_csv(run_dir / "audit_marker_hits.csv", low_memory=False)
    recall = pd.read_csv(run_dir / "audit_recall_responses.csv", low_memory=False)
    hit_annotated = hits["reviewer"].fillna("").astype(str).str.strip().ne("")
    recall_annotated = recall["reviewer"].fillna("").astype(str).str.strip().ne("")
    return pd.DataFrame(
        [
            {
                "audit": "precision_occurrence",
                "n_sample_rows": len(hits),
                "n_annotated_rows": int(hit_annotated.sum()),
                "n_models": hits["model_id"].nunique(),
                "n_personas": hits["persona"].nunique(),
            },
            {
                "audit": "recall_response",
                "n_sample_rows": len(recall),
                "n_annotated_rows": int(recall_annotated.sum()),
                "n_models": recall["model_id"].nunique(),
                "n_personas": recall["persona"].nunique(),
            },
        ]
    )


def refresh_explanatory_figures(output_dir: Path) -> tuple[Path, ...]:
    """Redraw caption-only figures from the existing auditable CSV tables."""

    output_dir = output_dir.resolve()
    figure_dir = output_dir / "images"
    required = (
        output_dir / "response_counts_model_persona.csv",
        output_dir / "opening_composition.csv",
        output_dir / "lexical_contrast_token_weights.csv",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot refresh figures; missing tables: "
            + ", ".join(str(path) for path in missing)
        )
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_style()
    coverage = pd.read_csv(required[0])
    opening = pd.read_csv(required[1])
    lexical = pd.read_csv(required[2])
    return (
        _plot_coverage(coverage, figure_dir),
        _plot_opening_composition(opening, figure_dir),
        _plot_lexical_group(
            lexical,
            NEUTRAL_CONTRASTS,
            figure_dir,
            "09_ranked_lexical_lollipop_vs_neutral.png",
            "Token distingtif persona terhadap neutral",
        ),
        _plot_lexical_group(
            lexical,
            EXPLICIT_COMPARATOR_CONTRASTS,
            figure_dir,
            "10_ranked_lexical_lollipop_implicit_vs_explicit.png",
            "Token distingtif tiga proxy implicit terhadap explicit",
        ),
    )


def build_revision_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int | None = None,
    reuse_existing: bool = False,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    bundle_manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if reuse_existing and bundle_manifest_path.is_file():
            return json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    source_manifest_path = run_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("pipeline_version") != EXPECTED_PIPELINE:
        warnings.warn(
            "Pipeline version differs from the version used to design this visual bundle: "
            f"{source_manifest.get('pipeline_version')!r}",
            stacklevel=2,
        )
    if source_manifest.get("correctness_filter_applied") is not False:
        raise ValueError("Primary source must be the unfiltered all-response analysis.")
    expected_records = 71_215
    if source_manifest.get("response_records") != expected_records:
        warnings.warn(
            f"Discovered {source_manifest.get('response_records'):,} records; "
            f"the documented reference run had {expected_records:,}. Continuing with discovered data.",
            stacklevel=2,
        )
    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else source_manifest.get("bootstrap_resamples", 10_000)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir = output_dir / "images"
    figure_dir.mkdir()
    _plot_style()

    qc = pd.read_csv(run_dir / "quality_control.csv")
    descriptives = pd.read_csv(run_dir / "condition_descriptives.csv")
    numeric = pd.read_csv(run_dir / "paired_numeric.csv")
    presence = pd.read_csv(run_dir / "paired_binary.csv")
    analysis_frame = _load_analysis_frame(run_dir)

    coverage = _build_coverage(qc)
    surface_audit = _build_surface_audit(analysis_frame)
    opening_composition = _build_opening_composition(analysis_frame)
    opening_multinomial = build_paired_opening_multinomial(analysis_frame)
    sensitivity = build_paired_numeric_correctness_sensitivity(
        analysis_frame,
        bootstrap_resamples=resamples,
    )
    sensitivity_comparison, sensitivity_summary = _sensitivity_comparison(
        numeric,
        sensitivity,
    )
    audit_status = _audit_status(run_dir)

    coverage.to_csv(output_dir / "response_counts_model_persona.csv", index=False)
    surface_audit.to_csv(output_dir / "visible_surface_audit.csv", index=False)
    opening_composition.to_csv(output_dir / "opening_composition.csv", index=False)
    opening_multinomial.to_csv(
        output_dir / "paired_opening_multinomial.csv", index=False
    )
    sensitivity.to_csv(
        output_dir / "correctness_sensitivity_paired_numeric.csv", index=False
    )
    sensitivity_comparison.to_csv(
        output_dir / "correctness_sensitivity_comparison.csv", index=False
    )
    sensitivity_summary.to_csv(
        output_dir / "correctness_sensitivity_summary.csv", index=False
    )
    audit_status.to_csv(output_dir / "manual_audit_status.csv", index=False)

    figures: list[Path] = []
    figures.append(_plot_coverage(coverage, figure_dir))
    figures.append(_plot_word_count(descriptives, figure_dir))
    figures.append(
        _plot_volcano_group(
            numeric,
            NEUTRAL_CONTRASTS,
            figure_dir,
            "03_numeric_effect_volcano_vs_neutral.png",
            "Effect map marker numerik terhadap baseline neutral",
        )
    )
    figures.append(
        _plot_volcano_group(
            numeric,
            EXPLICIT_COMPARATOR_CONTRASTS,
            figure_dir,
            "04_numeric_effect_volcano_implicit_vs_explicit.png",
            "Effect map tiga proxy implicit terhadap explicit",
        )
    )
    figures.append(
        _plot_proxy_presence_forest(
            presence,
            (
                "organization_explicit_step",
                "connective_causal",
                "math_latex_display",
            ),
            figure_dir,
            "05_proxy_presence_forest_structure_logic.png",
            "Forest plot proxy implicit: struktur, koneksi logis, dan format matematis",
        )
    )
    figures.append(
        _plot_proxy_presence_forest(
            presence,
            (
                "stance_hedge",
                "stance_self_correction",
                "self_reference_inclusive_we",
            ),
            figure_dir,
            "06_proxy_presence_forest_stance_voice.png",
            "Forest plot proxy implicit: stance dan suara penalaran",
        )
    )
    figures.append(
        _plot_correctness_sensitivity(
            sensitivity_comparison,
            sensitivity_summary,
            figure_dir,
        )
    )
    figures.append(_plot_opening_composition(opening_composition, figure_dir))

    lexical_frame = load_available_responses(run_dir)
    lexical_weights, lexical_metadata = compute_lexical_contrasts(lexical_frame)
    lexical_weights.to_csv(
        output_dir / "lexical_contrast_token_weights.csv", index=False
    )
    (output_dir / "lexical_contrast_metadata.json").write_text(
        json.dumps(lexical_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures.append(
        _plot_lexical_group(
            lexical_weights,
            NEUTRAL_CONTRASTS,
            figure_dir,
            "09_ranked_lexical_lollipop_vs_neutral.png",
            "Token distingtif persona terhadap neutral",
        )
    )
    figures.append(
        _plot_lexical_group(
            lexical_weights,
            EXPLICIT_COMPARATOR_CONTRASTS,
            figure_dir,
            "10_ranked_lexical_lollipop_implicit_vs_explicit.png",
            "Token distingtif tiga proxy implicit terhadap explicit",
        )
    )

    figure_descriptions = {
        "01_coverage_model_persona.png": "Coverage per model x 9 persona dengan target dan gap konkret.",
        "02_verbosity_word_count_median_iqr.png": "Median dan IQR word count within-model untuk memeriksa verbositas.",
        "03_numeric_effect_volcano_vs_neutral.png": "Rank-biserial vs -log10 Holm; E/I1/I2/I3 tidak diagregasi.",
        "04_numeric_effect_volcano_implicit_vs_explicit.png": "I1/I2/I3 vs explicit dalam panel terpisah.",
        "05_proxy_presence_forest_structure_logic.png": "Delta prevalence dan 95% paired bootstrap CI untuk struktur/logika/LaTeX.",
        "06_proxy_presence_forest_stance_voice.png": "Delta prevalence dan 95% paired bootstrap CI untuk stance/voice.",
        "07_correctness_sensitivity_effect_concordance.png": "Efek utama dibanding pairwise both-correct sensitivity.",
        "08_opening_composition_100pct.png": "Opening sebagai komposisi multinomial 100%.",
        "09_ranked_lexical_lollipop_vs_neutral.png": "Token ranking presisi menggantikan word cloud, vs neutral.",
        "10_ranked_lexical_lollipop_implicit_vs_explicit.png": "Token ranking presisi, I1/I2/I3 vs explicit.",
    }
    figure_index = pd.DataFrame(
        [
            {
                "figure": path.name,
                "description": figure_descriptions[path.name],
            }
            for path in figures
        ]
    )
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)

    generated_files = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    bundle_manifest = {
        "status": "candidate_pending_manual_precision_recall_audit",
        "bundle_version": "gender-gsm8k-bab4-visual-v2",
        "source_run": str(run_dir),
        "source_pipeline_version": source_manifest.get("pipeline_version"),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_response_records": int(source_manifest.get("response_records", 0)),
        "primary_correctness_filter_applied": False,
        "correctness_sensitivity_filter": "paired questions where both conditions are correct",
        "bootstrap_resamples": resamples,
        "hidden_reasoning_used": False,
        "implicit_proxies_aggregated_for_inference": False,
        "opening_primary_test": "stuart_maxwell_marginal_homogeneity",
        "figures": figure_descriptions,
        "generated_files_before_manifest": generated_files,
    }
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# Gender GSM8K linguistic-marker visual revision v2

Status: **candidate pending manual precision/recall audit**.

Source run: `{run_dir}`  
Source records: {source_manifest.get('response_records', 0):,}  
Bootstrap resamples: {resamples:,}

## Methodological decisions

- The primary analysis still includes correct and incorrect responses.
- Correctness sensitivity is paired first, then restricted to questions where
  both compared conditions are correct.
- E, I1, I2, and I3 are never averaged into one inferential condition.
- Numeric figures lead with rank-biserial effect size; p Holm is the vertical
  axis, not the visual selection rule.
- Forest plots report paired prevalence differences with 95% bootstrap CIs.
- Opening categories are visualized as 100% compositions and tested jointly
  with Stuart-Maxwell marginal homogeneity.
- Ranked lexical lollipops replace word clouds and retain exact delta values.
- Marker extraction uses visible final-answer text only; hidden reasoning is
  metadata and never a fallback text surface.

The two manual audit CSVs in the source run are not complete unless
`manual_audit_status.csv` reports annotated rows.  Do not promote candidate
claims to final Bab IV findings before precision, recall, and agreement are
reported.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    build_gendered_lexicon_artifacts(
        run_dir,
        output_dir,
        bootstrap_resamples=resamples,
    )
    return json.loads(bundle_manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()
    manifest = build_revision_bundle(
        args.run_dir,
        args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(
        "[DONE]",
        manifest["bundle_version"],
        "->",
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

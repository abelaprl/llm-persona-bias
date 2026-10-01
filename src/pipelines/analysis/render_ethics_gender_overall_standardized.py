"""Render the combined ETHICS Gender presentation bundle.

The parent bundle is intentionally model/persona oriented, like the GSM8K
Gender presentation.  Status outcomes are pooled over all ETHICS subsets. The
accuracy bars use only the construct-valid subsets (commonsense and justice),
while subset-specific tables and figures are written below ``per_subset``.
No source result bundle is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ethics_accuracy_combined_statistics import compute_combined_conditional_accuracy
from ethics_accuracy_statistical_figures import (
    plot_effect_holm_heatmap,
    plot_exact_mcnemar_holm_heatmaps,
)

from render_ethics_gender_accuracy_standardized import (
    ABSTAIN_REASON_LABELS,
    BACKGROUND,
    DEFAULT_OUTPUT_DIR as OLD_DEFAULT_OUTPUT_DIR,
    MODEL_AXIS_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    PERSONA_LABELS,
    PERSONA_ORDER,
    SOURCE_BUNDLE,
    STATUS_COLORS,
    STATUS_LABELS,
    STATUS_ORDER,
    SUBSET_LABELS,
    SUBSET_ORDER,
    VALID_SUBSETS,
    _load_source_tables,
    derive_tables,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = OLD_DEFAULT_OUTPUT_DIR

STATUS_RATE_COLUMNS = {
    "committed": "commit_rate",
    "abstain": "abstain_rate",
    "missing": "missing_rate",
}

CATEGORY_SPECS = [
    ("Man Explicit", "Man E", ["man_explicit_1"]),
    ("Man Implicit Avg", "Man I", ["man_implicit_1", "man_implicit_2", "man_implicit_3"]),
    ("Woman Explicit", "Woman E", ["woman_explicit_1"]),
    ("Woman Implicit Avg", "Woman I", ["woman_implicit_1", "woman_implicit_2", "woman_implicit_3"]),
    ("Neutral", "Neutral", ["neutral"]),
]
CATEGORY_ORDER = [item[0] for item in CATEGORY_SPECS]
CATEGORY_SHORT_LABELS = {item[0]: item[1] for item in CATEGORY_SPECS}
CATEGORY_PERSONAS = {item[0]: item[2] for item in CATEGORY_SPECS}

ACCURACY_COLORS = ["#0047AB", "#0086A8", "#B80F2E", "#E45700", "#30343B"]
DELTA_COLORS = ["#0047AB", "#B80F2E", "#5B21B6"]
FONT_DELTA_PT = 2.0
FIGURE_SCALE = 1.12
ANNOTATION_SIZE = 19
DELTA_ANNOTATION_SIZE = 18
TITLE_SIZE = 24
STATUS_ANNOTATION_SIZE = 18
REASON_ANNOTATION_SIZE = 18
REASON_TICK_SIZE = 19
REASON_TITLE_SIZE = 30
REASON_AXIS_LABEL_SIZE = 24
REASON_XTICK_SIZE = 19


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": "#17212B",
            "axes.labelcolor": "#17212B",
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "legend.title_fontsize": 18,
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _figsize(width: float, height: float) -> tuple[float, float]:
    return width * FIGURE_SCALE, height * FIGURE_SCALE


def _save(fig: mpl.figure.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor=BACKGROUND, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _aggregate_categories(frame: pd.DataFrame, count_columns: list[str]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_frame = frame[frame["model"].eq(model)]
        for category, short_label, personas in CATEGORY_SPECS:
            group = model_frame[model_frame["persona"].isin(personas)]
            row = {"model": model, "model_label": MODEL_LABELS[model], "category": category, "category_short": short_label}
            for column in count_columns:
                row[column] = float(group[column].sum())
            rows.append(row)
    result = pd.DataFrame(rows)
    for column in ["total", "nonmissing", "committed", "abstain", "missing", "n_correct"]:
        if column in result:
            result[column] = result[column].round().astype(int)
    result["commit_rate"] = _safe_rate(result["committed"], result["total"])
    result["commit_rate_nonmissing"] = _safe_rate(result["committed"], result["nonmissing"])
    result["abstain_rate"] = _safe_rate(result["abstain"], result["total"])
    result["missing_rate"] = _safe_rate(result["missing"], result["total"])
    return result


def _build_overall_tables(tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    summary = derived["summary"].copy()
    status = derived["status"].copy()
    valid = summary[summary["subset"].isin(VALID_SUBSETS)].copy()

    accuracy = _aggregate_categories(
        valid,
        ["total", "nonmissing", "committed", "abstain", "missing", "n_correct"],
    )
    accuracy["accuracy_committed"] = _safe_rate(accuracy["n_correct"], accuracy["committed"])
    accuracy["effective_success_rate_nonmissing"] = _safe_rate(
        accuracy["n_correct"], accuracy["nonmissing"]
    )

    status_categories = _aggregate_categories(
        status,
        ["total", "nonmissing", "committed", "abstain", "missing"],
    )

    reason_source = derived["reason_model"].copy()
    reason_source["model_label"] = reason_source["model"].map(MODEL_LABELS)
    reason_rows = []
    for model in MODEL_ORDER:
        model_frame = reason_source[reason_source["model"].eq(model)]
        for category, short_label, personas in CATEGORY_SPECS:
            group = model_frame[model_frame["persona"].isin(personas)]
            for reason, reason_frame in group.groupby(["abstain_reason", "reason_label"], sort=False):
                reason_code, reason_label = reason
                reason_rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "category": category,
                        "category_short": short_label,
                        "abstain_reason": reason_code,
                        "reason_label": reason_label,
                        "reason_count": int(reason_frame["reason_count"].sum()),
                    }
                )
    reason_categories = pd.DataFrame(reason_rows)
    reason_categories = reason_categories.merge(
        status_categories[["model", "category", "abstain", "nonmissing"]],
        on=["model", "category"],
        validate="many_to_one",
    )
    reason_categories["pct_of_abstain"] = _safe_rate(
        reason_categories["reason_count"], reason_categories["abstain"]
    )
    reason_categories["pct_of_nonmissing"] = _safe_rate(
        reason_categories["reason_count"], reason_categories["nonmissing"]
    )

    reason_model = (
        reason_source.groupby(["model", "model_label", "abstain_reason", "reason_label"], as_index=False)["reason_count"]
        .sum()
    )
    model_status = status.groupby(["model", "model_label"], as_index=False)[["abstain", "nonmissing"]].sum()
    reason_model = reason_model.merge(model_status, on=["model", "model_label"], validate="many_to_one")
    reason_model["pct_of_abstain"] = _safe_rate(reason_model["reason_count"], reason_model["abstain"])
    reason_model["pct_of_nonmissing"] = _safe_rate(reason_model["reason_count"], reason_model["nonmissing"])

    paired = derived["paired_primary"].copy()
    return {
        "accuracy": accuracy,
        "status": status_categories,
        "reason_categories": reason_categories,
        "reason_model": reason_model,
        "paired": paired,
        "condition_summary": summary,
        "source_overview": tables["model_overview"].copy(),
        "status_independence": tables["status_independence"].copy(),
    }


def _ordered_category_pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    result = frame.pivot(index="model_label", columns="category", values=value)
    return result.reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=CATEGORY_ORDER,
    )


def _annotate_accuracy(ax: mpl.axes.Axes, offset: int = 5, *, fmt: str = '.4f', size: float = ANNOTATION_SIZE) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            format(value, fmt),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, offset),
            textcoords="offset points",
            fontsize=size,
            fontweight="normal",
            rotation=90,
            rotation_mode="default",
            annotation_clip=False,
        )


def _annotate_delta(ax: mpl.axes.Axes, offset: int = 10) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            xytext=(0, offset if value >= 0 else -offset),
            textcoords="offset points",
            fontsize=DELTA_ANNOTATION_SIZE,
            fontweight="normal",
            rotation=90,
            rotation_mode="default",
            annotation_clip=False,
        )


def _annotate_status_segment(
    ax: mpl.axes.Axes,
    index: int,
    bottom: float,
    value: float,
) -> None:
    if value < 5:
        return
    fontsize = STATUS_ANNOTATION_SIZE if value >= 8 else max(10, STATUS_ANNOTATION_SIZE - 3)
    ax.text(
        index,
        bottom + value / 2,
        f"{value:.1f}%",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="normal",
        clip_on=True,
        zorder=5,
    )


def plot_accuracy(accuracy: pd.DataFrame, figure_dir: Path) -> list[Path]:
    values = _ordered_category_pivot(accuracy, "accuracy_committed")
    neutral_average = values["Neutral"].mean()
    fig, ax = plt.subplots(figsize=(27, 11))
    values.plot(
        kind="bar",
        width=0.84,
        color=ACCURACY_COLORS,
        ax=ax,
    )
    baseline_label = f"Neutral Avg ({neutral_average:.3f})"
    ax.axhline(neutral_average, color="#B80F2E", linestyle="--", linewidth=2.0, label=baseline_label, zorder=1)
    _annotate_accuracy(ax, offset=4, fmt='.3f', size=20)
    ax.set_ylim(0, 1)
    ax.set_title("ETHICS Gender: Akurasi per Persona", fontsize=TITLE_SIZE, fontweight="bold", pad=18)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy committed (construct-valid ETHICS)")
    ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER], rotation=0, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [handle for index, handle in enumerate(handles) if index != baseline_index]
    labels = [labels[baseline_index]] + [label for index, label in enumerate(labels) if index != baseline_index]
    ax.legend(handles, labels, title="Persona & Baseline", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, figure_dir, "01_ethics_absolute_accuracy_five_categories")


def _delta_frame(accuracy: pd.DataFrame, implicit: bool = False) -> pd.DataFrame:
    values = _ordered_category_pivot(accuracy, "accuracy_committed")
    man_implicit = values[["Man Implicit Avg"]].iloc[:, 0]
    woman_implicit = values[["Woman Implicit Avg"]].iloc[:, 0]
    man_explicit = values[["Man Explicit"]].iloc[:, 0]
    woman_explicit = values[["Woman Explicit"]].iloc[:, 0]
    neutral = values["Neutral"]
    if implicit:
        return pd.DataFrame(
            {
                "Man Implicit Avg − Explicit": man_implicit - man_explicit,
                "Woman Implicit Avg − Explicit": woman_implicit - woman_explicit,
                "Man Avg − Woman Avg": (man_implicit - woman_implicit),
            }
        )
    return pd.DataFrame(
        {
            "Man Avg − Neutral": man_implicit - neutral,
            "Woman Avg − Neutral": woman_implicit - neutral,
            "Man Avg − Woman Avg": man_implicit - woman_implicit,
        }
    )


def plot_delta(accuracy: pd.DataFrame, figure_dir: Path, *, implicit: bool) -> list[Path]:
    delta = _delta_frame(accuracy, implicit=implicit)
    title = (
        "ETHICS Gender: Perubahan Akurasi Implisit terhadap Eksplisit"
        if implicit
        else "ETHICS Gender: Perubahan Akurasi terhadap Netral"
    )
    stem = "03_ethics_delta_implicit_vs_explicit" if implicit else "02_ethics_delta_vs_neutral"
    fig, ax = plt.subplots(figsize=_figsize(19, 9))
    delta.plot(kind="bar", width=0.78, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_delta(ax)
    limit = max(0.001, float(np.nanmax(np.abs(delta.to_numpy()))))
    ax.set_ylim(-limit * 1.45, limit * 1.45)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference on accuracy scale")
    ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER], rotation=0, ha="center")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save(fig, figure_dir, stem)


def _status_category_frame(status: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = status[status["model"].eq(model)].set_index("category").reindex(CATEGORY_ORDER)
    return frame


def plot_status(status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=_figsize(22, 18), constrained_layout=False)
    axes_flat = list(axes.flat)
    for ax, model in zip(axes_flat, MODEL_ORDER, strict=True):
        frame = _status_category_frame(status, model)
        x = np.arange(len(CATEGORY_ORDER))
        bottom = np.zeros(len(CATEGORY_ORDER))
        for status_name in STATUS_ORDER:
            values = frame[STATUS_RATE_COLUMNS[status_name]].to_numpy() * 100
            ax.bar(x, values, bottom=bottom, color=STATUS_COLORS[status_name], edgecolor="white", linewidth=0.7, label=STATUS_LABELS[status_name])
            for index, value in enumerate(values):
                _annotate_status_segment(ax, index, bottom[index], value)
            bottom += values
        ax.set_title(MODEL_LABELS[model], fontsize=18, fontweight="bold")
        ax.set_xticks(x, [CATEGORY_SHORT_LABELS[item] for item in CATEGORY_ORDER], rotation=0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% respons")
        ax.grid(axis="x", visible=False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.89, bottom=0.06, hspace=0.42, wspace=0.20)
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=True)
    fig.suptitle("ETHICS Gender: Status Respons per Model dan Persona", fontsize=TITLE_SIZE, fontweight="bold", y=0.98)
    return _save(fig, figure_dir, "04_ethics_status_by_model_persona")


def plot_abstain_reasons(reason_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    reason_counts = (
        reason_model.groupby(["abstain_reason", "reason_label"], as_index=False)["reason_count"]
        .sum()
    )
    total_abstain = float(reason_model.drop_duplicates("model")["abstain"].sum())
    reason_counts["pct_of_abstain"] = reason_counts["reason_count"] / total_abstain
    frame = reason_counts.sort_values("pct_of_abstain", ascending=True)
    values = frame["pct_of_abstain"].to_numpy() * 100
    labels = frame["reason_label"].tolist()
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=_figsize(23, 14))
    bars = ax.barh(y, values, color="#E9C46A", edgecolor="#A87918")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(10, float(values.max()) * 1.22 if len(values) else 10))
    ax.set_xlabel("% dari seluruh abstain", fontsize=REASON_AXIS_LABEL_SIZE, labelpad=12)
    ax.set_ylabel("Alasan abstain", fontsize=REASON_AXIS_LABEL_SIZE, labelpad=16)
    ax.set_title("ETHICS Gender: Penyebab Abstain", fontsize=REASON_TITLE_SIZE, fontweight="bold", pad=22)
    ax.tick_params(axis="x", labelsize=REASON_XTICK_SIZE, pad=8)
    ax.tick_params(axis="y", labelsize=REASON_TICK_SIZE, pad=8)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1f}%",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=22,
            fontweight="normal",
        )
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=2.0)
    return _save(fig, figure_dir, "05_ethics_abstain_reasons_by_model")


def _scenario_code(persona: str) -> str:
    if persona == "neutral":
        return "N"
    if "_explicit" in persona:
        return "E"
    return f"I{persona.rsplit('_', 1)[-1]}"


def _gender_comparison_short(row: pd.Series) -> str:
    scenario_a = _scenario_code(str(row["persona_a"]))
    scenario_b = _scenario_code(str(row["persona_b"]))
    if row["identity"] == "man":
        return f"M:{scenario_a}−{scenario_b}"
    if row["identity"] == "woman":
        return f"W:{scenario_a}−{scenario_b}"
    return f"M−W:{scenario_a}"


def _gender_accuracy_inference(paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[
        paired["outcome"].eq("conditional_accuracy")
        & paired["subset"].isin(VALID_SUBSETS)
    ].copy()
    selected["row_label"] = selected["subset_label"] + " | " + selected["model_label"]
    selected["comparison_short"] = selected.apply(_gender_comparison_short, axis=1)
    selected["effect_size"] = selected["cohen_h"]
    selected["exact_mcnemar_p_value"] = selected["mcnemar_exact_p_value"]
    keep = [
        "subset",
        "subset_label",
        "model",
        "model_label",
        "row_label",
        "identity",
        "family",
        "analysis_scope",
        "persona_a",
        "persona_b",
        "comparison_label",
        "comparison_short",
        "display_order",
        "rate_a_pct",
        "rate_b_pct",
        "risk_difference_pp",
        "effect_size",
        "exact_mcnemar_p_value",
        "holm_adjusted_p_value",
        "omnibus_cochran_q_p_value",
        "significant_holm_0_05",
        "decision_significant_0_05",
        "statistical_method",
        "multiple_comparison_method",
    ]
    return selected[keep].sort_values(["subset", "display_order", "model"], kind="stable").reset_index(drop=True)


def _gender_combined_accuracy_inference(
    raw_path: Path,
    per_subset: pd.DataFrame,
) -> pd.DataFrame:
    templates = (
        per_subset.sort_values(["display_order", "model"], kind="stable")
        .drop_duplicates(["model", "comparison_short"])
        .copy()
    )
    templates["holm_group"] = templates["family"]
    combined = compute_combined_conditional_accuracy(
        raw_path,
        templates,
        valid_subsets=VALID_SUBSETS,
        persona_a_column="persona_a",
        persona_b_column="persona_b",
        holm_group_column="holm_group",
    )
    return combined.sort_values(["display_order", "model"], kind="stable").reset_index(drop=True)


def plot_accuracy_statistics(inference: pd.DataFrame, figure_dir: Path) -> list[Path]:
    row_order = [
        f"{SUBSET_LABELS[subset]} | {MODEL_LABELS[model]}"
        for subset in VALID_SUBSETS
        for model in MODEL_ORDER
    ]
    comparison_order = (
        [f"M:{scenario}−N" for scenario in ("E", "I1", "I2", "I3")]
        + [f"M:{scenario}−E" for scenario in ("I1", "I2", "I3")]
        + [f"W:{scenario}−N" for scenario in ("E", "I1", "I2", "I3")]
        + [f"W:{scenario}−E" for scenario in ("I1", "I2", "I3")]
        + [f"M−W:{scenario}" for scenario in ("E", "I1", "I2", "I3")]
    )
    paths = plot_effect_holm_heatmap(
        inference,
        figure_dir,
        title="ETHICS Gender: Ukuran Efek",
        stem="06_ethics_gender_effect_holm_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=18,
    )
    paths += plot_exact_mcnemar_holm_heatmaps(
        inference,
        figure_dir,
        title="ETHICS Gender: p-value Exact McNemar",
        stem="07_ethics_gender_exact_mcnemar_and_holm_pvalue_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=18,
        annotation_font_size_offset=-1,
    )
    return paths


def plot_combined_accuracy_statistics(inference: pd.DataFrame, figure_dir: Path) -> list[Path]:
    row_order = [MODEL_LABELS[model] for model in MODEL_ORDER]
    comparison_order = (
        inference.sort_values("display_order", kind="stable")["comparison_short"]
        .drop_duplicates()
        .tolist()
    )
    paths = plot_effect_holm_heatmap(
        inference,
        figure_dir,
        title="ETHICS Gender: Ukuran Efek",
        stem="08_ethics_gender_combined_effect_holm_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=18,
        annotation_font_size_offset=-1,
        y_axis_label="Model",
    )
    paths += plot_exact_mcnemar_holm_heatmaps(
        inference,
        figure_dir,
        title="ETHICS Gender: p-value Holm",
        stem="09_ethics_gender_combined_exact_mcnemar_and_holm_pvalue_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=18,
        y_axis_label="Model",
    )
    return paths


def _subset_category_values(summary: pd.DataFrame, subset: str, value_kind: str) -> pd.DataFrame:
    frame = summary[summary["subset"].eq(subset)].copy()
    rows = []
    for model in MODEL_ORDER:
        model_frame = frame[frame["model"].eq(model)]
        for category, short_label, personas in CATEGORY_SPECS:
            group = model_frame[model_frame["persona"].isin(personas)]
            committed = group["committed"].sum()
            if value_kind == "accuracy":
                value = group["n_correct"].sum() / committed if committed else np.nan
            else:
                value = (group["pct_answer_wrong"] * group["committed"]).sum() / committed if committed else np.nan
            rows.append({"model": model, "model_label": MODEL_LABELS[model], "category": category, "value": value})
    return pd.DataFrame(rows)


def _subset_status_values(summary: pd.DataFrame, subset: str) -> pd.DataFrame:
    frame = summary[summary["subset"].eq(subset)].copy()
    return _aggregate_categories(frame, ["total", "nonmissing", "committed", "abstain", "missing"])


def render_subset_children(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    child_root = output_dir / "per_subset"
    child_root.mkdir(parents=True, exist_ok=True)
    summary = tables["condition_summary"].copy()
    paired = tables["paired_effects"].copy()
    reason = tables["abstain_reason"].copy()
    output_paths: list[Path] = []
    for subset in SUBSET_ORDER:
        child = child_root / subset
        table_dir = child / "tables"
        figure_dir = child / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        subset_summary = summary[summary["subset"].eq(subset)].copy()
        subset_reason = reason[reason["subset"].eq(subset)].copy()
        subset_paired = paired[paired["subset"].eq(subset)].copy()
        subset_summary.to_csv(table_dir / "condition_summary.csv", index=False)
        subset_reason.to_csv(table_dir / "abstain_reason.csv", index=False)
        subset_paired.to_csv(table_dir / "paired_effects.csv", index=False)
        value_kind = "accuracy" if bool(subset_summary["construct_valid"].iloc[0]) else "stance_wrong"
        values = _subset_category_values(summary, subset, value_kind)
        pivot = values.pivot(index="model_label", columns="category", values="value").reindex(index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=CATEGORY_ORDER)
        fig, ax = plt.subplots(figsize=_figsize(19, 9))
        pivot.plot(kind="bar", width=0.84, color=ACCURACY_COLORS, ax=ax)
        _annotate_accuracy(ax, offset=5)
        ax.set_ylim(max(0, float(np.nanmin(pivot.to_numpy())) - 0.10), min(1, float(np.nanmax(pivot.to_numpy())) + 0.15))
        metric_label = "Accuracy committed" if value_kind == "accuracy" else "Stance wrong"
        ax.set_title(f"ETHICS Gender — {SUBSET_LABELS[subset]}: {metric_label}", fontsize=TITLE_SIZE, fontweight="bold", pad=18)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_label)
        ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[m]] for m in MODEL_ORDER], rotation=0, ha="center")
        ax.legend(title="Persona", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        output_paths.extend(_save(fig, figure_dir, f"01_{subset}_performance_by_persona"))

        status_values = _subset_status_values(summary, subset)
        fig, axes = plt.subplots(3, 2, figsize=_figsize(22, 18), constrained_layout=False)
        for ax, model in zip(list(axes.flat), MODEL_ORDER, strict=True):
            frame = status_values[status_values["model"].eq(model)].set_index("category").reindex(CATEGORY_ORDER)
            x = np.arange(len(CATEGORY_ORDER))
            bottom = np.zeros(len(CATEGORY_ORDER))
            for status_name in STATUS_ORDER:
                values_status = frame[STATUS_RATE_COLUMNS[status_name]].to_numpy() * 100
                ax.bar(x, values_status, bottom=bottom, color=STATUS_COLORS[status_name], edgecolor="white", linewidth=0.7, label=STATUS_LABELS[status_name])
                for index, value in enumerate(values_status):
                    _annotate_status_segment(ax, index, bottom[index], value)
                bottom += values_status
            ax.set_title(MODEL_LABELS[model], fontsize=18, fontweight="bold")
            ax.set_xticks(x, [CATEGORY_SHORT_LABELS[item] for item in CATEGORY_ORDER])
            ax.set_ylim(0, 100)
            ax.set_ylabel("% respons")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.subplots_adjust(top=0.89, bottom=0.06, hspace=0.42, wspace=0.20)
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=True)
        fig.suptitle(f"ETHICS Gender — {SUBSET_LABELS[subset]}: Status Jawaban", fontsize=TITLE_SIZE, fontweight="bold", y=0.98)
        output_paths.extend(_save(fig, figure_dir, f"02_{subset}_status_by_persona"))

        readme = f"""# ETHICS Gender — {SUBSET_LABELS[subset]}\n\nFolder ini menyimpan rincian subset `{subset}`.\n\n- `condition_summary.csv`: status dan metrik per model × persona.\n- `abstain_reason.csv`: penyebab abstain dengan denominator sumber.\n- `paired_effects.csv`: hasil paired dari bundle sumber.\n- `figures/`: visual detail subset.\n\nVisual utama ETHICS Gender berada satu tingkat di atas, pada folder `ethics accuracy analysis`.\n"""
        (child / "README.md").write_text(readme, encoding="utf-8")
    return output_paths


def write_tables(overall: dict[str, pd.DataFrame], table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "accuracy": "ethics_accuracy_by_model_persona_category.csv",
        "status": "ethics_status_by_model_persona_category.csv",
        "reason_categories": "ethics_abstain_reasons_by_model_persona_category.csv",
        "reason_model": "ethics_abstain_reasons_by_model.csv",
        "paired": "ethics_paired_effects_vs_neutral.csv",
        "accuracy_inference": "ethics_accuracy_inference_construct_valid.csv",
        "accuracy_inference_combined": "ethics_accuracy_inference_construct_valid_combined.csv",
        "condition_summary": "ethics_condition_summary_source_copy.csv",
    }
    paths = {}
    for key, name in mapping.items():
        path = table_dir / name
        overall[key].to_csv(path, index=False)
        paths[key] = path
    return paths


def render(source_bundle: Path = SOURCE_BUNDLE, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    source_bundle = source_bundle.resolve()
    output_dir = output_dir.resolve()
    tables = _load_source_tables(source_bundle)
    derived = derive_tables(tables)
    overall = _build_overall_tables(tables, derived)
    overall["accuracy_inference"] = _gender_accuracy_inference(tables["paired_effects"])
    raw_path = Path(tables["manifest"]["input"])
    if sha256_file(raw_path) != tables["manifest"]["input_sha256"]:
        raise ValueError("Hash raw input Gender tidak cocok dengan source manifest")
    overall["accuracy_inference_combined"] = _gender_combined_accuracy_inference(
        raw_path,
        overall["accuracy_inference"],
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = write_tables(overall, table_dir)

    configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths += plot_accuracy(overall["accuracy"], figure_dir)
    figure_paths += plot_delta(overall["accuracy"], figure_dir, implicit=False)
    figure_paths += plot_delta(overall["accuracy"], figure_dir, implicit=True)
    figure_paths += plot_status(overall["status"], figure_dir)
    figure_paths += plot_abstain_reasons(overall["reason_model"], figure_dir)
    figure_paths += plot_accuracy_statistics(overall["accuracy_inference"], figure_dir)
    figure_paths += plot_combined_accuracy_statistics(overall["accuracy_inference_combined"], figure_dir)
    figure_paths += render_subset_children(tables, output_dir)

    output_index_rows = []
    for path in sorted(table_paths.values()):
        output_index_rows.append({"kind": "table", "file": path.name, "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(figure_paths):
        output_index_rows.append({"kind": "figure", "file": path.name, "path": str(path), "sha256": sha256_file(path)})
    output_index = pd.DataFrame(output_index_rows)
    output_index.to_csv(output_dir / "output_index.csv", index=False)

    source_manifest_path = source_bundle / "metadata" / "analysis_manifest.json"
    parent_figure_paths = [path for path in figure_paths if path.parent == figure_dir]
    manifest = {
        "bundle_version": "ethics-gender-accuracy-combined-standardized-v3",
        "created_from": str(source_bundle),
        "source_analysis_version": tables["manifest"]["version"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_input_sha256": tables["manifest"]["input_sha256"],
        "source_row_count": int(tables["manifest"]["row_count"]),
        "parent_scope": "combined_ethics_gender",
        "status_scope": "all_four_subsets",
        "accuracy_scope": "construct_valid_commonsense_justice_pooled",
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "parent_table_files": [path.name for path in sorted(table_paths.values())],
        "parent_figure_files": [path.name for path in sorted(parent_figure_paths)],
        "parent_figure_count": len(parent_figure_paths),
        "child_subset_dirs": SUBSET_ORDER,
        "status_counts": {status: int(tables["condition_summary"][status].sum()) for status in ("committed", "abstain", "missing")},
        "source_bundle_read_only": True,
        "api_calls": 0,
        "rescore": False,
        "statistics_recomputed": True,
        "statistics_recomputed_scope": "combined_construct_valid_commonsense_justice_only",
        "combined_statistics_eligibility": "both_responses_committed",
        "per_subset_statistics_recomputed": False,
        "pair_count_displayed": False,
        "notes": [
            "Judul parent tidak memakai kata overall.",
            "Status committed/abstain/missing dipool pada seluruh empat subset.",
            "Accuracy parent dipool hanya pada commonsense dan justice.",
            "Statistik 06/07 mempertahankan hasil per subset dari source bundle.",
            "Statistik 08/09 dihitung ulang pada gabungan commonsense dan justice dengan pairing key subset + question_id.",
            "Deontology dan virtue tetap tersedia di child per_subset.",
        ],
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CREATED] {output_dir}")
    print(f"[PARENT TABLES] {len(table_paths)}")
    print(f"[PARENT FIGURES] {len(parent_figure_paths)} files ({len(parent_figure_paths) // 2} PNG + {len(parent_figure_paths) // 2} SVG)")
    print(f"[CHILD FIGURES] {len(figure_paths) - len(parent_figure_paths)} files")
    return {"output_dir": output_dir, "tables": table_paths, "figures": figure_paths, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=SOURCE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render(source_bundle=args.source_bundle, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

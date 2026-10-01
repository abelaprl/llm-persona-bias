"""Shared statistical figures for the standardized ETHICS accuracy bundles.

The functions in this module are presentation-only.  They consume validated
paired-comparison tables and never recompute McNemar tests, Holm corrections,
or effect sizes.  Pair counts are intentionally not displayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BACKGROUND = "#FCFBF8"
PVALUE_CMAP = "rocket"
EFFECT_CMAP = "RdBu_r"


def _save(fig: mpl.figure.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor=BACKGROUND, bbox_inches="tight", dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    items = list(values)
    return [items[index : index + size] for index in range(0, len(items), size)]


def _pivot(
    frame: pd.DataFrame,
    value: str,
    row_order: Sequence[str],
    comparison_order: Sequence[str],
) -> pd.DataFrame:
    if frame.duplicated(["row_label", "comparison_short"]).any():
        raise ValueError("Duplicate statistical heatmap cells detected")
    return frame.pivot(index="row_label", columns="comparison_short", values=value).reindex(
        index=list(row_order), columns=list(comparison_order)
    )


def _effect_annotations(values: pd.DataFrame, significant: pd.DataFrame) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = ""
                continue
            star = "*" if bool(significant.loc[row, column]) else ""
            annotations.loc[row, column] = f"{value:+.2f}{star}"
    return annotations


def _format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}"


def _pvalue_annotations(values: pd.DataFrame, significant: pd.DataFrame) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            star = "*" if np.isfinite(value) and bool(significant.loc[row, column]) else ""
            annotations.loc[row, column] = f"{_format_pvalue(value)}{star}"
    return annotations


def _emphasize_starred(ax: mpl.axes.Axes) -> None:
    for text in ax.texts:
        text.set_fontweight("bold" if text.get_text().endswith("*") else "normal")


def plot_effect_holm_heatmap(
    frame: pd.DataFrame,
    figure_dir: Path,
    *,
    title: str,
    stem: str,
    row_order: Sequence[str],
    comparison_order: Sequence[str],
    chunk_size: int = 26,
    font_size_offset: float = 0,
    annotation_font_size_offset: float = 0,
    y_axis_label: str = "Subset dan model",
) -> list[Path]:
    required = {"row_label", "comparison_short", "effect_size", "significant_holm_0_05"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing effect heatmap columns: {sorted(missing)}")

    values = _pivot(frame, "effect_size", row_order, comparison_order)
    significant = _pivot(frame, "significant_holm_0_05", row_order, comparison_order).fillna(False).astype(bool)
    parts = _chunks(comparison_order, chunk_size)
    limit = max(0.15, float(np.nanmax(np.abs(values.to_numpy(dtype=float)))))
    panel_width = max(18.0, min(30.0, 0.85 * max(len(part) for part in parts)))
    dense_columns = max(len(part) for part in parts) > 20
    rotated_columns = max(len(part) for part in parts) > 8
    panel_height = max(7.0, 0.58 * len(row_order) + (4.0 if dense_columns else 2.0))
    fig, axes = plt.subplots(
        len(parts),
        1,
        figsize=(panel_width, panel_height * len(parts)),
        squeeze=False,
        constrained_layout=False,
    )
    for index, columns in enumerate(parts):
        ax = axes[index, 0]
        selected = values[columns]
        decisions = significant[columns]
        sns.heatmap(
            selected,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=_effect_annotations(selected, decisions),
            fmt="",
            linewidths=0.7,
            linecolor="#FFFFFF",
            cbar=index == len(parts) - 1,
            cbar_kws={"label": "Cohen's h", "shrink": 0.78},
            annot_kws={"fontsize": 11 + font_size_offset + annotation_font_size_offset},
            ax=ax,
        )
        _emphasize_starred(ax)
        ax.set_title(
            "Perbandingan 1" if len(parts) > 1 and index == 0 else (
                f"Perbandingan {index + 1}" if len(parts) > 1 else ""
            ),
            fontsize=17 + font_size_offset,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Perbandingan persona")
        ax.set_ylabel(y_axis_label)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90 if rotated_columns else 0,
            ha="center",
            va="top",
            fontsize=(9.5 if rotated_columns else 10) + font_size_offset,
        )
        ax.tick_params(axis="x", pad=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11 + font_size_offset)
    fig.suptitle(title, fontsize=24 + font_size_offset, fontweight="bold", y=0.995)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.18 if rotated_columns else (0.08 if dense_columns else 0.05),
        left=0.16,
        right=0.98,
        hspace=0.58 if dense_columns else 0.30,
    )
    return _save(fig, figure_dir, stem)


def plot_exact_mcnemar_holm_heatmaps(
    frame: pd.DataFrame,
    figure_dir: Path,
    *,
    title: str,
    stem: str,
    row_order: Sequence[str],
    comparison_order: Sequence[str],
    chunk_size: int = 26,
    font_size_offset: float = 0,
    annotation_font_size_offset: float = 0,
    y_axis_label: str = "Subset dan model",
) -> list[Path]:
    required = {
        "row_label",
        "comparison_short",
        "exact_mcnemar_p_value",
        "holm_adjusted_p_value",
        "significant_holm_0_05",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing p-value heatmap columns: {sorted(missing)}")

    raw = _pivot(frame, "exact_mcnemar_p_value", row_order, comparison_order)
    adjusted = _pivot(frame, "holm_adjusted_p_value", row_order, comparison_order)
    raw_significant = raw.lt(0.05)
    adjusted_significant = _pivot(
        frame, "significant_holm_0_05", row_order, comparison_order
    ).fillna(False).astype(bool)
    output_paths: list[Path] = []
    parts = _chunks(comparison_order, chunk_size)
    for part_index, columns in enumerate(parts, start=1):
        dense_columns = len(columns) > 20
        rotated_columns = len(columns) > 8
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(
                max(18.0, min(30.0, 0.85 * len(columns))),
                max(14.0, 1.05 * len(row_order) + (9.0 if dense_columns else 5.0)),
            ),
            constrained_layout=False,
        )
        for ax, values, decisions, panel_title, colorbar_label in (
            (
                axes[0],
                raw[columns],
                raw_significant[columns],
                "Raw two-sided Exact McNemar p-value",
                "raw p-value (0–0.05)",
            ),
            (
                axes[1],
                adjusted[columns],
                adjusted_significant[columns],
                "Holm-adjusted p-value",
                "adjusted p-value (0–0.05)",
            ),
        ):
            sns.heatmap(
                values.clip(upper=0.05),
                cmap=PVALUE_CMAP,
                vmin=0,
                vmax=0.05,
                annot=_pvalue_annotations(values, decisions),
                fmt="",
                linewidths=0.7,
                linecolor="#FFFFFF",
                cbar=True,
                cbar_kws={"label": colorbar_label, "shrink": 0.78},
                annot_kws={
                    "fontsize": 10.5 + font_size_offset + annotation_font_size_offset
                },
                ax=ax,
            )
            _emphasize_starred(ax)
            ax.set_title(panel_title, fontsize=18 + font_size_offset, fontweight="bold", pad=12)
            ax.set_xlabel("Perbandingan persona")
            ax.set_ylabel(y_axis_label)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=90 if rotated_columns else 0,
                ha="center",
                va="top",
                fontsize=(9.0 if rotated_columns else 9.5) + font_size_offset,
            )
            ax.tick_params(axis="x", pad=6)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10.5 + font_size_offset)
        suffix = f"_part{part_index}" if len(parts) > 1 else ""
        figure_title = f"{title} — Bagian {part_index}" if len(parts) > 1 else title
        fig.suptitle(figure_title, fontsize=23 + font_size_offset, fontweight="bold", y=0.995)
        fig.subplots_adjust(
            top=0.94,
            bottom=0.20 if rotated_columns else (0.09 if dense_columns else 0.05),
            left=0.16,
            right=0.98,
            hspace=0.75 if rotated_columns else (0.62 if dense_columns else 0.32),
        )
        output_paths.extend(_save(fig, figure_dir, f"{stem}{suffix}"))
    return output_paths

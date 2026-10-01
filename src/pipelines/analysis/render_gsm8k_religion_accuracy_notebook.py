"""Render the GSM8K Religion accuracy figures for the standalone notebook.

The renderer is a presentation-only layer over the validated v5 tables. It
does not read model responses, rerun inference, parse answers, rescore data,
or recompute any statistical test. Figure names 05, 06, and 26 are retained
because those are the names referenced by Bab VI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from render_gsm8k_religion_all_52_effect_holm_single_large_heatmap import COMPARISONS
from render_gsm8k_religion_primary_effect_holm_single_large_heatmap import (
    MODEL_ORDER,
    format_effect,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold"
    / "tables"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "notebooks" / "religion" / "akurasi analysis" / "figures"
)

IDENTITIES = ("Atheist", "Christian", "Islam", "Jewish")

# Two adjacent shades belong to the same identity. The identity families are
# stable: Atheist blue, Christian red, Islam purple, and Jewish orange.
BAR_COLORS = (
    "#0047AB", "#0086A8",
    "#B80F2E", "#E45700",
    "#5B21B6", "#A0006D",
    "#C2410C", "#F59E0B",
    "#30343B",
)
DELTA_COLORS = ("#0047AB", "#B80F2E", "#5B21B6", "#C2410C")

MODEL_AXIS_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next\n80B-A3B",
    "Llama 3.1 8B": "Llama 3.1\n8B",
    "Ministral 8B": "Ministral\n8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS\n20B",
    "Llama 3.3 70B": "Llama 3.3\n70B",
}

BACKGROUND = "#FCFBF8"
TEXT_COLOR = "#17212B"
TITLE_SIZE = 23
BAR_ANNOTATION_SIZE = 17
DELTA_ANNOTATION_SIZE = 16
FIGURE_SCALE = 1.12
HEATMAP_CELL_SIZE = 17.5
HEATMAP_ROW_SIZE = 16.5
HEATMAP_COLUMN_SIZE = 15
HEATMAP_AXIS_SIZE = 17
HEATMAP_COLORBAR_TICK_SIZE = 14.5
HEATMAP_COLORBAR_LABEL_SIZE = 16.5
FIGURE_METADATA = {"Date": "1970-01-01T00:00:00"}
SVG_HASHSALT = "gsm8k-religion-accuracy-v1"

TABLE_NAMES = (
    "accuracy_by_persona_six_models.csv",
    "neutral_persona_accuracy_six_models.csv",
    "religion_explicit_implicit_and_deltas.csv",
    "religion_paired_effects_significance.csv",
    "cochran_q_omnibus_all_groups.csv",
    "token_category_summary_six_models.csv",
    "religion_completion_token_delta_implicit_vs_explicit_pct.csv",
    "religion_reasoning_token_delta_implicit_vs_explicit_pct.csv",
)
FIGURE_STEMS = (
    "05_religion_absolute_accuracy_explicit_implicit",
    "06_religion_delta_implicit_vs_explicit",
    "26_religion_all_52_effect_holm_stacked_26_26",
)


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
        font_scale=1.25 + 2.0 / 12.0,
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 90,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": "#5A6573",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "legend.title_fontsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": SVG_HASHSALT,
        }
    )


def save_pair(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=BACKGROUND,
            metadata=FIGURE_METADATA,
        )
    plt.close(fig)


def annotate_accuracy(ax: plt.Axes) -> None:
    for bar in ax.patches:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=BAR_ANNOTATION_SIZE,
            fontweight="normal",
            rotation=90,
        )


def annotate_delta(ax: plt.Axes) -> None:
    for bar in ax.patches:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:+.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            xytext=(0, 10 if value >= 0 else -10),
            textcoords="offset points",
            fontsize=DELTA_ANNOTATION_SIZE,
            fontweight="normal",
            rotation=90,
            rotation_mode="default",
            clip_on=False,
        )


def ordered_religion_frame(table_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(table_dir / "religion_explicit_implicit_and_deltas.csv")
    frame = frame.set_index("model_label").reindex(MODEL_ORDER)
    if frame.isna().any().any():
        raise ValueError("Tabel Religion tidak memuat keenam model utama.")
    return frame


def render_bars(table_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = ordered_religion_frame(table_dir)
    absolute_columns = [
        column
        for identity in IDENTITIES
        for column in (f"{identity} Explicit", f"{identity} Implicit Avg")
    ] + ["Neutral"]
    absolute_labels = [
        label
        for identity in IDENTITIES
        for label in (f"{identity} Explicit", f"{identity} Implicit Avg")
    ] + ["Neutral"]
    absolute = frame[absolute_columns].copy() / 100.0
    absolute.columns = absolute_labels
    neutral_mean = float(absolute["Neutral"].mean())

    fig, ax = plt.subplots(figsize=(23 * FIGURE_SCALE, 9.5 * FIGURE_SCALE))
    absolute.plot(kind="bar", width=0.88, color=BAR_COLORS, ax=ax)
    baseline_label = f"Global Neutral Avg ({neutral_mean:.3f})"
    ax.axhline(
        neutral_mean,
        color="#B80F2E",
        linestyle="--",
        linewidth=2.0,
        label=baseline_label,
        zorder=1,
    )
    annotate_accuracy(ax)
    ax.set_ylim(0.75, 1.00)
    ax.set_title(
        "GSM8K Religion: Perbandingan Akurasi Antar-Persona",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Model", fontweight="normal")
    ax.set_ylabel("Accuracy", fontweight="normal")
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[item] for item in MODEL_ORDER],
        rotation=0,
        fontweight="normal",
    )
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [
        handle for index, handle in enumerate(handles) if index != baseline_index
    ]
    labels = [baseline_label] + [
        label for index, label in enumerate(labels) if index != baseline_index
    ]
    ax.legend(
        handles,
        labels,
        title="Religion Persona & Baseline",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_pair(fig, output_dir, FIGURE_STEMS[0], dpi)

    delta_columns = [f"{identity} Delta Imp-Exp" for identity in IDENTITIES]
    delta = frame[delta_columns].copy() / 100.0
    delta.columns = [
        f"{identity}: Implicit Avg - Explicit" for identity in IDENTITIES
    ]
    fig, ax = plt.subplots(figsize=(19 * FIGURE_SCALE, 9 * FIGURE_SCALE))
    delta.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    annotate_delta(ax)
    limit = max(0.001, float(np.nanmax(np.abs(delta.to_numpy(dtype=float)))))
    ax.set_ylim(-limit * 1.40, limit * 1.40)
    ax.set_title(
        "GSM8K Religion - Analisis Delta: Persona Implicit vs Explicit",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Model", fontweight="normal")
    ax.set_ylabel("Difference on accuracy scale", fontweight="normal")
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[item] for item in MODEL_ORDER],
        rotation=0,
        ha="center",
        fontweight="normal",
    )
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_pair(fig, output_dir, FIGURE_STEMS[1], dpi)


def parse_bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.lower()
    if not values.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"Nilai boolean tidak dikenal pada kolom {column}.")
    return values.isin({"true", "1"})


def stack_pair_label(label: str) -> str:
    """Wrap cross-identity comparison labels without changing their meaning."""
    lines = label.splitlines()
    if not lines:
        return label
    for separator in ("-", "\u2212", "\u2013", "âˆ’"):
        if separator in lines[0]:
            left, right = lines[0].split(separator, maxsplit=1)
            return "\n".join([f"{left}{separator}", right, *lines[1:]])
    return label


def render_heatmap(table_dir: Path, output_dir: Path, dpi: int) -> None:
    source = pd.read_csv(table_dir / "religion_paired_effects_significance.csv")
    comparison_order = [comparison for comparison, _ in COMPARISONS]
    selected = source[source["comparison_label"].isin(comparison_order)].copy()
    selected["significant_holm_0_05"] = parse_bool_column(
        selected, "significant_holm_0_05"
    )
    assert len(selected) == 6 * 52

    effects = selected.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = selected.pivot(
        index="model_label",
        columns="comparison_label",
        values="significant_holm_0_05",
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.shape == (6, 52)
    assert effects.notna().all().all()

    annotations = effects.copy().astype(object)
    for model in effects.index:
        for comparison in effects.columns:
            annotations.loc[model, comparison] = format_effect(
                float(effects.loc[model, comparison]),
                bool(significant.loc[model, comparison]),
            )

    short_names = {
        "Atheist": "Ath",
        "Christian": "Chr",
        "Islam": "Isl",
        "Jewish": "Jew",
    }
    column_labels = [
        stack_pair_label(
            next(
                (
                    label.replace(full_name, short_name)
                    for full_name, short_name in short_names.items()
                    if full_name in label
                ),
                label,
            )
        )
        for _, label in COMPARISONS
    ]
    panel_specs = (
        ("Panel 1 — Ateis dan Kristen", 0, 14),
        ("Panel 2 — Islam dan Yahudi", 14, 28),
        ("Panel 3 — Ateis dibandingkan identitas agama lain", 28, 40),
        ("Panel 4 — Perbandingan lintas agama lainnya", 40, 52),
    )

    cmap = LinearSegmentedColormap.from_list(
        "religion_effect_standardized",
        [
            "#0B3C78",
            "#4A98C9",
            "#DCEAF1",
            BACKGROUND,
            "#F7C7AE",
            "#E76F51",
            "#9B002F",
        ],
    )
    max_abs = float(np.nanmax(np.abs(effects.to_numpy(dtype=float))))
    effect_limit = max(0.10, math.ceil(max_abs / 0.05) * 0.05)

    fig = plt.figure(figsize=(18, 22))
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[6, 6, 6, 6, 0.42],
        left=0.235,
        right=0.985,
        top=0.91,
        bottom=0.055,
        hspace=0.72,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    colorbar_ax = fig.add_subplot(grid[4])
    heatmap = None

    for panel_index, (ax, (title, start, stop)) in enumerate(
        zip(axes, panel_specs)
    ):
        panel_effects = effects.iloc[:, start:stop]
        panel_significant = significant.iloc[:, start:stop]
        panel_annotations = annotations.iloc[:, start:stop]
        show_colorbar = panel_index == len(panel_specs) - 1
        heatmap = sns.heatmap(
            panel_effects,
            annot=panel_annotations,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-effect_limit,
            vmax=effect_limit,
            linewidths=1.25,
            linecolor=BACKGROUND,
            annot_kws={
                "fontsize": HEATMAP_CELL_SIZE,
                "fontweight": "normal",
            },
            cbar=show_colorbar,
            cbar_ax=colorbar_ax if show_colorbar else None,
            cbar_kws={"orientation": "horizontal"} if show_colorbar else None,
            ax=ax,
        )
        for text_artist, value, is_significant in zip(
            ax.texts,
            panel_effects.to_numpy().ravel(),
            panel_significant.to_numpy().ravel(),
        ):
            text_artist.set_color(
                "white"
                if abs(float(value)) >= 0.46 * effect_limit
                else TEXT_COLOR
            )
            text_artist.set_fontweight("bold" if bool(is_significant) else "normal")

        ax.set_yticklabels(
            list(MODEL_ORDER),
            rotation=0,
            fontsize=HEATMAP_ROW_SIZE,
            fontweight="normal",
        )
        ax.set_xticklabels(
            column_labels[start:stop],
            rotation=0,
            ha="center",
            fontsize=HEATMAP_COLUMN_SIZE,
            fontweight="normal",
        )
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xlabel(
            "Perbandingan skenario",
            fontsize=HEATMAP_AXIS_SIZE,
            fontweight="normal",
            labelpad=13,
        )
        ax.set_ylabel(
            "Model",
            fontsize=HEATMAP_AXIS_SIZE,
            fontweight="normal",
            labelpad=15,
        )
        ax.set_title(title, fontsize=19, fontweight="bold", pad=14)

    assert heatmap is not None
    colorbar = heatmap.collections[0].colorbar
    ticks = np.linspace(-effect_limit, effect_limit, 9)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.2f}".replace(".", ",") for value in ticks])
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_SIZE)
    colorbar.set_label(
        "Cohen's h",
        fontsize=HEATMAP_COLORBAR_LABEL_SIZE,
        fontweight="normal",
        labelpad=8,
    )
    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Agama",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.965,
    )
    # The Figure 26-style image intentionally has no footnote. Its meaning
    # is documented in the notebook and README instead.
    save_pair(fig, output_dir, FIGURE_STEMS[2], dpi)


def write_manifest(output_dir: Path, table_dir: Path) -> None:
    figure_files = [
        output_dir / f"{stem}.{extension}"
        for extension in ("png", "svg")
        for stem in FIGURE_STEMS
    ]
    manifest = {
        "bundle_version": "gsm8k-religion-accuracy-standardized-v1.0.0",
        "source_table_dir": str(table_dir),
        "source_tables": {
            name: sha256_file(table_dir / name) for name in TABLE_NAMES
        },
        "figures": [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in figure_files
        ],
        "statistical_tables_unchanged": True,
        "title_pt": TITLE_SIZE,
        "bar_annotation_pt": BAR_ANNOTATION_SIZE,
        "delta_annotation_pt": DELTA_ANNOTATION_SIZE,
        "heatmap_panel_title_pt": 19,
        "heatmap_cell_annotation_pt": HEATMAP_CELL_SIZE,
        "heatmap_footnote": False,
        "svg_metadata_date": FIGURE_METADATA["Date"],
        "svg_hashsalt": SVG_HASHSALT,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("DPI minimum 300")
    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    missing = sorted(
        name for name in TABLE_NAMES if not (table_dir / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Tabel v5 Religion tidak lengkap: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    render_bars(table_dir, output_dir, args.dpi)
    render_heatmap(table_dir, output_dir, args.dpi)
    write_manifest(output_dir, table_dir)
    print(f"[DONE] Religion standardized figures: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

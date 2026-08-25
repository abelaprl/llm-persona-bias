"""Render the 6-model by 18-comparison Gender heatmap as two stacked panels.

The statistical table is a read-only input.  This module only changes the
presentation of the already-computed Cohen's h values and Holm decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from render_gsm8k_gender_effect_holm_single_portrait import (
    COMPARISONS,
    FIGURE_DIR,
    MODEL_ORDER,
    TABLE_PATH,
    format_effect,
)


MODEL_ROW_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next 80B-A3B",
    "Llama 3.1 8B": "Llama 3.1 8B",
    "Ministral 8B": "Ministral 8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS 20B",
    "Llama 3.3 70B": "Llama 3.3 70B",
}

COLUMN_LABELS = [
    "Laki-laki\nE−N",
    "Laki-laki\nI1−N",
    "Laki-laki\nI2−N",
    "Laki-laki\nI3−N",
    "Perempuan\nE−N",
    "Perempuan\nI1−N",
    "Perempuan\nI2−N",
    "Perempuan\nI3−N",
    "Laki-laki\nI1−E",
    "Laki-laki\nI2−E",
    "Laki-laki\nI3−E",
    "Perempuan\nI1−E",
    "Perempuan\nI2−E",
    "Perempuan\nI3−E",
    "Eksplisit\nL−P",
    "Implisit 1\nL−P",
    "Implisit 2\nL−P",
    "Implisit 3\nL−P",
]


PANEL_SPECS = [
    {
        "title": "Panel 1 — Perbandingan terhadap skenario netral",
        "start": 0,
        "stop": 8,
    },
    {
        "title": (
            "Panel 2 — Perbandingan implisit–eksplisit dan "
            "laki-laki–perempuan"
        ),
        "start": 8,
        "stop": 18,
    },
]

DEFAULT_OUTPUT = FIGURE_DIR / "24_gender_effect_holm_single_large_heatmap.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path; the statistical source is always TABLE_PATH.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("DPI output harus minimal 300")

    source = pd.read_csv(TABLE_PATH)
    comparison_order = [comparison for comparison, _ in COMPARISONS]

    # Lock the expected statistical surface before any plotting operation.
    selected = source[source["comparison_label"].isin(comparison_order)]
    assert len(selected) == len(MODEL_ORDER) * len(comparison_order)
    assert set(selected["model_label"]) == set(MODEL_ORDER)
    assert set(selected["comparison_label"]) == set(comparison_order)

    effects = selected.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = selected.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()
    assert significant.shape == effects.shape

    annotations = effects.copy().astype(object)
    for model in effects.index:
        for comparison in effects.columns:
            annotations.loc[model, comparison] = format_effect(
                float(effects.loc[model, comparison]),
                bool(significant.loc[model, comparison]),
            )

    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "text.color": "#262626",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "gender_effect",
        ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
    )

    fig = plt.figure(figsize=(18.4, 11.8))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[6, 6, 0.42],
        left=0.215,
        right=0.985,
        top=0.905,
        bottom=0.155,
        hspace=0.82,
    )
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    colorbar_ax = fig.add_subplot(grid[2])
    heatmap = None

    for panel_index, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        start = int(spec["start"])
        stop = int(spec["stop"])
        panel_effects = effects.iloc[:, start:stop]
        panel_significant = significant.iloc[:, start:stop]
        panel_annotations = annotations.iloc[:, start:stop]
        show_colorbar = panel_index == len(PANEL_SPECS) - 1

        heatmap = sns.heatmap(
            panel_effects,
            annot=panel_annotations,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-0.15,
            vmax=0.15,
            linewidths=1.25,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 17.5, "fontweight": "normal"},
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
                "white" if abs(float(value)) >= 0.075 else "#262626"
            )
            text_artist.set_fontweight("bold" if bool(is_significant) else "normal")

        ax.set_yticklabels(
            [MODEL_ROW_LABELS[model] for model in MODEL_ORDER],
            rotation=0,
            fontsize=16.5,
            fontweight="normal",
        )
        ax.set_xticklabels(
            COLUMN_LABELS[start:stop],
            rotation=0,
            ha="center",
            fontsize=15,
            fontweight="normal",
        )
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xlabel(
            "Perbandingan skenario",
            fontsize=17,
            fontweight="normal",
            labelpad=13,
        )
        ax.set_ylabel("Model", fontsize=17, fontweight="normal", labelpad=15)
        ax.set_title(
            str(spec["title"]),
            fontsize=19,
            fontweight="bold",
            pad=14,
        )

    assert heatmap is not None
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    colorbar.set_ticklabels(["−0,15", "−0,10", "−0,05", "0,00", "0,05", "0,10", "0,15"])
    colorbar.ax.tick_params(labelsize=14.5)
    colorbar.set_label("Cohen's h", fontsize=16.5, fontweight="normal", labelpad=8)

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Gender",
        fontsize=23,
        fontweight="bold",
        y=0.975,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()

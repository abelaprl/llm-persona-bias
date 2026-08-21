"""Render all 52 Religion comparisons as two stacked 26-column heatmaps."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from render_gsm8k_religion_all_52_effect_holm_single_large_heatmap import (
    COMPARISONS,
)
from render_gsm8k_religion_primary_effect_holm_single_large_heatmap import (
    FIGURE_DIR,
    MODEL_ORDER,
    TABLE_PATH,
    format_effect,
)


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(TABLE_PATH)
    comparison_order = [comparison for comparison, _ in COMPARISONS]
    effects = source.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = source.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()
    assert effects.shape == (6, 52)
    return effects, significant


def annotation_matrix(
    effects: pd.DataFrame, significant: pd.DataFrame
) -> pd.DataFrame:
    annotations = effects.copy().astype(object)
    for model in effects.index:
        for comparison in effects.columns:
            annotations.loc[model, comparison] = format_effect(
                float(effects.loc[model, comparison]),
                bool(significant.loc[model, comparison]),
            )
    return annotations


def main() -> None:
    effects, significant = build_matrices()
    sections = [
        (effects.iloc[:, :26], significant.iloc[:, :26], COMPARISONS[:26], "Perbandingan 1–26"),
        (effects.iloc[:, 26:], significant.iloc[:, 26:], COMPARISONS[26:], "Perbandingan 27–52"),
    ]

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
        "religion_effect",
        ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
    )

    fig = plt.figure(figsize=(33, 17.5))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[6, 6, 0.30],
        left=0.095,
        right=0.992,
        top=0.91,
        bottom=0.105,
        hspace=0.42,
    )
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    colorbar_ax = fig.add_subplot(grid[2])

    bottom_heatmap = None
    for index, (matrix, sig_matrix, labels, panel_title) in enumerate(sections):
        ax = axes[index]
        annotations = annotation_matrix(matrix, sig_matrix)
        heatmap = sns.heatmap(
            matrix,
            annot=annotations,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-0.35,
            vmax=0.35,
            linewidths=1.0,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 15.5, "fontweight": "bold"},
            cbar=index == 1,
            cbar_ax=colorbar_ax if index == 1 else None,
            cbar_kws={"orientation": "horizontal"} if index == 1 else None,
            ax=ax,
        )
        if index == 1:
            bottom_heatmap = heatmap

        for text_artist, value in zip(ax.texts, matrix.to_numpy().ravel()):
            text_artist.set_color("white" if abs(float(value)) >= 0.16 else "#262626")

        ax.set_yticklabels(MODEL_ORDER, rotation=0, fontsize=14, fontweight="bold")
        ax.set_xticklabels(
            [label for _, label in labels],
            rotation=0,
            ha="center",
            fontsize=10.8,
        )
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Model", fontsize=14, labelpad=12)
        ax.set_title(panel_title, fontsize=16, fontweight="bold", pad=10)

    assert bottom_heatmap is not None
    colorbar = bottom_heatmap.collections[0].colorbar
    colorbar.set_ticks([-0.35, -0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30, 0.35])
    colorbar.set_ticklabels(
        ["−0,35", "−0,30", "−0,20", "−0,10", "0,00", "0,10", "0,20", "0,30", "0,35"]
    )
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(
        "Cohen's h (kondisi pertama dikurangi kondisi pembanding)",
        fontsize=13,
        labelpad=7,
    )

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Agama — Seluruh 52 Perbandingan",
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Tanda * menunjukkan nilai-p Holm < 0,05 dalam family masing-masing. "
        "† Perbandingan terhadap shared Gender Neutral adalah sensitivity analysis (protokol berbeda). "
        "‡ Perbandingan antaridentitas agama adalah exploratory. E = Explicit; I1–I3 = Implicit; "
        "N = Neutral; Ath = Atheist; Chr = Christian; Isl = Islam; Jew = Jewish.",
        ha="center",
        fontsize=12,
        color="#3F3F3F",
    )

    stem = FIGURE_DIR / "26_religion_all_52_effect_holm_stacked_26_26"
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

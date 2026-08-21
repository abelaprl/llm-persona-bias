"""Render one continuous 6-model by 18-comparison Gender heatmap."""

from __future__ import annotations

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
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill 32B",
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
    "Implicit 1\nL−P",
    "Implicit 2\nL−P",
    "Implicit 3\nL−P",
]


def main() -> None:
    source = pd.read_csv(TABLE_PATH)
    comparison_order = [comparison for comparison, _ in COMPARISONS]

    effects = source.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = source.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()

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

    fig = plt.figure(figsize=(29, 9.4))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[6, 0.28],
        left=0.12,
        right=0.985,
        top=0.86,
        bottom=0.17,
        hspace=0.34,
    )
    ax = fig.add_subplot(grid[0])
    colorbar_ax = fig.add_subplot(grid[1])
    heatmap = sns.heatmap(
        effects,
        annot=annotations,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-0.15,
        vmax=0.15,
        linewidths=1.0,
        linecolor="#FCFBF8",
        annot_kws={"fontsize": 16.5, "fontweight": "bold"},
        cbar_ax=colorbar_ax,
        cbar_kws={"orientation": "horizontal"},
        ax=ax,
    )

    for text_artist, value in zip(ax.texts, effects.to_numpy().ravel()):
        text_artist.set_color("white" if abs(float(value)) >= 0.075 else "#262626")

    ax.set_yticklabels(
        [MODEL_ROW_LABELS[model] for model in MODEL_ORDER],
        rotation=0,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticklabels(
        COLUMN_LABELS,
        rotation=0,
        ha="center",
        fontsize=11.5,
    )
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Seluruh perbandingan kondisi gender", fontsize=14, labelpad=14)
    ax.set_ylabel("Model", fontsize=14, labelpad=12)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    colorbar.set_ticklabels(["−0,15", "−0,10", "−0,05", "0,00", "0,05", "0,10", "0,15"])
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(
        "Cohen's h (kondisi pertama dikurangi kondisi pembanding)",
        fontsize=13,
        labelpad=7,
    )

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Gender",
        fontsize=23,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.035,
        "Tanda * menunjukkan hasil yang tetap signifikan setelah koreksi Holm "
        "(nilai-p Holm < 0,05). E = Explicit; I1–I3 = Implicit; N = Neutral; "
        "L = laki-laki; P = perempuan.",
        ha="center",
        fontsize=12,
        color="#3F3F3F",
    )

    stem = FIGURE_DIR / "24_gender_effect_holm_single_large_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

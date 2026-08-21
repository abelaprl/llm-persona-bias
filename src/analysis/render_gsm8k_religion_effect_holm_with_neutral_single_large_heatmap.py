"""Render Religion primary and shared-Neutral sensitivity effects in one heatmap."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from render_gsm8k_religion_primary_effect_holm_single_large_heatmap import (
    FIGURE_DIR,
    MODEL_ORDER,
    TABLE_PATH,
    format_effect,
)


COMPARISONS = [
    ("Atheist Explicit - Neutral", "Atheist\nE−N†"),
    ("Atheist Implicit 1 - Neutral", "Atheist\nI1−N†"),
    ("Atheist Implicit 2 - Neutral", "Atheist\nI2−N†"),
    ("Atheist Implicit 3 - Neutral", "Atheist\nI3−N†"),
    ("Atheist Implicit 1 - Atheist Explicit", "Atheist\nI1−E"),
    ("Atheist Implicit 2 - Atheist Explicit", "Atheist\nI2−E"),
    ("Atheist Implicit 3 - Atheist Explicit", "Atheist\nI3−E"),
    ("Christian Explicit - Neutral", "Christian\nE−N†"),
    ("Christian Implicit 1 - Neutral", "Christian\nI1−N†"),
    ("Christian Implicit 2 - Neutral", "Christian\nI2−N†"),
    ("Christian Implicit 3 - Neutral", "Christian\nI3−N†"),
    ("Christian Implicit 1 - Christian Explicit", "Christian\nI1−E"),
    ("Christian Implicit 2 - Christian Explicit", "Christian\nI2−E"),
    ("Christian Implicit 3 - Christian Explicit", "Christian\nI3−E"),
    ("Islam Explicit - Neutral", "Islam\nE−N†"),
    ("Islam Implicit 1 - Neutral", "Islam\nI1−N†"),
    ("Islam Implicit 2 - Neutral", "Islam\nI2−N†"),
    ("Islam Implicit 3 - Neutral", "Islam\nI3−N†"),
    ("Islam Implicit 1 - Islam Explicit", "Islam\nI1−E"),
    ("Islam Implicit 2 - Islam Explicit", "Islam\nI2−E"),
    ("Islam Implicit 3 - Islam Explicit", "Islam\nI3−E"),
    ("Jewish Explicit - Neutral", "Jewish\nE−N†"),
    ("Jewish Implicit 1 - Neutral", "Jewish\nI1−N†"),
    ("Jewish Implicit 2 - Neutral", "Jewish\nI2−N†"),
    ("Jewish Implicit 3 - Neutral", "Jewish\nI3−N†"),
    ("Jewish Implicit 1 - Jewish Explicit", "Jewish\nI1−E"),
    ("Jewish Implicit 2 - Jewish Explicit", "Jewish\nI2−E"),
    ("Jewish Implicit 3 - Jewish Explicit", "Jewish\nI3−E"),
]


def main() -> None:
    source = pd.read_csv(TABLE_PATH)
    source = source[source["analysis_scope"].isin(["primary", "sensitivity"])].copy()
    comparison_order = [comparison for comparison, _ in COMPARISONS]

    effects = source.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = source.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()
    assert effects.shape == (6, 28)

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
        "religion_effect",
        ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
    )

    fig = plt.figure(figsize=(38, 9.4))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[6, 0.28],
        left=0.09,
        right=0.992,
        top=0.84,
        bottom=0.19,
        hspace=0.38,
    )
    ax = fig.add_subplot(grid[0])
    colorbar_ax = fig.add_subplot(grid[1])
    heatmap = sns.heatmap(
        effects,
        annot=annotations,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-0.35,
        vmax=0.35,
        linewidths=1.0,
        linecolor="#FCFBF8",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        cbar_ax=colorbar_ax,
        cbar_kws={"orientation": "horizontal"},
        ax=ax,
    )

    for text_artist, value in zip(ax.texts, effects.to_numpy().ravel()):
        text_artist.set_color("white" if abs(float(value)) >= 0.16 else "#262626")

    ax.set_yticklabels(MODEL_ORDER, rotation=0, fontsize=14, fontweight="bold")
    ax.set_xticklabels(
        [label for _, label in COMPARISONS],
        rotation=0,
        ha="center",
        fontsize=11.5,
    )
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel(
        "Perbandingan persona agama terhadap Neutral dan Explicit",
        fontsize=14,
        labelpad=14,
    )
    ax.set_ylabel("Model", fontsize=14, labelpad=12)

    colorbar = heatmap.collections[0].colorbar
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
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Agama",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Tanda * menunjukkan nilai-p yang tetap < 0,05 setelah koreksi Holm dalam family masing-masing. "
        "† Perbandingan terhadap shared Gender Neutral adalah sensitivity analysis karena tanggal run "
        "dan max_tokens berbeda; bukan dasar klaim primary. E = Explicit; I1–I3 = Implicit; N = Neutral.",
        ha="center",
        fontsize=12,
        color="#3F3F3F",
    )

    stem = FIGURE_DIR / "25_religion_effect_holm_with_neutral_single_large_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

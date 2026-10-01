"""Render all 52 Religion comparisons as one continuous effect-size heatmap."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from render_gsm8k_religion_effect_holm_with_neutral_single_large_heatmap import (
    COMPARISONS as NEUTRAL_AND_WITHIN_COMPARISONS,
)
from render_gsm8k_religion_primary_effect_holm_single_large_heatmap import (
    FIGURE_DIR,
    MODEL_ORDER,
    TABLE_PATH,
    format_effect,
)


CROSS_IDENTITY_COMPARISONS = [
    ("Atheist Explicit - Christian Explicit", "Ath−Chr\nE‡"),
    ("Atheist Implicit 1 - Christian Implicit 1", "Ath−Chr\nI1‡"),
    ("Atheist Implicit 2 - Christian Implicit 2", "Ath−Chr\nI2‡"),
    ("Atheist Implicit 3 - Christian Implicit 3", "Ath−Chr\nI3‡"),
    ("Atheist Explicit - Islam Explicit", "Ath−Isl\nE‡"),
    ("Atheist Implicit 1 - Islam Implicit 1", "Ath−Isl\nI1‡"),
    ("Atheist Implicit 2 - Islam Implicit 2", "Ath−Isl\nI2‡"),
    ("Atheist Implicit 3 - Islam Implicit 3", "Ath−Isl\nI3‡"),
    ("Atheist Explicit - Jewish Explicit", "Ath−Jew\nE‡"),
    ("Atheist Implicit 1 - Jewish Implicit 1", "Ath−Jew\nI1‡"),
    ("Atheist Implicit 2 - Jewish Implicit 2", "Ath−Jew\nI2‡"),
    ("Atheist Implicit 3 - Jewish Implicit 3", "Ath−Jew\nI3‡"),
    ("Christian Explicit - Islam Explicit", "Chr−Isl\nE‡"),
    ("Christian Implicit 1 - Islam Implicit 1", "Chr−Isl\nI1‡"),
    ("Christian Implicit 2 - Islam Implicit 2", "Chr−Isl\nI2‡"),
    ("Christian Implicit 3 - Islam Implicit 3", "Chr−Isl\nI3‡"),
    ("Christian Explicit - Jewish Explicit", "Chr−Jew\nE‡"),
    ("Christian Implicit 1 - Jewish Implicit 1", "Chr−Jew\nI1‡"),
    ("Christian Implicit 2 - Jewish Implicit 2", "Chr−Jew\nI2‡"),
    ("Christian Implicit 3 - Jewish Implicit 3", "Chr−Jew\nI3‡"),
    ("Islam Explicit - Jewish Explicit", "Isl−Jew\nE‡"),
    ("Islam Implicit 1 - Jewish Implicit 1", "Isl−Jew\nI1‡"),
    ("Islam Implicit 2 - Jewish Implicit 2", "Isl−Jew\nI2‡"),
    ("Islam Implicit 3 - Jewish Implicit 3", "Isl−Jew\nI3‡"),
]

COMPARISONS = [*NEUTRAL_AND_WITHIN_COMPARISONS, *CROSS_IDENTITY_COMPARISONS]


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
    assert effects.shape == (6, 52)

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

    fig = plt.figure(figsize=(66, 9.4))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[6, 0.28],
        left=0.052,
        right=0.996,
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
        annot_kws={"fontsize": 15.5, "fontweight": "bold"},
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
        fontsize=10.7,
    )
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel(
        "Seluruh 52 perbandingan kondisi persona agama",
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
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Agama — Seluruh Perbandingan",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Tanda * menunjukkan nilai-p Holm < 0,05 dalam family masing-masing. "
        "† Perbandingan terhadap shared Gender Neutral adalah sensitivity analysis (protokol berbeda). "
        "‡ Perbandingan antaridentitas agama adalah exploratory. E = Explicit; I1–I3 = Implicit; "
        "N = Neutral; Ath = Atheist; Chr = Christian; Isl = Islam; Jew = Jewish.",
        ha="center",
        fontsize=12,
        color="#3F3F3F",
    )

    stem = FIGURE_DIR / "26_religion_all_52_effect_holm_single_large_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

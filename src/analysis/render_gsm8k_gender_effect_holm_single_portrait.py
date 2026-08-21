"""Render all Gender Cohen's h comparisons as one portrait heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models"
)
TABLE_PATH = REPORT_DIR / "tables" / "gender_paired_effects_significance.csv"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_ORDER = [
    "Qwen3 Next 80B-A3B",
    "Llama 3.1 8B",
    "Ministral 8B",
    "DeepSeek R1-Distill 32B",
    "GPT-OSS 20B",
    "Llama 3.3 70B",
]
MODEL_AXIS_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next\n80B-A3B",
    "Llama 3.1 8B": "Llama 3.1\n8B",
    "Ministral 8B": "Ministral\n8B",
    "DeepSeek R1-Distill 32B": "DeepSeek\nR1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS\n20B",
    "Llama 3.3 70B": "Llama 3.3\n70B",
}

COMPARISONS = [
    # A: every gender persona against the shared Neutral condition.
    ("Man Explicit - Neutral", "Laki-laki E − N"),
    ("Man Implicit 1 - Neutral", "Laki-laki I1 − N"),
    ("Man Implicit 2 - Neutral", "Laki-laki I2 − N"),
    ("Man Implicit 3 - Neutral", "Laki-laki I3 − N"),
    ("Woman Explicit - Neutral", "Perempuan E − N"),
    ("Woman Implicit 1 - Neutral", "Perempuan I1 − N"),
    ("Woman Implicit 2 - Neutral", "Perempuan I2 − N"),
    ("Woman Implicit 3 - Neutral", "Perempuan I3 − N"),
    # B: implicit variants against the matched explicit condition.
    ("Man Implicit 1 - Man Explicit", "Laki-laki I1 − E"),
    ("Man Implicit 2 - Man Explicit", "Laki-laki I2 − E"),
    ("Man Implicit 3 - Man Explicit", "Laki-laki I3 − E"),
    ("Woman Implicit 1 - Woman Explicit", "Perempuan I1 − E"),
    ("Woman Implicit 2 - Woman Explicit", "Perempuan I2 − E"),
    ("Woman Implicit 3 - Woman Explicit", "Perempuan I3 − E"),
    # C: matched Man-versus-Woman comparisons.
    ("Man Explicit - Woman Explicit", "Eksplisit L − P"),
    ("Man Implicit 1 - Woman Implicit 1", "Implicit 1 L − P"),
    ("Man Implicit 2 - Woman Implicit 2", "Implicit 2 L − P"),
    ("Man Implicit 3 - Woman Implicit 3", "Implicit 3 L − P"),
]


def format_effect(value: float, significant: bool) -> str:
    if abs(value) < 0.005:
        text = "0,00"
    else:
        text = f"{value:+.2f}".replace(".", ",")
    return text + ("*" if significant else "")


def main() -> None:
    source = pd.read_csv(TABLE_PATH)
    comparison_order = [comparison for comparison, _ in COMPARISONS]
    row_labels = [label for _, label in COMPARISONS]

    effects = source.pivot(
        index="comparison_label", columns="model_label", values="cohen_h"
    ).reindex(index=comparison_order, columns=MODEL_ORDER)
    significant = source.pivot(
        index="comparison_label", columns="model_label", values="significant_holm_0_05"
    ).reindex(index=comparison_order, columns=MODEL_ORDER).astype(bool)
    assert effects.notna().all().all()
    assert significant.shape == effects.shape

    annotations = effects.copy().astype(object)
    for comparison in effects.index:
        for model in effects.columns:
            annotations.loc[comparison, model] = format_effect(
                float(effects.loc[comparison, model]),
                bool(significant.loc[comparison, model]),
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
        "gender_effect", ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"]
    )

    fig = plt.figure(figsize=(13.6, 15.8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[18, 0.55],
        left=0.31,
        right=0.985,
        top=0.88,
        bottom=0.105,
        hspace=0.085,
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
        linewidths=1.1,
        linecolor="#FCFBF8",
        annot_kws={"fontsize": 15, "fontweight": "bold"},
        cbar_kws={
            "label": "Cohen's h (kondisi pertama dikurangi kondisi pembanding)",
            "orientation": "horizontal",
        },
        cbar_ax=colorbar_ax,
        ax=ax,
    )

    # White annotation text on saturated cells keeps the larger numbers legible.
    for text_artist, value in zip(ax.texts, effects.to_numpy().ravel()):
        text_artist.set_color("white" if abs(float(value)) >= 0.075 else "#262626")

    ax.set_yticklabels(row_labels, rotation=0, fontsize=13.2)
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[model] for model in MODEL_ORDER],
        rotation=0,
        ha="center",
        fontsize=12.2,
        fontweight="bold",
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # One heatmap, with section boundaries and a second-level grouped y-axis.
    for boundary in (8, 14):
        ax.axhline(boundary, color="#262626", linewidth=2.4, clip_on=False)
    group_axis = ax.secondary_yaxis("left", functions=(lambda y: y, lambda y: y))
    group_axis.spines["left"].set_position(("outward", 190))
    group_axis.spines["left"].set_visible(False)
    group_axis.set_yticks([4, 11, 16])
    group_axis.set_yticklabels(
        [
            "A. Dibandingkan\ndengan netral",
            "B. Implicit dibandingkan\ndengan explicit",
            "C. Laki-laki dibandingkan\ndengan perempuan",
        ],
        fontsize=12.2,
        fontweight="bold",
        ha="center",
    )
    group_axis.tick_params(axis="y", length=0, pad=50)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks(np.arange(-0.15, 0.151, 0.05))
    colorbar.set_ticklabels(
        [f"{value:.2f}".replace(".", ",") for value in np.arange(-0.15, 0.151, 0.05)]
    )
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(
        "Cohen's h (kondisi pertama dikurangi kondisi pembanding)",
        fontsize=13,
        labelpad=8,
    )

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Gender",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.022,
        "Tanda * menunjukkan hasil yang tetap signifikan setelah koreksi Holm "
        "(nilai-p Holm < 0,05). E = Explicit; I1–I3 = Implicit; N = Neutral; "
        "L = laki-laki; P = perempuan.",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#3F3F3F",
        wrap=True,
    )
    stem = FIGURE_DIR / "24_gender_effect_holm_single_portrait_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Render one continuous Religion primary Cohen's h and Holm heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models"
)
TABLE_PATH = REPORT_DIR / "tables" / "religion_paired_effects_significance.csv"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_ORDER = [
    "Qwen3 Next 80B-A3B",
    "Llama 3.1 8B",
    "Ministral 8B",
    "DeepSeek R1-Distill 32B",
    "GPT-OSS 20B",
    "Llama 3.3 70B",
]

COMPARISONS = [
    ("Atheist Implicit 1 - Atheist Explicit", "Atheist\nI1−E"),
    ("Atheist Implicit 2 - Atheist Explicit", "Atheist\nI2−E"),
    ("Atheist Implicit 3 - Atheist Explicit", "Atheist\nI3−E"),
    ("Christian Implicit 1 - Christian Explicit", "Christian\nI1−E"),
    ("Christian Implicit 2 - Christian Explicit", "Christian\nI2−E"),
    ("Christian Implicit 3 - Christian Explicit", "Christian\nI3−E"),
    ("Islam Implicit 1 - Islam Explicit", "Islam\nI1−E"),
    ("Islam Implicit 2 - Islam Explicit", "Islam\nI2−E"),
    ("Islam Implicit 3 - Islam Explicit", "Islam\nI3−E"),
    ("Jewish Implicit 1 - Jewish Explicit", "Jewish\nI1−E"),
    ("Jewish Implicit 2 - Jewish Explicit", "Jewish\nI2−E"),
    ("Jewish Implicit 3 - Jewish Explicit", "Jewish\nI3−E"),
]


def format_effect(value: float, significant: bool) -> str:
    if abs(value) < 0.005:
        text = "0,00"
    else:
        text = f"{value:+.2f}".replace(".", ",")
    return text + ("*" if significant else "")


def main() -> None:
    source = pd.read_csv(TABLE_PATH)
    source = source[source["analysis_scope"].eq("primary")].copy()
    comparison_order = [comparison for comparison, _ in COMPARISONS]

    effects = source.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = source.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()
    assert effects.shape == (6, 12)

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

    fig = plt.figure(figsize=(21.5, 9.2))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[6, 0.28],
        left=0.16,
        right=0.985,
        top=0.84,
        bottom=0.18,
        hspace=0.36,
    )
    ax = fig.add_subplot(grid[0])
    colorbar_ax = fig.add_subplot(grid[1])
    heatmap = sns.heatmap(
        effects,
        annot=annotations,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-0.30,
        vmax=0.30,
        linewidths=1.0,
        linecolor="#FCFBF8",
        annot_kws={"fontsize": 17, "fontweight": "bold"},
        cbar_ax=colorbar_ax,
        cbar_kws={"orientation": "horizontal"},
        ax=ax,
    )

    for text_artist, value in zip(ax.texts, effects.to_numpy().ravel()):
        text_artist.set_color("white" if abs(float(value)) >= 0.14 else "#262626")

    ax.set_yticklabels(MODEL_ORDER, rotation=0, fontsize=14, fontweight="bold")
    ax.set_xticklabels(
        [label for _, label in COMPARISONS],
        rotation=0,
        ha="center",
        fontsize=13,
    )
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Perbandingan primary persona implicit terhadap explicit", fontsize=14, labelpad=14)
    ax.set_ylabel("Model", fontsize=14, labelpad=12)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30])
    colorbar.set_ticklabels(["−0,30", "−0,20", "−0,10", "0,00", "0,10", "0,20", "0,30"])
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(
        "Cohen's h (persona implicit dikurangi persona explicit)",
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
        0.04,
        "Tanda * menunjukkan hasil yang tetap signifikan setelah koreksi Holm "
        "(nilai-p Holm < 0,05). I1–I3 = Implicit; E = Explicit. "
        "Gambar ini hanya memuat perbandingan primary dalam identitas agama yang sama.",
        ha="center",
        fontsize=12,
        color="#3F3F3F",
    )

    stem = FIGURE_DIR / "25_religion_primary_effect_holm_single_large_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Render 52 Religion comparisons as stacked 14/14/12/12 heatmap panels.

The statistical table is a read-only input. This module only changes the
presentation of the already-computed Cohen's h values and Holm decisions.
The legacy ``26_26`` output stem is retained for LaTeX path compatibility.
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

from render_gsm8k_religion_all_52_effect_holm_single_large_heatmap import (
    COMPARISONS,
)
from render_gsm8k_religion_primary_effect_holm_single_large_heatmap import (
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

PANEL_SPECS = [
    {
        "title": "Panel 1 — Ateis dan Kristen",
        "start": 0,
        "stop": 14,
    },
    {
        "title": "Panel 2 — Islam dan Yahudi",
        "start": 14,
        "stop": 28,
    },
    {
        "title": "Panel 3 — Ateis dibandingkan identitas agama lain",
        "start": 28,
        "stop": 40,
    },
    {
        "title": "Panel 4 — Perbandingan lintas agama lainnya",
        "start": 40,
        "stop": 52,
    },
]

MAX_PANEL_COLUMNS = 14

DEFAULT_OUTPUT = FIGURE_DIR / "26_religion_all_52_effect_holm_stacked_26_26.png"


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


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(TABLE_PATH)
    comparison_order = [comparison for comparison, _ in COMPARISONS]
    selected = source[source["comparison_label"].isin(comparison_order)]

    # Lock the complete statistical surface before any plotting operation.
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
    assert effects.shape == (6, 52)
    assert significant.shape == effects.shape
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


def stack_pair_label(label: str) -> str:
    """Wrap cross-identity abbreviations without changing their meaning."""
    lines = label.splitlines()
    if lines and "−" in lines[0]:
        left, right = lines[0].split("−", maxsplit=1)
        return "\n".join([f"{left}−", right, *lines[1:]])
    return label


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("DPI output harus minimal 300")

    effects, significant = build_matrices()
    annotations = annotation_matrix(effects, significant)
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

    fig = plt.figure(figsize=(18, 24))
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[6, 6, 6, 6, 0.5],
        left=0.235,
        right=0.985,
        top=0.93,
        bottom=0.12,
        hspace=0.72,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    colorbar_ax = fig.add_subplot(grid[4])
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
            vmin=-0.35,
            vmax=0.35,
            linewidths=1.25,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 18, "fontweight": "normal"},
            cbar=show_colorbar,
            cbar_ax=colorbar_ax if show_colorbar else None,
            cbar_kws={"orientation": "horizontal"} if show_colorbar else None,
            ax=ax,
        )
        ax.set_xlim(0, MAX_PANEL_COLUMNS)

        for text_artist, value, is_significant in zip(
            ax.texts,
            panel_effects.to_numpy().ravel(),
            panel_significant.to_numpy().ravel(),
        ):
            text_artist.set_color(
                "white" if abs(float(value)) >= 0.16 else "#262626"
            )
            text_artist.set_fontweight(
                "bold" if bool(is_significant) else "normal"
            )

        ax.set_yticklabels(
            [MODEL_ROW_LABELS[model] for model in MODEL_ORDER],
            rotation=0,
            fontsize=18.5,
            fontweight="bold",
        )
        ax.set_xticklabels(
            column_labels[start:stop],
            rotation=0,
            ha="center",
            fontsize=17,
            fontweight="semibold",
        )
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xlabel(
            "Perbandingan skenario",
            fontsize=19.5,
            fontweight="normal",
            labelpad=12,
        )
        ax.set_ylabel("Model", fontsize=19.5, fontweight="normal", labelpad=15)
        ax.set_title(
            str(spec["title"]),
            fontsize=21,
            fontweight="bold",
            pad=13,
        )

    assert heatmap is not None
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([-0.35, -0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30, 0.35])
    colorbar.set_ticklabels(
        ["−0,35", "−0,30", "−0,20", "−0,10", "0,00", "0,10", "0,20", "0,30", "0,35"]
    )
    colorbar.ax.tick_params(labelsize=17)
    colorbar.set_label("Cohen's h", fontsize=19, fontweight="bold", labelpad=8)

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Agama",
        fontsize=27,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.025,
        "* = signifikan setelah koreksi Holm (p-Holm < 0,05). † = perbandingan terhadap Gender Neutral "
        "(analisis sensitivitas; protokol berbeda).\n"
        "‡ = perbandingan antaridentitas (eksploratif). E = eksplisit; I1–I3 = implisit; N = netral; "
        "Ath = ateis; Chr = Kristen; Isl = Islam; Jew = Yahudi.\n"
        "Cohen's h dihitung dari skenario pertama dikurangi skenario pembanding.",
        ha="center",
        va="bottom",
        fontsize=17.5,
        color="#3F3F3F",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()

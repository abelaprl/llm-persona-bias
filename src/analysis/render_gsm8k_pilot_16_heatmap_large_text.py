"""Re-render the 16-model GSM8K pilot heatmap with legible cell labels.

This renderer only changes figure typography and dimensions. It reads the
previously verified pilot tables and does not recompute statistical results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "pilot_model_selection_exact_mcnemar_16_models"
)
CONDITION_ORDER = ["M:E", "M:I1", "M:I2", "M:I3", "W:E", "W:I1", "W:I2", "W:I3"]


def main() -> None:
    pairwise = pd.read_csv(OUTPUT_DIR / "pilot_vs_neutral_exact_mcnemar.csv")
    summary = pd.read_csv(OUTPUT_DIR / "pilot_model_selection_summary.csv")

    model_order = summary["model_label"].tolist()
    values = pairwise.pivot(
        index="model_label", columns="condition_short", values="risk_difference_pp"
    ).reindex(index=model_order, columns=CONDITION_ORDER)
    adjusted_p = pairwise.pivot(
        index="model_label", columns="condition_short", values="holm_adjusted_p_value"
    ).reindex(index=model_order, columns=CONDITION_ORDER)

    annotations = values.copy().astype(object)
    for model in values.index:
        for condition in values.columns:
            value = values.loc[model, condition]
            p_value = adjusted_p.loc[model, condition]
            annotations.loc[model, condition] = (
                "N/A"
                if pd.isna(value)
                else f"{value:+.1f}{'*' if pd.notna(p_value) and p_value < 0.05 else ''}"
            )

    selected_labels = set(
        summary.loc[summary["selected_for_main_experiment"], "model_label"]
    )
    limit = max(1.0, float(np.nanmax(np.abs(values.to_numpy(dtype=float)))))

    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
        }
    )

    # The larger canvas preserves generous cell height for the 16 model rows.
    fig, ax = plt.subplots(figsize=(18.2, 14.6))
    heatmap = sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        center=0,
        vmin=-limit,
        vmax=limit,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        linewidths=0.8,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Perubahan akurasi vs Neutral (pp)", "pad": 0.05},
        annot_kws={"fontsize": 13.2, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_xticklabels(CONDITION_ORDER, rotation=0, ha="center", fontsize=12.5)
    ax.tick_params(axis="y", rotation=0, labelsize=12.2)
    for tick in ax.get_yticklabels():
        if tick.get_text() in selected_labels:
            tick.set_color("#0047AB")
            tick.set_fontweight("bold")

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=11.5)
    colorbar.set_label("Perubahan akurasi vs Neutral (pp)", fontsize=12.5, labelpad=12)

    ax.set_title(
        "Pilot GSM8K 300 Soal: Seluruh Kondisi Gender terhadap Neutral",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Kondisi persona", fontsize=13)
    ax.set_ylabel("Model (biru tebal = dipilih)", fontsize=13)
    fig.text(
        0.5,
        0.018,
        "Angka adalah paired risk difference. * Holm-adjusted two-sided Exact McNemar p < 0,05 "
        "dalam delapan perbandingan per model. M/W = Man/Woman; E = Explicit; I1-I3 = Implicit.",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))

    stem = OUTPUT_DIR / "02_pilot_all_conditions_exact_mcnemar_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

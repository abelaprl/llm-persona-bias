"""Restore the final GSM8K v5 figures from verified CSV tables.

The full notebook recomputes expensive bootstrap intervals before plotting. This
small renderer is intentionally limited to Figures 16, 22, and 23, whose source
tables had already been written successfully. No statistical result is changed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models"
)
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
}
MODEL_AXIS_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next\n80B-A3B",
    "Llama 3.1 8B": "Llama 3.1\n8B",
    "Ministral 8B": "Ministral\n8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS\n20B",
    "Llama 3.3 70B": "Llama 3.3\n70B",
}
REGION_ORDER = [
    "north_american",
    "east_asian",
    "south_asian",
    "sub_saharan_african",
]
REGION_LABELS = {
    "north_american": "North American",
    "east_asian": "East Asian",
    "south_asian": "South Asian",
    "sub_saharan_african": "Sub-Saharan African",
}
DELTA_COLORS = ["#0047AB", "#B80F2E", "#007A4D", "#5B21B6"]


def configure_plots() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind", font_scale=1.08)
    mpl.rcParams.update(
        {
            "figure.dpi": 90,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "text.color": "#17212B",
            "axes.labelcolor": "#17212B",
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "legend.title_fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def render_figure_16() -> None:
    source = pd.read_csv(
        TABLE_DIR / "regional_completion_token_delta_implicit_vs_explicit.csv"
    )
    matrix = source.pivot(
        index="model_label", columns="region", values="completion_token_difference_pct"
    ).reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=REGION_ORDER
    )
    matrix.columns = [
        f"{REGION_LABELS[region]}: Implicit Avg − Explicit" for region in REGION_ORDER
    ]

    fig, ax = plt.subplots(figsize=(19, 9))
    matrix.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            xytext=(0, 5 if value >= 0 else -5),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )
    values = matrix.to_numpy(dtype=float)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = max(1.0, maximum - minimum)
    ax.set_ylim(
        0 if minimum >= 0 else minimum - 0.16 * span,
        0 if maximum <= 0 else maximum + 0.16 * span,
    )
    ax.set_title(
        "GSM8K Regional Cultural: Completion Token Persona Implicit terhadap Explicit"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Change in average completion tokens (%)")
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[label] for label in matrix.index], rotation=0, ha="center"
    )
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "16_regional_completion_token_delta_implicit_vs_explicit")


def render_figure_22() -> None:
    frame = pd.read_csv(
        TABLE_DIR / "gender_completion_token_accuracy_tradeoff_implicit_vs_explicit.csv"
    )
    identity_colors = {"Man": "#0047AB", "Woman": "#B80F2E"}
    markers = {"Implicit 1": "o", "Implicit 2": "s", "Implicit 3": "D"}

    x_values = frame["token_delta_pct"]
    y_values = frame["accuracy_delta_pp"]
    x_min = min(-20.0, float(x_values.min()) - 25.0)
    x_max = float(x_values.max()) + 35.0
    y_padding = max(0.35, 0.10 * float(y_values.max() - y_values.min()))
    y_min = min(-0.35, float(y_values.min()) - y_padding)
    y_max = max(0.35, float(y_values.max()) + y_padding)

    fig, axes = plt.subplots(2, 3, figsize=(20.5, 12), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        model_rows = frame[frame["model"].eq(model)]
        ax.axhline(0, color="#30343B", linewidth=1.4, linestyle="--", zorder=1)
        ax.axvline(0, color="#30343B", linewidth=1.4, linestyle="--", zorder=1)
        ax.scatter(
            [0], [0], marker="X", s=130, color="#30343B", edgecolor="#FCFBF8",
            linewidth=1.1, zorder=5,
        )
        for identity, color in identity_colors.items():
            identity_rows = model_rows[model_rows["identity_label"].eq(identity)]
            for variant, marker in markers.items():
                point = identity_rows[identity_rows["implicit_variant"].eq(variant)]
                ax.scatter(
                    point["token_delta_pct"], point["accuracy_delta_pp"],
                    marker=marker, s=120, color=color, edgecolor="#FCFBF8",
                    linewidth=1.2, zorder=4,
                )
        rho = spearmanr(
            model_rows["token_delta_pct"], model_rows["accuracy_delta_pp"]
        ).statistic
        ax.set_title(
            f"{MODEL_LABELS[model]}\nSpearman's ρ = {rho:+.3f} (n = 6; descriptive)",
            fontsize=15.5,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="#D7DBDF", linewidth=0.8, alpha=0.8)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=color,
            markeredgecolor="#FCFBF8", markersize=10, label=identity,
        )
        for identity, color in identity_colors.items()
    ] + [
        Line2D(
            [0], [0], marker=marker, color="#30343B", linestyle="none",
            markersize=9, label=variant,
        )
        for variant, marker in markers.items()
    ] + [
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor="#30343B",
            markersize=10, label="Explicit baseline",
        )
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=6, frameon=True, title="Identity, Implicit Variant, and Baseline",
    )
    overall_rho = spearmanr(frame["token_delta_pct"], frame["accuracy_delta_pp"]).statistic
    fig.suptitle(
        "GSM8K Gender: Completion Token–Accuracy Trade-off untuk Implicit Variants vs Explicit",
        fontsize=18, fontweight="bold", y=0.995,
    )
    fig.supxlabel("Change in Average Completion Tokens vs Explicit (%)", y=0.035)
    fig.supylabel("Change in Accuracy vs Explicit (percentage points)", x=0.015)
    fig.text(
        0.5, 0.008,
        f"Exploratory delta-pooled Spearman's ρ = {overall_rho:+.3f} "
        f"(n = {len(frame)}); association only, not causal inference.",
        ha="center", va="bottom", fontsize=13, color="#30343B",
    )
    fig.tight_layout(rect=(0.035, 0.06, 1, 0.90), h_pad=2.0, w_pad=1.5)
    save_figure(fig, "22_gender_completion_token_accuracy_tradeoff_implicit_vs_explicit")


def render_figure_23() -> None:
    source = pd.read_csv(TABLE_DIR / "neutral_persona_accuracy_six_models.csv")
    source = source.set_index("model").reindex(MODEL_ORDER).reset_index()
    values = source["accuracy_pct"].to_numpy(dtype=float) / 100
    average = float(values.mean())

    fig, ax = plt.subplots(figsize=(17, 9))
    bars = ax.bar(
        np.arange(len(MODEL_ORDER)), values,
        color="#30343B", edgecolor="#FCFBF8", linewidth=1.4, width=0.68,
    )
    ax.axhline(
        average, color="#B80F2E", linestyle="--", linewidth=2.2,
        label=f"Neutral average ({average:.3f})",
    )
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=16, fontweight="bold",
        )
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER],
        rotation=0, ha="center", fontsize=12,
    )
    ax.set_ylim(0.80, 1.00)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("GSM8K Neutral Persona: Accuracy Enam Model Utama", fontsize=18)
    ax.grid(axis="y", color="#D7DBDF", linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, "23_neutral_persona_accuracy_six_models")


def main() -> None:
    configure_plots()
    render_figure_16()
    render_figure_22()
    render_figure_23()

    assert len(list(FIGURE_DIR.glob("*.png"))) == 23
    assert len(list(FIGURE_DIR.glob("*.svg"))) == 23


if __name__ == "__main__":
    main()

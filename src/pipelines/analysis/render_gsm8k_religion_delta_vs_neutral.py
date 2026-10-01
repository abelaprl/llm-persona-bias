"""Render the Chapter VI GSM8K religion average deltas versus Neutral.

The aggregation deliberately mirrors the gender figure:

    persona average = (explicit accuracy + mean(I1, I2, I3)) / 2

Input percentages are converted to the 0--1 accuracy scale used by
``02_gender_delta_vs_neutral``.  The chart is descriptive; paired
significance results remain in the dedicated McNemar/Holm figures.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DIR = Path(
    "D:/TugasAkhir/result/GMS8K/reports/"
    "persona_publication_analysis_v5_paired_significance_six_models"
)
SOURCE = REPORT_DIR / "tables" / "religion_explicit_implicit_and_deltas.csv"
OUTPUT_TABLE = REPORT_DIR / "tables" / "religion_average_delta_vs_neutral.csv"
OUTPUT_STEM = REPORT_DIR / "figures" / "28_religion_delta_vs_neutral"

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]

MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next\n80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1\n8B",
    "mistralai_ministral-8b-2512": "Ministral\n8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill\n32B",
    "openai_gpt-oss-20b": "GPT-OSS\n20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3\n70B",
}

# The order follows the v5 religion summary and keeps all four identities
# specified in Chapter IV.
PERSONAS = [
    ("Atheist", "Atheist Avg − Neutral", "#0b4fae"),
    ("Christian", "Christian Avg − Neutral", "#c8102e"),
    ("Islam", "Islam Avg − Neutral", "#5b21b6"),
    ("Jewish", "Jewish Avg − Neutral", "#e07a00"),
]


def load_deltas() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE).set_index("model").reindex(MODEL_ORDER)
    if frame.index.hasnans or frame["Neutral"].isna().any():
        raise ValueError("The religion summary is missing one or more main models.")

    output = frame[["model_label", "Neutral"]].copy()
    for persona, _, _ in PERSONAS:
        overall_col = f"{persona} Overall Avg"
        output[f"{persona} Avg"] = frame[overall_col]
        output[f"{persona} Avg − Neutral (pp)"] = (
            frame[overall_col] - frame["Neutral"]
        )
        output[f"{persona} Avg − Neutral (accuracy scale)"] = (
            output[f"{persona} Avg − Neutral (pp)"] / 100.0
        )

    output.reset_index().to_csv(OUTPUT_TABLE, index=False)
    return output


def render(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    all_values = []

    fig, ax = plt.subplots(figsize=(20.48, 9.6))
    for offset, (persona, legend_label, color) in zip(offsets, PERSONAS):
        values = frame[f"{persona} Avg − Neutral (accuracy scale)"].to_numpy(
            dtype=float
        )
        all_values.extend(values.tolist())
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=legend_label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=12.5,
                fontweight="bold",
                color="#17202a",
                rotation=90,
                rotation_mode="anchor",
                zorder=5,
            )

    max_abs = max(abs(value) for value in all_values)
    limit = math.ceil((max_abs + 0.007) * 200) / 200
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#111111", linewidth=1.35, zorder=4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)
    for position in x[:-1] + 0.5:
        ax.axvline(position, color="#9aa0a6", linewidth=0.8, alpha=0.45, zorder=1)

    ax.set_title(
        "GSM8K Agama: Analisis Delta terhadap Persona Netral",
        pad=18,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference on accuracy scale")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#56616f")
    ax.spines["bottom"].set_color("#56616f")
    fig.subplots_adjust(left=0.07, right=0.83, top=0.89, bottom=0.14)

    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT_STEM.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    frame = load_deltas()
    render(frame)
    console_summary = pd.DataFrame(
        {
            f"{persona}_avg_minus_neutral": frame[
                f"{persona} Avg − Neutral (accuracy scale)"
            ]
            for persona, _, _ in PERSONAS
        }
    )
    print(console_summary.round(4).to_string())
    print(f"table={OUTPUT_TABLE}")
    print(f"figure={OUTPUT_STEM}.png/.svg")


if __name__ == "__main__":
    main()

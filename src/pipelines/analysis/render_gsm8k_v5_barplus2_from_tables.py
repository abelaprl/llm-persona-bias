"""Render the nine GSM8K v5 accuracy/delta bars directly from saved tables.

This typography-only renderer avoids rerunning bootstrap and paired inference.
It reproduces Figures 01--03, 05--06, 12--13, 16, and 23 with the v5 source
font sizes increased by 2 pt and the established 1.12 figure-size scale.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
SOURCE_TABLE_DIR = (
    REPORT_ROOT
    / "persona_publication_analysis_v5_paired_significance_six_models"
    / "tables"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".codex_tmp" / "gsm8k_v5_barplus2_stage" / "figures"

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
MODEL_AXIS_ORDER = [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER]

ACCURACY_COLORS = [
    "#0047AB", "#0086A8", "#B80F2E", "#E45700",
    "#007A4D", "#65A30D", "#5B21B6", "#A0006D", "#30343B",
]
DELTA_COLORS = ["#0047AB", "#B80F2E", "#007A4D", "#5B21B6"]
GENDER_DELTA_COLORS = ["#0047AB", "#B80F2E", "#5B21B6"]
REGION_LABELS = {
    "north_american": "North American",
    "east_asian": "East Asian",
    "south_asian": "South Asian",
    "sub_saharan_african": "Sub-Saharan African",
}

FONT_DELTA_PT = 2.0
FIGURE_SCALE = 1.12
ANNOTATION_SIZE = 20
DELTA_ANNOTATION_SIZE = 16
TITLE_SIZE = 22


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=SOURCE_TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--font-delta", type=float, default=2.0)
    parser.add_argument("--figure-scale", type=float, default=1.12)
    parser.add_argument("--accuracy-annotation-size", type=float, default=20.0)
    parser.add_argument("--delta-annotation-size", type=float, default=16.0)
    parser.add_argument("--only", choices=("all", "figure16"), default="all")
    return parser


def _figsize(width: float, height: float) -> tuple[float, float]:
    return width * FIGURE_SCALE, height * FIGURE_SCALE


def configure_typography(
    font_delta: float,
    figure_scale: float,
    accuracy_annotation_size: float,
    delta_annotation_size: float,
) -> None:
    global FONT_DELTA_PT, FIGURE_SCALE, ANNOTATION_SIZE
    global DELTA_ANNOTATION_SIZE, TITLE_SIZE
    FONT_DELTA_PT = font_delta
    FIGURE_SCALE = figure_scale
    ANNOTATION_SIZE = accuracy_annotation_size
    DELTA_ANNOTATION_SIZE = delta_annotation_size
    TITLE_SIZE = 20 + font_delta


def _set_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.25 + FONT_DELTA_PT / 12.0,
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 90,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "text.color": "#17212B",
            "axes.labelcolor": "#17212B",
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": 20 + FONT_DELTA_PT,
            "axes.labelsize": 17 + FONT_DELTA_PT,
            "xtick.labelsize": 15 + FONT_DELTA_PT,
            "ytick.labelsize": 15 + FONT_DELTA_PT,
            "legend.fontsize": 15 + FONT_DELTA_PT,
            "legend.title_fontsize": 16 + FONT_DELTA_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model"] = result["model"].astype(str)
    result = result.set_index("model").reindex(MODEL_ORDER).reset_index()
    if result["model_label"].isna().any():
        raise ValueError("At least one required model is missing from a saved table.")
    return result


def _save(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[BAR +2] {stem}.png/.svg")


def _annotate_accuracy(ax: mpl.axes.Axes, offset: int = 4) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, offset),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            rotation=90,
            fontweight="normal",
        )


def _annotate_delta(
    ax: mpl.axes.Axes,
    fmt: str = ".4f",
    *,
    rotation: float = 90,
    offset: int = 5,
    fontsize: float | None = None,
    rotation_mode: str = "anchor",
) -> None:
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            format(value, fmt),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            xytext=(0, offset if value >= 0 else -offset),
            textcoords="offset points",
            fontsize=fontsize if fontsize is not None else DELTA_ANNOTATION_SIZE,
            fontweight="normal",
            rotation=rotation,
            rotation_mode=rotation_mode,
        )


def _render_gender(
    table_dir: Path,
    output_dir: Path,
    *,
    only_stems: set[str] | None = None,
) -> None:
    frame = _ordered(pd.read_csv(table_dir / "gender_five_category_and_deltas.csv"))
    categories = [
        "Man Explicit", "Man Implicit Avg", "Woman Explicit", "Woman Implicit Avg", "Neutral"
    ]
    accuracy = frame.set_index("model_label")[categories] / 100
    neutral_average = accuracy["Neutral"].mean()

    fig, ax = plt.subplots(figsize=_figsize(19, 9.5))
    accuracy.plot(kind="bar", width=0.84, color=[ACCURACY_COLORS[i] for i in [0, 1, 2, 3, 8]], ax=ax)
    baseline_label = f"Global Neutral Avg ({neutral_average:.3f})"
    ax.axhline(
        neutral_average, color="#B80F2E", linestyle="--", linewidth=2.0,
        label=baseline_label, zorder=1,
    )
    _annotate_accuracy(ax)
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [h for index, h in enumerate(handles) if index != baseline_index]
    labels = [labels[baseline_index]] + [label for index, label in enumerate(labels) if index != baseline_index]
    ax.set_ylim(0.80, 1.00)
    ax.set_title(
        "GSM8K Gender: Perbandingan Akurasi Antar-Persona (Skala 0,8–1,0)",
        fontsize=TITLE_SIZE, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center")
    ax.legend(handles, labels, title="Persona & Baseline", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    accuracy_stem = "01_gender_absolute_accuracy_five_categories"
    if only_stems is None or accuracy_stem in only_stems:
        _save(fig, output_dir, accuracy_stem)
    else:
        plt.close(fig)

    delta_specs = (
        (
            ["Delta Man vs Neutral", "Delta Woman vs Neutral", "Gender Gap"],
            ["Man Avg − Neutral", "Woman Avg − Neutral", "Man Avg − Woman Avg"],
            "GSM8K Gender: Analisis Delta terhadap Persona Neutral",
            "02_gender_delta_vs_neutral",
        ),
        (
            ["Delta Man Imp-Exp", "Delta Woman Imp-Exp", "Gender Gap"],
            ["Man Implicit Avg − Explicit", "Woman Implicit Avg − Explicit", "Man Avg − Woman Avg"],
            "GSM8K Gender — Analisis Delta: Performance Gap dan Gender Bias",
            "03_gender_delta_implicit_vs_explicit",
        ),
    )
    for columns, labels, title, stem in delta_specs:
        delta = frame.set_index("model_label")[columns] / 100
        delta.columns = labels
        fig, ax = plt.subplots(figsize=_figsize(19, 9))
        delta.plot(kind="bar", width=0.78, color=GENDER_DELTA_COLORS, ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        _annotate_delta(
            ax,
            rotation=90,
            offset=10,
            fontsize=DELTA_ANNOTATION_SIZE,
            rotation_mode="default",
        )
        limit = max(0.001, np.nanmax(np.abs(delta.to_numpy())))
        ax.set_ylim(-limit * 1.38, limit * 1.38)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=16)
        ax.set_xlabel("Model")
        ax.set_ylabel("Difference on accuracy scale")
        ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        if only_stems is None or stem in only_stems:
            _save(fig, output_dir, stem)
        else:
            plt.close(fig)


def _render_group_accuracy_delta(
    frame: pd.DataFrame,
    identities: list[str],
    output_dir: Path,
    *,
    domain: str,
    accuracy_stem: str,
    delta_stem: str,
    accuracy_title: str,
    delta_title: str,
    accuracy_width: float,
    accuracy_ylim: tuple[float, float],
    legend_title: str,
) -> None:
    accuracy_columns = [
        column for identity in identities
        for column in [f"{identity} Explicit", f"{identity} Implicit Avg"]
    ] + ["Neutral"]
    accuracy = frame.set_index("model_label")[accuracy_columns] / 100
    neutral_average = accuracy["Neutral"].mean()
    fig, ax = plt.subplots(figsize=_figsize(accuracy_width, 9.5))
    accuracy.plot(kind="bar", width=0.88, color=ACCURACY_COLORS, ax=ax)
    baseline_label = f"Global Neutral Avg ({neutral_average:.3f})"
    ax.axhline(
        neutral_average, color="#B80F2E", linestyle="--", linewidth=2.0,
        label=baseline_label, zorder=1,
    )
    _annotate_accuracy(ax, offset=3)
    ax.set_ylim(*accuracy_ylim)
    ax.set_title(accuracy_title, fontsize=TITLE_SIZE, fontweight="bold", pad=18)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [h for index, h in enumerate(handles) if index != baseline_index]
    labels = [labels[baseline_index]] + [label for index, label in enumerate(labels) if index != baseline_index]
    ax.legend(handles, labels, title=legend_title, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save(fig, output_dir, accuracy_stem)

    delta_columns = [f"{identity} Delta Imp-Exp" for identity in identities]
    delta = frame.set_index("model_label")[delta_columns] / 100
    delta.columns = [f"{identity}: Implicit Avg − Explicit" for identity in identities]
    fig, ax = plt.subplots(figsize=_figsize(19, 9))
    delta.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_delta(ax)
    limit = max(0.001, np.nanmax(np.abs(delta.to_numpy())))
    multiplier = 1.35 if domain == "Religion" else 1.4
    ax.set_ylim(-limit * multiplier, limit * multiplier)
    ax.set_title(delta_title, fontsize=TITLE_SIZE, fontweight="bold", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference on accuracy scale")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, delta_stem)


def _render_religion_and_regional(table_dir: Path, output_dir: Path) -> None:
    religion = _ordered(pd.read_csv(table_dir / "religion_explicit_implicit_and_deltas.csv"))
    _render_group_accuracy_delta(
        religion,
        ["Atheist", "Christian", "Islam", "Jewish"],
        output_dir,
        domain="Religion",
        accuracy_stem="05_religion_absolute_accuracy_explicit_implicit",
        delta_stem="06_religion_delta_implicit_vs_explicit",
        accuracy_title="GSM8K Religion: Perbandingan Akurasi Antar-Persona (Skala 0,75–1,0)",
        delta_title="GSM8K Religion — Analisis Delta: Persona Implicit vs Explicit",
        accuracy_width=22,
        accuracy_ylim=(0.75, 1.00),
        legend_title="Religion Persona & Baseline",
    )

    regional = _ordered(pd.read_csv(table_dir / "regional_accuracy_and_deltas.csv"))
    regional_columns = [
        column for identity in REGION_LABELS.values()
        for column in [f"{identity} Explicit", f"{identity} Implicit Avg"]
    ] + ["Neutral"]
    regional_min = max(
        0.70,
        math.floor(
            (np.nanmin((regional.set_index("model_label")[regional_columns] / 100).to_numpy()) - 0.02)
            * 20
        ) / 20,
    )
    _render_group_accuracy_delta(
        regional,
        list(REGION_LABELS.values()),
        output_dir,
        domain="Regional Cultural",
        accuracy_stem="12_regional_absolute_accuracy_explicit_implicit",
        delta_stem="13_regional_delta_implicit_vs_explicit",
        accuracy_title="GSM8K Regional Cultural: Perbandingan Akurasi Antar-Persona",
        delta_title="GSM8K Regional Cultural — Analisis Delta: Persona Implicit vs Explicit",
        accuracy_width=23,
        accuracy_ylim=(regional_min, 1.00),
        legend_title="Regional Persona & Baseline",
    )


def _render_regional_tokens(table_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(table_dir / "regional_completion_token_delta_implicit_vs_explicit.csv")
    frame["region_label"] = frame["region"].map(REGION_LABELS)
    matrix = (
        frame.pivot(index="model_label", columns="region_label", values="completion_token_difference_pct")
        .reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=list(REGION_LABELS.values()))
    )
    matrix.columns = [f"{label}: Implicit Avg − Explicit" for label in matrix.columns]

    fig, ax = plt.subplots(figsize=_figsize(19, 9))
    matrix.plot(kind="bar", width=0.82, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            f"{value:+.1f}%", (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom" if value >= 0 else "top",
            xytext=(0, 4 if value >= 0 else -4), textcoords="offset points",
            fontsize=DELTA_ANNOTATION_SIZE, fontweight="bold",
            rotation=90, rotation_mode="anchor",
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
        "GSM8K Regional Cultural: Perubahan Relatif Completion Tokens\n"
        "Persona Implicit vs Explicit (%)",
        pad=14,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Perubahan relatif rata-rata completion tokens (%)")
    ax.set_xticklabels(MODEL_AXIS_ORDER, rotation=0, ha="center")
    ax.legend(title="Region", loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.5,
        0.012,
        "Rumus: 100 × (Implicit Avg − Explicit) / Explicit. "
        "Metrik ini mengukur panjang jawaban, bukan akurasi.",
        ha="center",
        va="bottom",
        fontsize=13 + FONT_DELTA_PT,
        color="#30343B",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    _save(fig, output_dir, "16_regional_completion_token_delta_implicit_vs_explicit")


def _render_neutral(table_dir: Path, output_dir: Path) -> None:
    frame = _ordered(pd.read_csv(table_dir / "neutral_persona_accuracy_six_models.csv"))
    values = frame["accuracy_pct"].to_numpy() / 100
    average = float(values.mean())
    fig, ax = plt.subplots(figsize=_figsize(17, 9))
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
            f"{value:.3f}", xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
            fontsize=ANNOTATION_SIZE, fontweight="bold",
        )
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(
        MODEL_AXIS_ORDER,
        rotation=0,
        ha="center",
        fontsize=16 + FONT_DELTA_PT,
    )
    ax.set_ylim(0.80, 1.00)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "GSM8K Neutral Persona: Accuracy Enam Model Utama",
        fontsize=21 + FONT_DELTA_PT,
    )
    ax.grid(axis="y", color="#D7DBDF", linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save(fig, output_dir, "23_neutral_persona_accuracy_six_models")


def main() -> int:
    args = _parser().parse_args()
    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    configure_typography(
        args.font_delta,
        args.figure_scale,
        args.accuracy_annotation_size,
        args.delta_annotation_size,
    )
    required = (
        "gender_five_category_and_deltas.csv",
        "religion_explicit_implicit_and_deltas.csv",
        "regional_accuracy_and_deltas.csv",
        "regional_completion_token_delta_implicit_vs_explicit.csv",
        "neutral_persona_accuracy_six_models.csv",
    )
    missing = [name for name in required if not (table_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validated v5 tables: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    if args.only == "figure16":
        _render_regional_tokens(table_dir, output_dir)
        print(f"[DONE] Figure 16: {output_dir}")
        return 0

    _render_gender(table_dir, output_dir)
    _render_religion_and_regional(table_dir, output_dir)
    _render_regional_tokens(table_dir, output_dir)
    _render_neutral(table_dir, output_dir)
    png = list(output_dir.glob("*.png"))
    svg = list(output_dir.glob("*.svg"))
    if len(png) != 9 or len(svg) != 9:
        raise RuntimeError(f"Expected 9 PNG/SVG bars, found {len(png)}/{len(svg)}.")
    print(f"[DONE] 9 PNG + 9 SVG bar figures: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

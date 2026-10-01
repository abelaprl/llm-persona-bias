"""Render a standardized, read-only ETHICS Gender accuracy bundle.

The source of truth is the validated ETHICS Gender publication bundle.  This
module only reads its saved tables and writes presentation artefacts under the
notebook directory; it never reparses responses, rescored answers, calls an
API, or overwrites the source report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BUNDLE = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "_combined"
    / "persona_publication_analysis_v2_readable_paired_holm_six_models"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "notebooks" / "gender" / "ethics accuracy analysis"
)

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

PERSONA_ORDER = [
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
    "neutral",
]
PERSONA_LABELS = {
    "man_explicit_1": "Laki-laki E",
    "man_implicit_1": "Laki-laki I1",
    "man_implicit_2": "Laki-laki I2",
    "man_implicit_3": "Laki-laki I3",
    "woman_explicit_1": "Perempuan E",
    "woman_implicit_1": "Perempuan I1",
    "woman_implicit_2": "Perempuan I2",
    "woman_implicit_3": "Perempuan I3",
    "neutral": "Netral",
}
PERSONA_AXIS_LABELS = [PERSONA_LABELS[item] for item in PERSONA_ORDER]
PERSONA_COMPARISON_ORDER = [item for item in PERSONA_ORDER if item != "neutral"]

SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
VALID_SUBSETS = ["commonsense", "justice"]
INVALID_STANCE_SUBSETS = ["deontology", "virtue"]

STATUS_ORDER = ["committed", "abstain", "missing"]
STATUS_LABELS = {
    "committed": "Committed",
    "abstain": "Abstain",
    "missing": "Missing",
}
STATUS_COLORS = {
    "committed": "#2A9D8F",
    "abstain": "#E9C46A",
    "missing": "#E76F51",
}

ABSTAIN_REASON_LABELS = {
    "both_sides": "Kedua sisi / tidak berpihak",
    "analysis_without_verdict": "Analisis tanpa verdict",
    "no_opinion_stop": "Tidak punya opini lalu berhenti",
    "explicit_binary_refusal": "Menolak jawaban biner",
    "task_substitution": "Mengganti tugas",
    "ai_refusal": "Penolakan sebagai AI",
    "insufficient_context": "Konteks dianggap kurang",
    "placeholder_confusion": "Bingung placeholder",
    "returns_question": "Mengulang pertanyaan",
    "capability_mismatch": "Mengaku tidak mampu",
    "detector_disagreement": "Detektor tidak sepakat",
    "conditional_only": "Jawaban hanya bersyarat",
    "unspecified": "Tidak dispesifikasikan",
}

BACKGROUND = "#FCFBF8"
TEXT = "#17212B"
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 22
PANEL_TITLE_SIZE = 18
HEATMAP_ANNOTATION_SIZE = 12
COMMIT_RATE_ANNOTATION_SIZE = 16
COMMIT_RATE_PANEL_TITLE_SIZE = 20
COMMIT_RATE_TICK_SIZE = 15
COMMIT_RATE_COLORBAR_TICK_SIZE = 14
COMMIT_RATE_COLORBAR_LABEL_SIZE = 15


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
            "font.family": FONT_FAMILY,
            "font.weight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor=BACKGROUND, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _sort(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.Categorical(result[column], categories=order, ordered=True)
    return result.sort_values(column, kind="stable")


def _model_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model"] = pd.Categorical(
        result["model"], categories=MODEL_ORDER, ordered=True
    )
    return result.sort_values("model", kind="stable")


def _persona_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["persona"] = pd.Categorical(
        result["persona"], categories=PERSONA_ORDER, ordered=True
    )
    return result.sort_values("persona", kind="stable")


def _matrix(
    frame: pd.DataFrame,
    value: str,
    *,
    index: str = "model_label",
    columns: str = "persona_label",
    persona_order: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.pivot(index=index, columns=columns, values=value)
    index_order = [MODEL_LABELS[model] for model in MODEL_ORDER]
    column_order = [PERSONA_LABELS[item] for item in (persona_order or PERSONA_ORDER)]
    return result.reindex(index=index_order, columns=column_order)


def _style_heatmap(ax: plt.Axes, rotation: float = 42) -> None:
    ax.tick_params(axis="x", labelsize=13, rotation=rotation, pad=5)
    ax.tick_params(axis="y", labelsize=13, rotation=0, pad=4)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_fontweight("normal")
    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
    for text in ax.texts:
        text.set_fontweight("normal")
        text.set_fontsize(HEATMAP_ANNOTATION_SIZE)
    if ax.collections and ax.collections[0].colorbar is not None:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=12)
        colorbar.ax.yaxis.label.set_size(13)


def _load_source_tables(source_bundle: Path) -> dict[str, pd.DataFrame]:
    table_dir = source_bundle / "tables"
    names = {
        "condition_summary": "condition_summary_full.csv",
        "model_overview": "model_overview.csv",
        "status_independence": "persona_status_independence.csv",
        "abstain_reason": "abstain_reason_by_condition.csv",
        "behavior_category": "behavior_category_by_condition.csv",
        "response_flags": "response_flags_by_condition.csv",
        "paired_effects": "paired_effects_all_outcomes.csv",
        "omnibus": "cochran_q_omnibus.csv",
    }
    tables = {}
    for key, name in names.items():
        path = table_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[key] = pd.read_csv(path, low_memory=False)
    manifest_path = source_bundle / "metadata" / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    tables["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tables


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def derive_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    summary = tables["condition_summary"].copy()
    summary["construct_valid"] = summary["construct_valid"].astype(bool)

    status_count_columns = ["total", "nonmissing", "committed", "abstain", "missing"]
    status = (
        summary.groupby(
            ["model", "model_label", "persona", "persona_label"],
            as_index=False,
            sort=False,
        )[status_count_columns]
        .sum()
    )
    status["commit_rate"] = _safe_rate(status["committed"], status["total"])
    status["commit_rate_nonmissing"] = _safe_rate(
        status["committed"], status["nonmissing"]
    )
    status["abstain_rate"] = _safe_rate(status["abstain"], status["total"])
    status["missing_rate"] = _safe_rate(status["missing"], status["total"])
    status = _model_ordered(_persona_ordered(status))

    reason_source = tables["abstain_reason"].copy()
    reason_source["abstain_reason"] = reason_source["abstain_reason"].replace(
        "", "unspecified"
    )
    reason_columns = ["model", "persona", "abstain_reason"]
    reason_model = (
        reason_source.groupby(reason_columns, as_index=False, sort=False)["reason_count"]
        .sum()
        .merge(
            status[
                [
                    "model",
                    "persona",
                    "total",
                    "nonmissing",
                    "abstain",
                ]
            ],
            on=["model", "persona"],
            validate="many_to_one",
        )
    )
    reason_model["pct_of_abstain"] = _safe_rate(
        reason_model["reason_count"], reason_model["abstain"]
    )
    reason_model["pct_of_nonmissing"] = _safe_rate(
        reason_model["reason_count"], reason_model["nonmissing"]
    )
    reason_model["reason_label"] = reason_model["abstain_reason"].map(
        lambda value: ABSTAIN_REASON_LABELS.get(
            value, str(value).replace("_", " ").title()
        )
    )
    reason_model = _model_ordered(_persona_ordered(reason_model))

    reason_persona = (
        reason_model.groupby(
            ["persona", "reason_label", "abstain_reason"], as_index=False, sort=False
        )[["reason_count"]]
        .sum()
    )
    persona_denominator = (
        status.groupby("persona", as_index=False)[["total", "nonmissing", "abstain"]]
        .sum()
        .rename(
            columns={
                "total": "total_responses",
                "nonmissing": "total_nonmissing",
                "abstain": "total_abstain",
            }
        )
    )
    reason_persona = reason_persona.merge(
        persona_denominator, on="persona", validate="many_to_one"
    )
    reason_persona["pct_of_abstain"] = _safe_rate(
        reason_persona["reason_count"], reason_persona["total_abstain"]
    )
    reason_persona["pct_of_nonmissing"] = _safe_rate(
        reason_persona["reason_count"], reason_persona["total_nonmissing"]
    )
    reason_persona = _persona_ordered(reason_persona)

    accuracy_valid = summary[summary["subset"].isin(VALID_SUBSETS)].copy()
    stance_invalid = summary[summary["subset"].isin(INVALID_STANCE_SUBSETS)].copy()

    paired = tables["paired_effects"].copy()
    paired_primary = paired[
        paired["persona_b"].eq("neutral")
        & paired["persona_a"].isin(PERSONA_COMPARISON_ORDER)
    ].copy()

    return {
        "summary": summary,
        "status": status,
        "status_by_model_persona": status,
        "reason_model": reason_model,
        "reason_persona": reason_persona,
        "accuracy_valid": accuracy_valid,
        "stance_invalid": stance_invalid,
        "paired_primary": paired_primary,
    }


def write_derived_tables(derived: dict[str, pd.DataFrame], table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "status_by_model_persona": "status_by_model_persona.csv",
        "reason_model": "abstain_reason_by_model_persona.csv",
        "reason_persona": "abstain_reason_by_persona.csv",
        "accuracy_valid": "accuracy_construct_valid_subsets.csv",
        "stance_invalid": "stance_construct_invalid_subsets.csv",
        "paired_primary": "paired_primary_effects_vs_neutral.csv",
    }
    paths = {}
    for key, name in mapping.items():
        path = table_dir / name
        derived[key].to_csv(path, index=False)
        paths[key] = path
    return paths


def plot_status_composition(overview: pd.DataFrame, figure_dir: Path) -> list[Path]:
    data = overview.set_index("model_label").reindex(
        [MODEL_LABELS[model] for model in MODEL_ORDER]
    )
    fig, ax = plt.subplots(figsize=(17, 8.5))
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    rate_columns = {
        "committed": "commit_rate",
        "abstain": "abstain_rate",
        "missing": "missing_rate",
    }
    for status in STATUS_ORDER:
        values = 100 * data[rate_columns[status]].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=STATUS_LABELS[status],
            color=STATUS_COLORS[status],
            edgecolor="white",
            linewidth=0.8,
        )
        for index, value in enumerate(values):
            if value >= 2.0:
                ax.text(
                    index,
                    bottom[index] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="normal",
                )
        bottom += values
    ax.set_xticks(x, [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Persentase seluruh respons")
    ax.set_title(
        "ETHICS Gender: Komposisi Status Respons per Model",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=18,
    )
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, figure_dir, "01_ethics_status_composition_by_model")


def plot_status_heatmaps(status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    rate_columns = ["commit_rate", "abstain_rate", "missing_rate"]
    titles = ["Committed", "Abstain", "Missing"]
    cmaps = ["YlGnBu", "YlOrBr", "OrRd"]
    fig, axes = plt.subplots(1, 3, figsize=(31, 9), constrained_layout=True)
    for ax, value, title, cmap in zip(axes, rate_columns, titles, cmaps, strict=True):
        matrix = _matrix(status, value) * 100
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": HEATMAP_ANNOTATION_SIZE, "fontweight": "normal"},
            cmap=cmap,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "%"},
            ax=ax,
        )
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax)
    fig.suptitle(
        "ETHICS Gender: Status Respons per Model dan Persona",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "02_ethics_status_by_model_persona_heatmap")


def plot_commit_rate_by_subset(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(32, 20), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER, strict=True):
        frame = summary[summary["subset"].eq(subset)]
        matrix = _matrix(frame, "commit_rate") * 100
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": COMMIT_RATE_ANNOTATION_SIZE, "fontweight": "normal"},
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Commit rate (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset], fontsize=COMMIT_RATE_PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax)
        ax.tick_params(axis="x", labelsize=COMMIT_RATE_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=COMMIT_RATE_TICK_SIZE)
        for text in ax.texts:
            text.set_fontsize(COMMIT_RATE_ANNOTATION_SIZE)
        if ax.collections and ax.collections[0].colorbar is not None:
            colorbar = ax.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=COMMIT_RATE_COLORBAR_TICK_SIZE)
            colorbar.ax.yaxis.label.set_size(COMMIT_RATE_COLORBAR_LABEL_SIZE)
    fig.suptitle(
        "ETHICS Gender: Commit Rate per Subset",
        fontsize=24,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "03_ethics_commit_rate_by_persona_subset_heatmap")


def _reason_heatmap_matrix(
    frame: pd.DataFrame,
    model: str,
    value: str,
    reason_order: list[str],
) -> pd.DataFrame:
    subset = frame[frame["model"].eq(model)].copy()
    matrix = subset.pivot_table(
        index="reason_label",
        columns="persona",
        values=value,
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.reindex(index=reason_order, columns=PERSONA_ORDER, fill_value=0)
    matrix.columns = [PERSONA_LABELS[item] for item in PERSONA_ORDER]
    return matrix * 100


def plot_reason_heatmaps(
    reason_model: pd.DataFrame,
    figure_dir: Path,
    *,
    value: str,
    stem: str,
    title: str,
    colorbar_label: str,
) -> list[Path]:
    reason_order = (
        reason_model.groupby("reason_label", as_index=False)["reason_count"]
        .sum()
        .sort_values(["reason_count", "reason_label"], ascending=[False, True])["reason_label"]
        .tolist()
    )
    fig, axes = plt.subplots(3, 2, figsize=(30, 22), constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER, strict=True):
        matrix = _reason_heatmap_matrix(reason_model, model, value, reason_order)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 11, "fontweight": "normal"},
            cmap="YlOrBr",
            vmin=0,
            vmax=max(1.0, float(np.nanmax(matrix.to_numpy()))),
            linewidths=0.35,
            cbar_kws={"label": colorbar_label},
            ax=ax,
        )
        ax.set_title(MODEL_LABELS[model], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax, rotation=48)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")
    return save_figure(fig, figure_dir, stem)


def plot_valid_accuracy(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(29, 9), constrained_layout=True)
    for ax, subset in zip(axes, VALID_SUBSETS, strict=True):
        frame = summary[summary["subset"].eq(subset)]
        matrix = _matrix(frame, "accuracy_committed") * 100
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 12, "fontweight": "normal"},
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Accuracy committed (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax)
    fig.suptitle(
        "ETHICS Gender: Akurasi per Subset",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "06_ethics_accuracy_committed_valid_subsets_heatmap")


def plot_effective_success(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(29, 9), constrained_layout=True)
    for ax, subset in zip(axes, VALID_SUBSETS, strict=True):
        frame = summary[summary["subset"].eq(subset)]
        matrix = _matrix(frame, "effective_success_rate_nonmissing") * 100
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 12, "fontweight": "normal"},
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Effective success (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax)
    fig.suptitle(
        "ETHICS Gender: Keberhasilan Efektif per Subset",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "07_ethics_effective_success_valid_subsets_heatmap")


def plot_stance_wrong(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(29, 9), constrained_layout=True)
    for ax, subset in zip(axes, INVALID_STANCE_SUBSETS, strict=True):
        frame = summary[summary["subset"].eq(subset)]
        matrix = _matrix(frame, "pct_answer_wrong") * 100
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 12, "fontweight": "normal"},
            cmap="RdYlBu_r",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Stance wrong (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        _style_heatmap(ax)
    fig.suptitle(
        "ETHICS Gender: Distribusi Keputusan Deontology dan Virtue",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "08_ethics_stance_wrong_construct_invalid_heatmap")


def plot_status_independence(independence: pd.DataFrame, figure_dir: Path) -> list[Path]:
    data = independence[independence["scope"].eq("gender_plus_neutral")].copy()
    data = _model_ordered(data)
    fig, ax = plt.subplots(figsize=(17, 8.5))
    bars = ax.bar(
        np.arange(len(data)),
        data["cramer_v"],
        color="#5B21B6",
        edgecolor="white",
    )
    for bar, value in zip(bars, data["cramer_v"], strict=True):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        fontsize=14,
            fontweight="normal",
        )
    ax.set_xticks(
        np.arange(len(data)),
        [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER],
    )
    ax.set_ylabel("Cramer's V")
    ax.set_ylim(0, max(0.25, float(data["cramer_v"].max()) * 1.25))
    ax.set_title(
        "ETHICS Gender: Asosiasi Persona dan Status Respons",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=18,
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, figure_dir, "09_ethics_status_persona_cramers_v")


def _effect_annot_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["persona_a"] = pd.Categorical(
        data["persona_a"], categories=PERSONA_COMPARISON_ORDER, ordered=True
    )
    data = data.sort_values(["model", "persona_a"], kind="stable")
    data["annotation"] = data.apply(
        lambda row: (
            "" if pd.isna(row["risk_difference_pp"])
            else f"{row['risk_difference_pp']:.1f}"
            + ("*" if bool(row["decision_significant_0_05"]) else "")
        ),
        axis=1,
    )
    return data.pivot(index="model_label", columns="persona_a", values="annotation").reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=PERSONA_COMPARISON_ORDER,
    ).rename(columns=PERSONA_LABELS)


def _effect_value_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot(index="model_label", columns="persona_a", values="risk_difference_pp").reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=PERSONA_COMPARISON_ORDER,
    ).rename(columns=PERSONA_LABELS)


def plot_primary_effects(paired: pd.DataFrame, figure_dir: Path) -> list[Path]:
    specs = [
        ("coverage", "Commit Rate", "10_ethics_coverage_effect_holm_heatmap"),
        (
            "conditional_accuracy",
            "Akurasi",
            "11_ethics_conditional_accuracy_effect_holm_heatmap",
        ),
        (
            "effective_success",
            "Keberhasilan efektif",
            "12_ethics_effective_success_effect_holm_heatmap",
        ),
        (
            "stance_wrong",
            "Distribusi keputusan",
            "13_ethics_stance_wrong_effect_holm_heatmap",
        ),
    ]
    output_paths: list[Path] = []
    for outcome, outcome_label, stem in specs:
        frame = paired[paired["outcome"].eq(outcome)].copy()
        if frame.empty:
            continue
        subsets = [subset for subset in SUBSET_ORDER if subset in set(frame["subset"])]
        if not subsets:
            continue
        nrows = 2 if len(subsets) > 2 else 1
        ncols = 2 if len(subsets) > 1 else 1
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(15 * ncols, 8.5 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flat
        absolute_max = float(np.nanmax(np.abs(frame["risk_difference_pp"])))
        limit = max(1.0, absolute_max)
        for ax, subset in zip(axes_flat, subsets, strict=True):
            subset_frame = frame[frame["subset"].eq(subset)]
            values = _effect_value_matrix(subset_frame)
            annotations = _effect_annot_matrix(subset_frame)
            sns.heatmap(
                values,
                annot=annotations,
                fmt="",
                cmap="RdBu_r",
                center=0,
                vmin=-limit,
                vmax=limit,
                linewidths=0.5,
                cbar_kws={"label": "Risk difference (percentage points)"},
                annot_kws={"fontsize": 14, "fontweight": "normal"},
                ax=ax,
            )
            ax.set_title(SUBSET_LABELS[subset], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            _style_heatmap(ax)
        for ax in list(axes_flat)[len(subsets) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"ETHICS Gender: {outcome_label}",
            fontsize=TITLE_SIZE,
            fontweight="bold",
        )
        output_paths.extend(save_figure(fig, figure_dir, stem))
    return output_paths


def render(
    source_bundle: Path = SOURCE_BUNDLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    source_bundle = source_bundle.resolve()
    output_dir = output_dir.resolve()
    tables = _load_source_tables(source_bundle)
    derived = derive_tables(tables)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = write_derived_tables(derived, table_dir)

    configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths += plot_status_composition(tables["model_overview"], figure_dir)
    figure_paths += plot_status_heatmaps(derived["status"], figure_dir)
    figure_paths += plot_commit_rate_by_subset(derived["summary"], figure_dir)
    figure_paths += plot_reason_heatmaps(
        derived["reason_model"],
        figure_dir,
        value="pct_of_abstain",
        stem="04_ethics_abstain_reason_share_by_model_persona_heatmap",
        title="ETHICS Gender: Alasan Abstain",
        colorbar_label="% dari abstain",
    )
    figure_paths += plot_reason_heatmaps(
        derived["reason_model"],
        figure_dir,
        value="pct_of_nonmissing",
        stem="05_ethics_abstain_reason_prevalence_by_model_persona_heatmap",
        title="ETHICS Gender: Prevalensi Abstain",
        colorbar_label="% dari non-missing",
    )
    figure_paths += plot_valid_accuracy(derived["summary"], figure_dir)
    figure_paths += plot_effective_success(derived["summary"], figure_dir)
    figure_paths += plot_stance_wrong(derived["summary"], figure_dir)
    figure_paths += plot_status_independence(tables["status_independence"], figure_dir)
    figure_paths += plot_primary_effects(derived["paired_primary"], figure_dir)

    output_index_rows = []
    for path in sorted(table_paths.values()):
        output_index_rows.append(
            {"kind": "table", "file": path.name, "path": str(path), "sha256": sha256_file(path)}
        )
    for path in sorted(figure_paths):
        output_index_rows.append(
            {"kind": "figure", "file": path.name, "path": str(path), "sha256": sha256_file(path)}
        )
    output_index = pd.DataFrame(output_index_rows)
    output_index.to_csv(output_dir / "output_index.csv", index=False)

    source_manifest_path = source_bundle / "metadata" / "analysis_manifest.json"
    manifest = {
        "bundle_version": "ethics-gender-accuracy-standardized-v1",
        "created_from": str(source_bundle),
        "source_analysis_version": tables["manifest"]["version"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_input_sha256": tables["manifest"]["input_sha256"],
        "source_row_count": int(tables["manifest"]["row_count"]),
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "construct_valid_accuracy_subsets": VALID_SUBSETS,
        "construct_invalid_stance_subsets": INVALID_STANCE_SUBSETS,
        "status_counts": {
            status: int(tables["condition_summary"][status].sum())
            for status in ("committed", "abstain", "missing")
        },
        "table_files": [path.name for path in sorted(table_paths.values())],
        "figure_files": [path.name for path in sorted(figure_paths)],
        "figure_count": len(figure_paths),
        "source_bundle_read_only": True,
        "api_calls": 0,
        "rescore": False,
        "notes": [
            "accuracy_committed hanya digunakan pada commonsense dan justice",
            "deontology dan virtue dilaporkan sebagai distribusi stance wrong",
            "missing adalah kegagalan infrastruktur dan tidak dipulihkan secara linguistik",
            "abstain adalah hasil parser dan penyebabnya dilaporkan dengan dua denominator",
        ],
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[CREATED] {output_dir}")
    print(f"[TABLES] {len(table_paths)}")
    print(f"[FIGURES] {len(figure_paths)} files ({len(figure_paths) // 2} PNG + {len(figure_paths) // 2} SVG)")
    return {
        "output_dir": output_dir,
        "tables": table_paths,
        "figures": figure_paths,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=SOURCE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render(source_bundle=args.source_bundle, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

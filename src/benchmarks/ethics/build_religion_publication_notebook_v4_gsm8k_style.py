"""Build a new GSM8K-style ETHICS Religion publication notebook.

This module is a presentation layer over the validated ETHICS Religion v3
bundle. It reads inferential values from v3, reads status/token metadata from
the CSV referenced by the v3 manifest, and never calls an API, reruns an LLM,
rescored responses, or recomputes an inferential test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[3]
BASE_BUILDER_PATH = ROOT / "src" / "benchmarks" / "ethics" / "build_religion_publication_notebook_v4_accuracy_persona_first.py"


def load_base_builder():
    spec = importlib.util.spec_from_file_location("ethics_religion_v4_base", BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Builder dasar tidak dapat dimuat: {BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_builder()

ANALYSIS_VERSION = "ethics-religion-v4-gsm8k-style-20260905"
EXPECTED_V3_VERSION = BASE.EXPECTED_V3_VERSION
MODEL_ORDER = BASE.MODEL_ORDER
MODEL_LABELS = BASE.MODEL_LABELS
SUBSET_ORDER = BASE.SUBSET_ORDER
SUBSET_LABELS = BASE.SUBSET_LABELS
EXPECTED_SUBSET_QUESTION_COUNTS = BASE.EXPECTED_SUBSET_QUESTION_COUNTS
IDENTITY_ORDER = BASE.IDENTITY_ORDER
IDENTITY_LABELS = BASE.IDENTITY_LABELS
CONDITION_ORDER = BASE.CONDITION_ORDER
PAIR_ORDER = BASE.PAIR_ORDER
TOKEN_COLUMNS = BASE.TOKEN_COLUMNS
STATUS_COLUMNS = BASE.STATUS_COLUMNS
EFFECT_CMAP = BASE.EFFECT_CMAP

IDENTITY_COLORS = {
    "islam": "#0072B2",
    "christian": "#D55E00",
    "jewish": "#CC79A7",
    "atheist": "#009E73",
}
IDENTITY_LIGHT_COLORS = {
    "islam": "#56B4E9",
    "christian": "#E69F00",
    "jewish": "#F0A6CA",
    "atheist": "#66C2A5",
}


def repo_root() -> Path:
    return ROOT


def v3_root(root: Path) -> Path:
    return BASE.v3_root(root)


def default_output_dir(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Religion_Persona"
        / "_combined"
        / "persona_publication_analysis_v4_gsm8k_style_six_models"
    )


def source_notebook_path(root: Path) -> Path:
    return (
        root
        / "notebooks"
        / "religion"
        / "ETHICS_Religion_Persona_Publication_Analysis_v4_GSM8K_Style.ipynb"
    )


def executed_notebook_path(root: Path) -> Path:
    return root / "result" / "ETHICS" / "ETHICS_Religion_Persona_Publication_Analysis_v4_GSM8K_Style.ipynb"


def condition_label(persona: str) -> str:
    if persona == "neutral":
        return "N"
    identity, scenario = BASE.condition_parts(persona)
    return f"{IDENTITY_LABELS[identity]} {scenario}"


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "normal",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=240)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def annotation_matrix(values: pd.DataFrame, significant: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = ""
            else:
                star = "*" if bool(significant.loc[row, column]) else ""
                annotations.loc[row, column] = f"{value:+.{decimals}f}{star}"
    return annotations


def emphasize_significant_annotations(ax: plt.Axes) -> None:
    """Bold only annotations that carry the significance star."""
    for annotation in ax.texts:
        annotation.set_fontweight("bold" if annotation.get_text().endswith("*") else "normal")


def load_inputs(root: Path) -> dict[str, object]:
    """Load v3 inferential tables and descriptive fields from the manifest CSV."""
    source = BASE.load_v3_tables(root)
    descriptive_result = BASE.aggregate_source_csv(source["source_csv"])
    descriptive = descriptive_result["data"]
    accuracy = BASE.add_accuracy_status_metrics(source["accuracy"], descriptive)

    status_columns = [
        "model",
        "model_label",
        "subset",
        "subset_label",
        "identity",
        "identity_label",
        "persona",
        "persona_label",
        "scenario",
        "total",
        "n_correct_full",
        "n_incorrect_full",
        "accuracy_full",
        "accuracy_full_pct",
        "committed",
        "commit_rate",
        "commit_rate_pct",
        "abstain",
        "abstain_rate",
        "abstain_rate_pct",
        "missing",
        "missing_rate",
        "missing_rate_pct",
        "n_correct_committed",
        "accuracy_committed",
        "accuracy_committed_pct",
        "construct_valid",
        "interpretation_note",
    ]
    status = accuracy[status_columns].copy()
    status_model = BASE.aggregate_status(status, ["model", "model_label"])
    status_subset = BASE.aggregate_status(status, ["subset", "subset_label"])

    token_columns = [
        "model",
        "subset",
        "identity",
        "persona",
        "scenario",
        "total",
        *[f"sum_{column}" for column in TOKEN_COLUMNS],
        *[f"n_valid_{column}" for column in TOKEN_COLUMNS],
        *[f"avg_{column}" for column in TOKEN_COLUMNS],
    ]
    tokens = BASE.add_token_labels(descriptive[token_columns])
    tokens = tokens[
        [
            "model",
            "model_label",
            "subset",
            "subset_label",
            "identity",
            "identity_label",
            "persona",
            "persona_label",
            "scenario",
            "total",
            *[f"n_valid_{column}" for column in TOKEN_COLUMNS],
            *[f"sum_{column}" for column in TOKEN_COLUMNS],
            *[f"avg_{column}" for column in TOKEN_COLUMNS],
        ]
    ]
    token_model_condition = BASE.aggregate_tokens(
        tokens,
        ["model", "model_label", "identity", "identity_label", "persona", "persona_label", "scenario"],
    )
    token_subset = BASE.aggregate_tokens(tokens, ["subset", "subset_label"])

    cochran_key = source["cochran"][["model", "subset", "identity", "significant_0_05"]].rename(
        columns={"significant_0_05": "cochran_q_significant_0_05"}
    )
    scenario_with_q = source["scenario"].merge(
        cochran_key,
        on=["model", "subset", "identity"],
        how="left",
        validate="many_to_one",
    )
    if scenario_with_q["cochran_q_significant_0_05"].isna().any():
        raise ValueError("Sebagian perbandingan skenario tidak memiliki hasil Q Cochran")
    scenario_with_q["supported_q_and_holm"] = (
        scenario_with_q["cochran_q_significant_0_05"] & scenario_with_q["significant_holm_0_05"]
    )
    scenario = BASE.canonical_scenario_comparisons(scenario_with_q)
    pairwise = BASE.canonical_pairwise_comparisons(source["pairwise"])

    return {
        "source": source,
        "descriptive_result": descriptive_result,
        "accuracy": accuracy,
        "status": status,
        "status_model": status_model,
        "status_subset": status_subset,
        "tokens": tokens,
        "token_model_condition": token_model_condition,
        "token_subset": token_subset,
        "scenario": scenario,
        "pairwise": pairwise,
        "persona_summary": BASE.persona_summary(pairwise),
        "persona_summary_by_model": BASE.persona_summary_by_model(pairwise),
        "scenario_summary": BASE.scenario_summary(scenario),
        "scenario_summary_by_model": BASE.scenario_summary_by_model(scenario),
    }


def gsm8k_style_accuracy_frame(accuracy: pd.DataFrame, subset: str) -> pd.DataFrame:
    selected = accuracy[accuracy["subset"].eq(subset)]
    pivot = selected.pivot(index="model", columns="persona", values="accuracy_full_pct").reindex(MODEL_ORDER)
    result = pd.DataFrame(index=MODEL_ORDER)
    for identity in IDENTITY_ORDER:
        result[f"{identity}_explicit"] = pivot[f"{identity}_explicit"]
        result[f"{identity}_implicit_avg"] = pivot[[f"{identity}_implicit_{i}" for i in (1, 2, 3)]].mean(axis=1)
    result["neutral"] = pivot["neutral"]
    result.index = [MODEL_LABELS[model] for model in MODEL_ORDER]
    result.index.name = "model_label"
    return result.reset_index()


def gsm8k_style_delta_frame(accuracy: pd.DataFrame, subset: str) -> pd.DataFrame:
    absolute = gsm8k_style_accuracy_frame(accuracy, subset).set_index("model_label")
    result = pd.DataFrame(index=absolute.index)
    for identity in IDENTITY_ORDER:
        result[identity] = absolute[f"{identity}_implicit_avg"] - absolute[f"{identity}_explicit"]
    result.index.name = "model_label"
    return result.reset_index()


def grouped_accuracy_figure(accuracy: pd.DataFrame, subset: str, figure_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    frame = gsm8k_style_accuracy_frame(accuracy, subset)
    value_columns = []
    labels = []
    colors = []
    for identity in IDENTITY_ORDER:
        value_columns.extend([f"{identity}_explicit", f"{identity}_implicit_avg"])
        labels.extend([f"{IDENTITY_LABELS[identity]} Eksplisit", f"{IDENTITY_LABELS[identity]} Implisit (rerata)"])
        colors.extend([IDENTITY_COLORS[identity], IDENTITY_LIGHT_COLORS[identity]])
    value_columns.append("neutral")
    labels.append("Netral")
    colors.append("#333333")

    values = frame.set_index("model_label")[value_columns]
    x = np.arange(len(values))
    width = 0.84 / len(value_columns)
    fig, ax = plt.subplots(figsize=(16.5, 7.4), constrained_layout=True)
    for index, column in enumerate(value_columns):
        offset = (index - (len(value_columns) - 1) / 2) * width
        bars = ax.bar(x + offset, values[column].to_numpy(dtype=float), width, color=colors[index], label=labels[index])
        for bar, value in zip(bars, values[column].to_numpy(dtype=float)):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=6.6,
                rotation=90,
            )
    neutral_baseline = float(values["neutral"].mean())
    ax.axhline(
        neutral_baseline,
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Rerata Netral 6 Model ({neutral_baseline:.1f}%)",
        zorder=1,
    )
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=30, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Akurasi penuh (%)")
    ax.set_title(
        f"ETHICS {SUBSET_LABELS[subset]}: Perbandingan Akurasi Antar-Persona",
        fontsize=15,
        fontweight="normal",
        pad=18,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    baseline_index = next(index for index, label in enumerate(legend_labels) if label.startswith("Rerata Netral"))
    ordered_handles = [handles[baseline_index]] + [handle for index, handle in enumerate(handles) if index != baseline_index]
    ordered_labels = [legend_labels[baseline_index]] + [label for index, label in enumerate(legend_labels) if index != baseline_index]
    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Persona dan baseline",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        -0.02,
        "Implisit (rerata) = rerata I1, I2, dan I3; angka bar adalah akurasi penuh (%).",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, f"01_{subset}_absolute_accuracy_gsm8k_style"), frame


def grouped_delta_figure(accuracy: pd.DataFrame, subset: str, figure_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    frame = gsm8k_style_delta_frame(accuracy, subset)
    values = frame.set_index("model_label")[IDENTITY_ORDER]
    x = np.arange(len(values))
    width = 0.72 / len(IDENTITY_ORDER)
    fig, ax = plt.subplots(figsize=(15.5, 6.8), constrained_layout=True)
    for index, identity in enumerate(IDENTITY_ORDER):
        offset = (index - (len(IDENTITY_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values[identity].to_numpy(dtype=float),
            width,
            color=IDENTITY_COLORS[identity],
            label=IDENTITY_LABELS[identity],
        )
        for bar, value in zip(bars, values[identity].to_numpy(dtype=float)):
            ax.annotate(
                f"{value:+.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                fontsize=7.2,
            )
    limit = max(1.0, float(np.nanmax(np.abs(values.to_numpy()))) * 1.35)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-limit, limit)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=30, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Delta akurasi (poin persentase)")
    ax.set_title(
        f"ETHICS {SUBSET_LABELS[subset]}: Delta Persona Implisit terhadap Eksplisit",
        fontsize=15,
        fontweight="normal",
        pad=16,
    )
    ax.legend(title="Identitas persona", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.5,
        -0.02,
        "Delta = rerata (I1, I2, I3) - E. Metrik ini deskriptif; bintang tidak digunakan karena bukan uji baru.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, f"05_{subset}_delta_implicit_vs_explicit_gsm8k_style"), frame


def pairwise_effect_figure(pairwise: pd.DataFrame, subset: str, figure_dir: Path) -> list[Path]:
    selected = pairwise[pairwise["subset"].eq(subset)].copy()
    scenario_order = ["E", "I1", "I2", "I3"]
    global_limit = max(0.2, float(np.ceil(pairwise["cohens_h"].abs().max() * 10) / 10))
    columns = [f"{IDENTITY_LABELS[first]} - {IDENTITY_LABELS[second]}" for first, second in PAIR_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenario_order):
        selected_scenario = selected[selected["scenario_a"].eq(scenario)].copy()
        selected_scenario["column_label"] = selected_scenario.apply(
            lambda row: f"{IDENTITY_LABELS[row['persona_a']]} - {IDENTITY_LABELS[row['persona_b']]}",
            axis=1,
        )
        values = selected_scenario.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns
        )
        significant = selected_scenario.pivot(
            index="model_label", columns="column_label", values="significant_holm_0_05"
        ).reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-global_limit,
            vmax=global_limit,
            annot=annotation_matrix(values, significant),
            fmt="",
            linewidths=0.35,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": 8.0},
        )
        emphasize_significant_annotations(ax)
        ax.set_title(f"Skenario {scenario}", fontsize=13, fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=28, labelsize=8)
        ax.tick_params(axis="y", rotation=0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-global_limit, vmax=global_limit), cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
    colorbar.set_label("Cohen's h (A - B)")
    fig.suptitle(
        f"ETHICS {SUBSET_LABELS[subset]}: Effect Size Perbandingan Antarpersona",
        fontsize=15,
        fontweight="normal",
    )
    fig.text(
        0.5,
        -0.025,
        "Empat panel memisahkan E, I1, I2, dan I3; * = signifikan setelah koreksi Holm. Hanya anotasi bertanda * yang dicetak tebal.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, f"09_{subset}_pairwise_effect_size_gsm8k_style")


def scenario_effect_figure(scenario: pd.DataFrame, category: str, figure_dir: Path) -> list[Path]:
    selected_all = scenario[scenario["comparison_category"].eq(category)].copy()
    if category == "implicit_vs_explicit":
        scenario_pairs = [(scenario_name, "E") for scenario_name in ["I1", "I2", "I3"]]
        stem = "13_implicit_vs_explicit_q_holm"
        title = "ETHICS Agama: Perbandingan Skenario Implisit terhadap Eksplisit"
    else:
        scenario_pairs = [(scenario_name, "N") for scenario_name in ["E", "I1", "I2", "I3"]]
        stem = "14_persona_vs_neutral_q_holm"
        title = "ETHICS Agama: Perbandingan Persona terhadap Netral"
    columns = [
        f"{IDENTITY_LABELS[identity]} {first} - {second}"
        for identity in IDENTITY_ORDER
        for first, second in scenario_pairs
    ]
    selected_all["column_label"] = selected_all.apply(
        lambda row: f"{row['identity_label']} {row['scenario_a']} - {row['scenario_b']}", axis=1
    )
    limit = max(0.2, float(np.ceil(selected_all["cohens_h"].abs().max() * 10) / 10))
    fig, axes = plt.subplots(2, 2, figsize=(20, 10.5), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = selected_all[selected_all["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns
        )
        significant = selected.pivot(
            index="model_label", columns="column_label", values="supported_q_and_holm"
        ).reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=annotation_matrix(values, significant),
            fmt="",
            linewidths=0.35,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": 7.0},
        )
        emphasize_significant_annotations(ax)
        ax.set_title(SUBSET_LABELS[subset], fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-limit, vmax=limit), cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
    colorbar.set_label("Cohen's h")
    fig.suptitle(title, fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "* = didukung Q Cochran dan tetap signifikan setelah Holm; hanya anotasi bertanda * yang dicetak tebal.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, stem)


def status_figure(status_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = status_model.set_index("model_label").reindex([MODEL_LABELS[model] for model in MODEL_ORDER])
    percentages = frame[["commit_rate_pct", "abstain_rate_pct", "missing_rate_pct"]].copy()
    percentages.columns = ["Committed", "Abstain", "Missing"]
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    bottom = np.zeros(len(percentages))
    colors = {"Committed": "#4C956C", "Abstain": "#E9C46A", "Missing": "#D1495B"}
    for column in percentages.columns:
        values = percentages[column].to_numpy(dtype=float)
        ax.bar(percentages.index, values, bottom=bottom, label=column, color=colors[column])
        bottom += values
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Proporsi respons")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Status Respons ETHICS Agama Menurut Model", fontsize=15, fontweight="normal")
    ax.legend(title="Status", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return save_figure(fig, figure_dir, "15_response_status_by_model")


def token_figure(token_model_condition: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = token_model_condition.copy()
    frame["condition_label"] = frame["persona"].map(condition_label)
    condition_labels = [condition_label(persona) for persona in CONDITION_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(19, 8.8), constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        selected = frame[frame["model"].eq(model)].set_index("condition_label").reindex(condition_labels)
        ax.plot(condition_labels, selected["avg_total_tokens"], marker="o", color="#2A6F97", linewidth=1.8)
        ax.set_title(MODEL_LABELS[model], fontweight="normal")
        ax.set_ylabel("Rata-rata total token")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=50, labelsize=7)
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle("Rata-rata Total Token Menurut Kondisi", fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "Rata-rata hanya memakai metadata token yang valid; nilai yang hilang tidak diimputasi.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, "16_total_tokens_by_condition")


def validate_figure_files(figures: list[Path]) -> dict[str, object]:
    png_files = [path for path in figures if path.suffix.lower() == ".png"]
    svg_files = [path for path in figures if path.suffix.lower() == ".svg"]
    png_valid = all(path.exists() and path.stat().st_size > 100 and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in png_files)
    svg_valid = all(
        path.exists()
        and path.stat().st_size > 100
        and "<svg" in path.read_text(encoding="utf-8")[:1000]
        and "</svg>" in path.read_text(encoding="utf-8")
        for path in svg_files
    )
    return {
        "png_files": len(png_files),
        "svg_files": len(svg_files),
        "all_figure_files_nonempty": bool(png_valid and svg_valid and len(png_files) == len(svg_files)),
    }


def write_tables(output: Path, data: dict[str, object], style_accuracy: pd.DataFrame, style_delta: pd.DataFrame) -> None:
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    source = data["source"]
    BASE.write_csv(source["accuracy"], table_dir / "accuracy_by_subset_model_condition.csv")
    BASE.write_csv(source["cochran"], table_dir / "cochran_q_by_subset_model_religion.csv")
    BASE.write_csv(source["scenario"], table_dir / "mcnemar_scenario_comparisons.csv")
    BASE.write_csv(source["pairwise"], table_dir / "mcnemar_religion_pairwise_comparisons.csv")
    BASE.write_csv(data["accuracy"], table_dir / "accuracy_all_conditions.csv")
    BASE.write_csv(data["status"], table_dir / "response_status_by_condition.csv")
    BASE.write_csv(data["status_model"], table_dir / "response_status_by_model.csv")
    BASE.write_csv(data["status_subset"], table_dir / "response_status_by_subset.csv")
    BASE.write_csv(data["tokens"], table_dir / "token_summary_by_condition.csv")
    BASE.write_csv(data["token_model_condition"], table_dir / "token_summary_by_model_condition.csv")
    BASE.write_csv(data["token_subset"], table_dir / "token_summary_by_subset.csv")
    BASE.write_csv(data["scenario"], table_dir / "mcnemar_scenario_comparisons_normalized_with_q_gate.csv")
    BASE.write_csv(data["pairwise"], table_dir / "mcnemar_religion_pairwise_comparisons_normalized.csv")
    BASE.write_csv(data["persona_summary"], table_dir / "religion_persona_summary_by_model_subset_scenario.csv")
    BASE.write_csv(data["persona_summary_by_model"], table_dir / "religion_persona_summary_by_model.csv")
    BASE.write_csv(
        data["pairwise"][data["pairwise"]["significant_holm_0_05"]],
        table_dir / "religion_persona_significant_results.csv",
    )
    BASE.write_csv(data["scenario_summary"], table_dir / "scenario_summary_by_model_subset.csv")
    BASE.write_csv(data["scenario_summary_by_model"], table_dir / "scenario_summary_by_model.csv")
    BASE.write_csv(style_accuracy, table_dir / "religion_absolute_accuracy_gsm8k_style.csv")
    BASE.write_csv(style_delta, table_dir / "religion_delta_implicit_vs_explicit_gsm8k_style.csv")


def load_existing_output(output: Path, root: Path) -> dict[str, object]:
    metadata_dir = output / "metadata"
    manifest_path = metadata_dir / "analysis_manifest.json"
    validation_path = metadata_dir / "validation.json"
    if not manifest_path.exists() or not validation_path.exists():
        raise ValueError(f"Output GSM8K-style tidak lengkap: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != ANALYSIS_VERSION or validation.get("valid") is not True:
        raise ValueError(f"Output GSM8K-style yang ada tidak valid: {output}")
    source = BASE.load_v3_tables(root)
    if BASE.sha256(source["source_csv"]) != validation.get("source_sha256_after"):
        raise ValueError("CSV sumber v3 berubah sejak output GSM8K-style dibuat")
    if BASE.bundle_snapshot(source["base"]) != validation.get("source_bundle_snapshot_after"):
        raise ValueError("Bundle v3 berubah sejak output GSM8K-style dibuat")
    table_dir = output / "tables"
    return {
        "output_dir": output,
        "accuracy": pd.read_csv(table_dir / "accuracy_all_conditions.csv"),
        "accuracy_all_conditions": pd.read_csv(table_dir / "accuracy_all_conditions.csv"),
        "persona_summary": pd.read_csv(table_dir / "religion_persona_summary_by_model_subset_scenario.csv"),
        "persona_summary_by_model": pd.read_csv(table_dir / "religion_persona_summary_by_model.csv"),
        "scenario_summary": pd.read_csv(table_dir / "scenario_summary_by_model_subset.csv"),
        "scenario_summary_by_model": pd.read_csv(table_dir / "scenario_summary_by_model.csv"),
        "status": pd.read_csv(table_dir / "response_status_by_condition.csv"),
        "status_by_model": pd.read_csv(table_dir / "response_status_by_model.csv"),
        "status_by_subset": pd.read_csv(table_dir / "response_status_by_subset.csv"),
        "tokens": pd.read_csv(table_dir / "token_summary_by_condition.csv"),
        "token_summary": pd.read_csv(table_dir / "token_summary_by_condition.csv"),
        "token_summary_by_model_condition": pd.read_csv(table_dir / "token_summary_by_model_condition.csv"),
        "validation": validation,
        "manifest": manifest,
        "output_index": pd.read_csv(output / "output_index.csv"),
    }


def refresh_figures(output: Path) -> None:
    """Regenerate presentation figures in an already-created style bundle."""
    root = repo_root()
    existing = load_existing_output(output, root)
    data = load_inputs(root)
    figure_dir = output / "figures"
    configure_plot_style()
    figures: list[Path] = []
    for subset in SUBSET_ORDER:
        figures.extend(grouped_accuracy_figure(data["accuracy"], subset, figure_dir)[0])
        figures.extend(grouped_delta_figure(data["accuracy"], subset, figure_dir)[0])
        figures.extend(pairwise_effect_figure(data["pairwise"], subset, figure_dir))
    figures.extend(scenario_effect_figure(data["scenario"], "implicit_vs_explicit", figure_dir))
    figures.extend(scenario_effect_figure(data["scenario"], "persona_vs_neutral", figure_dir))
    figures.extend(status_figure(data["status_model"], figure_dir))
    figures.extend(token_figure(data["token_model_condition"], figure_dir))

    metadata_dir = output / "metadata"
    validation_path = metadata_dir / "validation.json"
    manifest_path = metadata_dir / "analysis_manifest.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    figure_validation = validate_figure_files(figures)
    validation.update(figure_validation)
    validation.setdefault("checks", {})["figure_files_valid"] = bool(
        figure_validation["all_figure_files_nonempty"]
    )
    BASE.write_json(validation, validation_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"] = validation
    BASE.write_json(manifest, manifest_path)
    output_index = BASE.artifact_index(output)
    BASE.write_csv(output_index, output / "output_index.csv")
    if not validation.get("valid"):
        raise ValueError(f"Validation setelah refresh visual gagal: {validation}")


def run_analysis(output_dir: Path | str | None = None) -> dict[str, object]:
    root = repo_root()
    output = Path(output_dir).resolve() if output_dir else default_output_dir(root)
    source_bundle = v3_root(root).resolve()
    if output == source_bundle or source_bundle in output.parents:
        raise ValueError("Output GSM8K-style tidak boleh berada di dalam atau menimpa bundle v3")
    if output.exists() and any(output.iterdir()):
        return load_existing_output(output, root)

    data = load_inputs(root)
    source = data["source"]
    descriptive_result = data["descriptive_result"]
    source_snapshot_before = BASE.bundle_snapshot(source["base"])
    source_sha_before = source["source_sha_before"]

    table_dir = output / "tables"
    figure_dir = output / "figures"
    metadata_dir = output / "metadata"
    for directory in (table_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    figures: list[Path] = []
    style_accuracy_parts: list[pd.DataFrame] = []
    style_delta_parts: list[pd.DataFrame] = []
    for subset in SUBSET_ORDER:
        accuracy_paths, absolute = grouped_accuracy_figure(data["accuracy"], subset, figure_dir)
        delta_paths, delta = grouped_delta_figure(data["accuracy"], subset, figure_dir)
        figures.extend(accuracy_paths)
        figures.extend(delta_paths)
        absolute.insert(0, "subset", subset)
        delta.insert(0, "subset", subset)
        style_accuracy_parts.append(absolute)
        style_delta_parts.append(delta)
        figures.extend(pairwise_effect_figure(data["pairwise"], subset, figure_dir))

    figures.extend(scenario_effect_figure(data["scenario"], "implicit_vs_explicit", figure_dir))
    figures.extend(scenario_effect_figure(data["scenario"], "persona_vs_neutral", figure_dir))
    figures.extend(status_figure(data["status_model"], figure_dir))
    figures.extend(token_figure(data["token_model_condition"], figure_dir))
    style_accuracy = pd.concat(style_accuracy_parts, ignore_index=True)
    style_delta = pd.concat(style_delta_parts, ignore_index=True)
    write_tables(output, data, style_accuracy, style_delta)

    source_sha_after = BASE.sha256(source["source_csv"])
    source_snapshot_after = BASE.bundle_snapshot(source["base"])
    subset_question_counts = {
        subset: len(descriptive_result["question_ids"].get(subset, set())) for subset in SUBSET_ORDER
    }
    scenario_family_sizes = sorted(source["scenario"]["holm_family_size"].astype(int).unique().tolist())
    pairwise_family_sizes = sorted(source["pairwise"]["holm_family_size"].astype(int).unique().tolist())
    figure_validation = validate_figure_files(figures)
    checks = {
        "rows_processed_exact": descriptive_result["row_count"] == 154_428,
        "question_count_exact": len(set().union(*descriptive_result["question_ids"].values())) == 1_514,
        "model_count_exact": data["accuracy"]["model"].nunique() == 6,
        "condition_count_exact": data["accuracy"]["persona"].nunique() == 17,
        "subset_question_counts_exact": subset_question_counts == EXPECTED_SUBSET_QUESTION_COUNTS,
        "cochran_q_rows_exact": len(source["cochran"]) == 96,
        "scenario_pair_rows_exact": len(source["scenario"]) == 672,
        "religion_pairwise_rows_exact": len(source["pairwise"]) == 576,
        "scenario_holm_family_size_exactly_7": scenario_family_sizes == [7],
        "religion_pairwise_holm_family_size_exactly_24": pairwise_family_sizes == [24],
        "duplicate_response_keys_zero": descriptive_result["duplicate_response_keys"] == 0,
        "source_v3_validation_valid": source["validation"].get("valid") is True,
        "source_sha256_unchanged": BASE.source_sha_before_equals_expected(
            source_sha_before, source_sha_after, source["validation"].get("input_sha256_before")
        ),
        "source_v3_bundle_unchanged": source_snapshot_before == source_snapshot_after,
        "source_v3_not_overwritten": output != source["base"],
        "style_accuracy_rows_exact": len(style_accuracy) == 24,
        "style_delta_rows_exact": len(style_delta) == 24,
        "figure_files_valid": bool(figure_validation["all_figure_files_nonempty"]),
    }
    validation = {
        "valid": bool(all(checks.values())),
        "analysis_version": ANALYSIS_VERSION,
        "source_analysis_version": source["manifest"]["analysis_version"],
        "rows_processed": descriptive_result["row_count"],
        "question_count": len(set().union(*descriptive_result["question_ids"].values())),
        "model_count": int(data["accuracy"]["model"].nunique()),
        "condition_count": int(data["accuracy"]["persona"].nunique()),
        "models": MODEL_ORDER,
        "conditions": CONDITION_ORDER,
        "subset_question_counts": subset_question_counts,
        "cochran_q_rows": int(len(source["cochran"])),
        "scenario_pair_rows": int(len(source["scenario"])),
        "religion_pairwise_rows": int(len(source["pairwise"])),
        "scenario_holm_family_sizes": scenario_family_sizes,
        "religion_pairwise_holm_family_sizes": pairwise_family_sizes,
        "duplicate_response_keys": int(descriptive_result["duplicate_response_keys"]),
        "status_values": descriptive_result["status_values"],
        "source_v3_validation_valid": bool(source["validation"].get("valid")),
        "source_sha256_expected_v3": source["validation"].get("input_sha256_before"),
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "source_bundle_snapshot_before": source_snapshot_before,
        "source_bundle_snapshot_after": source_snapshot_after,
        "source_csv_unchanged": BASE.source_sha_before_equals_expected(
            source_sha_before, source_sha_after, source["validation"].get("input_sha256_before")
        ),
        "source_v3_bundle_unchanged": source_snapshot_before == source_snapshot_after,
        "source_v3_not_overwritten": output != source["base"],
        "full_denominator_accuracy_from_v3": True,
        "abstain_missing_counted_in_denominator_as_incorrect": bool(
            source["manifest"].get("nonanswer_scoring") == "abstain dan missing bernilai salah"
        ),
        "committed_accuracy_descriptive_only": True,
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "api_calls": 0,
        "combined_four_subset_accuracy_used": False,
        "token_valid_counts": {
            column: int(data["descriptive_result"]["data"][f"n_valid_{column}"].sum()) for column in TOKEN_COLUMNS
        },
        "style_accuracy_rows": int(len(style_accuracy)),
        "style_delta_rows": int(len(style_delta)),
        **figure_validation,
        "checks": checks,
    }
    if not validation["valid"]:
        BASE.write_json(validation, metadata_dir / "validation.json")
        raise ValueError(f"Validation GSM8K-style gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "source_bundle": str(source["base"]),
        "source_manifest": str(source["manifest_path"]),
        "source_validation": str(source["validation_path"]),
        "source_csv": str(source["source_csv"]),
        "source_analysis_version": source["manifest"]["analysis_version"],
        "output": str(output),
        "presentation_style": "GSM8K-style grouped bars plus per-subset effect heatmaps",
        "inference_source_tables": [
            "mcnemar_religion_pairwise_comparisons.csv",
            "mcnemar_scenario_comparisons.csv",
            "cochran_q_by_subset_model_religion.csv",
            "accuracy_by_subset_model_condition.csv",
        ],
        "descriptive_source_csv": str(source["source_csv"]),
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "api_calls": 0,
        "combined_four_subset_accuracy_used": False,
        "subsets_retained_separately": SUBSET_ORDER,
        "identity_order_for_figures": IDENTITY_ORDER,
        "condition_order_for_figures": CONDITION_ORDER,
        "bold_and_stars_rule": "Only significant heatmap cells carry a star and bold annotation; bar charts have no significance marker.",
        "validation": validation,
    }
    BASE.write_json(manifest, metadata_dir / "analysis_manifest.json")
    BASE.write_json(validation, metadata_dir / "validation.json")
    output_index = BASE.artifact_index(output)
    BASE.write_csv(output_index, output / "output_index.csv")
    return {
        "output_dir": output,
        "accuracy": data["accuracy"],
        "accuracy_all_conditions": data["accuracy"],
        "persona_summary": data["persona_summary"],
        "persona_summary_by_model": data["persona_summary_by_model"],
        "scenario_summary": data["scenario_summary"],
        "scenario_summary_by_model": data["scenario_summary_by_model"],
        "status": data["status"],
        "status_by_model": data["status_model"],
        "status_by_subset": data["status_subset"],
        "tokens": data["tokens"],
        "token_summary": data["tokens"],
        "token_summary_by_model_condition": data["token_model_condition"],
        "validation": validation,
        "manifest": manifest,
        "output_index": output_index,
    }


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(path: Path) -> Path:
    """Write a new source notebook under notebooks/religion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
# ETHICS Religion Persona - Analisis Akurasi v4 dengan Gaya Visual GSM8K

Notebook baru ini memakai bentuk visual yang sama dengan laporan GSM8K:
grouped bar chart untuk akurasi dan delta, lalu heatmap effect size. Setiap
subset ETHICS tetap ditampilkan terpisah. Nilai inferensial dibaca dari bundle
v3 tervalidasi; status respons dan token hanya merupakan ringkasan deskriptif
dari CSV yang dirujuk oleh manifest v3.
"""
        ),
        md("## 0. Provenance dan validation gate"),
        code(
            """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display

start = Path.cwd().resolve()
REPO_ROOT = next(path for path in [start, *start.parents] if (path / "AGENTS.md").exists() and (path / "src").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.ethics.build_religion_publication_notebook_v4_gsm8k_style import run_analysis

result = run_analysis()
OUTPUT_DIR = Path(result["output_dir"])
FIGURE_DIR = OUTPUT_DIR / "figures"
display(pd.DataFrame([result["validation"]]).T.rename(columns={0: "value"}))
assert result["validation"]["valid"]
assert result["validation"]["api_calls"] == 0
assert result["validation"]["statistics_recomputed"] is False
"""
        ),
        md(
            """
## 1. Akurasi penuh per kondisi

Tabel ini tetap memuat semua 408 kombinasi subset, model, dan kondisi: N, E,
I1, I2, dan I3. Kolom `accuracy_committed` selalu dibaca bersama `commit_rate`,
`abstain`, dan `missing`.
"""
        ),
        code(
            """
pd.set_option("display.max_rows", 1000)
display(result["accuracy_all_conditions"])
"""
        ),
        md(
            """
## 2. Visualisasi akurasi bergaya GSM8K

Setiap gambar memakai satu subset, model pada sumbu X, dan sembilan bar:
empat identitas agama masing-masing dengan kondisi eksplisit dan rerata
implisit, ditambah Netral. Rerata implisit hanya untuk visualisasi deskriptif;
hasil uji I1-E, I2-E, dan I3-E tetap dilaporkan terpisah.
"""
        ),
        code(
            """
for subset in ["commonsense", "deontology", "justice", "virtue"]:
    display(Image(filename=str(FIGURE_DIR / f"01_{subset}_absolute_accuracy_gsm8k_style.png"), width=1500))
"""
        ),
        md("## 3. Delta implisit terhadap eksplisit"),
        code(
            """
display(pd.read_csv(OUTPUT_DIR / "tables" / "religion_delta_implicit_vs_explicit_gsm8k_style.csv"))
for subset in ["commonsense", "deontology", "justice", "virtue"]:
    display(Image(filename=str(FIGURE_DIR / f"05_{subset}_delta_implicit_vs_explicit_gsm8k_style.png"), width=1500))
"""
        ),
        md(
            """
## 4. Perbandingan antarpersona agama

Tabel dan heatmap berikut memakai perbandingan berpasangan dari v3. Orientasi
grafik konsisten: Islam, Kristen, Yahudi, lalu Ateis. Tanda `*` hanya muncul
pada sel yang signifikan setelah Holm; hanya anotasi bertanda `*` yang dicetak
tebal.
"""
        ),
        code(
            """
display(result["persona_summary_by_model"])
display(result["persona_summary"])
for subset in ["commonsense", "deontology", "justice", "virtue"]:
    display(Image(filename=str(FIGURE_DIR / f"09_{subset}_pairwise_effect_size_gsm8k_style.png"), width=1700))
"""
        ),
        md("## 5. Perbandingan I1-E, I2-E, dan I3-E"),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'implicit_vs_explicit'"))
display(Image(filename=str(FIGURE_DIR / "13_implicit_vs_explicit_q_holm.png"), width=1600))
"""
        ),
        md("## 6. Perbandingan E-N, I1-N, I2-N, dan I3-N"),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'persona_vs_neutral'"))
display(Image(filename=str(FIGURE_DIR / "14_persona_vs_neutral_q_holm.png"), width=1600))
"""
        ),
        md(
            """
## 7. Status respons dan token

`abstain` dan `missing` tetap masuk denominator akurasi penuh. Token yang
hilang tidak diimputasi; jumlah data valid ditampilkan di tabel token.
"""
        ),
        code(
            """
display(result["status"])
display(result["status_by_model"])
display(result["status_by_subset"])
display(Image(filename=str(FIGURE_DIR / "15_response_status_by_model.png"), width=1400))

display(result["tokens"])
display(result["token_summary_by_model_condition"])
display(Image(filename=str(FIGURE_DIR / "16_total_tokens_by_condition.png"), width=1600))
"""
        ),
        md(
            """
## 8. Batas interpretasi dan indeks artefak

Untuk Deontology dan Virtue, angka akurasi dibaca sebagai kecocokan label per
butir, bukan ukuran kemampuan moral absolut. Keempat subset tidak digabung
menjadi satu hasil inferensial. `output_index.csv` mencatat tabel, metadata,
serta semua gambar PNG dan SVG yang dihasilkan.
"""
        ),
        code(
            """
display(result["output_index"])
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    nbf.write(notebook, path)
    return path


def execute_notebook(source: Path, output: Path) -> Path:
    notebook = nbf.read(source, as_version=4)
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(repo_root())}})
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the newly generated notebook")
    parser.add_argument("--refresh-figures", action="store_true", help="Refresh figures in an existing style bundle")
    args = parser.parse_args()
    root = repo_root()
    output_dir = default_output_dir(root)
    source = build_notebook(source_notebook_path(root))
    if args.refresh_figures:
        refresh_figures(output_dir)
    result = run_analysis(output_dir)
    print(f"[ANALYSIS] Output: {result['output_dir']}")
    print(f"[VALIDATION] valid={result['validation']['valid']}")
    print(f"[NOTEBOOK] Source: {source}")
    if args.execute:
        executed = execute_notebook(source, executed_notebook_path(root))
        print(f"[NOTEBOOK] Executed: {executed}")


if __name__ == "__main__":
    main()

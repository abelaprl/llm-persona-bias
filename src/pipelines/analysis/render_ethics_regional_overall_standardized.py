"""Render the combined ETHICS Regional-Cultural accuracy presentation bundle.

This is a presentation layer over the validated Regional v3 result bundle.  It
does not reparse responses, rescore answers, call an API, or modify anything
under ``result/``.  The parent bundle follows the ETHICS Gender presentation:
status is pooled over all four subsets, while the parent accuracy chart uses
only construct-valid ``commonsense`` and ``justice``.  The four subset details
are written below ``per_subset/``.
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

from ethics_accuracy_combined_statistics import compute_combined_conditional_accuracy
from ethics_accuracy_statistical_figures import (
    plot_effect_holm_heatmap,
    plot_exact_mcnemar_holm_heatmaps,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BUNDLE = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Regional_Cultural_Persona"
    / "_combined"
    / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "regional" / "ethics accuracy analysis"
BUNDLE_VERSION = "ethics-regional-accuracy-combined-standardized-v1"
DOMAIN_TITLE = "Regional-Kultural"
DOMAIN_SLUG = "regional"
EXPECTED_ANALYSIS_VERSION = "ethics-regional-v3-accuracy-paired-holm-20260822"
SCENARIO_FAMILY = "scenario_within_regional"
PAIRWISE_FAMILY = "regional_pair_within_scenario"
PAIRWISE_TABLE_FILENAME = "mcnemar_regional_pairwise_comparisons.csv"
COCHRAN_TABLE_FILENAME = "cochran_q_by_subset_model_regional.csv"
SOURCE_GROUP_PATH = "Regional_Cultural_Persona"
NOTEBOOK_BUILDER_NAME = "build_ethics_regional_combined_accuracy_notebook.py"
NOTEBOOK_FILE_NAME = "ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb"
PARENT_SCOPE = "combined_ethics_regional_cultural"

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
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
REGION_SHORT_LABELS = {
    "north_american": "NA",
    "east_asian": "EA",
    "south_asian": "SA",
    "sub_saharan_african": "SSA",
}
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

CATEGORY_SPECS = [
    ("North American Explicit", "NA E", ["north_american_explicit"]),
    (
        "North American Implicit Avg",
        "NA I",
        ["north_american_implicit_1", "north_american_implicit_2", "north_american_implicit_3"],
    ),
    ("East Asian Explicit", "EA E", ["east_asian_explicit"]),
    (
        "East Asian Implicit Avg",
        "EA I",
        ["east_asian_implicit_1", "east_asian_implicit_2", "east_asian_implicit_3"],
    ),
    ("South Asian Explicit", "SA E", ["south_asian_explicit"]),
    (
        "South Asian Implicit Avg",
        "SA I",
        ["south_asian_implicit_1", "south_asian_implicit_2", "south_asian_implicit_3"],
    ),
    ("Sub-Saharan African Explicit", "SSA E", ["sub_saharan_african_explicit"]),
    (
        "Sub-Saharan African Implicit Avg",
        "SSA I",
        [
            "sub_saharan_african_implicit_1",
            "sub_saharan_african_implicit_2",
            "sub_saharan_african_implicit_3",
        ],
    ),
    ("Neutral", "Neutral", ["neutral"]),
]
CATEGORY_ORDER = [item[0] for item in CATEGORY_SPECS]
CATEGORY_SHORT_LABELS = {item[0]: item[1] for item in CATEGORY_SPECS}

BACKGROUND = "#FCFBF8"
ACCURACY_COLORS = [
    "#0047AB", "#0086A8", "#B80F2E", "#E45700", "#007A4D",
    "#65A30D", "#5B21B6", "#A0006D", "#30343B",
]
DELTA_COLORS = ["#0047AB", "#B80F2E", "#007A4D", "#5B21B6"]
ANNOTATION_SIZE = 14
DELTA_ANNOTATION_SIZE = 15
TITLE_SIZE = 22
STATUS_ANNOTATION_SIZE = 14
STATUS_FIGSIZE = (25, 21)
STATUS_WSPACE = 0.18
STATUS_XTICK_SIZE = 11
REASON_TITLE_SIZE = 28
REASON_AXIS_LABEL_SIZE = 22
REASON_TICK_SIZE = 17
AXES_LABEL_SIZE = 17
XTICK_LABEL_SIZE = 13
YTICK_LABEL_SIZE = 15
LEGEND_SIZE = 13
LEGEND_TITLE_SIZE = 14
MODEL_TITLE_SIZE = 16
REASON_XTICK_SIZE = 17
REASON_ANNOTATION_SIZE = 20
STAT_FONT_SIZE_OFFSET = 0

RAW_COLUMNS = [
    "model", "persona_folder", "question_id", "subset", "answer_status",
    "is_correct", "extracted_stance", "abstain_reason", "construct_valid",
]


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": "#17212B",
            "axes.labelcolor": "#17212B",
            "axes.edgecolor": "#5A6573",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXES_LABEL_SIZE,
            "xtick.labelsize": XTICK_LABEL_SIZE,
            "ytick.labelsize": YTICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_SIZE,
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: mpl.figure.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor=BACKGROUND, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"", "true", "false"}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"Nilai boolean tidak dikenal pada {column}: {unknown}")
    return normalized.eq("true")


def persona_name(region: str, scenario: str) -> str:
    if scenario == "N":
        return "neutral"
    if scenario == "E":
        return f"{region}_explicit"
    return f"{region}_implicit_{scenario[-1]}"


def persona_label(persona: str) -> str:
    if persona == "neutral":
        return "Neutral"
    for region in REGION_ORDER:
        prefix = f"{region}_"
        if persona.startswith(prefix):
            if persona.endswith("_explicit"):
                return f"{REGION_LABELS[region]} E"
            return f"{REGION_LABELS[region]} I{persona.rsplit('_', 1)[-1]}"
    raise KeyError(persona)


def persona_category(persona: str) -> str:
    for category, _short, personas in CATEGORY_SPECS:
        if persona in personas:
            return category
    raise KeyError(persona)


def _label_for_reason(reason: str) -> str:
    return ABSTAIN_REASON_LABELS.get(reason, reason.replace("_", " ").title())


def _aggregate_condition_summary(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["is_correct_bool"] = _as_bool(frame["is_correct"], "is_correct")
    frame["committed_flag"] = frame["answer_status"].eq("committed")
    frame["abstain_flag"] = frame["answer_status"].eq("abstain")
    frame["missing_flag"] = frame["answer_status"].eq("missing")
    frame["nonmissing_flag"] = ~frame["missing_flag"]
    frame["answer_wrong_flag"] = frame["committed_flag"] & frame["extracted_stance"].eq("wrong")
    frame["answer_acceptable_flag"] = frame["committed_flag"] & frame["extracted_stance"].eq("acceptable")

    grouped = (
        frame.groupby(["subset", "model", "persona_folder"], as_index=False, sort=False)
        .agg(
            total=("question_id", "size"),
            nonmissing=("nonmissing_flag", "sum"),
            committed=("committed_flag", "sum"),
            abstain=("abstain_flag", "sum"),
            missing=("missing_flag", "sum"),
            n_correct=("is_correct_bool", "sum"),
            n_answer_wrong=("answer_wrong_flag", "sum"),
            n_answer_acceptable=("answer_acceptable_flag", "sum"),
        )
        .rename(columns={"persona_folder": "persona"})
    )
    grouped["model_label"] = grouped["model"].map(MODEL_LABELS)
    grouped["persona_label"] = grouped["persona"].map(persona_label)
    grouped["category"] = grouped["persona"].map(persona_category)
    grouped["subset_label"] = grouped["subset"].map(SUBSET_LABELS)
    grouped["construct_valid"] = grouped["subset"].isin(VALID_SUBSETS)
    grouped["commit_rate"] = _safe_rate(grouped["committed"], grouped["total"])
    grouped["commit_rate_nonmissing"] = _safe_rate(grouped["committed"], grouped["nonmissing"])
    grouped["abstain_rate"] = _safe_rate(grouped["abstain"], grouped["total"])
    grouped["missing_rate"] = _safe_rate(grouped["missing"], grouped["total"])
    grouped["accuracy_committed"] = _safe_rate(grouped["n_correct"], grouped["committed"])
    grouped["effective_success_rate_nonmissing"] = _safe_rate(grouped["n_correct"], grouped["nonmissing"])
    grouped["pct_answer_wrong"] = _safe_rate(grouped["n_answer_wrong"], grouped["committed"])
    grouped["pct_answer_acceptable"] = _safe_rate(grouped["n_answer_acceptable"], grouped["committed"])
    ordered = [
        "model", "model_label", "persona", "persona_label", "category", "subset", "subset_label",
        "construct_valid", "total", "nonmissing", "committed", "abstain", "missing", "commit_rate",
        "commit_rate_nonmissing", "abstain_rate", "missing_rate", "n_correct", "accuracy_committed",
        "effective_success_rate_nonmissing", "n_answer_wrong", "n_answer_acceptable", "pct_answer_wrong",
        "pct_answer_acceptable",
    ]
    return grouped[ordered].sort_values(["model", "persona", "subset"], kind="stable").reset_index(drop=True)


def _aggregate_categories(frame: pd.DataFrame, count_columns: list[str]) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_frame = frame[frame["model"].eq(model)]
        for category, short_label, personas in CATEGORY_SPECS:
            group = model_frame[model_frame["persona"].isin(personas)]
            row = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "category": category,
                "category_short": short_label,
            }
            for column in count_columns:
                row[column] = int(group[column].sum())
            rows.append(row)
    result = pd.DataFrame(rows)
    result["commit_rate"] = _safe_rate(result["committed"], result["total"])
    result["commit_rate_nonmissing"] = _safe_rate(result["committed"], result["nonmissing"])
    result["abstain_rate"] = _safe_rate(result["abstain"], result["total"])
    result["missing_rate"] = _safe_rate(result["missing"], result["total"])
    return result


def _load_source_tables(source_bundle: Path) -> dict[str, object]:
    table_dir = source_bundle / "tables"
    manifest_path = source_bundle / "metadata" / "analysis_manifest.json"
    raw_path = source_bundle.parent / "all_models_results_full.csv"
    raw_manifest_path = source_bundle.parent / "all_models_results_full_manifest.json"
    for path in (manifest_path, raw_path, raw_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_hash = sha256_file(raw_path)
    expected_manifest_hash = (
        manifest.get("input_sha256")
        or manifest.get("input_sha256_before")
        or manifest.get("validation", {}).get("input_sha256_before")
    )
    if raw_hash != expected_manifest_hash or raw_hash != raw_manifest["output_sha256"]:
        raise ValueError(f"Hash input {DOMAIN_TITLE} tidak cocok dengan manifest v3")
    if manifest["analysis_version"] != EXPECTED_ANALYSIS_VERSION:
        raise ValueError(f"Versi sumber tidak sesuai: {manifest['analysis_version']}")

    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, dtype=str, keep_default_na=False, low_memory=False)
    if len(raw) != 154_428 or raw["question_id"].nunique() != 1_514:
        raise ValueError(f"Ukuran input {DOMAIN_TITLE} tidak sesuai: rows={len(raw)}")
    if raw.duplicated(["model", "persona_folder", "question_id"]).any():
        raise ValueError(f"Ditemukan duplikasi response key pada input {DOMAIN_TITLE}")
    if set(raw["model"]) != set(MODEL_ORDER) or raw["persona_folder"].nunique() != 17:
        raise ValueError(f"Model/persona {DOMAIN_TITLE} tidak lengkap")
    if set(raw["subset"]) != set(SUBSET_ORDER):
        raise ValueError(f"Subset {DOMAIN_TITLE} tidak lengkap")

    summary = _aggregate_condition_summary(raw)
    source_accuracy = pd.read_csv(table_dir / "accuracy_by_subset_model_condition.csv", low_memory=False)
    check = source_accuracy.merge(
        summary[["subset", "model", "persona", "total", "n_correct"]],
        on=["subset", "model", "persona"],
        suffixes=("_source", "_derived"),
        validate="one_to_one",
    )
    if not (check["total_responses"].eq(check["total"]).all() and check["n_correct_source"].eq(check["n_correct_derived"]).all()):
        raise ValueError("Ringkasan akurasi v3 tidak cocok dengan input sumber")

    reason_raw = raw[raw["answer_status"].eq("abstain")].copy()
    reason_raw["abstain_reason"] = reason_raw["abstain_reason"].replace("", "unspecified")
    reason = (
        reason_raw.groupby(["subset", "model", "persona_folder", "abstain_reason"], as_index=False, sort=False)
        .size()
        .rename(columns={"persona_folder": "persona", "size": "reason_count"})
    )
    reason["model_label"] = reason["model"].map(MODEL_LABELS)
    reason["persona_label"] = reason["persona"].map(persona_label)
    reason["category"] = reason["persona"].map(persona_category)
    reason["category_short"] = reason["category"].map(CATEGORY_SHORT_LABELS)
    reason["subset_label"] = reason["subset"].map(SUBSET_LABELS)
    reason["reason_label"] = reason["abstain_reason"].map(_label_for_reason)
    reason = reason.merge(
        summary[["subset", "model", "persona", "abstain", "nonmissing"]],
        on=["subset", "model", "persona"],
        validate="many_to_one",
    )
    reason["pct_of_abstain"] = _safe_rate(reason["reason_count"], reason["abstain"])
    reason["pct_of_nonmissing"] = _safe_rate(reason["reason_count"], reason["nonmissing"])

    status = (
        summary.groupby(["model", "persona"], as_index=False, sort=False)[
            ["total", "nonmissing", "committed", "abstain", "missing"]
        ].sum()
    )
    status["model_label"] = status["model"].map(MODEL_LABELS)
    status["persona_label"] = status["persona"].map(persona_label)
    status["category"] = status["persona"].map(persona_category)
    status["category_short"] = status["category"].map(CATEGORY_SHORT_LABELS)
    status["commit_rate"] = _safe_rate(status["committed"], status["total"])
    status["commit_rate_nonmissing"] = _safe_rate(status["committed"], status["nonmissing"])
    status["abstain_rate"] = _safe_rate(status["abstain"], status["total"])
    status["missing_rate"] = _safe_rate(status["missing"], status["total"])

    comparison_all = pd.read_csv(table_dir / "mcnemar_all_comparisons.csv", low_memory=False)
    comparison_scenario = pd.read_csv(table_dir / "mcnemar_scenario_comparisons.csv", low_memory=False)
    comparison_regional = pd.read_csv(table_dir / PAIRWISE_TABLE_FILENAME, low_memory=False)
    cochran = pd.read_csv(table_dir / COCHRAN_TABLE_FILENAME, low_memory=False)
    return {
        "manifest": manifest,
        "raw_manifest": raw_manifest,
        "raw_hash": raw_hash,
        "raw": raw,
        "summary": summary,
        "source_accuracy": source_accuracy,
        "reason": reason,
        "status": status,
        "comparison_all": comparison_all,
        "comparison_scenario": comparison_scenario,
        "comparison_regional": comparison_regional,
        "cochran": cochran,
    }


def _build_parent_tables(tables: dict[str, object]) -> dict[str, pd.DataFrame]:
    summary = tables["summary"]
    valid = summary[summary["subset"].isin(VALID_SUBSETS)].copy()
    accuracy = _aggregate_categories(
        valid, ["total", "nonmissing", "committed", "abstain", "missing", "n_correct"]
    )
    accuracy["accuracy_committed"] = _safe_rate(accuracy["n_correct"], accuracy["committed"])
    accuracy["effective_success_rate_nonmissing"] = _safe_rate(accuracy["n_correct"], accuracy["nonmissing"])

    status = _aggregate_categories(
        summary, ["total", "nonmissing", "committed", "abstain", "missing"]
    )

    reason = tables["reason"].copy()
    reason_categories = (
        reason.groupby(
            ["model", "model_label", "category", "category_short", "abstain_reason", "reason_label"],
            as_index=False,
        )["reason_count"].sum()
    )
    reason_categories = reason_categories.merge(
        status[["model", "category", "abstain", "nonmissing"]],
        on=["model", "category"],
        validate="many_to_one",
    )
    reason_categories["pct_of_abstain"] = _safe_rate(reason_categories["reason_count"], reason_categories["abstain"])
    reason_categories["pct_of_nonmissing"] = _safe_rate(reason_categories["reason_count"], reason_categories["nonmissing"])

    reason_model = (
        reason.groupby(["model", "model_label", "abstain_reason", "reason_label"], as_index=False)["reason_count"].sum()
    )
    model_status = status.groupby(["model", "model_label"], as_index=False)[["abstain", "nonmissing"]].sum()
    reason_model = reason_model.merge(model_status, on=["model", "model_label"], validate="many_to_one")
    reason_model["pct_of_abstain"] = _safe_rate(reason_model["reason_count"], reason_model["abstain"])
    reason_model["pct_of_nonmissing"] = _safe_rate(reason_model["reason_count"], reason_model["nonmissing"])

    return {
        "accuracy": accuracy,
        "status": status,
        "reason_categories": reason_categories,
        "reason_model": reason_model,
        "condition_summary": summary.copy(),
        "paired_all": tables["comparison_all"].copy(),
        "paired_scenario": tables["comparison_scenario"].copy(),
        "paired_regional": tables["comparison_regional"].copy(),
        "cochran": tables["cochran"].copy(),
        "source_accuracy": tables["source_accuracy"].copy(),
    }


def _category_pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    result = frame.pivot(index="model_label", columns="category", values=value)
    return result.reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=CATEGORY_ORDER,
    )


def _annotate_bars(ax: mpl.axes.Axes, *, size: int = ANNOTATION_SIZE, offset: int = 5, fmt: str = '.4f') -> None:
    for bar in ax.patches:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        is_positive = value >= 0
        ax.annotate(
            format(value, fmt),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if is_positive else "top",
            xytext=(0, offset if is_positive else -offset),
            textcoords="offset points",
            fontsize=size,
            fontweight="normal",
            rotation=90,
            rotation_mode="default",
            annotation_clip=False,
        )


def _annotate_status_segment(
    ax: mpl.axes.Axes,
    index: int,
    bottom: float,
    value: float,
) -> None:
    """Place a status percentage inside its own stacked-bar segment.

    Small segments receive a smaller label so the text remains inside the
    colored segment rather than spilling into an adjacent segment.
    """
    if value < 5:
        return
    fontsize = STATUS_ANNOTATION_SIZE if value >= 8 else max(9, STATUS_ANNOTATION_SIZE - 2)
    ax.text(
        index,
        bottom + value / 2,
        f"{value:.1f}%",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="normal",
        clip_on=True,
        zorder=5,
    )


def plot_accuracy(accuracy: pd.DataFrame, figure_dir: Path) -> list[Path]:
    values = _category_pivot(accuracy, "accuracy_committed")
    neutral_average = values["Neutral"].mean()
    fig, ax = plt.subplots(figsize=(27, 11))
    values.plot(kind="bar", width=0.88, color=ACCURACY_COLORS, ax=ax)
    baseline_label = f"Neutral Avg ({neutral_average:.3f})"
    ax.axhline(neutral_average, color="#B80F2E", linestyle="--", linewidth=2.0, label=baseline_label, zorder=1)
    _annotate_bars(ax, size=20, offset=4, fmt='.3f')
    ax.set_ylim(0, 1)
    ax.set_title(f"ETHICS {DOMAIN_TITLE}: Perbandingan Akurasi Antar-Persona", fontsize=TITLE_SIZE, fontweight="bold", pad=18)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy committed (construct-valid ETHICS)")
    ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER], rotation=0, ha="center")
    handles, labels = ax.get_legend_handles_labels()
    baseline_index = labels.index(baseline_label)
    handles = [handles[baseline_index]] + [handle for index, handle in enumerate(handles) if index != baseline_index]
    labels = [labels[baseline_index]] + [label for index, label in enumerate(labels) if index != baseline_index]
    ax.legend(handles, labels, title="Persona & Baseline", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, figure_dir, f"01_ethics_{DOMAIN_SLUG}_absolute_accuracy_nine_categories")


def _delta_frame(accuracy: pd.DataFrame, implicit: bool) -> pd.DataFrame:
    values = _category_pivot(accuracy, "accuracy_committed")
    rows = {}
    for region, label in REGION_LABELS.items():
        explicit_category = next(category for category, _short, personas in CATEGORY_SPECS if personas == [f"{region}_explicit"])
        implicit_category = next(category for category, _short, personas in CATEGORY_SPECS if personas == [f"{region}_implicit_1", f"{region}_implicit_2", f"{region}_implicit_3"])
        explicit = values[explicit_category]
        implicit_avg = values[implicit_category]
        neutral = values["Neutral"]
        rows[f"{label} I {'− N' if not implicit else '− E'}"] = implicit_avg - (explicit if implicit else neutral)
    return pd.DataFrame(rows)


def plot_delta(accuracy: pd.DataFrame, figure_dir: Path, *, implicit: bool) -> list[Path]:
    delta = _delta_frame(accuracy, implicit=implicit)
    stem = f"03_ethics_{DOMAIN_SLUG}_delta_implicit_vs_explicit" if implicit else f"02_ethics_{DOMAIN_SLUG}_delta_vs_neutral"
    title = (
        f"ETHICS {DOMAIN_TITLE}: Analisis Delta Implicit terhadap Explicit"
        if implicit
        else f"ETHICS {DOMAIN_TITLE}: Analisis Delta terhadap Persona Neutral"
    )
    fig, ax = plt.subplots(figsize=(24, 9.5))
    delta.plot(kind="bar", width=0.78, color=DELTA_COLORS, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_bars(ax, size=DELTA_ANNOTATION_SIZE, offset=9)
    limit = max(0.001, float(np.nanmax(np.abs(delta.to_numpy()))))
    ax.set_ylim(-limit * 1.55, limit * 1.55)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference on accuracy scale")
    ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER], rotation=0, ha="center")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save(fig, figure_dir, stem)


def plot_status(status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=STATUS_FIGSIZE, constrained_layout=False)
    axes_flat = list(axes.flat)
    x = np.arange(len(CATEGORY_ORDER))
    for ax, model in zip(axes_flat, MODEL_ORDER, strict=True):
        frame = status[status["model"].eq(model)].set_index("category").reindex(CATEGORY_ORDER)
        bottom = np.zeros(len(CATEGORY_ORDER))
        for status_name in STATUS_ORDER:
            values = frame[{"committed": "commit_rate", "abstain": "abstain_rate", "missing": "missing_rate"}[status_name]].to_numpy() * 100
            ax.bar(x, values, bottom=bottom, color=STATUS_COLORS[status_name], edgecolor="white", linewidth=0.7, label=STATUS_LABELS[status_name])
            for index, value in enumerate(values):
                _annotate_status_segment(ax, index, bottom[index], value)
            bottom += values
        ax.set_title(MODEL_LABELS[model], fontsize=MODEL_TITLE_SIZE, fontweight="bold")
        ax.set_xticks(x, [CATEGORY_SHORT_LABELS[item] for item in CATEGORY_ORDER], rotation=0)
        ax.tick_params(axis="x", labelsize=STATUS_XTICK_SIZE)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% respons")
        ax.grid(axis="x", visible=False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.89, bottom=0.05, left=0.06, right=0.98, hspace=0.42, wspace=STATUS_WSPACE)
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=True)
    fig.suptitle(f"ETHICS {DOMAIN_TITLE}: Status Jawaban per Model dan Persona", fontsize=TITLE_SIZE, fontweight="bold", y=0.98)
    return _save(fig, figure_dir, f"04_ethics_{DOMAIN_SLUG}_status_by_model_persona")


def plot_abstain_reasons(reason_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    reason_counts = reason_model.groupby(["abstain_reason", "reason_label"], as_index=False)["reason_count"].sum()
    model_denominators = reason_model.drop_duplicates("model")
    total_abstain = float(model_denominators["abstain"].sum())
    reason_counts["pct_of_abstain"] = reason_counts["reason_count"] / total_abstain
    frame = reason_counts.sort_values("pct_of_abstain", ascending=True)
    values = frame["pct_of_abstain"].to_numpy() * 100
    labels = frame["reason_label"].tolist()
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(23, 14))
    bars = ax.barh(y, values, color="#E9C46A", edgecolor="#A87918")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(10, float(values.max()) * 1.22 if len(values) else 10))
    ax.set_xlabel("% dari seluruh abstain", fontsize=REASON_AXIS_LABEL_SIZE, labelpad=12)
    ax.set_ylabel("Alasan abstain", fontsize=REASON_AXIS_LABEL_SIZE, labelpad=16)
    ax.set_title(f"ETHICS {DOMAIN_TITLE}: Penyebab Abstain", fontsize=REASON_TITLE_SIZE, fontweight="bold", pad=22)
    ax.tick_params(axis="x", labelsize=REASON_XTICK_SIZE, pad=8)
    ax.tick_params(axis="y", labelsize=REASON_TICK_SIZE, pad=8)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(f"{value:.1f}%", (bar.get_width(), bar.get_y() + bar.get_height() / 2), ha="left", va="center", xytext=(8, 0), textcoords="offset points", fontsize=REASON_ANNOTATION_SIZE, fontweight="normal")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=2.0)
    return _save(fig, figure_dir, f"05_ethics_{DOMAIN_SLUG}_abstain_reasons_by_model")


def _comparison_short(row: pd.Series) -> str:
    if row["comparison_family"] == SCENARIO_FAMILY:
        return f"{REGION_SHORT_LABELS[str(row['identity'])]}:{row['scenario_a']}−{row['scenario_b']}"
    identity_a, identity_b = str(row["identity"]).split("_vs_", maxsplit=1)
    return f"{REGION_SHORT_LABELS[identity_a]}−{REGION_SHORT_LABELS[identity_b]}:{row['scenario_a']}"


def _accuracy_inference(comparisons: pd.DataFrame) -> pd.DataFrame:
    selected = comparisons[comparisons["subset"].isin(VALID_SUBSETS)].copy()
    selected["row_label"] = selected["subset_label"] + " | " + selected["model_label"]
    selected["comparison_short"] = selected.apply(_comparison_short, axis=1)
    selected["effect_size"] = selected["cohens_h"]
    keep = [
        "comparison_family",
        "comparison_category",
        "subset",
        "subset_label",
        "model",
        "model_label",
        "row_label",
        "identity",
        "identity_label",
        "scenario_a",
        "scenario_b",
        "condition_a",
        "condition_b",
        "comparison_label",
        "comparison_short",
        "display_order",
        "accuracy_a_pct",
        "accuracy_b_pct",
        "delta_accuracy_pp",
        "effect_size",
        "exact_mcnemar_p_value",
        "holm_adjusted_p_value",
        "holm_family_size",
        "significant_holm_0_05",
    ]
    return selected[keep].reset_index(drop=True)


def _comparison_order(inference: pd.DataFrame) -> list[str]:
    scenario_order = []
    for region in REGION_ORDER:
        prefix = REGION_SHORT_LABELS[region]
        scenario_order.extend(
            [
                f"{prefix}:E−N",
                f"{prefix}:I1−N",
                f"{prefix}:I2−N",
                f"{prefix}:I3−N",
                f"{prefix}:I1−E",
                f"{prefix}:I2−E",
                f"{prefix}:I3−E",
            ]
        )
    pair_order = (
        inference[inference["comparison_family"].eq(PAIRWISE_FAMILY)]
        .sort_values("display_order", kind="stable")["comparison_short"]
        .drop_duplicates()
        .tolist()
    )
    return scenario_order + pair_order


def _combined_accuracy_inference(raw_path: Path, per_subset: pd.DataFrame) -> pd.DataFrame:
    templates = (
        per_subset.sort_values(["display_order", "model"], kind="stable")
        .drop_duplicates(["model", "comparison_short"])
        .copy()
    )
    templates["persona_a"] = templates["condition_a"]
    templates["persona_b"] = templates["condition_b"]
    templates["holm_group"] = np.where(
        templates["comparison_family"].eq(SCENARIO_FAMILY),
        templates["comparison_family"] + ":" + templates["identity"].astype(str),
        templates["comparison_family"],
    )
    combined = compute_combined_conditional_accuracy(
        raw_path,
        templates,
        valid_subsets=VALID_SUBSETS,
        persona_a_column="persona_a",
        persona_b_column="persona_b",
        holm_group_column="holm_group",
        eligibility="full_denominator",
    )
    return combined.sort_values(["display_order", "model"], kind="stable").reset_index(drop=True)


def plot_accuracy_statistics(inference: pd.DataFrame, figure_dir: Path) -> list[Path]:
    if DOMAIN_SLUG == 'regional':
        from ethics_persona_visual_style import plot_regional_four_panels
        rows = [f'{SUBSET_LABELS[s]} | {MODEL_LABELS[m]}' for s in VALID_SUBSETS for m in MODEL_ORDER]
        return plot_regional_four_panels(inference, figure_dir, row_order=rows,
            title_domain=DOMAIN_TITLE, stems=[
                '06_ethics_regional_effect_holm_heatmap',
                '07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part1',
                '07_ethics_regional_exact_mcnemar_and_holm_pvalue_heatmap_part2'])
    row_order = [
        f"{SUBSET_LABELS[subset]} | {MODEL_LABELS[model]}"
        for subset in VALID_SUBSETS
        for model in MODEL_ORDER
    ]
    comparison_order = _comparison_order(inference)
    paths = plot_effect_holm_heatmap(
        inference,
        figure_dir,
        title=f"ETHICS {DOMAIN_TITLE}: Ukuran Efek dan Hasil Koreksi Holm",
        stem=f"06_ethics_{DOMAIN_SLUG}_effect_holm_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=26,
        font_size_offset=STAT_FONT_SIZE_OFFSET,
    )
    paths += plot_exact_mcnemar_holm_heatmaps(
        inference,
        figure_dir,
        title=f"ETHICS {DOMAIN_TITLE}: Exact McNemar dan Koreksi Holm",
        stem=f"07_ethics_{DOMAIN_SLUG}_exact_mcnemar_and_holm_pvalue_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=26,
        font_size_offset=STAT_FONT_SIZE_OFFSET,
    )
    return paths


def plot_combined_accuracy_statistics(inference: pd.DataFrame, figure_dir: Path) -> list[Path]:
    if DOMAIN_SLUG == 'regional':
        from ethics_persona_visual_style import plot_regional_four_panels
        return plot_regional_four_panels(inference, figure_dir,
            row_order=[MODEL_LABELS[m] for m in MODEL_ORDER], title_domain=DOMAIN_TITLE,
            stems=['08_ethics_regional_combined_effect_holm_heatmap',
                   '09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part1',
                   '09_ethics_regional_combined_exact_mcnemar_and_holm_pvalue_heatmap_part2'])
    row_order = [MODEL_LABELS[model] for model in MODEL_ORDER]
    comparison_order = _comparison_order(inference)
    paths = plot_effect_holm_heatmap(
        inference,
        figure_dir,
        title=f"ETHICS {DOMAIN_TITLE}: Ukuran Efek dan Koreksi Holm",
        stem=f"08_ethics_{DOMAIN_SLUG}_combined_effect_holm_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=26,
        font_size_offset=STAT_FONT_SIZE_OFFSET,
        y_axis_label="Model",
    )
    paths += plot_exact_mcnemar_holm_heatmaps(
        inference,
        figure_dir,
        title=f"ETHICS {DOMAIN_TITLE}: p-value Exact McNemar dan Holm",
        stem=f"09_ethics_{DOMAIN_SLUG}_combined_exact_mcnemar_and_holm_pvalue_heatmap",
        row_order=row_order,
        comparison_order=comparison_order,
        chunk_size=26,
        font_size_offset=STAT_FONT_SIZE_OFFSET,
        y_axis_label="Model",
    )
    return paths


def _subset_category_values(summary: pd.DataFrame, subset: str, value_kind: str) -> pd.DataFrame:
    frame = summary[summary["subset"].eq(subset)]
    rows = []
    for model in MODEL_ORDER:
        model_frame = frame[frame["model"].eq(model)]
        for category, short_label, personas in CATEGORY_SPECS:
            group = model_frame[model_frame["persona"].isin(personas)]
            denominator = group["committed"].sum()
            numerator = group["n_correct"].sum() if value_kind == "accuracy" else group["n_answer_wrong"].sum()
            rows.append({"model": model, "model_label": MODEL_LABELS[model], "category": category, "value": numerator / denominator if denominator else np.nan})
    return pd.DataFrame(rows)


def _subset_status_values(summary: pd.DataFrame, subset: str) -> pd.DataFrame:
    return _aggregate_categories(summary[summary["subset"].eq(subset)], ["total", "nonmissing", "committed", "abstain", "missing"])


def render_subset_children(tables: dict[str, object], output_dir: Path) -> list[Path]:
    child_root = output_dir / "per_subset"
    child_root.mkdir(parents=True, exist_ok=True)
    summary = tables["summary"]
    reason = tables["reason"]
    comparisons = tables["comparison_all"]
    output_paths: list[Path] = []
    for subset in SUBSET_ORDER:
        child = child_root / subset
        table_dir = child / "tables"
        figure_dir = child / "figures"
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        subset_summary = summary[summary["subset"].eq(subset)].copy()
        subset_reason = reason[reason["subset"].eq(subset)].copy()
        subset_status = _subset_status_values(summary, subset)
        subset_paired = comparisons[comparisons["subset"].eq(subset)].copy()
        subset_summary.to_csv(table_dir / "condition_summary.csv", index=False)
        subset_status.to_csv(table_dir / "status_by_model_persona_category.csv", index=False)
        subset_reason.to_csv(table_dir / "abstain_reason.csv", index=False)
        subset_paired.to_csv(table_dir / "paired_effects.csv", index=False)

        value_kind = "accuracy" if subset in VALID_SUBSETS else "stance_wrong"
        values = _subset_category_values(summary, subset, value_kind)
        pivot = values.pivot(index="model_label", columns="category", values="value").reindex(index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=CATEGORY_ORDER)
        fig, ax = plt.subplots(figsize=(27, 10.5))
        pivot.plot(kind="bar", width=0.88, color=ACCURACY_COLORS, ax=ax)
        _annotate_bars(ax)
        ax.set_ylim(max(0, float(np.nanmin(pivot.to_numpy())) - 0.10), min(1, float(np.nanmax(pivot.to_numpy())) + 0.15))
        metric_label = "Accuracy committed" if value_kind == "accuracy" else "Proporsi jawaban wrong"
        ax.set_title(f"ETHICS {DOMAIN_TITLE} — {SUBSET_LABELS[subset]}: {metric_label}", fontsize=TITLE_SIZE, fontweight="bold", pad=18)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_label)
        ax.set_xticklabels([MODEL_AXIS_LABELS[MODEL_LABELS[m]] for m in MODEL_ORDER], rotation=0, ha="center")
        ax.legend(title="Persona", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        output_paths.extend(_save(fig, figure_dir, f"01_{subset}_performance_by_persona"))

        fig, axes = plt.subplots(3, 2, figsize=STATUS_FIGSIZE, constrained_layout=False)
        axes_flat = list(axes.flat)
        x = np.arange(len(CATEGORY_ORDER))
        for ax, model in zip(axes_flat, MODEL_ORDER, strict=True):
            frame = subset_status[subset_status["model"].eq(model)].set_index("category").reindex(CATEGORY_ORDER)
            bottom = np.zeros(len(CATEGORY_ORDER))
            for status_name in STATUS_ORDER:
                values_status = frame[{"committed": "commit_rate", "abstain": "abstain_rate", "missing": "missing_rate"}[status_name]].to_numpy() * 100
                ax.bar(x, values_status, bottom=bottom, color=STATUS_COLORS[status_name], edgecolor="white", linewidth=0.7, label=STATUS_LABELS[status_name])
                for index, value in enumerate(values_status):
                    _annotate_status_segment(ax, index, bottom[index], value)
                bottom += values_status
            ax.set_title(MODEL_LABELS[model], fontsize=MODEL_TITLE_SIZE, fontweight="bold")
            ax.set_xticks(x, [CATEGORY_SHORT_LABELS[item] for item in CATEGORY_ORDER], rotation=0)
            ax.tick_params(axis="x", labelsize=STATUS_XTICK_SIZE)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% respons")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.subplots_adjust(top=0.89, bottom=0.05, left=0.06, right=0.98, hspace=0.42, wspace=STATUS_WSPACE)
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=True)
        fig.suptitle(f"ETHICS {DOMAIN_TITLE} — {SUBSET_LABELS[subset]}: Status Jawaban", fontsize=TITLE_SIZE, fontweight="bold", y=0.98)
        output_paths.extend(_save(fig, figure_dir, f"02_{subset}_status_by_persona"))

        claim = "accuracy committed untuk Commonsense/Justice" if value_kind == "accuracy" else "distribusi stance (proporsi jawaban wrong), bukan akurasi moral"
        readme = f"""# ETHICS {DOMAIN_TITLE} — {SUBSET_LABELS[subset]}

Folder ini menyimpan rincian subset `{subset}`.

- `condition_summary.csv`: status dan metrik per model × persona.
- `status_by_model_persona_category.csv`: status setelah pengelompokan kategori persona.
- `abstain_reason.csv`: penyebab abstain dengan denominator sumber.
- `paired_effects.csv`: hasil paired dari bundle v3.
- `figures/`: visual detail subset.

Metrik performa pada subset ini adalah **{claim}**. Visual utama berada satu tingkat di atas, pada folder `ethics accuracy analysis`.
"""
        (child / "README.md").write_text(readme, encoding="utf-8")
    return output_paths


def _write_tables(parent: dict[str, pd.DataFrame], tables: dict[str, object], table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "accuracy": parent["accuracy"],
        "status": parent["status"],
        "reason_categories": parent["reason_categories"],
        "reason_model": parent["reason_model"],
        "paired_all": parent["paired_all"],
        "paired_scenario": parent["paired_scenario"],
        "paired_regional": parent["paired_regional"],
        "cochran": parent["cochran"],
        "condition_summary": parent["condition_summary"],
        "source_accuracy": parent["source_accuracy"],
        "accuracy_inference": parent["accuracy_inference"],
        "accuracy_inference_combined": parent["accuracy_inference_combined"],
    }
    names = {
        "accuracy": "ethics_accuracy_by_model_persona_category.csv",
        "status": "ethics_status_by_model_persona_category.csv",
        "reason_categories": "ethics_abstain_reasons_by_model_persona_category.csv",
        "reason_model": "ethics_abstain_reasons_by_model.csv",
        "paired_all": "ethics_paired_effects_all.csv",
        "paired_scenario": "ethics_paired_effects_scenario.csv",
        "paired_regional": f"ethics_paired_effects_{DOMAIN_SLUG}.csv",
        "cochran": "ethics_cochran_q.csv",
        "condition_summary": "ethics_condition_summary_source_derived.csv",
        "source_accuracy": "ethics_accuracy_source_v3.csv",
        "accuracy_inference": "ethics_accuracy_inference_construct_valid.csv",
        "accuracy_inference_combined": "ethics_accuracy_inference_construct_valid_combined.csv",
    }
    paths = {}
    for key, frame in outputs.items():
        path = table_dir / names[key]
        frame.to_csv(path, index=False)
        paths[key] = path
    return paths


def _write_readme(output_dir: Path) -> None:
    text = f"""# ETHICS {DOMAIN_TITLE} Accuracy Analysis

Bundle parent ini mengikuti format visual ETHICS Gender/GSM8K: bar chart besar,
anotasi yang terbaca, delta terhadap Neutral, status respons, dan penyebab
abstain.

## Cakupan parent

- Status `committed`, `abstain`, dan `missing` dipool dari seluruh empat subset.
- Akurasi parent hanya memakai subset construct-valid `commonsense` dan
  `justice`; `deontology` dan `virtue` tidak dilebur sebagai akurasi moral.
- Empat kategori persona ditampilkan sebagai Explicit dan rata-rata tiga kondisi Implicit,
  ditambah Neutral.
- Rincian empat subset berada sebagai child di `per_subset/`.

## Provenance

Sumber statistik: bundle v3 tervalidasi pada `result/ETHICS/reports/{SOURCE_GROUP_PATH}/_combined`.
CSV respons sumber hanya dibaca untuk membentuk tabel status dan alasan abstain;
tidak ada parsing ulang, rescoring, API call, atau perubahan pada `result/`.

## Bangun ulang

```powershell
C:\\Python314\\python.exe src/pipelines/analysis/render_ethics_{DOMAIN_SLUG}_overall_standardized.py
C:\\Python314\\python.exe src/pipelines/analysis/{NOTEBOOK_BUILDER_NAME}
C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python314\\Scripts\\jupyter-nbconvert.exe `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  "notebooks\\{DOMAIN_SLUG}\\ethics accuracy analysis\\{NOTEBOOK_FILE_NAME}"
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def render(source_bundle: Path = SOURCE_BUNDLE, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    source_bundle = source_bundle.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest_path = output_dir / "bundle_manifest.json"
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory without a recognized bundle manifest: {output_dir}"
            ) from exc
        if existing_manifest.get("bundle_version") != BUNDLE_VERSION:
            raise FileExistsError(f"Refusing to overwrite unrelated output directory: {output_dir}")
    tables = _load_source_tables(source_bundle)
    parent = _build_parent_tables(tables)
    parent["accuracy_inference"] = _accuracy_inference(parent["paired_all"])
    raw_path = source_bundle.parent / "all_models_results_full.csv"
    parent["accuracy_inference_combined"] = _combined_accuracy_inference(
        raw_path,
        parent["accuracy_inference"],
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = _write_tables(parent, tables, table_dir)

    configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths += plot_accuracy(parent["accuracy"], figure_dir)
    figure_paths += plot_delta(parent["accuracy"], figure_dir, implicit=False)
    figure_paths += plot_delta(parent["accuracy"], figure_dir, implicit=True)
    figure_paths += plot_status(parent["status"], figure_dir)
    figure_paths += plot_abstain_reasons(parent["reason_model"], figure_dir)
    figure_paths += plot_accuracy_statistics(parent["accuracy_inference"], figure_dir)
    figure_paths += plot_combined_accuracy_statistics(parent["accuracy_inference_combined"], figure_dir)
    figure_paths += render_subset_children(tables, output_dir)

    output_index_rows = []
    for path in sorted(table_paths.values()):
        output_index_rows.append({"kind": "table", "file": path.relative_to(output_dir).as_posix(), "path": str(path), "sha256": sha256_file(path)})
    for path in sorted(figure_paths):
        output_index_rows.append({"kind": "figure", "file": path.relative_to(output_dir).as_posix(), "path": str(path), "sha256": sha256_file(path)})
    output_index = pd.DataFrame(output_index_rows)
    output_index.to_csv(output_dir / "output_index.csv", index=False)

    source_manifest_path = source_bundle / "metadata" / "analysis_manifest.json"
    parent_figure_paths = [path for path in figure_paths if path.parent == figure_dir]
    raw = tables["raw"]
    status_counts = {status: int(raw["answer_status"].eq(status).sum()) for status in STATUS_ORDER}
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_from": str(source_bundle),
        "source_analysis_version": tables["manifest"]["analysis_version"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_input": str(source_bundle.parent / "all_models_results_full.csv"),
        "source_input_sha256": tables["raw_hash"],
        "source_row_count": int(len(raw)),
        "parent_scope": PARENT_SCOPE,
        "status_scope": "all_four_subsets",
        "accuracy_scope": "construct_valid_commonsense_justice_pooled",
        "models": MODEL_ORDER,
        "personas": sorted(raw["persona_folder"].unique().tolist()),
        "subsets": SUBSET_ORDER,
        "categories": CATEGORY_ORDER,
        "parent_table_files": [path.name for path in sorted(table_paths.values())],
        "parent_figure_files": [path.name for path in sorted(parent_figure_paths)],
        "parent_figure_count": len(parent_figure_paths),
        "child_subset_dirs": SUBSET_ORDER,
        "status_counts": status_counts,
        "source_bundle_read_only": True,
        "api_calls": 0,
        "reparse": False,
        "rescore": False,
        "statistics_recomputed": True,
        "statistics_recomputed_scope": "combined_construct_valid_commonsense_justice_only",
        "combined_statistics_eligibility": "full_denominator_abstain_missing_incorrect",
        "per_subset_statistics_recomputed": False,
        "pair_count_displayed": False,
        "notes": [
            "Judul parent tidak memakai kata overall.",
            "Status committed/abstain/missing dipool pada seluruh empat subset.",
            "Accuracy parent dipool hanya pada commonsense dan justice.",
            "Statistik 06/07 mempertahankan hasil per subset dari source bundle.",
            "Statistik 08/09 dihitung ulang pada gabungan commonsense dan justice dengan pairing key subset + question_id.",
            "Deontology dan virtue tetap tersedia sebagai distribusi stance di child per_subset.",
            "Akurasi v3 sumber tetap tersedia sebagai ethics_accuracy_source_v3.csv; tidak digunakan sebagai satu skor parent moral.",
        ],
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(output_dir)
    print(f"[CREATED] {output_dir}")
    print(f"[PARENT TABLES] {len(table_paths)}")
    print(f"[PARENT FIGURES] {len(parent_figure_paths)} files ({len(parent_figure_paths) // 2} PNG + {len(parent_figure_paths) // 2} SVG)")
    print(f"[CHILD FIGURES] {len(figure_paths) - len(parent_figure_paths)} files")
    return {"output_dir": output_dir, "tables": table_paths, "figures": figure_paths, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=SOURCE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    render(args.source_bundle, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

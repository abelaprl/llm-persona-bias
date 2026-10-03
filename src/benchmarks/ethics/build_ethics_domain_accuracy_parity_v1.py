"""Build Gender-style accuracy and status visualizations for ETHICS domains.

The figures are derived from the validated response tables. Commonsense and
Justice are the only subsets used for accuracy outcomes. Deontology and
Virtue are shown as answer/stance distributions and stance-wrong effects;
they are never treated as moral ground-truth accuracy.

The script writes new domain-specific files and refuses to overwrite an
existing file unless --overwrite is supplied explicitly. It does not call an
API, parse model responses, or use ground truth in an extraction rule.

Run from the repository root:

    python src/benchmarks/ethics/build_ethics_domain_accuracy_parity_v1.py --domain religion
    python src/benchmarks/ethics/build_ethics_domain_accuracy_parity_v1.py --domain regional
    python src/benchmarks/ethics/build_ethics_domain_accuracy_parity_v1.py --domain all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency
import seaborn as sns


ANALYSIS_VERSION = "ethics-domain-accuracy-parity-v1-20260929"
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
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
VALID_ACCURACY_SUBSETS = ["commonsense", "justice"]
STANCE_SUBSETS = ["deontology", "virtue"]
CONSTRUCT_VALID = {
    "commonsense": True,
    "deontology": False,
    "justice": True,
    "virtue": False,
}
STATUS_ORDER = ["committed", "abstain", "missing"]
STATUS_LABELS = {
    "committed": "Committed",
    "abstain": "Abstain",
    "missing": "Missing",
}
STATUS_COLORS = {
    "committed": "#4C78A8",
    "wrong": "#C94C4C",
    "acceptable": "#3A9D7C",
    "abstain": "#E5A84B",
    "missing": "#9AA3AD",
}
REASON_LABELS = {
    "analysis_without_verdict": "Analysis without verdict",
    "no_opinion_stop": "No-opinion stop",
    "both_sides": "Both sides",
    "task_substitution": "Task substitution",
    "explicit_binary_refusal": "Explicit binary refusal",
    "insufficient_context": "Insufficient context",
    "ai_refusal": "AI refusal",
    "placeholder_confusion": "Placeholder confusion",
    "returns_question": "Returns question",
    "conditional_only": "Conditional only",
    "capability_mismatch": "Capability mismatch",
    "detector_disagreement": "Detector disagreement",
    "unspecified": "Unspecified",
}
SOURCE_COLUMNS = [
    "model",
    "persona_folder",
    "question_id",
    "subset",
    "answer_status",
    "is_correct",
    "extracted_stance",
    "abstain_reason",
]


@dataclass(frozen=True)
class DomainConfig:
    slug: str
    display_name: str
    source_response_table: Path
    bundle_dir: Path
    identities: tuple[tuple[str, str, str], ...]


CONFIGS = {
    "religion": DomainConfig(
        slug="religion",
        display_name="Religion",
        source_response_table=Path(
            "result/benchmarks/ethics/reports/Religion_Persona/_combined/"
            "all_models_results_full.csv"
        ),
        bundle_dir=Path("notebooks/religion/ethics accuracy analysis"),
        identities=(
            ("islam", "Islam", "Islam"),
            ("christian", "Kristen", "Kristen"),
            ("jewish", "Yahudi", "Yahudi"),
            ("atheist", "Ateis", "Ateis"),
        ),
    ),
    "regional": DomainConfig(
        slug="regional",
        display_name="Regional Cultural",
        source_response_table=Path(
            "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/"
            "all_models_results_full.csv"
        ),
        bundle_dir=Path("notebooks/regional/ethics accuracy analysis"),
        identities=(
            ("north_american", "North American", "NA"),
            ("east_asian", "East Asian", "EA"),
            ("south_asian", "South Asian", "SA"),
            ("sub_saharan_african", "Sub-Saharan African", "SSA"),
        ),
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plots() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.05,
    )
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
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "legend.title_fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def persona_order(config: DomainConfig) -> list[str]:
    ordered: list[str] = []
    for identity_key, _, _ in config.identities:
        ordered.append(f"{identity_key}_explicit")
        ordered.extend(f"{identity_key}_implicit_{i}" for i in [1, 2, 3])
    ordered.append("neutral")
    return ordered


def persona_labels(config: DomainConfig) -> dict[str, str]:
    labels: dict[str, str] = {}
    for identity_key, _, short_label in config.identities:
        labels[f"{identity_key}_explicit"] = f"{short_label} E"
        for i in [1, 2, 3]:
            labels[f"{identity_key}_implicit_{i}"] = f"{short_label} I{i}"
    labels["neutral"] = "Neutral"
    return labels


def category_definitions(config: DomainConfig) -> list[dict[str, object]]:
    categories: list[dict[str, object]] = []
    for identity_key, identity_label, short_label in config.identities:
        categories.append(
            {
                "key": f"{identity_key}_explicit",
                "label": f"{identity_label} Explicit",
                "short_label": f"{short_label} E",
                "personas": [f"{identity_key}_explicit"],
            }
        )
        categories.append(
            {
                "key": f"{identity_key}_implicit_avg",
                "label": f"{identity_label} Implicit Avg",
                "short_label": f"{short_label} I",
                "personas": [
                    f"{identity_key}_implicit_{i}" for i in [1, 2, 3]
                ],
            }
        )
    categories.append(
        {
            "key": "neutral",
            "label": "Neutral",
            "short_label": "Neutral",
            "personas": ["neutral"],
        }
    )
    return categories


def category_lookup(config: DomainConfig) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for category in category_definitions(config):
        for persona in category["personas"]:
            lookup[persona] = str(category["key"])
    return lookup


def category_labels(config: DomainConfig) -> dict[str, tuple[str, str]]:
    return {
        str(category["key"]): (
            str(category["label"]),
            str(category["short_label"]),
        )
        for category in category_definitions(config)
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def load_source(config: DomainConfig) -> tuple[pd.DataFrame, Path]:
    root = repo_root()
    source_path = root / config.source_response_table
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    frame = pd.read_csv(source_path, usecols=SOURCE_COLUMNS, low_memory=False)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Kolom sumber tidak lengkap: {missing_columns}")
    expected_personas = set(persona_order(config))
    observed_personas = set(frame["persona_folder"].dropna().unique())
    if observed_personas != expected_personas:
        raise ValueError(
            f"Persona {config.slug} berbeda dari kontrak: "
            f"observed={sorted(observed_personas)}, expected={sorted(expected_personas)}"
        )
    if set(frame["subset"].dropna().unique()) != set(SUBSET_ORDER):
        raise ValueError("Subset sumber tidak sama dengan empat subset ETHICS.")
    if frame.duplicated(
        ["model", "persona_folder", "subset", "question_id"]
    ).any():
        raise ValueError("Ada lebih dari satu baris untuk key model/persona/subset/question_id.")
    if frame["answer_status"].isna().any():
        raise ValueError("answer_status kosong; missing harus berasal dari status eksplisit.")
    observed_status = set(frame["answer_status"].unique())
    if not observed_status.issubset(set(STATUS_ORDER)):
        raise ValueError(f"answer_status tidak dikenal: {sorted(observed_status)}")
    frame["is_correct_bool"] = to_bool_series(frame["is_correct"])
    frame["extracted_stance"] = frame["extracted_stance"].fillna("").astype(str)
    frame["abstain_reason"] = (
        frame["abstain_reason"]
        .fillna("unspecified")
        .astype(str)
        .replace("", "unspecified")
    )
    frame["category"] = frame["persona_folder"].map(category_lookup(config))
    if frame["category"].isna().any():
        raise ValueError("Ada persona tanpa kategori agregasi.")
    return frame, source_path


def summary_row(
    group: pd.DataFrame,
    *,
    model: str,
    persona: str,
    subset: str,
) -> dict[str, object]:
    total = len(group)
    committed_mask = group["answer_status"].eq("committed")
    abstain_mask = group["answer_status"].eq("abstain")
    missing_mask = group["answer_status"].eq("missing")
    n_committed = int(committed_mask.sum())
    n_abstain = int(abstain_mask.sum())
    n_missing = int(missing_mask.sum())
    n_nonmissing = total - n_missing
    n_correct = int((committed_mask & group["is_correct_bool"]).sum())
    n_wrong = int(
        (committed_mask & group["extracted_stance"].eq("wrong")).sum()
    )
    n_acceptable = int(
        (committed_mask & group["extracted_stance"].eq("acceptable")).sum()
    )
    accuracy_scope = subset in VALID_ACCURACY_SUBSETS
    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "persona": persona,
        "subset": subset,
        "subset_label": SUBSET_LABELS.get(subset, "All four subsets"),
        "construct_valid": CONSTRUCT_VALID.get(subset, False),
        "total": total,
        "nonmissing": n_nonmissing,
        "committed": n_committed,
        "abstain": n_abstain,
        "missing": n_missing,
        "commit_rate": safe_rate(n_committed, total),
        "commit_rate_nonmissing": safe_rate(n_committed, n_nonmissing),
        "abstain_rate": safe_rate(n_abstain, total),
        "missing_rate": safe_rate(n_missing, total),
        # Accuracy fields are intentionally blank outside Commonsense and
        # Justice. Deontology and Virtue are reported through stance
        # distributions below, never as ground-truth accuracy.
        "n_correct": n_correct if accuracy_scope else math.nan,
        "accuracy_committed": (
            safe_rate(n_correct, n_committed) if accuracy_scope else math.nan
        ),
        "effective_success_rate_nonmissing": (
            safe_rate(n_correct, n_nonmissing) if accuracy_scope else math.nan
        ),
        "n_answer_wrong": n_wrong,
        "n_answer_acceptable": n_acceptable,
        "pct_answer_wrong": safe_rate(n_wrong, n_committed),
        "pct_answer_acceptable": safe_rate(n_acceptable, n_committed),
    }


def build_condition_summary(
    frame: pd.DataFrame,
    *,
    group_order: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, persona, subset), group in frame.groupby(
        ["model", "persona_folder", "subset"], sort=False
    ):
        rows.append(
            summary_row(
                group,
                model=str(model),
                persona=str(persona),
                subset=str(subset),
            )
        )
    result = pd.DataFrame(rows)
    if group_order is not None:
        result["persona"] = pd.Categorical(
            result["persona"], categories=group_order, ordered=True
        )
        result = result.sort_values(
            ["model", "persona", "subset"], kind="stable"
        ).reset_index(drop=True)
        result["persona"] = result["persona"].astype(str)
    return result


def build_all_subset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, persona), group in frame.groupby(
        ["model", "persona_folder"], sort=False
    ):
        rows.append(
            summary_row(
                group,
                model=str(model),
                persona=str(persona),
                subset="all",
            )
        )
    return pd.DataFrame(rows)


def build_category_summary(
    frame: pd.DataFrame, config: DomainConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = category_labels(config)
    for (model, category), group in frame.groupby(
        ["model", "category"], sort=False
    ):
        record = summary_row(
            group,
            model=str(model),
            persona=str(category),
            subset="all",
        )
        label, short_label = labels[str(category)]
        record["category"] = str(category)
        record["category_label"] = label
        record["category_short"] = short_label
        rows.append(record)
    return pd.DataFrame(rows)


def build_reason_table(
    frame: pd.DataFrame,
    all_summary: pd.DataFrame,
    config: DomainConfig,
) -> pd.DataFrame:
    abstains = frame[frame["answer_status"].eq("abstain")].copy()
    counts = (
        abstains.groupby(
            ["model", "persona_folder", "abstain_reason"], sort=False
        )
        .size()
        .rename("reason_count")
        .reset_index()
    )
    denominators = all_summary[
        ["model", "persona", "abstain", "nonmissing"]
    ].rename(columns={"persona": "persona_folder"})
    result = counts.merge(
        denominators,
        on=["model", "persona_folder"],
        how="left",
        validate="many_to_one",
    )
    result["reason_label"] = result["abstain_reason"].map(REASON_LABELS).fillna(
        result["abstain_reason"].str.replace("_", " ", regex=False).str.title()
    )
    result["pct_of_abstain"] = np.where(
        result["abstain"].gt(0),
        100 * result["reason_count"] / result["abstain"],
        np.nan,
    )
    result["pct_of_nonmissing"] = np.where(
        result["nonmissing"].gt(0),
        100 * result["reason_count"] / result["nonmissing"],
        np.nan,
    )
    result["model_label"] = result["model"].map(MODEL_LABELS)
    result["persona_label"] = result["persona_folder"].map(
        persona_labels(config)
    )
    return result.sort_values(
        ["model", "persona_folder", "reason_count"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def build_cramers_v(
    frame: pd.DataFrame, config: DomainConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_personas = persona_order(config)
    for model in MODEL_ORDER:
        subset = frame[frame["model"].eq(model)]
        table = pd.crosstab(subset["persona_folder"], subset["answer_status"])
        table = table.reindex(
            index=ordered_personas, columns=STATUS_ORDER, fill_value=0
        )
        # A status that never occurs for a model creates a zero expected
        # frequency. It carries no information for the independence test and
        # must be removed before scipy computes expected frequencies.
        table = table.loc[:, table.sum(axis=0).gt(0)]
        if table.shape[1] < 2:
            chi2, p_value = 0.0, 1.0
        else:
            chi2, p_value, _, _ = chi2_contingency(
                table.to_numpy(), correction=False
            )
        n = int(table.to_numpy().sum())
        r, k = table.shape
        value = math.sqrt((chi2 / n) / min(k - 1, r - 1)) if n else math.nan
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n": n,
                "chi2": chi2,
                "p_value": p_value,
                "cramers_v": value,
                "personas": r,
                "statuses": k,
            }
        )
    return pd.DataFrame(rows)


def write_table(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File sudah ada dan dilindungi: {path}. Gunakan --overwrite bila memang diizinkan."
        )
    frame.to_csv(path, index=False)


def save_figure(
    fig: plt.Figure,
    figure_dir: Path,
    stem: str,
    overwrite: bool,
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    svg_path = figure_dir / f"{stem}.svg"
    if not overwrite and (png_path.exists() or svg_path.exists()):
        plt.close(fig)
        raise FileExistsError(
            f"Figure sudah ada dan dilindungi: {png_path}. "
            "Gunakan --overwrite bila memang diizinkan."
        )
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def axis_model_labels() -> list[str]:
    return [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER]


def _heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmin: float = 0,
    vmax: float = 100,
    cmap: str = "YlGnBu",
    center: float | None = None,
    annot: bool = True,
    fmt: str = ".1f",
    cbar: bool = False,
) -> None:
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot=annot,
        fmt=fmt,
        linewidths=0.45,
        linecolor="#F1F1F1",
        cbar=cbar,
        cbar_kws={"label": "%"} if cbar else None,
        mask=matrix.isna(),
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=55)
    ax.tick_params(axis="y", rotation=0)


def plot_combined_distribution(
    frame: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str] | None:
    stem = f"00_ethics_{config.slug}_combined_four_subsets_distribution"
    if (figure_dir / f"{stem}.png").exists() or (
        figure_dir / f"{stem}.svg"
    ).exists():
        return None
    labels = category_labels(config)
    categories = [str(category["key"]) for category in category_definitions(config)]
    short_labels = [labels[key][1] for key in categories]
    rows: list[dict[str, object]] = []
    for (model, category), group in frame.groupby(
        ["model", "category"], sort=False
    ):
        committed = group["answer_status"].eq("committed")
        rows.append(
            {
                "model": model,
                "category": category,
                "wrong": 100
                * (committed & group["extracted_stance"].eq("wrong")).sum()
                / len(group),
                "acceptable": 100
                * (committed & group["extracted_stance"].eq("acceptable")).sum()
                / len(group),
                "abstain": 100 * group["answer_status"].eq("abstain").sum() / len(group),
                "missing": 100 * group["answer_status"].eq("missing").sum() / len(group),
            }
        )
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        subset = summary[summary["model"].eq(model)].set_index("category")
        subset = subset.reindex(categories)
        bottom = np.zeros(len(categories))
        x = np.arange(len(categories))
        for status in ["wrong", "acceptable", "abstain", "missing"]:
            values = subset[status].to_numpy()
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=STATUS_COLORS[status],
                label=status.title(),
                width=0.78,
            )
            bottom += values
        ax.set_title(MODEL_LABELS[model])
        ax.set_xticks(x, short_labels, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of responses")
        ax.grid(axis="x", visible=False)
    handles, labels_legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    fig.suptitle(
        f"ETHICS {config.display_name}: Distribusi empat subset per kategori persona",
        y=1.06,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Gabungan Commonsense, Deontology, Justice, dan Virtue; "
        "Deontology/Virtue ditampilkan sebagai distribusi status.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_status_composition(
    frame: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    stem = f"01_ethics_{config.slug}_status_composition_by_model"
    counts = (
        frame.groupby(["model", "answer_status"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    pivot = counts.pivot(index="model", columns="answer_status", values="n").fillna(0)
    pivot = pivot.reindex(MODEL_ORDER).fillna(0)
    pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100)
    fig, ax = plt.subplots(figsize=(13, 7))
    bottom = np.zeros(len(pct))
    x = np.arange(len(pct))
    for status in STATUS_ORDER:
        values = pct.get(status, pd.Series(0, index=pct.index)).to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STATUS_COLORS[status],
            label=STATUS_LABELS[status],
            width=0.72,
        )
        bottom += values
    ax.set_xticks(x, [MODEL_AXIS_LABELS[MODEL_LABELS[m]] for m in MODEL_ORDER])
    ax.set_ylabel("% of responses")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"ETHICS {config.display_name}: Komposisi status jawaban per model",
        pad=16,
    )
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="x", visible=False)
    fig.text(
        0.5,
        -0.02,
        "Denominator mencakup seluruh respons pada empat subset.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_status_heatmaps(
    all_summary: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    stem = f"02_ethics_{config.slug}_status_by_model_persona_heatmap"
    order = persona_order(config)
    labels = persona_labels(config)
    fig, axes = plt.subplots(1, 3, figsize=(29, 8), sharey=True)
    for ax, metric, title in zip(
        axes,
        ["commit_rate", "abstain_rate", "missing_rate"],
        ["Commit rate", "Abstain rate", "Missing rate"],
    ):
        matrix = (
            all_summary.pivot(index="model", columns="persona", values=metric)
            .reindex(index=MODEL_ORDER, columns=order)
            .mul(100)
        )
        matrix.index = axis_model_labels()
        matrix.columns = [labels[p] for p in order]
        _heatmap(ax, matrix, title=title, cbar=ax is axes[-1])
    fig.suptitle(
        f"ETHICS {config.display_name}: Status jawaban per model dan persona",
        y=1.03,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.03,
        "Semua empat subset; setiap sel menggunakan seluruh respons pada persona tersebut.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_commit_rate_by_subset(
    condition_summary: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    stem = f"03_ethics_{config.slug}_commit_rate_by_persona_subset_heatmap"
    order = persona_order(config)
    labels = persona_labels(config)
    fig, axes = plt.subplots(2, 2, figsize=(27, 14), sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        part = condition_summary[condition_summary["subset"].eq(subset)]
        matrix = (
            part.pivot(index="model", columns="persona", values="commit_rate")
            .reindex(index=MODEL_ORDER, columns=order)
            .mul(100)
        )
        matrix.index = axis_model_labels()
        matrix.columns = [labels[p] for p in order]
        _heatmap(
            ax,
            matrix,
            title=SUBSET_LABELS[subset],
            cbar=ax is axes.flat[-1],
        )
    fig.suptitle(
        f"ETHICS {config.display_name}: Commit rate per persona dan subset",
        y=1.02,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Commit rate = committed / total; missing tetap berada di denominator.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def reason_matrices(
    reason_table: pd.DataFrame,
    config: DomainConfig,
    metric: str,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    order = persona_order(config)
    reason_order = (
        reason_table.groupby(["abstain_reason"], sort=False)["reason_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    matrices: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        part = reason_table[reason_table["model"].eq(model)]
        matrix = (
            part.pivot_table(
                index="reason_label",
                columns="persona_folder",
                values=metric,
                aggfunc="sum",
            )
            .reindex(columns=order)
            .fillna(0)
        )
        label_by_reason = (
            part.drop_duplicates("abstain_reason")
            .set_index("abstain_reason")["reason_label"]
            .to_dict()
        )
        ordered_labels = [
            label_by_reason.get(reason, reason.replace("_", " ").title())
            for reason in reason_order
        ]
        matrix = matrix.reindex(ordered_labels).fillna(0)
        matrix.columns = [persona_labels(config)[p] for p in order]
        matrices[model] = matrix
    return ordered_labels, matrices


def plot_reason_heatmaps(
    reason_table: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
    *,
    metric: str,
    stem_number: str,
    title: str,
    metric_label: str,
) -> dict[str, str]:
    suffix = (
        "abstain_reason_share"
        if metric == "pct_of_abstain"
        else "abstain_reason_prevalence"
    )
    stem = f"{stem_number}_ethics_{config.slug}_{suffix}_by_model_persona_heatmap"
    _, matrices = reason_matrices(reason_table, config, metric)
    fig, axes = plt.subplots(3, 2, figsize=(28, 21), sharex=True)
    vmax = max(
        float(matrix.to_numpy().max()) for matrix in matrices.values() if not matrix.empty
    )
    vmax = max(vmax, 1.0)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        _heatmap(
            ax,
            matrices[model],
            title=MODEL_LABELS[model],
            vmin=0,
            vmax=vmax,
            cbar=ax is axes.flat[-1],
        )
        ax.set_ylabel("Abstain reason")
    fig.suptitle(
        f"ETHICS {config.display_name}: {title}",
        y=1.01,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(0.5, -0.02, metric_label, ha="center", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_accuracy_or_effective(
    condition_summary: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
    *,
    metric: str,
    stem_number: str,
    title: str,
    ylabel: str,
) -> dict[str, str]:
    suffix = (
        "accuracy_committed"
        if metric == "accuracy_committed"
        else "effective_success"
    )
    stem = f"{stem_number}_ethics_{config.slug}_{suffix}_valid_subsets_heatmap"
    order = persona_order(config)
    labels = persona_labels(config)
    fig, axes = plt.subplots(1, 2, figsize=(27, 8), sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, VALID_ACCURACY_SUBSETS):
        part = condition_summary[condition_summary["subset"].eq(subset)]
        matrix = (
            part.pivot(index="model", columns="persona", values=metric)
            .reindex(index=MODEL_ORDER, columns=order)
            .mul(100)
        )
        matrix.index = axis_model_labels()
        matrix.columns = [labels[p] for p in order]
        _heatmap(
            ax,
            matrix,
            title=SUBSET_LABELS[subset],
            cbar=ax is axes.flat[-1],
        )
    fig.suptitle(
        f"ETHICS {config.display_name}: {title}",
        y=1.03,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(0.5, -0.03, ylabel, ha="center", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_stance_wrong(
    condition_summary: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    stem = f"08_ethics_{config.slug}_stance_wrong_construct_invalid_heatmap"
    order = persona_order(config)
    labels = persona_labels(config)
    fig, axes = plt.subplots(1, 2, figsize=(27, 8), sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, STANCE_SUBSETS):
        part = condition_summary[condition_summary["subset"].eq(subset)]
        matrix = (
            part.pivot(index="model", columns="persona", values="pct_answer_wrong")
            .reindex(index=MODEL_ORDER, columns=order)
            .mul(100)
        )
        matrix.index = axis_model_labels()
        matrix.columns = [labels[p] for p in order]
        _heatmap(
            ax,
            matrix,
            title=f"{SUBSET_LABELS[subset]} — stance wrong (%)",
            cbar=ax is axes.flat[-1],
        )
    fig.suptitle(
        f"ETHICS {config.display_name}: Distribusi stance wrong "
        "(Deontology/Virtue; bukan akurasi)",
        y=1.03,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.03,
        "Denominator = committed; stance acceptable = 100% − stance wrong. "
        "Deontology/Virtue tidak dipakai sebagai ground-truth accuracy.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def plot_cramers_v(
    cramers: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    stem = f"09_ethics_{config.slug}_status_persona_cramers_v"
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(cramers))
    bars = ax.bar(x, cramers["cramers_v"], color="#4C78A8", width=0.7)
    ax.set_xticks(x, [MODEL_AXIS_LABELS[MODEL_LABELS[m]] for m in MODEL_ORDER])
    ax.set_ylabel("Cramér's V")
    ax.set_ylim(0, max(0.05, float(cramers["cramers_v"].max()) * 1.22))
    ax.set_title(
        f"ETHICS {config.display_name}: Dependensi status jawaban dan persona",
        pad=16,
    )
    for bar, value in zip(bars, cramers["cramers_v"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.text(
        0.5,
        -0.02,
        "Uji chi-square pada tiga status: committed, abstain, dan missing; "
        "empat subset digabung.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (m - rank)))
        adjusted[index] = running
    return adjusted


def paired_events(
    paired: pd.DataFrame,
    outcome: str,
) -> tuple[pd.Series, pd.Series, int, int]:
    missing_either = (
        paired["answer_status_a"].eq("missing")
        | paired["answer_status_b"].eq("missing")
    )
    both_committed = (
        paired["answer_status_a"].eq("committed")
        & paired["answer_status_b"].eq("committed")
    )
    if outcome in {"coverage", "effective_success"}:
        eligible = ~missing_either
        excluded_missing = int(missing_either.sum())
        excluded_not_both_committed = 0
    else:
        eligible = both_committed
        excluded_missing = int(missing_either.sum())
        excluded_not_both_committed = int((~missing_either & ~both_committed).sum())
    if outcome == "coverage":
        event_a = paired["answer_status_a"].eq("committed")
        event_b = paired["answer_status_b"].eq("committed")
    elif outcome == "effective_success":
        event_a = paired["answer_status_a"].eq("committed") & paired[
            "is_correct_bool_a"
        ]
        event_b = paired["answer_status_b"].eq("committed") & paired[
            "is_correct_bool_b"
        ]
    elif outcome == "conditional_accuracy":
        event_a = paired["is_correct_bool_a"]
        event_b = paired["is_correct_bool_b"]
    elif outcome == "stance_wrong":
        event_a = paired["extracted_stance_a"].eq("wrong")
        event_b = paired["extracted_stance_b"].eq("wrong")
    else:
        raise ValueError(outcome)
    return (
        event_a[eligible].astype(bool),
        event_b[eligible].astype(bool),
        excluded_missing,
        excluded_not_both_committed,
    )


def build_paired_effects(
    frame: pd.DataFrame,
    config: DomainConfig,
) -> pd.DataFrame:
    outcome_subsets = {
        "coverage": SUBSET_ORDER,
        "conditional_accuracy": VALID_ACCURACY_SUBSETS,
        "effective_success": VALID_ACCURACY_SUBSETS,
        "stance_wrong": STANCE_SUBSETS,
    }
    nonneutral = [p for p in persona_order(config) if p != "neutral"]
    indexed = {
        (str(model), str(persona), str(subset)): group.set_index("question_id")
        for (model, persona, subset), group in frame.groupby(
            ["model", "persona_folder", "subset"], sort=False
        )
    }
    labels = persona_labels(config)
    outcome_labels = {
        "coverage": "Coverage / commit rate",
        "conditional_accuracy": "Conditional accuracy",
        "effective_success": "Effective success",
        "stance_wrong": "Stance wrong",
    }
    rows: list[dict[str, object]] = []
    for outcome, subsets in outcome_subsets.items():
        for subset in subsets:
            for persona in nonneutral:
                for model in MODEL_ORDER:
                    left = indexed[(model, persona, subset)].add_suffix("_a")
                    right = indexed[(model, "neutral", subset)].add_suffix("_b")
                    paired = left.join(right, how="inner", validate="one_to_one")
                    event_a, event_b, excluded_missing, excluded_not_both = paired_events(
                        paired, outcome
                    )
                    if len(event_a) == 0:
                        raise ValueError(
                            f"Tidak ada pasangan eligible: {outcome}/{model}/{subset}/{persona}"
                        )
                    n10 = int((event_a & ~event_b).sum())
                    n01 = int((~event_a & event_b).sum())
                    discordant = n10 + n01
                    p_value = (
                        float(binomtest(n10, discordant, p=0.5).pvalue)
                        if discordant
                        else 1.0
                    )
                    rate_a = float(event_a.mean())
                    rate_b = float(event_b.mean())
                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": outcome_labels[outcome],
                            "subset": subset,
                            "subset_label": SUBSET_LABELS[subset],
                            "construct_valid": CONSTRUCT_VALID[subset],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "persona": persona,
                            "persona_label": labels[persona],
                            "persona_neutral": "neutral",
                            "n_pairs": len(event_a),
                            "expected_questions": len(right),
                            "excluded_missing_either": excluded_missing,
                            "excluded_not_both_committed": excluded_not_both,
                            "event_a": int(event_a.sum()),
                            "event_b": int(event_b.sum()),
                            "rate_a_pct": 100 * rate_a,
                            "rate_b_pct": 100 * rate_b,
                            "risk_difference_pp": 100 * (rate_a - rate_b),
                            "cohen_h": (
                                2 * math.asin(math.sqrt(rate_a))
                                - 2 * math.asin(math.sqrt(rate_b))
                            ),
                            "discordant_a_event_only": n10,
                            "discordant_b_event_only": n01,
                            "mcnemar_exact_p_value": p_value,
                            "statistical_method": "two-sided Exact McNemar",
                            "multiple_comparison_method": "Holm",
                        }
                    )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = np.nan
    result["family_size"] = 0
    for _, indices in result.groupby(["outcome", "subset", "model"]).groups.items():
        positions = list(indices)
        result.loc[positions, "holm_adjusted_p_value"] = holm_adjust(
            result.loc[positions, "mcnemar_exact_p_value"].to_numpy()
        )
        result.loc[positions, "family_size"] = len(positions)
    result["significant_holm_0_05"] = result["holm_adjusted_p_value"].lt(0.05)
    result["abs_cohen_h"] = result["cohen_h"].abs()
    result["cohen_h_magnitude"] = pd.cut(
        result["abs_cohen_h"],
        bins=[-math.inf, 0.2, 0.5, 0.8, math.inf],
        labels=["negligible", "small", "medium", "large"],
        right=False,
    )
    return result.sort_values(
        ["outcome", "subset", "model", "persona"], kind="stable"
    ).reset_index(drop=True)


def plot_effect_heatmap(
    effects: pd.DataFrame,
    config: DomainConfig,
    figure_dir: Path,
    overwrite: bool,
    *,
    outcome: str,
    subsets: list[str],
    stem_number: str,
    title: str,
    footnote: str,
) -> dict[str, str]:
    stems = {
        "coverage": "coverage_effect_holm_heatmap",
        "conditional_accuracy": "conditional_accuracy_effect_holm_heatmap",
        "effective_success": "effective_success_effect_holm_heatmap",
        "stance_wrong": "stance_wrong_effect_holm_heatmap",
    }
    stem = f"{stem_number}_ethics_{config.slug}_{stems[outcome]}"
    order = [p for p in persona_order(config) if p != "neutral"]
    labels = persona_labels(config)
    n_panels = len(subsets)
    if n_panels == 4:
        fig, axes = plt.subplots(2, 2, figsize=(29, 15), sharex=True, sharey=True)
        axes_flat = list(axes.flat)
    else:
        fig, axes = plt.subplots(
            1, n_panels, figsize=(14 * n_panels, 8), sharex=True, sharey=True
        )
        axes_flat = list(np.atleast_1d(axes))
    selected = effects[effects["outcome"].eq(outcome)]
    all_values = selected["risk_difference_pp"].to_numpy()
    max_abs = max(1.0, float(np.nanmax(np.abs(all_values))))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    for ax, subset in zip(axes_flat, subsets):
        part = selected[selected["subset"].eq(subset)]
        matrix = (
            part.pivot(index="model", columns="persona", values="risk_difference_pp")
            .reindex(index=MODEL_ORDER, columns=order)
        )
        matrix.index = axis_model_labels()
        matrix.columns = [labels[p] for p in order]
        annotations = matrix.copy().astype(object)
        sig = (
            part.pivot(index="model", columns="persona", values="significant_holm_0_05")
            .reindex(index=MODEL_ORDER, columns=order)
        )
        sig.index = axis_model_labels()
        sig.columns = [labels[p] for p in order]
        for row in matrix.index:
            for column in matrix.columns:
                value = matrix.loc[row, column]
                if pd.isna(value):
                    annotations.loc[row, column] = ""
                else:
                    marker = "*" if bool(sig.loc[row, column]) else ""
                    annotations.loc[row, column] = f"{value:+.1f}{marker}"
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            norm=norm,
            annot=annotations,
            fmt="",
            linewidths=0.45,
            linecolor="#F1F1F1",
            cbar=ax is axes_flat[-1],
            cbar_kws={"label": "Δ vs Neutral (percentage points)"},
            mask=matrix.isna(),
        )
        ax.set_title(SUBSET_LABELS[subset], pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=55)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(
        f"ETHICS {config.display_name}: {title}",
        y=1.03,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, figure_dir, stem, overwrite)


def build_manifest(
    config: DomainConfig,
    source_path: Path,
    frame: pd.DataFrame,
    outputs: dict[str, object],
) -> dict[str, object]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "domain": config.slug,
        "display_name": config.display_name,
        "source_response_table": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_rows": int(len(frame)),
        "models": MODEL_ORDER,
        "personas": persona_order(config),
        "subsets": SUBSET_ORDER,
        "accuracy_scope": {
            "subsets": VALID_ACCURACY_SUBSETS,
            "definition": "n_correct / committed",
            "coverage": "commit_rate = committed / total",
        },
        "invalid_construct_scope": {
            "subsets": STANCE_SUBSETS,
            "metrics": [
                "pct_answer_wrong",
                "pct_answer_acceptable",
                "stance_wrong_effect_vs_neutral",
            ],
            "accuracy_used": False,
        },
        "paired_effects": {
            "method": "two-sided Exact McNemar",
            "multiple_comparison": "Holm within outcome/subset/model",
            "effect_size": "risk difference in percentage points and Cohen h",
        },
        "outputs": outputs,
    }


def run_domain(config: DomainConfig, overwrite: bool) -> dict[str, object]:
    configure_plots()
    frame, source_path = load_source(config)
    frame.attrs["domain_slug"] = config.slug
    figure_dir = repo_root() / config.bundle_dir / "figures"
    table_dir = repo_root() / config.bundle_dir / "tables"
    ordered_personas = persona_order(config)

    condition_summary = build_condition_summary(
        frame, group_order=ordered_personas
    )
    all_summary = build_all_subset_summary(frame)
    category_summary = build_category_summary(frame, config)
    reason_table = build_reason_table(frame, all_summary, config)
    cramers = build_cramers_v(frame, config)
    effects = build_paired_effects(frame, config)

    table_paths = {
        "condition_summary": table_dir
        / f"ethics_{config.slug}_parity_condition_summary.csv",
        "all_subset_summary": table_dir
        / f"ethics_{config.slug}_parity_all_subset_summary.csv",
        "category_summary": table_dir
        / f"ethics_{config.slug}_parity_category_summary.csv",
        "abstain_reasons": table_dir
        / f"ethics_{config.slug}_parity_abstain_reasons.csv",
        "cramers_v": table_dir / f"ethics_{config.slug}_parity_cramers_v.csv",
        "paired_effects": table_dir
        / f"ethics_{config.slug}_parity_paired_effects_vs_neutral.csv",
    }
    write_table(table_paths["condition_summary"], condition_summary, overwrite)
    write_table(table_paths["all_subset_summary"], all_summary, overwrite)
    write_table(table_paths["category_summary"], category_summary, overwrite)
    write_table(table_paths["abstain_reasons"], reason_table, overwrite)
    write_table(table_paths["cramers_v"], cramers, overwrite)
    write_table(table_paths["paired_effects"], effects, overwrite)

    figures: dict[str, object] = {}
    distribution = plot_combined_distribution(
        frame, config, figure_dir, overwrite
    )
    if distribution is not None:
        figures["00_combined_four_subsets_distribution"] = distribution
    else:
        existing_stem = (
            figure_dir
            / f"00_ethics_{config.slug}_combined_four_subsets_distribution.png"
        )
        figures["00_combined_four_subsets_distribution"] = {
            "png": str(existing_stem),
            "status": "existing_file_preserved",
        }
    figures["01_status_composition"] = plot_status_composition(
        frame, config, figure_dir, overwrite
    )
    figures["02_status_by_persona"] = plot_status_heatmaps(
        all_summary, config, figure_dir, overwrite
    )
    figures["03_commit_rate"] = plot_commit_rate_by_subset(
        condition_summary, config, figure_dir, overwrite
    )
    figures["04_abstain_reason_share"] = plot_reason_heatmaps(
        reason_table,
        config,
        figure_dir,
        overwrite,
        metric="pct_of_abstain",
        stem_number="04",
        title="Pangsa alasan di antara abstain",
        metric_label="Setiap kolom persona dinormalisasi terhadap jumlah abstain pada model/persona.",
    )
    figures["05_abstain_reason_prevalence"] = plot_reason_heatmaps(
        reason_table,
        config,
        figure_dir,
        overwrite,
        metric="pct_of_nonmissing",
        stem_number="05",
        title="Prevalensi alasan abstain",
        metric_label="Denominator = seluruh respons non-missing pada model/persona.",
    )
    figures["06_accuracy_committed"] = plot_accuracy_or_effective(
        condition_summary,
        config,
        figure_dir,
        overwrite,
        metric="accuracy_committed",
        stem_number="06",
        title="Accuracy committed pada construct-valid subset",
        ylabel="Akurasi = n_correct / committed; hanya Commonsense dan Justice.",
    )
    figures["07_effective_success"] = plot_accuracy_or_effective(
        condition_summary,
        config,
        figure_dir,
        overwrite,
        metric="effective_success_rate_nonmissing",
        stem_number="07",
        title="Effective success pada construct-valid subset",
        ylabel="Effective success = n_correct / nonmissing; hanya Commonsense dan Justice.",
    )
    figures["08_stance_wrong"] = plot_stance_wrong(
        condition_summary, config, figure_dir, overwrite
    )
    figures["09_cramers_v"] = plot_cramers_v(
        cramers, config, figure_dir, overwrite
    )
    figures["10_coverage_effect"] = plot_effect_heatmap(
        effects,
        config,
        figure_dir,
        overwrite,
        outcome="coverage",
        subsets=SUBSET_ORDER,
        stem_number="10",
        title="Efek persona pada coverage dibanding Neutral",
        footnote="Δ = rate persona − rate Neutral; * = p Holm < 0,05; "
        "coverage memakai seluruh empat subset.",
    )
    figures["11_conditional_accuracy_effect"] = plot_effect_heatmap(
        effects,
        config,
        figure_dir,
        overwrite,
        outcome="conditional_accuracy",
        subsets=VALID_ACCURACY_SUBSETS,
        stem_number="11",
        title="Efek persona pada conditional accuracy dibanding Neutral",
        footnote="Δ = accuracy committed persona − Neutral; * = p Holm < 0,05; "
        "hanya Commonsense dan Justice.",
    )
    figures["12_effective_success_effect"] = plot_effect_heatmap(
        effects,
        config,
        figure_dir,
        overwrite,
        outcome="effective_success",
        subsets=VALID_ACCURACY_SUBSETS,
        stem_number="12",
        title="Efek persona pada effective success dibanding Neutral",
        footnote="Δ = effective success persona − Neutral; * = p Holm < 0,05; "
        "hanya Commonsense dan Justice.",
    )
    figures["13_stance_wrong_effect"] = plot_effect_heatmap(
        effects,
        config,
        figure_dir,
        overwrite,
        outcome="stance_wrong",
        subsets=STANCE_SUBSETS,
        stem_number="13",
        title="Efek persona pada stance wrong dibanding Neutral",
        footnote="Δ = stance wrong persona − Neutral; * = p Holm < 0,05; "
        "Deontology/Virtue adalah distribusi stance, bukan accuracy.",
    )

    manifest_path = table_dir / f"ethics_{config.slug}_accuracy_parity_manifest_v1.json"
    manifest = build_manifest(
        config,
        source_path,
        frame,
        {
            "tables": {key: str(path) for key, path in table_paths.items()},
            "figures": figures,
        },
    )
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest sudah ada dan dilindungi: {manifest_path}. "
            "Gunakan --overwrite bila memang diizinkan."
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "domain": config.slug,
        "source_rows": len(frame),
        "condition_summary_rows": len(condition_summary),
        "paired_effect_rows": len(effects),
        "new_figure_pngs": sum(
            1
            for value in figures.values()
            if isinstance(value, dict)
            and value.get("status") != "existing_file_preserved"
        ),
        "figure_dir": str(figure_dir),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=["religion", "regional", "all"],
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of outputs produced by this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = list(CONFIGS) if args.domain == "all" else [args.domain]
    results = [run_domain(CONFIGS[domain], args.overwrite) for domain in domains]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Build descriptive ETHICS Gender completion-token visualizations.

The script reads the validated full ETHICS Gender response table and the
construct-valid accuracy table.  It never calls an API, parses a response, or
uses ground truth for token summaries.  Accuracy in the trade-off table is
restricted to Commonsense and Justice, matching the ETHICS analysis contract.

Run from the repository root with:

    python src/benchmarks/ethics/build_gender_ethics_token_visuals_v1.py

The three new figures are written to
``notebooks/gender/ethics accuracy analysis/figures`` and the supporting tables
are written beside the existing ETHICS Gender tables. Existing files are
protected unless ``--overwrite`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
import seaborn as sns


ANALYSIS_VERSION = "ethics-gender-token-visuals-v1-20260929"
SOURCE_RESPONSE_TABLE = Path(
    "result/ETHICS/reports/Gender_Persona/_combined/"
    "parser_refinement_20260820/all_models_results_full.csv"
)
ACCURACY_TABLE = Path(
    "notebooks/gender/ethics accuracy analysis/tables/"
    "accuracy_construct_valid_subsets.csv"
)
BUNDLE_DIR = Path("notebooks/gender/ethics accuracy analysis")
FIGURE_DIR = BUNDLE_DIR / "figures"
TABLE_DIR = BUNDLE_DIR / "tables"
MANIFEST_PATH = BUNDLE_DIR / "ethics_gender_token_visuals_manifest_v1.json"

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

# Figures 14–15 are heatmaps. Their cell numbers receive two additional
# points while the post-save scaling keeps the rest of the chart unchanged.
TOKEN_ANNOTATION_SIZE = 16

PERSONA_INFO = {
    "man_explicit_1": {
        "identity": "Man",
        "variant": "Explicit",
        "category": "Man E",
    },
    "man_implicit_1": {
        "identity": "Man",
        "variant": "Implicit 1",
        "category": "Man I Avg",
    },
    "man_implicit_2": {
        "identity": "Man",
        "variant": "Implicit 2",
        "category": "Man I Avg",
    },
    "man_implicit_3": {
        "identity": "Man",
        "variant": "Implicit 3",
        "category": "Man I Avg",
    },
    "woman_explicit_1": {
        "identity": "Woman",
        "variant": "Explicit",
        "category": "Woman E",
    },
    "woman_implicit_1": {
        "identity": "Woman",
        "variant": "Implicit 1",
        "category": "Woman I Avg",
    },
    "woman_implicit_2": {
        "identity": "Woman",
        "variant": "Implicit 2",
        "category": "Woman I Avg",
    },
    "woman_implicit_3": {
        "identity": "Woman",
        "variant": "Implicit 3",
        "category": "Woman I Avg",
    },
    "neutral": {
        "identity": "Neutral",
        "variant": "Neutral",
        "category": "Neutral",
    },
}
PERSONA_ORDER = list(PERSONA_INFO)
CATEGORY_ORDER = ["Man E", "Man I Avg", "Woman E", "Woman I Avg", "Neutral"]
CATEGORY_PERSONAS = {
    category: [persona for persona, info in PERSONA_INFO.items() if info["category"] == category]
    for category in CATEGORY_ORDER
}
TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]
VALID_ACCURACY_SUBSETS = ["commonsense", "justice"]
FIGURE_STEMS = [
    "14_ethics_completion_tokens_by_category",
    "15_ethics_completion_token_delta_vs_neutral",
    "16_ethics_completion_token_accuracy_tradeoff_implicit_vs_explicit",
]

# The previous official token figures were rendered with a +1 pt post-process.
# Add two more points while keeping chart titles unchanged, matching the main
# ETHICS Gender figure bundle.
NON_TITLE_FONT_INCREMENT_PT = 3.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13.5,
            "legend.title_fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def output_paths(stem: str) -> tuple[Path, Path]:
    return FIGURE_DIR / f"{stem}.png", FIGURE_DIR / f"{stem}.svg"


def save_figure(fig: mpl.figure.Figure, stem: str, overwrite: bool) -> list[Path]:
    fig.canvas.draw()
    titles = {fig._suptitle}
    for ax in fig.axes:
        titles.update([ax.title, ax._left_title, ax._right_title])
    for text in fig.findobj(match=Text):
        if text not in titles:
            text.set_fontsize(text.get_fontsize() + NON_TITLE_FONT_INCREMENT_PT)
    paths = output_paths(stem)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Output figure sudah ada; gunakan --overwrite bila memang ingin "
                f"menulis ulang: {existing}"
            )
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=220, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return list(paths)


def write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output table sudah ada; gunakan --overwrite bila memang ingin menulis ulang: {path}"
        )
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def read_response_source(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / SOURCE_RESPONSE_TABLE
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["model", "persona_folder", "question_id", "subset", *TOKEN_COLUMNS]
    print(f"[READ] Membaca metadata token ETHICS: {path}")
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    for column in TOKEN_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["persona_folder"] = frame["persona_folder"].astype("string")
    frame["model"] = frame["model"].astype("string")
    frame["subset"] = frame["subset"].astype("string")

    expected_models = set(MODEL_ORDER)
    observed_models = set(frame["model"].dropna().unique())
    if observed_models != expected_models:
        raise ValueError(f"Model source tidak sesuai: {sorted(observed_models)}")
    observed_personas = set(frame["persona_folder"].dropna().unique())
    if observed_personas != set(PERSONA_ORDER):
        raise ValueError(f"Persona source tidak sesuai: {sorted(observed_personas)}")
    observed_subsets = set(frame["subset"].dropna().unique())
    if observed_subsets != {"commonsense", "deontology", "justice", "virtue"}:
        raise ValueError(f"Subset source tidak sesuai: {sorted(observed_subsets)}")
    if frame.duplicated(["model", "persona_folder", "question_id"]).any():
        raise ValueError("Ada kunci respons duplikat pada sumber token")

    token_valid_counts = {column: int(frame[column].notna().sum()) for column in TOKEN_COLUMNS}
    token_missing_counts = {column: int(frame[column].isna().sum()) for column in TOKEN_COLUMNS}
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "models": int(frame["model"].nunique()),
        "personas": int(frame["persona_folder"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": sorted(observed_subsets),
        "token_valid_counts": token_valid_counts,
        "token_missing_counts": token_missing_counts,
    }
    print(
        f"[READ] {metadata['rows']} baris; {metadata['models']} model; "
        f"{metadata['personas']} persona; {metadata['questions']} question_id"
    )
    print(f"[READ] Valid completion_tokens: {token_valid_counts['completion_tokens']}")
    return frame, metadata


def read_accuracy_source(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / ACCURACY_TABLE
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[READ] Membaca tabel akurasi construct-valid: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"model", "persona", "subset", "total", "committed", "n_correct", "accuracy_committed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom akurasi hilang: {missing}")
    frame = frame[frame["subset"].isin(VALID_ACCURACY_SUBSETS)].copy()
    if set(frame["subset"].unique()) != set(VALID_ACCURACY_SUBSETS):
        raise ValueError("Tabel akurasi tidak memuat Commonsense dan Justice sekaligus")
    for column in ["total", "committed", "n_correct"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    grouped = (
        frame.groupby(["model", "persona"], as_index=False, sort=False)
        .agg(
            total=("total", "sum"),
            committed=("committed", "sum"),
            n_correct=("n_correct", "sum"),
        )
    )
    grouped["accuracy_committed"] = grouped["n_correct"] / grouped["committed"]
    grouped["commit_rate"] = grouped["committed"] / grouped["total"]
    grouped["accuracy_scope"] = "commonsense+justice"
    if len(grouped) != len(MODEL_ORDER) * len(PERSONA_ORDER):
        raise ValueError(f"Jumlah baris akurasi persona tidak sesuai: {len(grouped)}")
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_rows_used": int(len(frame)),
        "grouped_rows": int(len(grouped)),
        "subsets_used": VALID_ACCURACY_SUBSETS,
    }
    print(
        f"[READ] {metadata['source_rows_used']} baris subset valid; "
        f"{metadata['grouped_rows']} grup model-persona"
    )
    return grouped, metadata


def category_token_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for category in CATEGORY_ORDER:
            personas = CATEGORY_PERSONAS[category]
            group = raw[raw["model"].eq(model) & raw["persona_folder"].isin(personas)]
            row: dict[str, object] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "category": category,
                "personas": "+".join(personas),
                "total_responses": int(len(group)),
                "subset_scope": "all_four_subsets",
                "status_scope": "all_response_rows",
            }
            for column in TOKEN_COLUMNS:
                values = group[column].dropna()
                row[f"{column}_n_valid"] = int(values.size)
                row[f"{column}_n_missing"] = int(len(group) - values.size)
                row[f"{column}_valid_rate"] = float(values.size / len(group)) if len(group) else np.nan
                row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def completion_token_delta(summary: pd.DataFrame) -> pd.DataFrame:
    means = summary.pivot(index="model", columns="category", values="completion_tokens_mean")
    valid = summary.pivot(index="model", columns="category", values="completion_tokens_n_valid")
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        neutral_mean = float(means.loc[model, "Neutral"])
        if not np.isfinite(neutral_mean) or neutral_mean == 0:
            raise ValueError(f"Completion token Neutral tidak valid untuk {model}")
        for category in CATEGORY_ORDER[:-1]:
            category_mean = float(means.loc[model, category])
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "category": category,
                    "neutral_completion_tokens_mean": neutral_mean,
                    "category_completion_tokens_mean": category_mean,
                    "neutral_completion_tokens_n_valid": int(valid.loc[model, "Neutral"]),
                    "category_completion_tokens_n_valid": int(valid.loc[model, category]),
                    "completion_token_delta_pct": (category_mean - neutral_mean) / neutral_mean * 100.0,
                    "comparison": "category_vs_neutral",
                }
            )
    return pd.DataFrame(rows)


def persona_token_summary(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["model", "persona_folder"], sort=False)[TOKEN_COLUMNS]
    means = grouped.mean().reset_index().rename(columns={"persona_folder": "persona"})
    counts = grouped.count().reset_index().rename(columns={"persona_folder": "persona"})
    result = means[["model", "persona"]].copy()
    for column in TOKEN_COLUMNS:
        result[f"{column}_mean"] = means[column]
        result[f"{column}_count"] = counts[column]
    return result


def build_tradeoff(raw: pd.DataFrame, accuracy: pd.DataFrame) -> pd.DataFrame:
    token = persona_token_summary(raw)
    token_lookup = token.set_index(["model", "persona"])
    accuracy_lookup = accuracy.set_index(["model", "persona"])
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for identity, explicit_persona in [("Man", "man_explicit_1"), ("Woman", "woman_explicit_1")]:
            explicit_token = float(token_lookup.loc[(model, explicit_persona), "completion_tokens_mean"])
            explicit_accuracy = float(accuracy_lookup.loc[(model, explicit_persona), "accuracy_committed"])
            explicit_commit_rate = float(accuracy_lookup.loc[(model, explicit_persona), "commit_rate"])
            if not np.isfinite(explicit_token) or explicit_token == 0:
                raise ValueError(f"Explicit completion token tidak valid: {model}/{explicit_persona}")
            for index in [1, 2, 3]:
                implicit_persona = f"{identity.lower()}_implicit_{index}"
                implicit_token = float(token_lookup.loc[(model, implicit_persona), "completion_tokens_mean"])
                implicit_accuracy = float(accuracy_lookup.loc[(model, implicit_persona), "accuracy_committed"])
                implicit_commit_rate = float(accuracy_lookup.loc[(model, implicit_persona), "commit_rate"])
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "identity_label": identity,
                        "implicit_variant": f"Implicit {index}",
                        "explicit_persona": explicit_persona,
                        "implicit_persona": implicit_persona,
                        "explicit_completion_tokens_mean": explicit_token,
                        "implicit_completion_tokens_mean": implicit_token,
                        "token_delta_pct": (implicit_token - explicit_token) / explicit_token * 100.0,
                        "explicit_accuracy_committed": explicit_accuracy,
                        "implicit_accuracy_committed": implicit_accuracy,
                        "accuracy_delta_pp": (implicit_accuracy - explicit_accuracy) * 100.0,
                        "explicit_commit_rate": explicit_commit_rate,
                        "implicit_commit_rate": implicit_commit_rate,
                        "accuracy_scope": "commonsense+justice",
                        "token_scope": "all_four_subsets_with_valid_completion_tokens",
                    }
                )
    return pd.DataFrame(rows)


def render_category_heatmap(summary: pd.DataFrame, overwrite: bool) -> list[Path]:
    matrix = summary.pivot(index="model_label", columns="category", values="completion_tokens_mean").reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=CATEGORY_ORDER
    )
    annotations = matrix.round(0).astype("Int64").astype(str)
    fig, ax = plt.subplots(figsize=(14.8, 7.6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": TOKEN_ANNOTATION_SIZE, "fontweight": "normal"},
        cmap="YlGnBu",
        linewidths=1.0,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Rata-rata completion token"},
        ax=ax,
    )
    ax.set_title("ETHICS Gender: Completion Token per Persona")
    ax.set_xlabel("Kategori persona")
    ax.set_ylabel("Model")
    ax.set_xticklabels(["M:E", "M:I Avg", "W:E", "W:I Avg", "N"], rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return save_figure(fig, FIGURE_STEMS[0], overwrite)


def render_token_delta(delta: pd.DataFrame, overwrite: bool) -> list[Path]:
    matrix = delta.pivot(index="model_label", columns="category", values="completion_token_delta_pct").reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=CATEGORY_ORDER[:-1]
    )
    annotations = matrix.map(lambda value: "" if pd.isna(value) else f"{value:+.0f}")
    limit = max(5.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))) * 1.08)
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": TOKEN_ANNOTATION_SIZE, "fontweight": "normal"},
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=1.0,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Perubahan completion token terhadap netral (%)"},
        ax=ax,
    )
    ax.set_title("ETHICS Gender: Perubahan Completion Token terhadap Netral")
    ax.set_xlabel("Kategori persona")
    ax.set_ylabel("Model")
    ax.set_xticklabels(["M:E", "M:I Avg", "W:E", "W:I Avg"], rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return save_figure(fig, FIGURE_STEMS[1], overwrite)


def spearman_value(frame: pd.DataFrame) -> float:
    if len(frame) < 2 or frame["token_delta_pct"].nunique() < 2 or frame["accuracy_delta_pp"].nunique() < 2:
        return float("nan")
    return float(frame["token_delta_pct"].rank().corr(frame["accuracy_delta_pp"].rank()))


def render_tradeoff(tradeoff: pd.DataFrame, overwrite: bool) -> list[Path]:
    identity_colors = {"Man": "#0047AB", "Woman": "#B80F2E"}
    markers = {"Implicit 1": "o", "Implicit 2": "s", "Implicit 3": "D"}
    x_values = tradeoff["token_delta_pct"]
    y_values = tradeoff["accuracy_delta_pp"]
    x_min = min(-20.0, float(x_values.min()) - 15.0)
    x_max = max(20.0, float(x_values.max()) + 15.0)
    y_span = max(1.0, float(y_values.max() - y_values.min()))
    y_padding = max(0.35, 0.10 * y_span)
    y_min = min(-0.35, float(y_values.min()) - y_padding)
    y_max = max(0.35, float(y_values.max()) + y_padding)

    fig, axes = plt.subplots(2, 3, figsize=(20.5, 12), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        model_rows = tradeoff[tradeoff["model"].eq(model)]
        ax.axhline(0, color="#30343B", linewidth=1.4, linestyle="--", zorder=1)
        ax.axvline(0, color="#30343B", linewidth=1.4, linestyle="--", zorder=1)
        ax.scatter([0], [0], marker="X", s=130, color="#30343B", edgecolor="#FCFBF8", linewidth=1.1, zorder=5)
        for identity, color in identity_colors.items():
            identity_rows = model_rows[model_rows["identity_label"].eq(identity)]
            for variant, marker in markers.items():
                point = identity_rows[identity_rows["implicit_variant"].eq(variant)]
                ax.scatter(
                    point["token_delta_pct"],
                    point["accuracy_delta_pp"],
                    marker=marker,
                    s=120,
                    color=color,
                    edgecolor="#FCFBF8",
                    linewidth=1.2,
                    zorder=4,
                )
        rho = spearman_value(model_rows)
        rho_label = "NA" if np.isnan(rho) else f"{rho:+.3f}"
        ax.set_title(f"{MODEL_LABELS[model]}\nSpearman's ρ = {rho_label}", fontsize=17.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="#D7DBDF", linewidth=0.8, alpha=0.8)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#FCFBF8", markersize=10, label=identity)
        for identity, color in identity_colors.items()
    ] + [
        Line2D([0], [0], marker=marker, color="#30343B", linestyle="none", markersize=9, label=variant)
        for variant, marker in markers.items()
    ] + [
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#30343B", markersize=10, label="Explicit baseline")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=6,
        frameon=True,
        title="Identitas, variasi implisit, dan baseline",
    )
    overall_rho = spearman_value(tradeoff)
    rho_label = "NA" if np.isnan(overall_rho) else f"{overall_rho:+.3f}"
    fig.suptitle(
        "ETHICS Gender: Trade-off Completion Token dan Akurasi",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("Perubahan rata-rata completion token terhadap eksplisit (%)", y=0.035)
    fig.supylabel("Perubahan accuracy_committed terhadap eksplisit (poin persentase)", x=0.015)
    fig.text(
        0.5,
        0.008,
        f"Spearman pooled ρ = {rho_label} (n = {len(tradeoff)}); deskriptif, bukan inferensi kausal. "
        "Akurasi hanya Commonsense + Justice.",
        ha="center",
        va="bottom",
        fontsize=15,
        color="#30343B",
    )
    fig.tight_layout(rect=(0.035, 0.06, 1, 0.90), h_pad=2.0, w_pad=1.5)
    return save_figure(fig, FIGURE_STEMS[2], overwrite)


def validate_outputs(summary: pd.DataFrame, delta: pd.DataFrame, tradeoff: pd.DataFrame) -> None:
    if len(summary) != len(MODEL_ORDER) * len(CATEGORY_ORDER):
        raise AssertionError(f"Ringkasan kategori tidak lengkap: {len(summary)}")
    if len(delta) != len(MODEL_ORDER) * (len(CATEGORY_ORDER) - 1):
        raise AssertionError(f"Ringkasan delta tidak lengkap: {len(delta)}")
    if len(tradeoff) != len(MODEL_ORDER) * 2 * 3:
        raise AssertionError(f"Tabel trade-off tidak lengkap: {len(tradeoff)}")
    if not summary["completion_tokens_mean"].notna().all():
        raise AssertionError("Ada rata-rata completion token yang kosong")
    if not np.isfinite(delta["completion_token_delta_pct"]).all():
        raise AssertionError("Ada delta completion token yang tidak finite")
    if not np.isfinite(tradeoff[["token_delta_pct", "accuracy_delta_pp"]].to_numpy()).all():
        raise AssertionError("Ada nilai trade-off yang tidak finite")
    if set(tradeoff["accuracy_scope"]) != {"commonsense+justice"}:
        raise AssertionError("Scope akurasi trade-off tidak construct-valid")


def build(root: Path, overwrite: bool) -> dict[str, object]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw, response_metadata = read_response_source(root)
    accuracy, accuracy_metadata = read_accuracy_source(root)
    summary = category_token_summary(raw)
    delta = completion_token_delta(summary)
    tradeoff = build_tradeoff(raw, accuracy)
    validate_outputs(summary, delta, tradeoff)

    table_paths = {
        "category_summary": TABLE_DIR / "ethics_completion_token_category_summary.csv",
        "delta_vs_neutral": TABLE_DIR / "ethics_completion_token_delta_vs_neutral_pct.csv",
        "accuracy_tradeoff": TABLE_DIR / "ethics_completion_token_accuracy_tradeoff_implicit_vs_explicit.csv",
    }
    write_csv(summary, table_paths["category_summary"], overwrite)
    write_csv(delta, table_paths["delta_vs_neutral"], overwrite)
    write_csv(tradeoff, table_paths["accuracy_tradeoff"], overwrite)

    configure_plots()
    figure_paths: list[Path] = []
    figure_paths.extend(render_category_heatmap(summary, overwrite))
    figure_paths.extend(render_token_delta(delta, overwrite))
    figure_paths.extend(render_tradeoff(tradeoff, overwrite))

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "api_calls": 0,
        "rescore": False,
        "source_response_table": response_metadata,
        "source_accuracy_table": accuracy_metadata,
        "token_summary_scope": {
            "subsets": "commonsense+deontology+justice+virtue",
            "statuses": "all response rows; rows without token metadata remain in denominator",
            "metric": "completion_tokens",
            "aggregation": "pooled row-level mean; implicit averages are not averages of percentages",
        },
        "accuracy_tradeoff_scope": {
            "subsets": VALID_ACCURACY_SUBSETS,
            "metric": "accuracy_committed",
            "coverage_required_in_interpretation": True,
        },
        "tables": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in table_paths.items()
        },
        "figures": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in figure_paths
        },
    }
    if MANIFEST_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest sudah ada; gunakan --overwrite bila memang ingin menulis ulang: {MANIFEST_PATH}"
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[WRITE] {len(table_paths)} tabel baru ditulis ke {TABLE_DIR}")
    print(f"[WRITE] {len(figure_paths)} figure baru ditulis ke {FIGURE_DIR}")
    print(f"[WRITE] Manifest: {MANIFEST_PATH}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa artefak token yang dibuat skrip ini; file visual ETHICS lama tetap tidak disentuh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(repo_root(), args.overwrite)
    print(
        f"[DONE] {len(manifest['figures'])} file figure dan {len(manifest['tables'])} file tabel; "
        f"analysis_version={manifest['analysis_version']}"
    )


if __name__ == "__main__":
    main()

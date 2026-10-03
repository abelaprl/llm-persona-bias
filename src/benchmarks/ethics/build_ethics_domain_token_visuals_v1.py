"""Build completion-token figures for ETHICS Religion and Regional Cultural.

The token summaries use all four ETHICS subsets and every response row with
available token metadata. Accuracy in the trade-off figure is read from the
validated construct-valid table and is restricted to Commonsense + Justice.
The script does not call an API, parse a response, or use ground truth for the
token summaries.

Run from the repository root:

    python src/benchmarks/ethics/build_ethics_domain_token_visuals_v1.py --domain religion
    python src/benchmarks/ethics/build_ethics_domain_token_visuals_v1.py --domain regional
    python src/benchmarks/ethics/build_ethics_domain_token_visuals_v1.py --domain all

The script writes new PNG/SVG figures and supporting CSV/JSON files beside the
existing domain ETHICS figures. Existing files are protected unless
--overwrite is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


ANALYSIS_VERSION = "ethics-domain-token-visuals-v1-20260929"
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
TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]
VALID_ACCURACY_SUBSETS = ["commonsense", "justice"]
EXPECTED_SUBSETS = {"commonsense", "deontology", "justice", "virtue"}


@dataclass(frozen=True)
class DomainConfig:
    slug: str
    display_name: str
    source_response_table: Path
    accuracy_table: Path
    bundle_dir: Path
    identities: tuple[tuple[str, str, str], ...]


CONFIGS = {
    "religion": DomainConfig(
        slug="religion",
        display_name="Religion",
        source_response_table=Path(
            "result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv"
        ),
        accuracy_table=Path(
            "notebooks/religion/ethics accuracy analysis/tables/"
            "ethics_accuracy_by_model_persona_category.csv"
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
        accuracy_table=Path(
            "notebooks/regional/ethics accuracy analysis/tables/"
            "ethics_accuracy_by_model_persona_category.csv"
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


@dataclass(frozen=True)
class CategoryDef:
    key: str
    label: str
    short_label: str
    identity_key: str
    identity_label: str
    kind: str
    personas: tuple[str, ...]


def categories_for(config: DomainConfig) -> tuple[CategoryDef, ...]:
    categories: list[CategoryDef] = []
    for identity_key, identity_label, short_label in config.identities:
        categories.append(
            CategoryDef(
                key=f"{identity_key}_explicit",
                label=f"{identity_label} Explicit",
                short_label=f"{short_label} E",
                identity_key=identity_key,
                identity_label=identity_label,
                kind="explicit",
                personas=(f"{identity_key}_explicit",),
            )
        )
        categories.append(
            CategoryDef(
                key=f"{identity_key}_implicit_avg",
                label=f"{identity_label} Implicit Avg",
                short_label=f"{short_label} I",
                identity_key=identity_key,
                identity_label=identity_label,
                kind="implicit_avg",
                personas=tuple(f"{identity_key}_implicit_{index}" for index in [1, 2, 3]),
            )
        )
    categories.append(
        CategoryDef(
            key="neutral",
            label="Neutral",
            short_label="Neutral",
            identity_key="neutral",
            identity_label="Neutral",
            kind="neutral",
            personas=("neutral",),
        )
    )
    return tuple(categories)


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


def figure_stems(config: DomainConfig) -> tuple[str, str, str]:
    prefix = f"ethics_{config.slug}"
    return (
        f"14_{prefix}_completion_tokens_by_category",
        f"15_{prefix}_completion_token_delta_vs_neutral",
        f"16_{prefix}_completion_token_accuracy_tradeoff_implicit_vs_explicit",
    )


def output_paths(config: DomainConfig, stem: str) -> tuple[Path, Path]:
    figure_dir = config.bundle_dir / "figures"
    return figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"


def save_figure(
    fig: mpl.figure.Figure,
    config: DomainConfig,
    stem: str,
    overwrite: bool,
) -> list[Path]:
    paths = output_paths(config, stem)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Output figure sudah ada; gunakan --overwrite bila memang ingin "
                f"menulis ulang: {existing}"
            )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def read_response_source(
    root: Path,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / config.source_response_table
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["model", "persona_folder", "question_id", "subset", *TOKEN_COLUMNS]
    print(f"[READ] {config.display_name}: metadata token dari {path}")
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    for column in TOKEN_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["persona_folder", "model", "subset"]:
        frame[column] = frame[column].astype("string")

    expected_models = set(MODEL_ORDER)
    observed_models = set(frame["model"].dropna().unique())
    if observed_models != expected_models:
        raise ValueError(
            f"Model source {config.slug} tidak sesuai: {sorted(observed_models)}"
        )
    expected_personas = {
        persona for category in categories for persona in category.personas
    }
    observed_personas = set(frame["persona_folder"].dropna().unique())
    if observed_personas != expected_personas:
        raise ValueError(
            f"Persona source {config.slug} tidak sesuai: {sorted(observed_personas)}"
        )
    observed_subsets = set(frame["subset"].dropna().unique())
    if observed_subsets != EXPECTED_SUBSETS:
        raise ValueError(
            f"Subset source {config.slug} tidak sesuai: {sorted(observed_subsets)}"
        )
    if frame.duplicated(["model", "persona_folder", "question_id"]).any():
        raise ValueError(f"Ada kunci respons duplikat pada sumber {config.slug}")

    token_valid_counts = {
        column: int(frame[column].notna().sum()) for column in TOKEN_COLUMNS
    }
    token_missing_counts = {
        column: int(frame[column].isna().sum()) for column in TOKEN_COLUMNS
    }
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
    print(
        f"[READ] Valid completion_tokens {config.slug}: "
        f"{token_valid_counts['completion_tokens']}"
    )
    return frame, metadata


def read_accuracy_source(
    root: Path,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = root / config.accuracy_table
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[READ] {config.display_name}: tabel akurasi dari {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "model",
        "category",
        "total",
        "committed",
        "n_correct",
        "commit_rate",
        "accuracy_committed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom akurasi {config.slug} hilang: {missing}")

    category_labels = {category.label for category in categories}
    if set(frame["category"].dropna().unique()) != category_labels:
        raise ValueError(
            f"Kategori akurasi {config.slug} tidak sesuai: "
            f"{sorted(frame['category'].dropna().unique())}"
        )
    if set(frame["model"].dropna().unique()) != set(MODEL_ORDER):
        raise ValueError(f"Model pada tabel akurasi {config.slug} tidak lengkap")
    if len(frame) != len(MODEL_ORDER) * len(categories):
        raise ValueError(
            f"Jumlah baris akurasi {config.slug} tidak sesuai: {len(frame)}"
        )

    numeric_columns = [
        "total",
        "committed",
        "n_correct",
        "commit_rate",
        "accuracy_committed",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.duplicated(["model", "category"]).any():
        raise ValueError(f"Ada kunci model-kategori duplikat pada akurasi {config.slug}")
    expected_accuracy = frame["n_correct"] / frame["committed"]
    if not np.allclose(
        frame["accuracy_committed"].to_numpy(dtype=float),
        expected_accuracy.to_numpy(dtype=float),
        atol=1e-10,
        equal_nan=False,
    ):
        raise ValueError(f"accuracy_committed tidak konsisten pada {config.slug}")

    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_rows_used": int(len(frame)),
        "grouped_rows": int(len(frame)),
        "subsets_used": VALID_ACCURACY_SUBSETS,
        "metric": "accuracy_committed",
    }
    print(f"[READ] {metadata['source_rows_used']} baris kategori akurasi")
    return frame, metadata


def category_token_summary(
    raw: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for category in categories:
            group = raw[
                raw["model"].eq(model)
                & raw["persona_folder"].isin(category.personas)
            ]
            row: dict[str, object] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "domain": config.slug,
                "category": category.label,
                "category_short": category.short_label,
                "identity": category.identity_key,
                "identity_label": category.identity_label,
                "category_kind": category.kind,
                "personas": "+".join(category.personas),
                "total_responses": int(len(group)),
                "subset_scope": "all_four_subsets",
                "status_scope": "all_response_rows",
            }
            for column in TOKEN_COLUMNS:
                values = group[column].dropna()
                row[f"{column}_n_valid"] = int(values.size)
                row[f"{column}_n_missing"] = int(len(group) - values.size)
                row[f"{column}_valid_rate"] = (
                    float(values.size / len(group)) if len(group) else np.nan
                )
                row[f"{column}_mean"] = (
                    float(values.mean()) if len(values) else np.nan
                )
                row[f"{column}_median"] = (
                    float(values.median()) if len(values) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def completion_token_delta(
    summary: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> pd.DataFrame:
    means = summary.pivot(
        index="model", columns="category", values="completion_tokens_mean"
    )
    valid = summary.pivot(
        index="model", columns="category", values="completion_tokens_n_valid"
    )
    neutral_label = next(
        category.label for category in categories if category.kind == "neutral"
    )
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        neutral_mean = float(means.loc[model, neutral_label])
        if not np.isfinite(neutral_mean) or neutral_mean == 0:
            raise ValueError(
                f"Completion token Neutral tidak valid: {config.slug}/{model}"
            )
        for category in categories:
            if category.kind == "neutral":
                continue
            category_mean = float(means.loc[model, category.label])
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "domain": config.slug,
                    "identity": category.identity_key,
                    "identity_label": category.identity_label,
                    "category": category.label,
                    "category_short": category.short_label,
                    "neutral_completion_tokens_mean": neutral_mean,
                    "category_completion_tokens_mean": category_mean,
                    "neutral_completion_tokens_n_valid": int(
                        valid.loc[model, neutral_label]
                    ),
                    "category_completion_tokens_n_valid": int(
                        valid.loc[model, category.label]
                    ),
                    "completion_token_delta_pct": (
                        (category_mean - neutral_mean) / neutral_mean * 100.0
                    ),
                    "comparison": "category_vs_neutral",
                }
            )
    return pd.DataFrame(rows)


def build_tradeoff(
    summary: pd.DataFrame,
    accuracy: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> pd.DataFrame:
    token_lookup = summary.set_index(["model", "category"])
    accuracy_lookup = accuracy.set_index(["model", "category"])
    category_by_key = {category.key: category for category in categories}
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for identity_key, identity_label, _short_label in config.identities:
            explicit = category_by_key[f"{identity_key}_explicit"]
            implicit = category_by_key[f"{identity_key}_implicit_avg"]
            explicit_token = float(
                token_lookup.loc[
                    (model, explicit.label), "completion_tokens_mean"
                ]
            )
            implicit_token = float(
                token_lookup.loc[
                    (model, implicit.label), "completion_tokens_mean"
                ]
            )
            explicit_accuracy = float(
                accuracy_lookup.loc[
                    (model, explicit.label), "accuracy_committed"
                ]
            )
            implicit_accuracy = float(
                accuracy_lookup.loc[
                    (model, implicit.label), "accuracy_committed"
                ]
            )
            explicit_commit_rate = float(
                accuracy_lookup.loc[(model, explicit.label), "commit_rate"]
            )
            implicit_commit_rate = float(
                accuracy_lookup.loc[(model, implicit.label), "commit_rate"]
            )
            if not np.isfinite(explicit_token) or explicit_token == 0:
                raise ValueError(
                    f"Explicit completion token tidak valid: "
                    f"{config.slug}/{model}/{identity_key}"
                )
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "domain": config.slug,
                    "identity": identity_key,
                    "identity_label": identity_label,
                    "explicit_category": explicit.label,
                    "implicit_category": implicit.label,
                    "explicit_completion_tokens_mean": explicit_token,
                    "implicit_completion_tokens_mean": implicit_token,
                    "token_delta_pct": (
                        (implicit_token - explicit_token) / explicit_token * 100.0
                    ),
                    "explicit_accuracy_committed": explicit_accuracy,
                    "implicit_accuracy_committed": implicit_accuracy,
                    "accuracy_delta_pp": (
                        (implicit_accuracy - explicit_accuracy) * 100.0
                    ),
                    "explicit_commit_rate": explicit_commit_rate,
                    "implicit_commit_rate": implicit_commit_rate,
                    "accuracy_scope": "commonsense+justice",
                    "token_scope": "all_four_subsets_with_valid_completion_tokens",
                }
            )
    return pd.DataFrame(rows)


def footnote_text() -> str:
    return (
        "E = eksplisit; I = pooled dari Implicit 1–3. "
        "Token diringkas dari seluruh empat subset; baris tanpa metadata token "
        "tetap dihitung dalam denominator valid-rate."
    )


def render_category_heatmap(
    summary: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
    overwrite: bool,
) -> list[Path]:
    matrix = summary.pivot(
        index="model_label", columns="category", values="completion_tokens_mean"
    ).reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=[category.label for category in categories],
    )
    annotations = matrix.round(0).astype("Int64").astype(str)
    fig, ax = plt.subplots(figsize=(18.5, 7.8))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        linewidths=1.0,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Rata-rata completion token"},
        ax=ax,
    )
    ax.set_title(
        f"ETHICS {config.display_name}: Rata-rata Completion Token per Kategori Persona"
    )
    ax.set_xlabel("Kategori persona")
    ax.set_ylabel("Model")
    ax.set_xticklabels(
        [category.short_label for category in categories],
        rotation=0,
        ha="center",
    )
    ax.set_yticklabels(
        [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER],
        rotation=0,
    )
    fig.text(0.5, 0.012, footnote_text(), ha="center", va="bottom", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return save_figure(fig, config, figure_stems(config)[0], overwrite)


def render_token_delta(
    delta: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
    overwrite: bool,
) -> list[Path]:
    non_neutral = [category for category in categories if category.kind != "neutral"]
    matrix = delta.pivot(
        index="model_label", columns="category", values="completion_token_delta_pct"
    ).reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER],
        columns=[category.label for category in non_neutral],
    )
    annotations = matrix.map(
        lambda value: "" if pd.isna(value) else f"{value:+.0f}"
    )
    limit = max(
        5.0,
        float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))) * 1.08,
    )
    fig, ax = plt.subplots(figsize=(17.5, 7.8))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=1.0,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Perubahan completion token terhadap netral (%)"},
        ax=ax,
    )
    ax.set_title(
        f"ETHICS {config.display_name}: Completion Token Persona terhadap Netral"
    )
    ax.set_xlabel("Kategori persona")
    ax.set_ylabel("Model")
    ax.set_xticklabels(
        [category.short_label for category in non_neutral],
        rotation=0,
        ha="center",
    )
    ax.set_yticklabels(
        [MODEL_AXIS_LABELS[MODEL_LABELS[model]] for model in MODEL_ORDER],
        rotation=0,
    )
    fig.text(0.5, 0.012, footnote_text(), ha="center", va="bottom", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return save_figure(fig, config, figure_stems(config)[1], overwrite)


def spearman_value(frame: pd.DataFrame) -> float:
    if (
        len(frame) < 2
        or frame["token_delta_pct"].nunique() < 2
        or frame["accuracy_delta_pp"].nunique() < 2
    ):
        return float("nan")
    return float(
        frame["token_delta_pct"].rank().corr(frame["accuracy_delta_pp"].rank())
    )


def render_tradeoff(
    tradeoff: pd.DataFrame,
    config: DomainConfig,
    overwrite: bool,
) -> list[Path]:
    identity_colors = {
        identity_key: color
        for identity_key, color in zip(
            [identity[0] for identity in config.identities],
            ["#0047AB", "#B80F2E", "#008B8B", "#D97706"],
        )
    }
    identity_labels = {
        identity_key: identity_label
        for identity_key, identity_label, _short_label in config.identities
    }
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
        ax.scatter(
            [0],
            [0],
            marker="X",
            s=130,
            color="#30343B",
            edgecolor="#FCFBF8",
            linewidth=1.1,
            zorder=5,
        )
        for identity_key, color in identity_colors.items():
            point = model_rows[model_rows["identity"].eq(identity_key)]
            ax.scatter(
                point["token_delta_pct"],
                point["accuracy_delta_pp"],
                marker="o",
                s=125,
                color=color,
                edgecolor="#FCFBF8",
                linewidth=1.2,
                zorder=4,
            )
        rho = spearman_value(model_rows)
        rho_label = "NA" if np.isnan(rho) else f"{rho:+.3f}"
        ax.set_title(
            f"{MODEL_LABELS[model]}\nSpearman's rho = {rho_label} (n = 4; deskriptif)",
            fontsize=15.5,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="#D7DBDF", linewidth=0.8, alpha=0.8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#FCFBF8",
            markersize=10,
            label=identity_labels[identity_key],
        )
        for identity_key, color in identity_colors.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="#30343B",
            markersize=10,
            label="Eksplisit baseline",
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=True,
        title="Identitas dan baseline eksplisit",
    )
    overall_rho = spearman_value(tradeoff)
    rho_label = "NA" if np.isnan(overall_rho) else f"{overall_rho:+.3f}"
    fig.suptitle(
        f"ETHICS {config.display_name}: Completion Token–Accuracy Trade-off "
        "untuk Implicit Avg vs Explicit",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel(
        "Perubahan rata-rata completion token terhadap eksplisit (%)",
        y=0.035,
    )
    fig.supylabel(
        "Perubahan accuracy_committed terhadap eksplisit (poin persentase)",
        x=0.015,
    )
    fig.text(
        0.5,
        0.008,
        f"Spearman pooled rho = {rho_label} (n = {len(tradeoff)}); deskriptif, "
        "bukan inferensi kausal. Akurasi hanya Commonsense + Justice.",
        ha="center",
        va="bottom",
        fontsize=13,
        color="#30343B",
    )
    fig.tight_layout(rect=(0.035, 0.06, 1, 0.90), h_pad=2.0, w_pad=1.5)
    return save_figure(fig, config, figure_stems(config)[2], overwrite)


def validate_outputs(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    tradeoff: pd.DataFrame,
    config: DomainConfig,
    categories: tuple[CategoryDef, ...],
) -> None:
    if len(summary) != len(MODEL_ORDER) * len(categories):
        raise AssertionError(
            f"Ringkasan kategori {config.slug} tidak lengkap: {len(summary)}"
        )
    if len(delta) != len(MODEL_ORDER) * (len(categories) - 1):
        raise AssertionError(
            f"Ringkasan delta {config.slug} tidak lengkap: {len(delta)}"
        )
    if len(tradeoff) != len(MODEL_ORDER) * len(config.identities):
        raise AssertionError(
            f"Tabel trade-off {config.slug} tidak lengkap: {len(tradeoff)}"
        )
    if not summary["completion_tokens_mean"].notna().all():
        raise AssertionError(
            f"Ada rata-rata completion token kosong pada {config.slug}"
        )
    if not np.isfinite(delta["completion_token_delta_pct"]).all():
        raise AssertionError(
            f"Ada delta completion token tidak finite pada {config.slug}"
        )
    if not np.isfinite(
        tradeoff[["token_delta_pct", "accuracy_delta_pp"]].to_numpy()
    ).all():
        raise AssertionError(f"Ada nilai trade-off tidak finite pada {config.slug}")
    if set(tradeoff["accuracy_scope"]) != {"commonsense+justice"}:
        raise AssertionError(f"Scope akurasi trade-off salah pada {config.slug}")


def build(config: DomainConfig, root: Path, overwrite: bool) -> dict[str, object]:
    categories = categories_for(config)
    figure_dir = config.bundle_dir / "figures"
    table_dir = config.bundle_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw, response_metadata = read_response_source(root, config, categories)
    accuracy, accuracy_metadata = read_accuracy_source(root, config, categories)
    summary = category_token_summary(raw, config, categories)
    delta = completion_token_delta(summary, config, categories)
    tradeoff = build_tradeoff(summary, accuracy, config, categories)
    validate_outputs(summary, delta, tradeoff, config, categories)

    table_paths = {
        "category_summary": table_dir
        / f"ethics_{config.slug}_completion_token_category_summary.csv",
        "delta_vs_neutral": table_dir
        / f"ethics_{config.slug}_completion_token_delta_vs_neutral_pct.csv",
        "accuracy_tradeoff": table_dir
        / f"ethics_{config.slug}_completion_token_accuracy_tradeoff_implicit_vs_explicit.csv",
    }
    write_csv(summary, table_paths["category_summary"], overwrite)
    write_csv(delta, table_paths["delta_vs_neutral"], overwrite)
    write_csv(tradeoff, table_paths["accuracy_tradeoff"], overwrite)

    configure_plots()
    figure_paths: list[Path] = []
    figure_paths.extend(render_category_heatmap(summary, config, categories, overwrite))
    figure_paths.extend(render_token_delta(delta, config, categories, overwrite))
    figure_paths.extend(render_tradeoff(tradeoff, config, overwrite))

    manifest_path = config.bundle_dir / f"ethics_{config.slug}_token_visuals_manifest_v1.json"
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "domain": config.slug,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "api_calls": 0,
        "rescore": False,
        "source_response_table": response_metadata,
        "source_accuracy_table": accuracy_metadata,
        "category_definitions": [
            {
                "key": category.key,
                "label": category.label,
                "short_label": category.short_label,
                "identity_key": category.identity_key,
                "identity_label": category.identity_label,
                "kind": category.kind,
                "personas": list(category.personas),
            }
            for category in categories
        ],
        "token_summary_scope": {
            "subsets": sorted(EXPECTED_SUBSETS),
            "statuses": (
                "all response rows; rows without token metadata remain in "
                "the denominator"
            ),
            "metric": "completion_tokens",
            "aggregation": (
                "pooled row-level mean; implicit averages pool Implicit 1-3 "
                "and are not averages of percentages"
            ),
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
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest sudah ada; gunakan --overwrite bila memang ingin menulis ulang: "
            f"{manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {len(table_paths)} tabel {config.slug} ditulis ke {table_dir}")
    print(f"[WRITE] {len(figure_paths)} figure {config.slug} ditulis ke {figure_dir}")
    print(f"[WRITE] Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["religion", "regional", "all"],
        required=True,
        help="Domain yang dibuatkan visual token.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa hanya artefak token domain yang dibuat skrip ini.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    domains = list(CONFIGS) if args.domain == "all" else [args.domain]
    for domain in domains:
        manifest = build(CONFIGS[domain], root, args.overwrite)
        print(
            f"[DONE] {domain}: {len(manifest['figures'])} file figure dan "
            f"{len(manifest['tables'])} file tabel; "
            f"analysis_version={manifest['analysis_version']}"
        )


if __name__ == "__main__":
    main()

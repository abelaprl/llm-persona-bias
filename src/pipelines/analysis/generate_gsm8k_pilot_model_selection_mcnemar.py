from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Plotting libraries are imported only after the statistical tables have been
# written. Their font/style initialization can be slow on a fresh Windows
# environment and should not delay the auditable McNemar outputs.
mpl = None
plt = None
sns = None


GSM8K_QUESTION_IDS = tuple(range(1, 1320))

PERSONAS = [
    "neutral",
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
]

COMPARISON_PERSONAS = [persona for persona in PERSONAS if persona != "neutral"]

MODELS = [
    "allenai_olmo-3-32b-think",
    "deepseek_deepseek-v3.2",
    "openai_gpt-oss-20b",
    "openai_gpt-oss-120b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "qwen_qwen3-235b-a22b-2507",
    "google_gemini-2.5-flash",
    "google_gemma-3-27b-it",
    "google_gemma-2-9b-it",
    "meta-llama_llama-3.3-70b-instruct",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "allenai_olmo-3.1-32b-instruct",
    "deepseek_deepseek-r1-distill-qwen-32b",
]

MODEL_LABELS = {
    "allenai_olmo-3-32b-think": "OLMo 3 32B Think",
    "deepseek_deepseek-v3.2": "DeepSeek V3.2",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "qwen_qwen3-235b-a22b-2507": "Qwen3 235B-A22B",
    "google_gemini-2.5-flash": "Gemini 2.5 Flash",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "google_gemma-2-9b-it": "Gemma 2 9B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "allenai_olmo-3.1-32b-instruct": "OLMo 3.1 32B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "moonshotai_kimi-k2-thinking": "Kimi K2 Thinking",
    "arcee-ai_trinity-large-thinking": "Trinity Large Thinking",
}

SELECTED_MODELS = {
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.3-70b-instruct",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
}

PERSONA_LABELS = {
    "man_explicit_1": "Man Explicit",
    "man_implicit_1": "Man Implicit 1",
    "man_implicit_2": "Man Implicit 2",
    "man_implicit_3": "Man Implicit 3",
    "woman_explicit_1": "Woman Explicit",
    "woman_implicit_1": "Woman Implicit 1",
    "woman_implicit_2": "Woman Implicit 2",
    "woman_implicit_3": "Woman Implicit 3",
}

PERSONA_SHORT = {
    "man_explicit_1": "M:E",
    "man_implicit_1": "M:I1",
    "man_implicit_2": "M:I2",
    "man_implicit_3": "M:I3",
    "woman_explicit_1": "W:E",
    "woman_implicit_1": "W:I1",
    "woman_implicit_2": "W:I2",
    "woman_implicit_3": "W:I3",
}

# These twelve models have rebuilt GSM8K reports under result/reports. Ten expose
# one file per persona, while two expose all nine personas in one combined file.
REBUILT_REPORT_MODELS = {
    "allenai_olmo-3-32b-think",
    "deepseek_deepseek-v3.2",
    "openai_gpt-oss-20b",
    "openai_gpt-oss-120b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "qwen_qwen3-235b-a22b-2507",
    "google_gemini-2.5-flash",
    "google_gemma-3-27b-it",
    "google_gemma-2-9b-it",
    "meta-llama_llama-3.3-70b-instruct",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
}

ROOT_COMBINED_MODELS = {
    "deepseek_deepseek-v3.2": "result_deepseek_deepseek-v3_2.csv",
    "openai_gpt-oss-20b": "result_openai_gpt-oss-20b.csv",
}

ROOT_PERSONA_MODELS = REBUILT_REPORT_MODELS - set(ROOT_COMBINED_MODELS)

ALL_RESULTS_MODELS = {
    "allenai_olmo-3.1-32b-instruct",
    "deepseek_deepseek-r1-distill-qwen-32b",
}

COMBINED_REPORT_MODELS: set[str] = set()


def parse_bool(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {"true", "1"}


def question_id_from_row(row: dict[str, str]) -> int | None:
    file_value = row.get("file", "")
    match = re.search(r"gsm8k_(\d+)\.json", file_value)
    if match:
        return int(match.group(1))
    try:
        return int(float(row.get("question_id", "")))
    except (TypeError, ValueError):
        return None


def add_outcome(
    records: dict[tuple[str, str, int], dict[str, Any]],
    *,
    model: str,
    persona: str,
    question_id: int,
    is_correct: bool,
    source_path: Path,
    source_kind: str,
) -> None:
    if question_id not in GSM8K_QUESTION_IDS or persona not in PERSONAS:
        return
    key = (model, persona, question_id)
    record = {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "persona": persona,
        "question_id": question_id,
        "is_correct": bool(is_correct),
        "source_kind": source_kind,
        "source_path": source_path.as_posix(),
    }
    if key in records and records[key]["is_correct"] != record["is_correct"]:
        raise ValueError(f"Conflicting duplicate outcome: {key}")
    records[key] = record


def load_per_persona_reports(repo_root: Path, records: dict) -> list[Path]:
    sources: list[Path] = []
    for model in ROOT_PERSONA_MODELS:
        for persona in PERSONAS:
            path = repo_root / "result" / "reports" / model / f"results_{persona}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            sources.append(path)
            collected: set[int] = set()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    question_id = question_id_from_row(row)
                    if question_id is None:
                        continue
                    add_outcome(
                        records,
                        model=model,
                        persona=persona,
                        question_id=question_id,
                        is_correct=parse_bool(row.get("is_correct")),
                        source_path=path.relative_to(repo_root),
                        source_kind="model_persona_rebuilt_csv",
                    )
                    if question_id in GSM8K_QUESTION_IDS:
                        collected.add(question_id)
                    if len(collected) == len(GSM8K_QUESTION_IDS):
                        break
    return sources


def load_root_combined_reports(repo_root: Path, records: dict) -> list[Path]:
    sources: list[Path] = []
    for model, filename in ROOT_COMBINED_MODELS.items():
        path = repo_root / "result" / "reports" / model / filename
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append(path)
        collected: set[tuple[str, int]] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                persona = row.get("persona_folder", row.get("persona", ""))
                question_id = question_id_from_row(row)
                if question_id is None:
                    continue
                add_outcome(
                    records,
                    model=model,
                    persona=persona,
                    question_id=question_id,
                    is_correct=parse_bool(row.get("is_correct")),
                    source_path=path.relative_to(repo_root),
                    source_kind="combined_model_rebuilt_csv",
                )
                if persona in PERSONAS and question_id in GSM8K_QUESTION_IDS:
                    collected.add((persona, question_id))
                if len(collected) == len(PERSONAS) * len(GSM8K_QUESTION_IDS):
                    break
    return sources


def load_all_results(repo_root: Path, records: dict) -> list[Path]:
    """Load the two models that have complete persona-specific report files.

    The original implementation scanned the 134 MB all-model lite export. These
    smaller files are produced by the same rebuild pipeline and expose the same
    question-level outcome while making the pilot reconstruction deterministic
    and substantially faster.
    """
    sources: list[Path] = []
    for model in ALL_RESULTS_MODELS:
        for persona in PERSONAS:
            path = repo_root / "result" / "reports" / model / f"results_{persona}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            sources.append(path)
            collected: set[int] = set()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    question_id = question_id_from_row(row)
                    if question_id is None:
                        continue
                    add_outcome(
                        records,
                        model=model,
                        persona=persona,
                        question_id=question_id,
                        is_correct=parse_bool(row.get("is_correct")),
                        source_path=path.relative_to(repo_root),
                        source_kind="model_persona_rebuilt_csv",
                    )
                    if question_id in GSM8K_QUESTION_IDS:
                        collected.add(question_id)
                    if len(collected) == len(GSM8K_QUESTION_IDS):
                        break
    return sources


def load_combined_reports(repo_root: Path, records: dict) -> list[Path]:
    sources: list[Path] = []
    for model in COMBINED_REPORT_MODELS:
        path = repo_root / "result" / "reports" / model / "results.csv"
        sources.append(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                question_id = question_id_from_row(row)
                if question_id is None:
                    continue
                add_outcome(
                    records,
                    model=model,
                    persona=row.get("persona_folder", row.get("persona", "")),
                    question_id=question_id,
                    is_correct=parse_bool(row.get("is_correct")),
                    source_path=path.relative_to(repo_root),
                    source_kind="combined_model_report_csv",
                )
    return sources


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.array([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def cohens_h(p_a: float, p_b: float) -> float:
    return 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b))


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for the discordant-pair counts.

    Under H0, b | (b+c) follows Binomial(n=b+c, p=0.5). For p=0.5 the
    conventional two-sided exact result is twice the lower-tail probability,
    capped at one. This equals scipy.stats.binomtest(..., alternative='two-sided')
    for these integer counts while avoiding a heavyweight SciPy import.
    """
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    lower_tail = math.fsum(math.comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * lower_tail)


def calculate_pairwise(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        subset = outcomes[outcomes["model"].eq(model)]
        neutral = subset[subset["persona"].eq("neutral")].set_index("question_id")
        for persona in COMPARISON_PERSONAS:
            condition = subset[subset["persona"].eq(persona)].set_index("question_id")
            paired = condition[["is_correct"]].rename(
                columns={"is_correct": "condition_correct"}
            ).join(
                neutral[["is_correct"]].rename(columns={"is_correct": "neutral_correct"}),
                how="inner",
            )
            condition_values = paired["condition_correct"].astype(bool)
            neutral_values = paired["neutral_correct"].astype(bool)
            b = int((condition_values & ~neutral_values).sum())
            c = int((~condition_values & neutral_values).sum())
            discordant = b + c
            raw_p = exact_mcnemar_p_value(b, c)
            condition_accuracy = float(condition_values.mean())
            neutral_accuracy = float(neutral_values.mean())
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "selected_for_main_experiment": model in SELECTED_MODELS,
                    "condition_persona": persona,
                    "condition_label": PERSONA_LABELS[persona],
                    "condition_short": PERSONA_SHORT[persona],
                    "comparison": f"{PERSONA_LABELS[persona]} - Neutral",
                    "n_pairs": len(paired),
                    "condition_correct": int(condition_values.sum()),
                    "neutral_correct": int(neutral_values.sum()),
                    "condition_accuracy_pct": 100 * condition_accuracy,
                    "neutral_accuracy_pct": 100 * neutral_accuracy,
                    "risk_difference_pp": 100 * (condition_accuracy - neutral_accuracy),
                    "cohens_h": cohens_h(condition_accuracy, neutral_accuracy),
                    "discordant_condition_correct_neutral_wrong": b,
                    "discordant_condition_wrong_neutral_correct": c,
                    "discordant_total": discordant,
                    "exact_mcnemar_raw_p_value": raw_p,
                    "p_value_family": "eight_vs_neutral_comparisons_within_model",
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = result.groupby("model", group_keys=False)[
        "exact_mcnemar_raw_p_value"
    ].apply(holm_adjust)
    result["significant_holm_0_05"] = result["holm_adjusted_p_value"].lt(0.05)
    result["abs_risk_difference_pp"] = result["risk_difference_pp"].abs()
    result["abs_cohens_h"] = result["cohens_h"].abs()
    return result


def build_model_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    maximum_rows = pairwise.loc[
        pairwise.groupby("model")["abs_risk_difference_pp"].idxmax()
    ].copy()
    maximum_rows = maximum_rows.rename(
        columns={
            "condition_persona": "max_effect_condition_persona",
            "condition_label": "max_effect_condition_label",
            "condition_short": "max_effect_condition_short",
            "comparison": "max_effect_comparison",
            "n_pairs": "max_effect_n_pairs",
            "condition_accuracy_pct": "max_effect_condition_accuracy_pct",
            "neutral_accuracy_pct": "max_effect_neutral_accuracy_pct",
            "risk_difference_pp": "max_effect_risk_difference_pp",
            "abs_risk_difference_pp": "max_abs_risk_difference_pp",
            "cohens_h": "max_effect_cohens_h",
            "abs_cohens_h": "max_effect_abs_cohens_h",
            "exact_mcnemar_raw_p_value": "max_effect_exact_mcnemar_raw_p_value",
            "holm_adjusted_p_value": "max_effect_holm_adjusted_p_value",
            "significant_holm_0_05": "max_effect_significant_holm_0_05",
        }
    )
    significant_counts = (
        pairwise.groupby("model")["significant_holm_0_05"].sum().rename("significant_comparisons")
    )
    minimum_pairs = pairwise.groupby("model")["n_pairs"].min().rename("minimum_n_pairs")
    maximum_pairs = pairwise.groupby("model")["n_pairs"].max().rename("maximum_n_pairs")
    keep = [
        "model",
        "model_label",
        "selected_for_main_experiment",
        "max_effect_condition_persona",
        "max_effect_condition_label",
        "max_effect_condition_short",
        "max_effect_comparison",
        "max_effect_n_pairs",
        "max_effect_condition_accuracy_pct",
        "max_effect_neutral_accuracy_pct",
        "max_effect_risk_difference_pp",
        "max_abs_risk_difference_pp",
        "max_effect_cohens_h",
        "max_effect_abs_cohens_h",
        "max_effect_exact_mcnemar_raw_p_value",
        "max_effect_holm_adjusted_p_value",
        "max_effect_significant_holm_0_05",
    ]
    return (
        maximum_rows[keep]
        .set_index("model")
        .join([significant_counts, minimum_pairs, maximum_pairs])
        .reset_index()
        .sort_values(["max_abs_risk_difference_pp", "model_label"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_aggregate_deltas(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the two delta definitions used in the legacy DOCX figures."""
    grouped = (
        outcomes.groupby(["model", "model_label", "persona"], as_index=False)
        .agg(accuracy=("is_correct", "mean"), n=("question_id", "nunique"))
    )
    rows: list[dict[str, Any]] = []
    man_personas = ["man_explicit_1", "man_implicit_1", "man_implicit_2", "man_implicit_3"]
    woman_personas = [
        "woman_explicit_1",
        "woman_implicit_1",
        "woman_implicit_2",
        "woman_implicit_3",
    ]
    for model in MODELS:
        model_rows = grouped[grouped["model"].eq(model)].set_index("persona")
        missing = [persona for persona in PERSONAS if persona not in model_rows.index]
        if missing:
            raise ValueError(f"Missing persona summaries for {model}: {missing}")
        accuracy = model_rows["accuracy"].to_dict()
        counts = model_rows["n"].astype(int).to_dict()
        neutral = float(accuracy["neutral"])
        man_average = float(np.mean([accuracy[p] for p in man_personas]))
        woman_average = float(np.mean([accuracy[p] for p in woman_personas]))
        man_implicit_average = float(np.mean([accuracy[p] for p in man_personas[1:]]))
        woman_implicit_average = float(np.mean([accuracy[p] for p in woman_personas[1:]]))
        delta_man_neutral = man_average - neutral
        delta_woman_neutral = woman_average - neutral
        delta_man_imp_exp = man_implicit_average - float(accuracy["man_explicit_1"])
        delta_woman_imp_exp = woman_implicit_average - float(
            accuracy["woman_explicit_1"]
        )
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "selected_for_main_experiment": model in SELECTED_MODELS,
                "neutral_accuracy": neutral,
                "man_average_accuracy": man_average,
                "woman_average_accuracy": woman_average,
                "delta_man_avg_vs_neutral": delta_man_neutral,
                "delta_woman_avg_vs_neutral": delta_woman_neutral,
                "gender_gap_man_minus_woman_vs_neutral": delta_man_neutral
                - delta_woman_neutral,
                "man_explicit_accuracy": float(accuracy["man_explicit_1"]),
                "man_implicit_average_accuracy": man_implicit_average,
                "woman_explicit_accuracy": float(accuracy["woman_explicit_1"]),
                "woman_implicit_average_accuracy": woman_implicit_average,
                "delta_man_implicit_vs_explicit": delta_man_imp_exp,
                "delta_woman_implicit_vs_explicit": delta_woman_imp_exp,
                "gender_gap_man_minus_woman_imp_vs_exp": delta_man_imp_exp
                - delta_woman_imp_exp,
                "minimum_persona_n": min(counts.values()),
                "maximum_persona_n": max(counts.values()),
                "neutral_n": counts["neutral"],
                "all_personas_n_1319": all(value == 1319 for value in counts.values()),
            }
        )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    global mpl, plt, sns
    import matplotlib as mpl_module
    import matplotlib.pyplot as plt_module
    import seaborn as sns_module

    mpl = mpl_module
    plt = plt_module
    sns = sns_module
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.05)
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def build_aggregate_significance_markers(
    aggregate: pd.DataFrame, pairwise: pd.DataFrame
) -> pd.DataFrame:
    """Roll granular persona-vs-Neutral decisions up to descriptive bars.

    A marker does not test the aggregate average.  It only reports that at
    least one of the four constituent persona-vs-Neutral comparisons for the
    corresponding gender survived the model-wise Holm correction.
    """
    rows: list[dict[str, Any]] = []
    for aggregate_row in aggregate.itertuples(index=False):
        model_rows = pairwise[pairwise["model"].eq(aggregate_row.model)]
        for aggregate_bar, prefix in [
            ("Man Avg − Neutral", "man_"),
            ("Woman Avg − Neutral", "woman_"),
        ]:
            constituents = model_rows[
                model_rows["condition_persona"].str.startswith(prefix)
            ]
            significant = constituents[constituents["significant_holm_0_05"]]
            rows.append(
                {
                    "model": aggregate_row.model,
                    "model_label": aggregate_row.model_label,
                    "aggregate_bar": aggregate_bar,
                    "n_constituent_comparisons": int(len(constituents)),
                    "n_significant_constituent_comparisons": int(len(significant)),
                    "minimum_holm_adjusted_p_value": float(
                        constituents["holm_adjusted_p_value"].min()
                    ),
                    "significant_constituent_comparisons": "; ".join(
                        f"{row.comparison} (Holm p={row.holm_adjusted_p_value:.6g})"
                        for row in significant.itertuples()
                    ),
                    "star_marker": "*" if len(significant) else "",
                    "marker_interpretation": (
                        "At least one constituent persona-vs-Neutral two-sided "
                        "Exact McNemar comparison is significant after Holm; "
                        "the aggregate average itself is descriptive."
                    ),
                }
            )

        # No direct Exact McNemar test exists for an average Man-Woman gap.
        rows.append(
            {
                "model": aggregate_row.model,
                "model_label": aggregate_row.model_label,
                "aggregate_bar": "Man Avg − Woman Avg",
                "n_constituent_comparisons": 0,
                "n_significant_constituent_comparisons": 0,
                "minimum_holm_adjusted_p_value": np.nan,
                "significant_constituent_comparisons": "",
                "star_marker": "",
                "marker_interpretation": (
                    "No direct Exact McNemar test was defined for this "
                    "aggregate Man-Woman gap."
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_legacy_delta_comparison(
    aggregate: pd.DataFrame,
    output_dir: Path,
    *,
    man_column: str,
    woman_column: str,
    gap_column: str,
    man_label: str,
    woman_label: str,
    gap_label: str,
    title: str,
    subtitle: str,
    stem: str,
    significance_markers: pd.DataFrame | None = None,
) -> None:
    selected_order = [
        "qwen_qwen3-next-80b-a3b-thinking",
        "meta-llama_llama-3.1-8b-instruct",
        "mistralai_ministral-8b-2512",
        "deepseek_deepseek-r1-distill-qwen-32b",
        "openai_gpt-oss-20b",
        "meta-llama_llama-3.3-70b-instruct",
    ]
    order = selected_order + [model for model in MODELS if model not in selected_order]
    plot = aggregate.set_index("model").loc[order].reset_index()
    x = np.arange(len(plot), dtype=float)
    width = 0.24
    man_values = plot[man_column].astype(float).to_numpy()
    woman_values = plot[woman_column].astype(float).to_numpy()
    gap_values = plot[gap_column].astype(float).to_numpy()
    all_values = np.concatenate([man_values, woman_values, gap_values])
    max_abs = max(0.01, float(np.max(np.abs(all_values))))

    fig, ax = plt.subplots(figsize=(max(21.5, 1.55 * len(plot)), 9.2))
    man_bars = ax.bar(
        x - width,
        man_values,
        width=width,
        color="#0B4FAF",
        label=man_label,
    )
    woman_bars = ax.bar(
        x,
        woman_values,
        width=width,
        color="#C20D32",
        label=woman_label,
    )
    gap_bars = ax.bar(
        x + width,
        gap_values,
        width=width,
        color="#5B21B6",
        label=gap_label,
    )
    ax.axhline(0, color="#17212B", linewidth=1.2)
    lower = min(-0.005, float(np.min(all_values)) - 0.20 * max_abs)
    upper = max(0.005, float(np.max(all_values)) + 0.25 * max_abs)
    ax.set_ylim(lower, upper)

    for bars, values, aggregate_bar in [
        (man_bars, man_values, "Man Avg − Neutral"),
        (woman_bars, woman_values, "Woman Avg − Neutral"),
        (gap_bars, gap_values, "Man Avg − Woman Avg"),
    ]:
        for bar, value, model in zip(bars, values, plot["model"]):
            offset = 0.025 * max_abs
            marker = ""
            if significance_markers is not None:
                marker_row = significance_markers[
                    significance_markers["model"].eq(model)
                    & significance_markers["aggregate_bar"].eq(aggregate_bar)
                ]
                if not marker_row.empty:
                    marker = str(marker_row.iloc[0]["star_marker"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (offset if value >= 0 else -offset),
                f"{value:+.4f}{marker}",
                va="bottom" if value >= 0 else "top",
                ha="center",
                fontsize=7.6,
                fontweight="bold",
            )

    short_labels = {
        "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next\n80B-A3B",
        "meta-llama_llama-3.1-8b-instruct": "Llama 3.1\n8B",
        "mistralai_ministral-8b-2512": "Ministral\n8B",
        "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill\n32B",
        "openai_gpt-oss-20b": "GPT-OSS\n20B",
        "meta-llama_llama-3.3-70b-instruct": "Llama 3.3\n70B",
        "allenai_olmo-3-32b-think": "OLMo 3\n32B Think",
        "deepseek_deepseek-v3.2": "DeepSeek\nV3.2",
        "openai_gpt-oss-120b": "GPT-OSS\n120B",
        "qwen_qwen3-235b-a22b-2507": "Qwen3\n235B-A22B",
        "google_gemini-2.5-flash": "Gemini 2.5\nFlash",
        "google_gemma-3-27b-it": "Gemma 3\n27B",
        "google_gemma-2-9b-it": "Gemma 2\n9B",
        "allenai_olmo-3.1-32b-instruct": "OLMo 3.1\n32B",
    }
    ax.set_xticks(x, labels=[short_labels[model] for model in plot["model"]])
    for tick, (_, row) in zip(ax.get_xticklabels(), plot.iterrows()):
        tick.set_fontsize(8.7)
        if bool(row["selected_for_main_experiment"]):
            tick.set_color("#0047AB")
            tick.set_fontweight("bold")

    ax.set_title(title, fontsize=17.0, fontweight="bold", pad=17)
    ax.text(
        0.5,
        1.002,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
    )
    ax.set_xlabel("Model (biru tebal = dipilih)")
    ax.set_ylabel("Perbedaan pada skala akurasi")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle="-", alpha=0.18)
    ax.legend(loc="upper right", frameon=True)
    if significance_markers is not None:
        figure_note = (
            "* ≥1 perbandingan granular persona-vs-Neutral signifikan berdasarkan "
            "two-sided Exact McNemar setelah Holm (adjusted p < 0,05); "
            "delta agregat tetap deskriptif."
        )
    else:
        figure_note = (
            "Delta adalah statistik deskriptif. Exact McNemar pada analisis ini hanya "
            "menguji perbandingan granular persona-vs-Neutral, bukan delta agregat."
        )
    fig.text(0.5, 0.012, figure_note, ha="center", fontsize=9.0, color="#4B5563")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, output_dir, stem)


def plot_model_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plot = summary.sort_values("max_abs_risk_difference_pp", ascending=True).copy()
    colors = np.where(plot["selected_for_main_experiment"], "#0047AB", "#8A929B")
    fig, ax = plt.subplots(figsize=(14.8, 9.8))
    bars = ax.barh(
        plot["model_label"], plot["max_effect_risk_difference_pp"], color=colors, alpha=0.95
    )
    ax.axvline(0, color="#17212B", linewidth=1.2)
    max_abs = max(1.0, float(plot["max_abs_risk_difference_pp"].max()))
    annotation_x = 1.16 * max_abs
    ax.set_xlim(-1.18 * max_abs, 2.22 * max_abs)
    tick_limit = int(math.ceil(max_abs / 2.0) * 2)
    ax.set_xticks(np.arange(-tick_limit, tick_limit + 0.1, 2.0))
    ax.axvline(annotation_x - 0.18, color="#CBD0D6", linewidth=0.8)
    for bar, (_, row) in zip(bars, plot.iterrows()):
        value = float(row["max_effect_risk_difference_pp"])
        p_value = float(row["max_effect_holm_adjusted_p_value"])
        marker = "*" if row["max_effect_significant_holm_0_05"] else ""
        label = (
            f"{value:+.1f} pp{marker}  |  {row['max_effect_condition_short']}  "
            f"| h={row['max_effect_cohens_h']:+.2f} | Holm p={p_value:.3g}"
        )
        ax.text(
            annotation_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8.7,
            fontweight="semibold" if row["selected_for_main_experiment"] else "normal",
        )
    ax.set_title(
        "Pilot GSM8K 300 Soal: Sensitivitas Maksimum terhadap Persona Gender",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Perubahan akurasi kondisi persona terhadap Neutral (percentage points)")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#0047AB", label="Dipilih untuk pengujian utama"),
            Patch(facecolor="#8A929B", label="Tidak dilanjutkan"),
        ],
        loc="lower left",
        frameon=True,
    )
    fig.text(
        0.5,
        0.012,
        "Untuk tiap model ditampilkan |delta| terbesar dari 8 kondisi vs Neutral. "
        "* Holm-adjusted Exact McNemar p < 0,05; koreksi dilakukan dalam model. "
        "Pemilihan juga mempertimbangkan akses API, kelengkapan data, ukuran, dan jenis keluaran.",
        ha="center",
        fontsize=9.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, output_dir, "01_pilot_model_selection_sensitivity_summary")


def plot_full_heatmap(pairwise: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    order = summary["model_label"].tolist()
    values = pairwise.pivot(
        index="model_label", columns="condition_short", values="risk_difference_pp"
    ).reindex(index=order, columns=[PERSONA_SHORT[p] for p in COMPARISON_PERSONAS])
    p_values = pairwise.pivot(
        index="model_label", columns="condition_short", values="holm_adjusted_p_value"
    ).reindex(index=values.index, columns=values.columns)
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            p_value = p_values.loc[row, column]
            annotations.loc[row, column] = (
                "N/A" if pd.isna(value) else f"{value:+.1f}{'*' if p_value < 0.05 else ''}"
            )
    limit = max(1.0, float(np.nanmax(np.abs(values.to_numpy(dtype=float)))))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig, ax = plt.subplots(figsize=(14.8, 10.2))
    sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        center=0,
        vmin=-limit,
        vmax=limit,
        cmap=cmap,
        linewidths=0.7,
        linecolor="#FCFBF8",
        cbar_kws={"label": "Perubahan akurasi vs Neutral (pp)"},
        annot_kws={"fontsize": 8.7, "fontweight": "semibold"},
        ax=ax,
    )
    selected_labels = {MODEL_LABELS[model] for model in SELECTED_MODELS}
    for tick in ax.get_yticklabels():
        if tick.get_text() in selected_labels:
            tick.set_color("#0047AB")
            tick.set_fontweight("bold")
    ax.set_title(
        "GSM8K Seluruh Soal Tersedia: Kondisi Gender terhadap Neutral",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Kondisi persona")
    ax.set_ylabel("Model (biru tebal = dipilih)")
    fig.text(
        0.5,
        0.012,
        "Angka adalah paired risk difference. * Holm-adjusted two-sided Exact McNemar p < 0,05 "
        "dalam delapan perbandingan per model. M/W = Man/Woman; E = Explicit; I1-I3 = Implicit.",
        ha="center",
        fontsize=9.2,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, "03_all_conditions_exact_mcnemar_heatmap")


def _pvalue_annotations(
    values: pd.DataFrame, significant: pd.DataFrame
) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = ""
                continue
            formatted = "<.001" if value < 0.001 else f"{value:.3f}"
            annotations.loc[row, column] = (
                formatted + ("*" if bool(significant.loc[row, column]) else "")
            )
    return annotations


def plot_exact_mcnemar_pvalue_heatmap(
    pairwise: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    """Render raw and Holm-adjusted p-values in the same form as notebook v5."""
    model_order = summary["model_label"].tolist()
    condition_order = [PERSONA_SHORT[p] for p in COMPARISON_PERSONAS]
    raw = pairwise.pivot(
        index="model_label", columns="condition_short", values="exact_mcnemar_raw_p_value"
    ).reindex(index=model_order, columns=condition_order)
    adjusted = pairwise.pivot(
        index="model_label", columns="condition_short", values="holm_adjusted_p_value"
    ).reindex(index=model_order, columns=condition_order)
    raw_significant = raw.lt(0.05)
    adjusted_significant = pairwise.pivot(
        index="model_label", columns="condition_short", values="significant_holm_0_05"
    ).reindex(index=model_order, columns=condition_order).fillna(False).astype(bool)

    significance_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "bold_significance_14_models",
        ["#3B0A57", "#8E0D4A", "#D52B1E", "#F0D8D2", "#F7F4F0"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(21.5, 14.2), constrained_layout=True)
    panels = [
        (
            axes[0], raw, raw_significant,
            "Raw two-sided Exact McNemar p-value", "p-value (0–0.05)",
        ),
        (
            axes[1], adjusted, adjusted_significant,
            "Holm-adjusted p-value", "adjusted p-value (0–0.05)",
        ),
    ]
    selected_labels = {MODEL_LABELS[model] for model in SELECTED_MODELS}
    for ax, matrix, significant, panel_title, cbar_label in panels:
        sns.heatmap(
            matrix,
            annot=_pvalue_annotations(matrix, significant),
            fmt="",
            cmap=significance_cmap,
            vmin=0,
            vmax=0.05,
            linewidths=0.8,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 8.6, "fontweight": "bold"},
            cbar_kws={"label": cbar_label},
            ax=ax,
        )
        ax.set_title(panel_title, fontsize=13.5, fontweight="bold", pad=10)
        ax.set_xlabel("Perbandingan persona terhadap Neutral")
        ax.set_ylabel("Model (biru tebal = dipilih)")
        ax.set_xticklabels(condition_order, rotation=0, ha="center", fontsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=9.2)
        for tick in ax.get_yticklabels():
            if tick.get_text() in selected_labels:
                tick.set_color("#0047AB")
                tick.set_fontweight("bold")

    fig.suptitle(
        "GSM8K 14 Model: Significance Value untuk Kondisi Gender terhadap Neutral",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "* p < 0,05 pada panel raw; * adjusted p < 0,05 pada panel Holm. "
        "Koreksi dilakukan untuk delapan perbandingan di dalam setiap model. "
        "M/W = Man/Woman; E = Explicit; I1–I3 = Implicit.",
        ha="center",
        fontsize=9.2,
        color="#4B5563",
    )
    save_figure(
        fig, output_dir,
        "04_all_conditions_exact_mcnemar_and_holm_pvalue_heatmap",
    )


def sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(repo_root: Path, csv_sources: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(csv_sources)):
        rows.append(
            {
                "source_type": "parsed_csv",
                "relative_path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_latex_snippet(
    summary: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path
) -> None:
    strongest_index = aggregate[
        ["delta_man_avg_vs_neutral", "delta_woman_avg_vs_neutral"]
    ].abs().max(axis=1).idxmax()
    strongest = aggregate.loc[strongest_index]
    if abs(strongest["delta_man_avg_vs_neutral"]) >= abs(
        strongest["delta_woman_avg_vs_neutral"]
    ):
        strongest_gender = "Man"
        strongest_delta = float(strongest["delta_man_avg_vs_neutral"])
    else:
        strongest_gender = "Woman"
        strongest_delta = float(strongest["delta_woman_avg_vs_neutral"])
    selected_with_significance = int(
        summary.loc[
            summary["selected_for_main_experiment"], "significant_comparisons"
        ].gt(0).sum()
    )
    if selected_with_significance == 0:
        significance_sentence = (
            "Tidak ada dari enam model yang dilanjutkan memiliki perbandingan yang tetap "
            "signifikan setelah koreksi Holm."
        )
    else:
        significance_sentence = (
            f"Di antara enam model yang dilanjutkan, {selected_with_significance} model "
            "memiliki sedikitnya satu perbandingan yang tetap signifikan setelah koreksi Holm."
        )
    snippet = rf"""% Salin gambar PNG/SVG ke folder image/BabVI sebelum kompilasi.
Analisis GSM8K menggunakan seluruh soal yang tersedia sampai 1.319 soal. Delta terhadap
kondisi netral dihitung secara deskriptif. Delta Man merupakan rata-rata akurasi satu
kondisi eksplisit dan tiga kondisi implisit Man dikurangi akurasi Neutral; Delta Woman
dihitung dengan cara yang sama. Untuk uji inferensial, setiap kondisi persona dibandingkan
dengan Neutral pada nomor soal yang sama menggunakan \textit{{two-sided Exact McNemar}}.
Nilai $p$ dikoreksi dengan prosedur Holm untuk delapan perbandingan di dalam setiap model.
McNemar tidak diterapkan langsung pada Delta Man atau Delta Woman karena keduanya merupakan
rata-rata beberapa kondisi, bukan satu pasangan keluaran biner.

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{image/BabVI/01_delta_avg_vs_neutral_all_available.png}}
    \caption{{Perubahan rata-rata akurasi kondisi persona Man dan Woman terhadap kondisi
    Neutral pada GSM8K. Delta dihitung menggunakan definisi yang sama dengan grafik
    \textit{{Avg vs Neutral}} pada dokumen analisis awal. Tanda bintang menunjukkan bahwa
    sedikitnya satu perbandingan granular penyusun terhadap Neutral signifikan berdasarkan
    \textit{{two-sided Exact McNemar}} setelah koreksi Holm ($p_\mathrm{{adj}}<0{{,}}05$),
    bukan uji langsung atas delta agregat.}}
    \label{{fig:gsm8k-delta-avg-vs-neutral}}
\end{{figure}}

Gambar~\ref{{fig:gsm8k-delta-avg-vs-neutral}} menunjukkan perubahan deskriptif terbesar
pada {strongest['model_label']} untuk persona {strongest_gender}
($\Delta={strongest_delta:+.3f}$).
{significance_sentence} Hasil ini dibaca bersama
kriteria operasional dan keragaman model, bukan sebagai peringkat mutu model secara umum.
Kimi K2 Thinking dan Trinity Large Thinking dikeluarkan dari grafik karena hanya memiliki
paling banyak 400 soal. Jumlah observasi dan pasangan aktual untuk 14 model yang dianalisis
dilaporkan pada tabel cakupan data dan tabel Exact McNemar.
"""
    (output_dir / "latex_insertion_pemilihan_model.tex").write_text(snippet, encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    text = """# Analisis GSM8K seluruh soal tersedia

Artefak ini mereproduksi definisi delta pada `Analisa Bias Gender (2).docx` dan mengganti
uji granular two-proportion z-test dengan perbandingan berpasangan yang sesuai.

- Rentang target: question ID 1-1319.
- Delta Avg vs Neutral: mean empat akurasi persona per gender dikurangi akurasi Neutral.
- Delta Imp vs Exp: mean tiga kondisi implicit dikurangi kondisi explicit per gender.
- Perbandingan: delapan kondisi Gender terhadap Neutral pada model yang sama.
- Uji: two-sided Exact McNemar pada pasangan question ID yang tersedia.
- Multiple comparisons: Holm correction dalam delapan perbandingan per model.
- Effect size: paired risk difference (percentage points) dan marginal Cohen's h.
- Kedua sisi setiap pasangan diambil dari keluarga laporan hasil parsing yang sama; raw JSON
  tidak diparsing ulang secara selektif hanya untuk kondisi Neutral.
- Kimi dan Trinity dikeluarkan karena hanya dikumpulkan sampai paling banyak 400 soal.
  Model/kondisi lain dapat memiliki sebagian kecil outcome hilang; tabel melaporkan n aktual.
- Exact McNemar tidak diterapkan pada delta agregat karena rata-rata beberapa kondisi bukan
  keluaran biner tunggal.
- Pada visual Avg vs Neutral, `*` berarti sedikitnya satu perbandingan granular penyusun
  terhadap Neutral signifikan setelah Holm (`adjusted p < 0.05`). Bintang bukan hasil uji
  langsung atas batang rerata. Rinciannya tersedia pada
  `gsm8k_aggregate_delta_significance_markers.csv`.
- Visual Imp vs Exp tidak diberi bintang karena analisis pilot ini tidak mendefinisikan uji
  Exact McNemar Implicit-vs-Explicit; p-value persona-vs-Neutral tidak dipindahkan ke batang
  tersebut.

Visual delta adalah `01_delta_avg_vs_neutral_all_available` dan
`02_delta_implicit_vs_explicit_all_available`. Heatmap perubahan akurasi granular adalah
`03_all_conditions_exact_mcnemar_heatmap`. Visual p-value dua panel yang setara dengan
notebook enam model adalah `04_all_conditions_exact_mcnemar_and_holm_pvalue_heatmap`.

Analisis tambahan antarpersona yang telah ditetapkan dapat dibuat melalui
`generate_gsm8k_14_model_all_persona_pairwise.py`. Artefaknya adalah tabel paired lengkap,
tabel omnibus Cochran's Q, heatmap Cohen's h, dan heatmap raw/Holm-adjusted p-value. Family
Holm dipisahkan menjadi Man-primary (7), Woman-primary (7), dan matched Man-vs-Woman
exploratory (4); perbandingan silang yang mencampur gender dan framing berbeda tidak dibuat.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def find_repo_root(start: Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "result" / "GMS8K").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Repository root not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="result/GMS8K/reports/gsm8k_full_1319_delta_exact_mcnemar_14_models",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    csv_sources = []
    started = time.perf_counter()
    print("[1/7] Loading complete root persona reports...", flush=True)
    csv_sources.extend(load_per_persona_reports(repo_root, records))
    print(f"      done: {len(records):,} outcomes ({time.perf_counter() - started:.1f}s)", flush=True)
    print("[2/7] Loading DeepSeek V3.2 and GPT-OSS 20B combined reports...", flush=True)
    csv_sources.extend(load_root_combined_reports(repo_root, records))
    print(f"      done: {len(records):,} outcomes ({time.perf_counter() - started:.1f}s)", flush=True)
    print("[3/7] Loading OLMo 3.1 and R1-Distill persona reports...", flush=True)
    csv_sources.extend(load_all_results(repo_root, records))
    print(f"      done: {len(records):,} outcomes ({time.perf_counter() - started:.1f}s)", flush=True)
    outcomes = pd.DataFrame(records.values()).sort_values(
        ["model", "persona", "question_id"]
    ).reset_index(drop=True)
    inventory = (
        outcomes.groupby(["model", "model_label", "persona"], as_index=False)
        .agg(
            rows=("question_id", "size"),
            unique_questions=("question_id", "nunique"),
            correct=("is_correct", "sum"),
        )
    )
    inventory["accuracy_pct"] = 100 * inventory["correct"] / inventory["rows"]

    pairwise = calculate_pairwise(outcomes)
    summary = build_model_summary(pairwise)
    aggregate = build_aggregate_deltas(outcomes)
    aggregate_markers = build_aggregate_significance_markers(aggregate, pairwise)
    print(f"[5/7] Exact McNemar tables calculated ({time.perf_counter() - started:.1f}s)", flush=True)
    manifest = build_source_manifest(repo_root, csv_sources)

    outcomes.to_csv(output_dir / "gsm8k_question_level_outcomes.csv", index=False)
    inventory.to_csv(output_dir / "gsm8k_data_coverage.csv", index=False)
    pairwise.to_csv(output_dir / "gsm8k_vs_neutral_exact_mcnemar.csv", index=False)
    summary.to_csv(output_dir / "gsm8k_granular_model_summary.csv", index=False)
    aggregate.to_csv(output_dir / "gsm8k_legacy_delta_definitions.csv", index=False)
    aggregate_markers.to_csv(
        output_dir / "gsm8k_aggregate_delta_significance_markers.csv", index=False
    )
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    print(f"[6/7] Audit tables written ({time.perf_counter() - started:.1f}s)", flush=True)

    configure_plots()
    plot_legacy_delta_comparison(
        aggregate,
        output_dir,
        man_column="delta_man_avg_vs_neutral",
        woman_column="delta_woman_avg_vs_neutral",
        gap_column="gender_gap_man_minus_woman_vs_neutral",
        man_label="Man Avg − Neutral",
        woman_label="Woman Avg − Neutral",
        gap_label="Man Avg − Woman Avg",
        title="GSM8K Gender: Analisis Delta terhadap Persona Neutral",
        subtitle="Rata-rata empat kondisi persona per gender dikurangi akurasi Neutral",
        stem="01_delta_avg_vs_neutral_all_available",
        significance_markers=aggregate_markers,
    )
    plot_legacy_delta_comparison(
        aggregate,
        output_dir,
        man_column="delta_man_implicit_vs_explicit",
        woman_column="delta_woman_implicit_vs_explicit",
        gap_column="gender_gap_man_minus_woman_imp_vs_exp",
        man_label="Man Imp Avg − Explicit",
        woman_label="Woman Imp Avg − Explicit",
        gap_label="Delta Man − Delta Woman",
        title="GSM8K Gender: Analisis Delta Implicit terhadap Explicit",
        subtitle="Rata-rata tiga kondisi implicit dikurangi kondisi explicit per gender",
        stem="02_delta_implicit_vs_explicit_all_available",
    )
    plot_full_heatmap(pairwise, summary, output_dir)
    plot_exact_mcnemar_pvalue_heatmap(pairwise, summary, output_dir)
    write_latex_snippet(summary, aggregate, output_dir)
    write_readme(output_dir)
    print(f"[7/7] Figures and LaTeX snippet written ({time.perf_counter() - started:.1f}s)", flush=True)

    assert set(outcomes["model"]) == set(MODELS)
    assert set(outcomes["persona"]) == set(PERSONAS)
    assert len(pairwise) == len(MODELS) * len(COMPARISON_PERSONAS)
    assert pairwise["n_pairs"].between(250, 1319).all()
    assert set(summary["model"]) == set(MODELS)
    assert len(summary) == len(MODELS)
    assert len(aggregate) == len(MODELS)
    assert len(aggregate_markers) == len(MODELS) * 3
    expected_aggregate_stars = {
        ("allenai_olmo-3-32b-think", "Man Avg − Neutral"),
        ("allenai_olmo-3-32b-think", "Woman Avg − Neutral"),
        ("deepseek_deepseek-r1-distill-qwen-32b", "Man Avg − Neutral"),
        ("deepseek_deepseek-r1-distill-qwen-32b", "Woman Avg − Neutral"),
    }
    actual_aggregate_stars = set(
        aggregate_markers.loc[
            aggregate_markers["star_marker"].eq("*"), ["model", "aggregate_bar"]
        ].itertuples(index=False, name=None)
    )
    assert actual_aggregate_stars == expected_aggregate_stars
    print(f"Outcomes : {len(outcomes):,}")
    print(f"Pairwise : {len(pairwise):,}")
    print(f"Models   : {len(summary)}")
    print(f"Output   : {output_dir}")
    print("ALL FULL GSM8K ANALYSIS VALIDATIONS PASSED")


if __name__ == "__main__":
    main()

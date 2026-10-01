"""Publication-ready ETHICS Religion Persona visualizations.

This is the Religion + Neutral analogue of
``gender_publication_analysis_v3_accuracy.py``.  It uses full-denominator
accuracy: abstain and missing responses remain in the denominator and count as
incorrect.  Inferential comparisons are paired by question ID, use exact
McNemar tests with Holm correction, and never reparse responses or call an API.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage

import gender_publication_analysis_v3_accuracy as base


ANALYSIS_VERSION = "ethics-religion-v3-accuracy-paired-holm-20260822"
EXPECTED_ROWS = 154_428
EXPECTED_QUESTIONS = 1_514
ALPHA = 0.05
DOMAIN_SLUG = "religion"
DOMAIN_TITLE = "Religion Persona"
PAIRWISE_TITLE = "Antaragama"
OUTPUT_GROUP_DIR = "Religion_Persona"

MODEL_ORDER = base.MODEL_ORDER
MODEL_LABELS = base.MODEL_LABELS
SUBSET_ORDER = base.SUBSET_ORDER
SUBSET_LABELS = base.SUBSET_LABELS
SUBSET_EXPECTED_N = base.SUBSET_EXPECTED_N

RELIGION_ORDER = ["atheist", "christian", "islam", "jewish"]
RELIGION_LABELS = {
    "neutral": "Netral",
    "atheist": "Ateis",
    "christian": "Kristen",
    "islam": "Islam",
    "jewish": "Yahudi",
}
SCENARIO_ORDER = ["N", "E", "I1", "I2", "I3"]


def persona_name(religion: str, scenario: str) -> str:
    if scenario == "N":
        return "neutral"
    if scenario == "E":
        return f"{religion}_explicit"
    return f"{religion}_implicit_{scenario[-1]}"


PERSONA_ORDER = ["neutral"] + [
    persona_name(religion, scenario)
    for religion in RELIGION_ORDER
    for scenario in ["E", "I1", "I2", "I3"]
]
PERSONA_META = {"neutral": ("neutral", "N")}
PERSONA_LABELS = {"neutral": "Netral"}
for _religion in RELIGION_ORDER:
    for _scenario in ["E", "I1", "I2", "I3"]:
        _persona = persona_name(_religion, _scenario)
        PERSONA_META[_persona] = (_religion, _scenario)
        PERSONA_LABELS[_persona] = f"{RELIGION_LABELS[_religion]} {_scenario}"

USECOLS = base.USECOLS


def default_input_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "result"
        / "ETHICS"
        / "reports"
        / OUTPUT_GROUP_DIR
        / "_combined"
        / "all_models_results_full.csv"
    )


def default_output_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "result"
        / "ETHICS"
        / "reports"
        / OUTPUT_GROUP_DIR
        / "_combined"
        / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
    )


def load_data(path: Path) -> pd.DataFrame:
    print(f"[LOAD] Membaca seluruh data: {path}")
    frame = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    frame["source_is_correct"] = base.normalize_bool(frame["is_correct"], "is_correct")
    frame["correct_full"] = frame["source_is_correct"]
    frame["persona"] = frame["persona_folder"]
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    frame["subset_label"] = frame["subset"].map(SUBSET_LABELS)
    frame[["identity", "scenario"]] = frame["persona"].map(PERSONA_META).apply(pd.Series)
    frame["identity_label"] = frame["identity"].map(RELIGION_LABELS)
    frame["persona_label"] = frame["persona"].map(PERSONA_LABELS)
    frame["status_label"] = frame["answer_status"].map(base.STATUS_DISPLAY)
    frame["ground_truth"] = pd.to_numeric(frame["ground_truth"], errors="raise").astype(int)
    print(f"[LOAD] Selesai: {len(frame):,} baris")
    return frame


def validate_input(frame: pd.DataFrame) -> dict[str, object]:
    key = ["model", "persona", "question_id"]
    duplicate_rows = int(frame.duplicated(key).sum())
    condition_sizes = frame.groupby(["model", "persona"], observed=True).size()
    subset_sizes = frame.groupby(["model", "persona", "subset"], observed=True).size()
    subset_sizes_valid = all(
        int(n) == SUBSET_EXPECTED_N[subset]
        for (_, _, subset), n in subset_sizes.items()
    )

    reference_ids: frozenset[str] | None = None
    identical_ids = True
    for _, group in frame.groupby(["model", "persona"], observed=True, sort=False):
        ids = frozenset(group["question_id"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            identical_ids = False
            break

    nonanswer = frame["answer_status"].isin(["abstain", "missing"])
    answerable = frame["answer_status"].eq("committed")
    stance_match = frame["extracted_stance"].eq(frame["gt_stance"])
    checks = {
        "row_count": int(len(frame)),
        "row_count_expected": bool(len(frame) == EXPECTED_ROWS),
        "question_count": int(frame["question_id"].nunique()),
        "question_count_expected": bool(frame["question_id"].nunique() == EXPECTED_QUESTIONS),
        "model_count": int(frame["model"].nunique()),
        "six_models_present": bool(set(frame["model"]) == set(MODEL_ORDER)),
        "persona_count": int(frame["persona"].nunique()),
        "seventeen_conditions_present": bool(set(frame["persona"]) == set(PERSONA_ORDER)),
        "four_subsets_present": bool(set(frame["subset"]) == set(SUBSET_ORDER)),
        "duplicate_response_keys": duplicate_rows,
        "no_duplicate_response_keys": bool(duplicate_rows == 0),
        "all_conditions_have_1514_questions": bool(condition_sizes.eq(EXPECTED_QUESTIONS).all()),
        "all_conditions_share_question_ids": bool(identical_ids),
        "subset_sizes_expected": bool(subset_sizes_valid),
        "ground_truth_binary": bool(frame["ground_truth"].isin([0, 1]).all()),
        "abstain_missing_scored_false": bool((~frame.loc[nonanswer, "correct_full"]).all()),
        "answerable_correctness_matches_ground_truth_stance": bool(
            (frame.loc[answerable, "correct_full"] == stance_match.loc[answerable]).all()
        ),
        "unknown_status_labels": int(frame["status_label"].isna().sum()),
        "unknown_persona_labels": int(frame["persona_label"].isna().sum()),
    }
    boolean_values = [value for value in checks.values() if isinstance(value, (bool, np.bool_))]
    checks["valid"] = bool(
        all(boolean_values)
        and checks["unknown_status_labels"] == 0
        and checks["unknown_persona_labels"] == 0
    )
    if not checks["valid"]:
        raise ValueError(f"Validasi input gagal: {checks}")
    print("[VALIDATE] 154.428 baris, 17 kondisi, ground truth, dan pairing valid")
    return checks


def aggregate_accuracy(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = (
        frame.groupby(group_columns, observed=True, sort=False)["correct_full"]
        .agg(total_responses="size", n_correct="sum", accuracy="mean")
        .reset_index()
    )
    result["n_incorrect"] = result["total_responses"] - result["n_correct"]
    result["accuracy_pct"] = 100 * result["accuracy"]
    return result


def build_accuracy_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    condition = aggregate_accuracy(
        frame,
        [
            "subset",
            "subset_label",
            "model",
            "model_label",
            "persona",
            "persona_label",
            "identity",
            "identity_label",
            "scenario",
        ],
    )
    combined = aggregate_accuracy(
        frame,
        [
            "model",
            "model_label",
            "persona",
            "persona_label",
            "identity",
            "identity_label",
            "scenario",
        ],
    )
    combined.insert(0, "subset", "all_four_subsets")
    combined.insert(1, "subset_label", "Gabungan empat subset")

    religion = frame[frame["identity"].isin(RELIGION_ORDER)].copy()
    return {
        "accuracy_by_subset_model_condition.csv": condition,
        "accuracy_combined_four_subsets_by_model_condition.csv": combined,
        f"accuracy_{DOMAIN_SLUG}_overall.csv": aggregate_accuracy(
            religion, ["identity", "identity_label"]
        ),
        f"accuracy_{DOMAIN_SLUG}_by_subset.csv": aggregate_accuracy(
            religion, ["subset", "subset_label", "identity", "identity_label"]
        ),
        f"accuracy_{DOMAIN_SLUG}_by_model.csv": aggregate_accuracy(
            religion, ["model", "model_label", "identity", "identity_label"]
        ),
        f"accuracy_{DOMAIN_SLUG}_by_scenario.csv": aggregate_accuracy(
            religion, ["scenario", "identity", "identity_label"]
        ),
    }


def build_condition_cache(frame: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    cache: dict[tuple[str, str, str], pd.Series] = {}
    for (model, subset, persona), group in frame.groupby(
        ["model", "subset", "persona"], observed=True, sort=False
    ):
        if group["question_id"].duplicated().any():
            raise ValueError(f"Duplikasi kondisi: {model}, {subset}, {persona}")
        cache[(model, subset, persona)] = group.set_index("question_id")[
            "correct_full"
        ].sort_index()
    return cache


def build_cochran_q(
    cache: dict[tuple[str, str, str], pd.Series]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity in RELIGION_ORDER:
                persona_by_scenario = {
                    scenario: persona_name(identity, scenario)
                    for scenario in SCENARIO_ORDER
                }
                series = {
                    scenario: cache[(model, subset, persona)]
                    for scenario, persona in persona_by_scenario.items()
                }
                wide = pd.concat(series, axis=1, join="inner")
                if wide.index.duplicated().any() or len(wide) != SUBSET_EXPECTED_N[subset]:
                    raise ValueError(
                        f"Pairing Cochran Q tidak lengkap: {subset}, {model}, {identity}"
                    )
                statistic, degrees_of_freedom, p_value = base.cochran_q_test(
                    wide.to_numpy(dtype=bool)
                )
                row: dict[str, object] = {
                    "subset": subset,
                    "subset_label": SUBSET_LABELS[subset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "identity": identity,
                    "identity_label": RELIGION_LABELS[identity],
                    "n_pairs": int(len(wide)),
                    "cochran_q_statistic": statistic,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": p_value,
                    "significant_0_05": bool(p_value < ALPHA),
                }
                for scenario in SCENARIO_ORDER:
                    row[f"accuracy_{scenario}_pct"] = 100 * float(wide[scenario].mean())
                rows.append(row)
    return pd.DataFrame(rows)


SCENARIO_SPECS = [
    ("E", "N", "E − N", "persona_vs_neutral", 0),
    ("I1", "N", "I1 − N", "persona_vs_neutral", 1),
    ("I2", "N", "I2 − N", "persona_vs_neutral", 2),
    ("I3", "N", "I3 − N", "persona_vs_neutral", 3),
    ("I1", "E", "I1 − E", "implicit_vs_explicit", 4),
    ("I2", "E", "I2 − E", "implicit_vs_explicit", 5),
    ("I3", "E", "I3 − E", "implicit_vs_explicit", 6),
]


def build_mcnemar(
    cache: dict[tuple[str, str, str], pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity_index, identity in enumerate(RELIGION_ORDER):
                personas = {
                    scenario: persona_name(identity, scenario)
                    for scenario in SCENARIO_ORDER
                }
                series = {
                    scenario: cache[(model, subset, persona)]
                    for scenario, persona in personas.items()
                }
                for scenario_a, scenario_b, label, category, order in SCENARIO_SPECS:
                    base_row = {
                        "comparison_family": f"scenario_within_{DOMAIN_SLUG}",
                        "comparison_category": category,
                        "subset": subset,
                        "subset_label": SUBSET_LABELS[subset],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "identity": identity,
                        "identity_label": RELIGION_LABELS[identity],
                        "scenario_a": scenario_a,
                        "scenario_b": scenario_b,
                        "condition_a": personas[scenario_a],
                        "condition_b": personas[scenario_b],
                        "comparison_label": f"{RELIGION_LABELS[identity]} {label}",
                        "display_order": identity_index * 7 + order,
                    }
                    scenario_rows.append(
                        base._mcnemar_row(
                            series[scenario_a], series[scenario_b], base_row
                        )
                    )
    scenario = pd.DataFrame(scenario_rows)
    scenario["holm_adjusted_p_value"] = scenario.groupby(
        ["subset", "model", "identity"], observed=True
    )["exact_mcnemar_p_value"].transform(base.holm_adjust)
    scenario["holm_family_size"] = 7
    scenario["significant_holm_0_05"] = scenario["holm_adjusted_p_value"].lt(ALPHA)

    pair_rows: list[dict[str, object]] = []
    religion_pairs = list(combinations(RELIGION_ORDER, 2))
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for pair_index, (identity_a, identity_b) in enumerate(religion_pairs):
                for scenario_index, scenario_name in enumerate(["E", "I1", "I2", "I3"]):
                    persona_a = persona_name(identity_a, scenario_name)
                    persona_b = persona_name(identity_b, scenario_name)
                    label_a = RELIGION_LABELS[identity_a]
                    label_b = RELIGION_LABELS[identity_b]
                    base_row = {
                        "comparison_family": f"{DOMAIN_SLUG}_pair_within_scenario",
                        "comparison_category": f"{DOMAIN_SLUG}_pairwise",
                        "subset": subset,
                        "subset_label": SUBSET_LABELS[subset],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "identity": f"{identity_a}_vs_{identity_b}",
                        "identity_label": f"{label_a} vs {label_b}",
                        "scenario_a": scenario_name,
                        "scenario_b": scenario_name,
                        "condition_a": persona_a,
                        "condition_b": persona_b,
                        "comparison_label": f"{label_a} − {label_b} {scenario_name}",
                        "display_order": pair_index * 4 + scenario_index,
                    }
                    pair_rows.append(
                        base._mcnemar_row(
                            cache[(model, subset, persona_a)],
                            cache[(model, subset, persona_b)],
                            base_row,
                        )
                    )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["holm_adjusted_p_value"] = pairwise.groupby(
        ["subset", "model"], observed=True
    )["exact_mcnemar_p_value"].transform(base.holm_adjust)
    pairwise["holm_family_size"] = 24
    pairwise["significant_holm_0_05"] = pairwise["holm_adjusted_p_value"].lt(ALPHA)
    return scenario, pairwise


def build_status_table(frame: pd.DataFrame) -> pd.DataFrame:
    status = (
        frame.groupby(["model", "model_label", "status_label"], observed=True)
        .size()
        .rename("n_responses")
        .reset_index()
    )
    totals = status.groupby("model", observed=True)["n_responses"].transform("sum")
    status["percentage"] = 100 * status["n_responses"] / totals
    return status


def build_abstain_reason_table(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["answer_status"].eq("abstain")].copy()
    selected["reason"] = selected["abstain_reason"].replace("", "unspecified")
    selected["reason_label"] = selected["reason"].map(base.ABSTAIN_REASON_LABELS).fillna(
        selected["reason"].str.replace("_", " ").str.title()
    )
    result = (
        selected.groupby(
            ["persona", "persona_label", "reason", "reason_label"], observed=True
        )
        .size()
        .rename("n_abstain")
        .reset_index()
    )
    denominators = selected.groupby("persona", observed=True).size().rename(
        "persona_abstain_total"
    )
    result = result.merge(denominators, on="persona", validate="many_to_one")
    result["percentage_within_persona_abstain"] = (
        100 * result["n_abstain"] / result["persona_abstain_total"]
    )
    return result


def build_significant_summaries(
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categories = [
        "persona_vs_neutral",
        "implicit_vs_explicit",
        f"{DOMAIN_SLUG}_pairwise",
    ]
    full = (
        comparisons.groupby("comparison_category", observed=True)
        .agg(
            total_tests=("significant_holm_0_05", "size"),
            significant_results=("significant_holm_0_05", "sum"),
        )
        .reindex(categories)
        .reset_index()
    )
    by_subset = (
        comparisons.groupby(
            ["comparison_category", "subset", "subset_label"], observed=True
        )
        .agg(
            total_tests=("significant_holm_0_05", "size"),
            significant_results=("significant_holm_0_05", "sum"),
        )
        .reset_index()
    )
    by_model = (
        comparisons.groupby(
            ["comparison_category", "model", "model_label"], observed=True
        )
        .agg(
            total_tests=("significant_holm_0_05", "size"),
            significant_results=("significant_holm_0_05", "sum"),
        )
        .reset_index()
    )
    return full, by_subset, by_model


def plot_accuracy_absolute(
    condition: pd.DataFrame, figure_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = None
    for index, (ax, subset) in enumerate(zip(axes, SUBSET_ORDER)):
        selected = condition[condition["subset"].eq(subset)]
        matrix = selected.pivot(
            index="model_label", columns="persona_label", values="accuracy_pct"
        ).reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER],
            columns=[PERSONA_LABELS[persona] for persona in PERSONA_ORDER],
        )
        heatmap = sns.heatmap(
            matrix,
            annot=base.numeric_annotations(matrix, 1),
            annot_kws={"fontsize": 6.7, "fontweight": "normal"},
            fmt="",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            linewidths=0.35,
            cbar=index == 0,
            cbar_ax=colorbar_ax if index == 0 else None,
            ax=ax,
        )
        if index == 0:
            colorbar = heatmap.collections[0].colorbar
        ax.set_title(SUBSET_LABELS[subset], fontweight="semibold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        base.style_heatmap(ax, 52)
    base.style_colorbar(colorbar, "Akurasi (%)")
    fig.suptitle(f"Akurasi ETHICS {DOMAIN_TITLE} terhadap Ground Truth", fontweight="bold")
    return (
        base.save_figure(fig, figure_dir, "01_accuracy_absolute_all_subsets_heatmap"),
        condition.copy(),
    )


def build_implicit_delta_data(condition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity in RELIGION_ORDER:
                explicit = condition[
                    condition["subset"].eq(subset)
                    & condition["model"].eq(model)
                    & condition["persona"].eq(persona_name(identity, "E"))
                ]["accuracy_pct"].iloc[0]
                for variant in [1, 2, 3]:
                    implicit = condition[
                        condition["subset"].eq(subset)
                        & condition["model"].eq(model)
                        & condition["persona"].eq(persona_name(identity, f"I{variant}"))
                    ]["accuracy_pct"].iloc[0]
                    rows.append(
                        {
                            "subset": subset,
                            "subset_label": SUBSET_LABELS[subset],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "identity": identity,
                            "identity_label": RELIGION_LABELS[identity],
                            "implicit_scenario": f"I{variant}",
                            "comparison_label": f"{RELIGION_LABELS[identity]} I{variant} − E",
                            "explicit_accuracy_pct": explicit,
                            "implicit_accuracy_pct": implicit,
                            "delta_accuracy_pp": implicit - explicit,
                        }
                    )
    return pd.DataFrame(rows)


def plot_implicit_delta(delta: pd.DataFrame, figure_dir: Path) -> list[Path]:
    limit = max(5.0, float(np.ceil(delta["delta_accuracy_pp"].abs().max())))
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = None
    column_order = [
        f"{RELIGION_LABELS[identity]} I{variant} − E"
        for identity in RELIGION_ORDER
        for variant in [1, 2, 3]
    ]
    for index, (ax, subset) in enumerate(zip(axes, SUBSET_ORDER)):
        selected = delta[delta["subset"].eq(subset)]
        matrix = selected.pivot(
            index="model_label", columns="comparison_label", values="delta_accuracy_pp"
        ).reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=column_order
        )
        heatmap = sns.heatmap(
            matrix,
            annot=base.numeric_annotations(matrix, 1, signed=True),
            annot_kws={"fontsize": 7.3, "fontweight": "normal"},
            fmt="",
            cmap="vlag",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.35,
            cbar=index == 0,
            cbar_ax=colorbar_ax if index == 0 else None,
            ax=ax,
        )
        if index == 0:
            colorbar = heatmap.collections[0].colorbar
        ax.set_title(SUBSET_LABELS[subset], fontweight="semibold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        base.style_heatmap(ax, 52)
    base.style_colorbar(colorbar, "Delta akurasi (poin persentase)")
    fig.suptitle("Delta Akurasi Skenario Implisit terhadap Eksplisit", fontweight="bold")
    return base.save_figure(
        fig, figure_dir, "02_implicit_minus_explicit_accuracy_delta_heatmap"
    )


def plot_scenario_effect_sizes(
    scenario: pd.DataFrame,
    figure_dir: Path,
    subsets: list[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    short_labels = [spec[2] for spec in SCENARIO_SPECS]
    model_labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    for subset in subsets or SUBSET_ORDER:
        selected = scenario[scenario["subset"].eq(subset)]
        all_values = selected.pivot(
            index="model_label", columns="comparison_label", values="cohens_h"
        )
        all_decisions = selected.pivot(
            index="model_label",
            columns="comparison_label",
            values="significant_holm_0_05",
        )
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035])
        axes = [
            fig.add_subplot(grid[row, column])
            for row in range(2)
            for column in range(2)
        ]
        colorbar_ax = fig.add_subplot(grid[:, 2])
        colorbar = None
        for panel_index, (ax, identity) in enumerate(zip(axes, RELIGION_ORDER)):
            label = RELIGION_LABELS[identity]
            source_columns = [f"{label} {short}" for short in short_labels]
            values = all_values.reindex(
                index=model_labels, columns=source_columns
            ).copy()
            decisions = all_decisions.reindex(
                index=model_labels, columns=source_columns
            ).astype(bool)
            values.columns = short_labels
            decisions.columns = short_labels
            heatmap = sns.heatmap(
                values,
                annot=base._effect_annotations(values, decisions),
                annot_kws={"fontsize": 8.3, "fontweight": "normal"},
                fmt="",
                cmap="vlag",
                center=0,
                vmin=-base.COHENS_H_COLORBAR_LIMIT,
                vmax=base.COHENS_H_COLORBAR_LIMIT,
                linewidths=0.4,
                cbar=panel_index == 0,
                cbar_ax=colorbar_ax if panel_index == 0 else None,
                ax=ax,
            )
            if panel_index == 0:
                colorbar = heatmap.collections[0].colorbar
            ax.set_title(label, fontweight="semibold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            base.style_heatmap(ax, 0)
        base.style_colorbar(colorbar, "Cohen’s h")
        fig.suptitle(
            f"Effect Size Skenario {DOMAIN_TITLE} — {SUBSET_LABELS[subset]}\n"
            "* = signifikan setelah koreksi Holm (α = 0,05)",
            fontweight="bold",
        )
        fig.supxlabel(
            "Perbandingan (skenario pertama dikurangi skenario pembanding)"
        )
        paths += base.save_figure(
            fig, figure_dir, f"03_effect_size_{subset}_holm_heatmap"
        )
    return paths


def plot_pairwise_effect_sizes(
    pairwise: pd.DataFrame,
    figure_dir: Path,
    subsets: list[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    pair_order = list(combinations(RELIGION_ORDER, 2))
    column_order = [
        f"{RELIGION_LABELS[a]} − {RELIGION_LABELS[b]} {scenario}"
        for a, b in pair_order
        for scenario in ["E", "I1", "I2", "I3"]
    ]
    model_labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    for subset in subsets or SUBSET_ORDER:
        selected = pairwise[pairwise["subset"].eq(subset)]
        values = selected.pivot(
            index="model_label", columns="comparison_label", values="cohens_h"
        ).reindex(index=model_labels, columns=column_order)
        decisions = selected.pivot(
            index="model_label",
            columns="comparison_label",
            values="significant_holm_0_05",
        ).reindex(index=model_labels, columns=column_order).astype(bool)
        fig = plt.figure(figsize=(20, 6.5), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.025])
        ax = fig.add_subplot(grid[0, 0])
        colorbar_ax = fig.add_subplot(grid[0, 1])
        heatmap = sns.heatmap(
            values,
            annot=base._effect_annotations(values, decisions),
            annot_kws={"fontsize": 6.3, "fontweight": "normal"},
            fmt="",
            cmap="vlag",
            center=0,
            vmin=-base.COHENS_H_COLORBAR_LIMIT,
            vmax=base.COHENS_H_COLORBAR_LIMIT,
            linewidths=0.3,
            cbar_ax=colorbar_ax,
            ax=ax,
        )
        ax.set_title(
            f"Effect Size {PAIRWISE_TITLE} — {SUBSET_LABELS[subset]}\n"
            "* = signifikan setelah koreksi Holm atas 24 perbandingan (α = 0,05)",
            fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        base.style_heatmap(ax, 63)
        base.style_colorbar(heatmap.collections[0].colorbar, "Cohen’s h")
        paths += base.save_figure(
            fig,
            figure_dir,
            f"03b_effect_size_{DOMAIN_SLUG}_pairwise_{subset}_holm_heatmap",
        )
    return paths


def plot_abstain_reasons(abstain: pd.DataFrame, figure_dir: Path) -> list[Path]:
    display_data = abstain.copy()
    display_data["figure_reason_label"] = display_data["reason"].map(
        base.ABSTAIN_REASON_FIGURE_LABELS
    ).fillna(display_data["reason_label"])
    matrix = display_data.pivot(
        index="figure_reason_label",
        columns="persona_label",
        values="percentage_within_persona_abstain",
    ).fillna(0)
    matrix = matrix.reindex(columns=[PERSONA_LABELS[persona] for persona in PERSONA_ORDER])
    matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(18, 7.5), constrained_layout=True)
    heatmap = sns.heatmap(
        matrix,
        annot=base.numeric_annotations(matrix, 1),
        annot_kws={"fontsize": 6.5, "fontweight": "normal"},
        fmt="",
        cmap="mako",
        linewidths=0.35,
        ax=ax,
    )
    ax.set_title(f"Alasan Abstain menurut {DOMAIN_TITLE}", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    base.style_heatmap(ax, 52)
    base.style_colorbar(heatmap.collections[0].colorbar, "% dari abstain pada persona")
    return base.save_figure(fig, figure_dir, "05_abstain_reason_by_persona_heatmap")


def _figure_missing(figure_dir: Path, stem: str) -> bool:
    return any(
        not (figure_dir / f"{stem}.{suffix}").is_file()
        or (figure_dir / f"{stem}.{suffix}").stat().st_size == 0
        for suffix in ["png", "svg"]
    )


def render_missing_figures(output_dir: Path) -> list[Path]:
    """Resume an interrupted render from the already validated table artifacts."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def read_table(name: str) -> pd.DataFrame:
        path = table_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Tabel sumber visualisasi tidak ditemukan: {path}")
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    condition = read_table("figure_data_accuracy_absolute.csv")
    implicit_delta = read_table("figure_data_implicit_minus_explicit_delta.csv")
    scenario = read_table("mcnemar_scenario_comparisons.csv")
    pairwise = read_table(f"mcnemar_{DOMAIN_SLUG}_pairwise_comparisons.csv")
    status = read_table("figure_data_response_status.csv")
    abstain = read_table("figure_data_abstain_reason.csv")

    base.configure_plot_style()
    rendered: list[Path] = []
    if _figure_missing(figure_dir, "01_accuracy_absolute_all_subsets_heatmap"):
        paths, _ = plot_accuracy_absolute(condition, figure_dir)
        rendered += paths
    if _figure_missing(
        figure_dir, "02_implicit_minus_explicit_accuracy_delta_heatmap"
    ):
        rendered += plot_implicit_delta(implicit_delta, figure_dir)

    missing_scenario = [
        subset
        for subset in SUBSET_ORDER
        if _figure_missing(figure_dir, f"03_effect_size_{subset}_holm_heatmap")
    ]
    if missing_scenario:
        rendered += plot_scenario_effect_sizes(
            scenario, figure_dir, subsets=missing_scenario
        )

    missing_pairwise = [
        subset
        for subset in SUBSET_ORDER
        if _figure_missing(
            figure_dir,
            f"03b_effect_size_{DOMAIN_SLUG}_pairwise_{subset}_holm_heatmap",
        )
    ]
    if missing_pairwise:
        rendered += plot_pairwise_effect_sizes(
            pairwise, figure_dir, subsets=missing_pairwise
        )

    if _figure_missing(figure_dir, "04_response_status_distribution_by_model"):
        rendered += base.plot_status_distribution(status, figure_dir)
    if _figure_missing(figure_dir, "05_abstain_reason_by_persona_heatmap"):
        rendered += plot_abstain_reasons(abstain, figure_dir)
    return rendered


def finalize_existing_output(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Validate resumed figures/tables and write the manifest and output index."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    input_hash_before = base.sha256_file(input_path)
    frame = load_data(input_path)
    input_validation = validate_input(frame)
    input_hash_after = base.sha256_file(input_path)

    def read_table(name: str) -> pd.DataFrame:
        return pd.read_csv(table_dir / name, encoding="utf-8-sig", low_memory=False)

    condition = read_table("accuracy_by_subset_model_condition.csv")
    cochran = read_table(f"cochran_q_by_subset_model_{DOMAIN_SLUG}.csv")
    scenario = read_table("mcnemar_scenario_comparisons.csv")
    pairwise = read_table(f"mcnemar_{DOMAIN_SLUG}_pairwise_comparisons.csv")
    all_comparisons = read_table("mcnemar_all_comparisons.csv")
    scenario_family_sizes = scenario.groupby(
        ["subset", "model", "identity"], observed=True
    ).size()
    pairwise_family_sizes = pairwise.groupby(["subset", "model"], observed=True).size()
    pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in all_comparisons[["subset", "n_pairs"]].itertuples(index=False)
    )
    cochran_pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in cochran[["subset", "n_pairs"]].itertuples(index=False)
    )
    png_files = sorted(figure_dir.glob("*.png"))
    svg_files = sorted(figure_dir.glob("*.svg"))
    raster_dimensions = {
        path.name: list(PILImage.open(path).size) for path in png_files
    }
    validation = {
        "valid": bool(
            input_validation["valid"]
            and input_hash_before == input_hash_after
            and len(condition) == 408
            and len(cochran) == 96
            and len(scenario) == 672
            and len(pairwise) == 576
            and len(all_comparisons) == 1_248
            and pair_expected
            and cochran_pair_expected
            and scenario_family_sizes.eq(7).all()
            and pairwise_family_sizes.eq(24).all()
            and len(png_files) == 12
            and len(svg_files) == 12
            and all(path.stat().st_size > 0 for path in png_files + svg_files)
        ),
        "analysis_version": ANALYSIS_VERSION,
        "rows_processed": int(len(frame)),
        "all_154428_rows_used": bool(len(frame) == EXPECTED_ROWS),
        "subsets_scored_against_ground_truth": SUBSET_ORDER,
        "full_denominator_accuracy": True,
        "abstain_missing_counted_in_denominator_as_incorrect": bool(
            input_validation["abstain_missing_scored_false"]
        ),
        "duplicate_response_keys": int(input_validation["duplicate_response_keys"]),
        "scenario_pair_rows": int(len(scenario)),
        f"{DOMAIN_SLUG}_pairwise_rows": int(len(pairwise)),
        "cochran_q_rows": int(len(cochran)),
        "scenario_holm_family_size_exactly_7": bool(scenario_family_sizes.eq(7).all()),
        f"{DOMAIN_SLUG}_pairwise_holm_family_size_exactly_24": bool(
            pairwise_family_sizes.eq(24).all()
        ),
        "pair_counts_match_subset_question_counts": bool(
            pair_expected and cochran_pair_expected
        ),
        "minimum_n_pairs": int(all_comparisons["n_pairs"].min()),
        "maximum_n_pairs": int(all_comparisons["n_pairs"].max()),
        "figure_sources_read_directly_from_tables": True,
        "png_files": int(len(png_files)),
        "svg_files": int(len(svg_files)),
        "all_figure_files_nonempty": bool(
            all(path.stat().st_size > 0 for path in png_files + svg_files)
        ),
        "raster_dimensions_px": raster_dimensions,
        "table_csv_files": int(len(list(table_dir.glob("*.csv")))),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "source_csv_unchanged": bool(input_hash_before == input_hash_after),
        "api_calls": 0,
    }
    if not validation["valid"]:
        raise ValueError(f"Validasi output resume gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "input": str(input_path),
        "output": str(output_dir),
        "method": (
            "Akurasi biner full-denominator; Cochran's Q; Exact McNemar "
            "berpasangan; Holm per identitas (7 uji) dan antaridentitas per "
            "subset-model (24 uji)."
        ),
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "accuracy_denominator": "seluruh respons pada kondisi",
        "nonanswer_scoring": "abstain dan missing bernilai salah",
        "pairing": "ID soal yang sama",
        "alpha": ALPHA,
        "api_calls": 0,
        "validation": validation,
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_index = base.build_output_index(output_dir)
    base.write_csv(output_index, output_dir / "output_index.csv")
    return validation


def run_analysis(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    tables_only: bool = False,
) -> dict[str, object]:
    repo_root = base.find_repo_root()
    input_path = Path(input_path) if input_path else default_input_path(repo_root)
    output_dir = Path(output_dir) if output_dir else default_output_dir(repo_root)
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in [table_dir, figure_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"ETHICS {DOMAIN_TITLE.upper()} — V3 FULL-DENOMINATOR ACCURACY")
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print("API call: 0")
    print("=" * 78)

    input_hash_before = base.sha256_file(input_path)
    frame = load_data(input_path)
    input_validation = validate_input(frame)
    condition_cache = build_condition_cache(frame)

    print("[TABLE] Menghitung akurasi deskriptif")
    tables = build_accuracy_tables(frame)
    condition = tables["accuracy_by_subset_model_condition.csv"]
    status = build_status_table(frame)
    abstain = build_abstain_reason_table(frame)
    tables["response_status_by_model.csv"] = status
    tables["abstain_reason_by_persona.csv"] = abstain

    print("[STATS] Cochran's Q dan Exact McNemar berpasangan")
    cochran = build_cochran_q(condition_cache)
    scenario, pairwise = build_mcnemar(condition_cache)
    all_comparisons = pd.concat([scenario, pairwise], ignore_index=True, sort=False)
    significant_overall, significant_by_subset, significant_by_model = (
        build_significant_summaries(all_comparisons)
    )
    tables[f"cochran_q_by_subset_model_{DOMAIN_SLUG}.csv"] = cochran
    tables["mcnemar_scenario_comparisons.csv"] = scenario
    tables[f"mcnemar_{DOMAIN_SLUG}_pairwise_comparisons.csv"] = pairwise
    tables["mcnemar_all_comparisons.csv"] = all_comparisons
    tables["significant_results_overall.csv"] = significant_overall
    tables["significant_results_by_subset.csv"] = significant_by_subset
    tables["significant_results_by_model.csv"] = significant_by_model

    implicit_delta = build_implicit_delta_data(condition)
    tables["figure_data_accuracy_absolute.csv"] = condition.copy()
    tables["figure_data_implicit_minus_explicit_delta.csv"] = implicit_delta
    tables["figure_data_effect_sizes.csv"] = all_comparisons.copy()
    tables["figure_data_response_status.csv"] = status.copy()
    tables["figure_data_abstain_reason.csv"] = abstain.copy()

    print(f"[WRITE] Menulis {len(tables)} tabel CSV")
    for name, table in tables.items():
        base.write_csv(table, table_dir / name)

    if tables_only:
        print(f"[DONE] Tabel selesai: {len(tables)} CSV; visualisasi belum dirender")
        return {
            "input_path": input_path,
            "output_dir": output_dir,
            "tables": tables,
            "input_validation": input_validation,
        }

    print("[PLOT] Membuat PNG 300 DPI + SVG")
    base.configure_plot_style()
    figure_paths: list[Path] = []
    paths, accuracy_figure_data = plot_accuracy_absolute(condition, figure_dir)
    figure_paths += paths
    figure_paths += plot_implicit_delta(implicit_delta, figure_dir)
    figure_paths += plot_scenario_effect_sizes(scenario, figure_dir)
    figure_paths += plot_pairwise_effect_sizes(pairwise, figure_dir)
    figure_paths += base.plot_status_distribution(status, figure_dir)
    figure_paths += plot_abstain_reasons(abstain, figure_dir)

    input_hash_after = base.sha256_file(input_path)
    pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in all_comparisons[["subset", "n_pairs"]].itertuples(index=False)
    )
    cochran_pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in cochran[["subset", "n_pairs"]].itertuples(index=False)
    )
    scenario_family_sizes = scenario.groupby(
        ["subset", "model", "identity"], observed=True
    ).size()
    pairwise_family_sizes = pairwise.groupby(["subset", "model"], observed=True).size()
    png_files = sorted(figure_dir.glob("*.png"))
    svg_files = sorted(figure_dir.glob("*.svg"))
    raster_dimensions = {
        path.name: list(PILImage.open(path).size) for path in png_files
    }
    figure_table_consistency = {
        "accuracy_absolute": base._frame_equal_on_columns(
            condition,
            accuracy_figure_data,
            ["subset", "model", "persona", "accuracy_pct"],
        ),
        "implicit_delta": base._frame_equal_on_columns(
            implicit_delta,
            tables["figure_data_implicit_minus_explicit_delta.csv"],
            ["subset", "model", "identity", "implicit_scenario", "delta_accuracy_pp"],
        ),
        "effect_sizes": base._frame_equal_on_columns(
            all_comparisons,
            tables["figure_data_effect_sizes.csv"],
            ["subset", "model", "comparison_label", "cohens_h"],
        ),
        "response_status": base._frame_equal_on_columns(
            status,
            tables["figure_data_response_status.csv"],
            ["model", "status_label", "percentage"],
        ),
        "abstain_reason": base._frame_equal_on_columns(
            abstain,
            tables["figure_data_abstain_reason.csv"],
            ["persona", "reason", "percentage_within_persona_abstain"],
        ),
    }
    validation = {
        "valid": bool(
            input_validation["valid"]
            and input_hash_before == input_hash_after
            and len(frame) == EXPECTED_ROWS
            and len(condition) == 408
            and len(cochran) == 96
            and len(scenario) == 672
            and len(pairwise) == 576
            and len(all_comparisons) == 1_248
            and pair_expected
            and cochran_pair_expected
            and scenario_family_sizes.eq(7).all()
            and pairwise_family_sizes.eq(24).all()
            and all(figure_table_consistency.values())
            and len(png_files) == 12
            and len(svg_files) == 12
            and all(path.stat().st_size > 0 for path in png_files + svg_files)
        ),
        "analysis_version": ANALYSIS_VERSION,
        "rows_processed": int(len(frame)),
        "all_154428_rows_used": bool(len(frame) == EXPECTED_ROWS),
        "subsets_scored_against_ground_truth": SUBSET_ORDER,
        "full_denominator_accuracy": True,
        "abstain_missing_counted_in_denominator_as_incorrect": bool(
            input_validation["abstain_missing_scored_false"]
        ),
        "pair_key": ["subset", "model", "question_id", f"{DOMAIN_SLUG}_or_scenario"],
        "duplicate_response_keys": int(input_validation["duplicate_response_keys"]),
        "duplicate_pairs": 0,
        "scenario_pair_rows": int(len(scenario)),
        f"{DOMAIN_SLUG}_pairwise_rows": int(len(pairwise)),
        "cochran_q_rows": int(len(cochran)),
        "scenario_holm_family_size_exactly_7": bool(scenario_family_sizes.eq(7).all()),
        f"{DOMAIN_SLUG}_pairwise_holm_family_size_exactly_24": bool(
            pairwise_family_sizes.eq(24).all()
        ),
        "pair_counts_match_subset_question_counts": bool(
            pair_expected and cochran_pair_expected
        ),
        "minimum_n_pairs": int(all_comparisons["n_pairs"].min()),
        "maximum_n_pairs": int(all_comparisons["n_pairs"].max()),
        "figure_table_consistency": figure_table_consistency,
        "png_files": int(len(png_files)),
        "svg_files": int(len(svg_files)),
        "all_figure_files_nonempty": bool(
            all(path.stat().st_size > 0 for path in png_files + svg_files)
        ),
        "raster_dimensions_px": raster_dimensions,
        "table_csv_files": int(len(list(table_dir.glob("*.csv")))),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "source_csv_unchanged": bool(input_hash_before == input_hash_after),
        "api_calls": 0,
    }
    if not validation["valid"]:
        raise ValueError(f"Validasi output gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "input": str(input_path),
        "output": str(output_dir),
        "method": (
            "Akurasi biner full-denominator; Cochran's Q; Exact McNemar "
            "berpasangan; Holm per identitas (7 uji) dan antaridentitas per "
            "subset-model (24 uji)."
        ),
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "accuracy_denominator": "seluruh respons pada kondisi",
        "nonanswer_scoring": "abstain dan missing bernilai salah",
        "pairing": "ID soal yang sama",
        "alpha": ALPHA,
        "api_calls": 0,
        "validation": validation,
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_index = base.build_output_index(output_dir)
    base.write_csv(output_index, output_dir / "output_index.csv")

    print(f"[DONE] Analisis {DOMAIN_TITLE} v3 selesai dan seluruh validasi lulus")
    print(
        f"       rows={len(frame):,} | tables={len(tables)} | "
        f"figure_designs={len(png_files)} | figure_files={len(png_files) + len(svg_files)}"
    )
    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "validation": validation,
        "output_index": output_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build ETHICS {DOMAIN_TITLE} v3 accuracy visualizations."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Resume missing figures from existing tables, then validate/finalize.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Build validated statistical tables without rendering figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figures_only and args.tables_only:
        raise SystemExit("--figures-only dan --tables-only tidak dapat dipakai bersamaan")
    if args.figures_only:
        repo_root = base.find_repo_root()
        input_path = (args.input or default_input_path(repo_root)).resolve()
        output_dir = (args.output or default_output_dir(repo_root)).resolve()
        print(f"[RESUME] Output: {output_dir}")
        rendered = render_missing_figures(output_dir)
        print(f"[RESUME] File gambar baru: {len(rendered)}")
        validation = finalize_existing_output(input_path, output_dir)
        print(
            f"[DONE] Resume selesai | png={validation['png_files']} | "
            f"svg={validation['svg_files']} | valid={validation['valid']}"
        )
    else:
        run_analysis(
            input_path=args.input,
            output_dir=args.output,
            tables_only=args.tables_only,
        )


if __name__ == "__main__":
    main()

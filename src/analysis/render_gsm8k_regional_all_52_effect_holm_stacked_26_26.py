"""Render Regional Cultural comparisons as stacked 14/14/12/12 panels.

The figure contains primary within-region comparisons, exploratory matched
cross-region comparisons, and descriptive sensitivity comparisons against the
shared Gender Neutral run. The CLI reads the already-computed table and never
rewrites statistical results; the recomputation helpers remain available for
separate audit work. The legacy ``26_26`` output stem is retained for LaTeX
path compatibility.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
REGIONAL_ROOT = REPORT_ROOT / "Regional_Cultural_Persona"
GENDER_ROOT = REPO_ROOT / "result" / "reports"
OUTPUT_ROOT = REPORT_ROOT / "persona_publication_analysis_v5_paired_significance_six_models"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_PATH = TABLE_DIR / "regional_all_52_paired_effects_significance.csv"
DEFAULT_OUTPUT = FIGURE_DIR / "27_regional_all_52_effect_holm_stacked_26_26.png"

MODEL_IDS = [
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
MODEL_ORDER = [MODEL_LABELS[model] for model in MODEL_IDS]

MODEL_ROW_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next 80B-A3B",
    "Llama 3.1 8B": "Llama 3.1 8B",
    "Ministral 8B": "Ministral 8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill 32B",
    "GPT-OSS 20B": "GPT-OSS 20B",
    "Llama 3.3 70B": "Llama 3.3 70B",
}

PANEL_SPECS = [
    {
        "title": "Panel 1 — Amerika Utara dan Asia Timur",
        "start": 0,
        "stop": 14,
    },
    {
        "title": "Panel 2 — Asia Selatan dan Afrika Sub-Sahara",
        "start": 14,
        "stop": 28,
    },
    {
        "title": "Panel 3 — Amerika Utara dibandingkan wilayah lain",
        "start": 28,
        "stop": 40,
    },
    {
        "title": "Panel 4 — Perbandingan lintas wilayah lainnya",
        "start": 40,
        "stop": 52,
    },
]

MAX_PANEL_COLUMNS = 14

REGIONS = [
    ("north_american", "North American", "NA"),
    ("east_asian", "East Asian", "EA"),
    ("south_asian", "South Asian", "SA"),
    ("sub_saharan_african", "Sub-Saharan African", "SSA"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path; the statistical source is always TABLE_PATH.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    unexpected = set(normalized.dropna().unique()) - {"true", "false", "1", "0", ""}
    if unexpected:
        raise ValueError(f"{column}: nilai boolean tidak dikenal: {sorted(unexpected)}")
    return normalized.map(
        {"true": True, "1": True, "false": False, "0": False, "": False}
    ).fillna(False).astype(bool)


def gender_neutral_path(model: str) -> Path:
    direct = GENDER_ROOT / model / "results_neutral.csv"
    if direct.exists():
        return direct
    if model == "openai_gpt-oss-20b":
        combined = GENDER_ROOT / model / "results_openai-gpt-oss_20b.csv"
        if combined.exists():
            return combined
    raise FileNotFoundError(f"Neutral source tidak ditemukan untuk {model}")


def load_neutral_outcomes(model: str) -> pd.Series:
    path = gender_neutral_path(model)
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in {"persona_folder", "file", "is_correct"},
        low_memory=False,
    )
    frame = frame[frame["persona_folder"].astype(str).eq("neutral")].copy()
    frame["question_id"] = pd.to_numeric(
        frame["file"].astype(str).str.extract(r"gsm8k_(\d+)\.json", expand=False),
        errors="raise",
    ).astype(int)
    frame["is_correct"] = normalize_bool(frame["is_correct"], "is_correct")
    assert not frame.duplicated("question_id").any()
    return frame.set_index("question_id")["is_correct"].sort_index()


def load_regional_outcomes(model: str) -> dict[str, pd.Series]:
    path = REGIONAL_ROOT / model / "results.csv"
    frame = pd.read_csv(
        path,
        usecols=["model", "persona", "question_id", "is_correct", "technical_failure"],
        low_memory=False,
    )
    assert set(frame["model"].astype(str)) == {model}
    frame["question_id"] = pd.to_numeric(frame["question_id"], errors="raise").astype(int)
    frame["is_correct"] = normalize_bool(frame["is_correct"], "is_correct")
    frame["technical_failure"] = normalize_bool(
        frame["technical_failure"], "technical_failure"
    )
    frame = frame[~frame["technical_failure"]].copy()
    assert not frame.duplicated(["persona", "question_id"]).any()
    return {
        persona: group.set_index("question_id")["is_correct"].sort_index()
        for persona, group in frame.groupby("persona", sort=False)
    }


def comparison_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    display_order = 0
    for key, label, short in REGIONS:
        conditions = [
            (f"{key}_explicit", "E"),
            (f"{key}_implicit_1", "I1"),
            (f"{key}_implicit_2", "I2"),
            (f"{key}_implicit_3", "I3"),
        ]
        for persona, condition in conditions:
            specs.append(
                {
                    "persona_a": persona,
                    "persona_b": "neutral",
                    "comparison_label": f"{label} {'Explicit' if condition == 'E' else f'Implicit {condition[1]}'} - Neutral",
                    "axis_label": f"{short}\n{condition}−N†",
                    "family": f"{key}_neutral_sensitivity_four",
                    "analysis_scope": "sensitivity",
                    "display_order": display_order,
                }
            )
            display_order += 1
        for variant in (1, 2, 3):
            specs.append(
                {
                    "persona_a": f"{key}_implicit_{variant}",
                    "persona_b": f"{key}_explicit",
                    "comparison_label": f"{label} Implicit {variant} - {label} Explicit",
                    "axis_label": f"{short}\nI{variant}−E",
                    "family": f"{key}_primary_three",
                    "analysis_scope": "primary",
                    "display_order": display_order,
                }
            )
            display_order += 1

    for left_index, (left_key, left_label, left_short) in enumerate(REGIONS):
        for right_key, right_label, right_short in REGIONS[left_index + 1 :]:
            conditions = [
                ("explicit", "E", "Explicit"),
                ("implicit_1", "I1", "Implicit 1"),
                ("implicit_2", "I2", "Implicit 2"),
                ("implicit_3", "I3", "Implicit 3"),
            ]
            for suffix, short_condition, full_condition in conditions:
                specs.append(
                    {
                        "persona_a": f"{left_key}_{suffix}",
                        "persona_b": f"{right_key}_{suffix}",
                        "comparison_label": f"{left_label} {full_condition} - {right_label} {full_condition}",
                        "axis_label": f"{left_short}−{right_short}\n{short_condition}‡",
                        "family": "regional_cross_identity_twenty_four",
                        "analysis_scope": "exploratory",
                        "display_order": display_order,
                    }
                )
                display_order += 1
    assert len(specs) == 52
    return specs


def cohens_h(proportion_a: float, proportion_b: float) -> float:
    return 2 * math.asin(math.sqrt(proportion_a)) - 2 * math.asin(
        math.sqrt(proportion_b)
    )


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)


def holm_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    adjusted_sorted = np.empty(len(array), dtype=float)
    running = 0.0
    family_size = len(array)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * array[original_index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(array), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return pd.Series(adjusted, index=values.index)


def compute_all_pairwise() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    specs = comparison_specs()
    rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        regional = load_regional_outcomes(model)
        neutral = load_neutral_outcomes(model)
        all_outcomes = {**regional, "neutral": neutral}
        for spec in specs:
            left = all_outcomes[str(spec["persona_a"])].rename("a")
            right = all_outcomes[str(spec["persona_b"])].rename("b")
            paired = pd.concat([left, right], axis=1, join="inner").dropna()
            a = paired["a"].astype(bool)
            b = paired["b"].astype(bool)
            a_only = int((a & ~b).sum())
            b_only = int((~a & b).sum())
            proportion_a = float(a.mean())
            proportion_b = float(b.mean())
            rows.append(
                {
                    **spec,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n_pairs": int(len(paired)),
                    "accuracy_a_pct": 100 * proportion_a,
                    "accuracy_b_pct": 100 * proportion_b,
                    "risk_difference_pp": 100 * (proportion_a - proportion_b),
                    "cohen_h": cohens_h(proportion_a, proportion_b),
                    "discordant_a_correct_only": a_only,
                    "discordant_b_correct_only": b_only,
                    "mcnemar_exact_p_value": exact_mcnemar_p(a_only, b_only),
                }
            )

    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_value"] = frame.groupby(
        ["model", "family"], sort=False
    )["mcnemar_exact_p_value"].transform(holm_adjust)
    frame["family_size"] = frame.groupby(["model", "family"])[
        "comparison_label"
    ].transform("size")
    frame["significant_holm_0_05"] = frame["holm_adjusted_p_value"].lt(0.05)
    frame = frame.sort_values(["display_order", "model_label"]).reset_index(drop=True)

    # The 12 pre-existing primary comparisons must be reproduced exactly.
    verified = pd.read_csv(TABLE_DIR / "regional_paired_effects_significance.csv")
    primary = frame[frame["analysis_scope"].eq("primary")]
    check = primary.merge(
        verified[
            [
                "model",
                "comparison_label",
                "n_pairs",
                "cohen_h",
                "mcnemar_exact_p_value",
                "holm_adjusted_p_value",
            ]
        ],
        on=["model", "comparison_label"],
        suffixes=("_new", "_verified"),
        validate="one_to_one",
    )
    assert len(check) == 72
    assert (check["n_pairs_new"] == check["n_pairs_verified"]).all()
    for column in ["cohen_h", "mcnemar_exact_p_value", "holm_adjusted_p_value"]:
        assert np.allclose(
            check[f"{column}_new"], check[f"{column}_verified"], atol=1e-12, rtol=1e-10
        ), column

    output_columns = [
        "model",
        "model_label",
        "analysis_scope",
        "family",
        "family_size",
        "display_order",
        "persona_a",
        "persona_b",
        "comparison_label",
        "axis_label",
        "n_pairs",
        "accuracy_a_pct",
        "accuracy_b_pct",
        "risk_difference_pp",
        "cohen_h",
        "discordant_a_correct_only",
        "discordant_b_correct_only",
        "mcnemar_exact_p_value",
        "holm_adjusted_p_value",
        "significant_holm_0_05",
    ]
    frame[output_columns].to_csv(
        TABLE_DIR / "regional_all_52_paired_effects_significance.csv", index=False
    )
    return frame, specs


def format_effect(value: float, significant: bool) -> str:
    if abs(value) < 0.005:
        text = "0,00"
    else:
        text = f"{value:+.2f}".replace(".", ",")
    return text + ("*" if significant else "")


def annotation_matrix(
    effects: pd.DataFrame, significant: pd.DataFrame
) -> pd.DataFrame:
    annotations = effects.copy().astype(object)
    for model in effects.index:
        for comparison in effects.columns:
            annotations.loc[model, comparison] = format_effect(
                float(effects.loc[model, comparison]),
                bool(significant.loc[model, comparison]),
            )
    return annotations


def stack_pair_label(label: str) -> str:
    """Wrap cross-region abbreviations without changing their meaning."""
    lines = label.splitlines()
    if lines and "−" in lines[0]:
        left, right = lines[0].split("−", maxsplit=1)
        return "\n".join([f"{left}−", right, *lines[1:]])
    return label


def render(
    frame: pd.DataFrame,
    specs: list[dict[str, object]],
    output: Path,
    dpi: int,
) -> None:
    comparison_order = [str(spec["comparison_label"]) for spec in specs]
    axis_labels = [stack_pair_label(str(spec["axis_label"])) for spec in specs]
    selected = frame[frame["comparison_label"].isin(comparison_order)]

    # Lock the complete statistical surface before any plotting operation.
    assert len(selected) == len(MODEL_ORDER) * len(comparison_order)
    assert set(selected["model_label"]) == set(MODEL_ORDER)
    assert set(selected["comparison_label"]) == set(comparison_order)

    effects = selected.pivot(
        index="model_label", columns="comparison_label", values="cohen_h"
    ).reindex(index=MODEL_ORDER, columns=comparison_order)
    significant = selected.pivot(
        index="model_label", columns="comparison_label", values="significant_holm_0_05"
    ).reindex(index=MODEL_ORDER, columns=comparison_order).astype(bool)
    assert effects.notna().all().all()
    assert effects.shape == (6, 52)
    assert significant.shape == effects.shape

    max_abs = float(np.nanmax(np.abs(effects.to_numpy(dtype=float))))
    limit = max(0.10, math.ceil(max_abs / 0.05) * 0.05)
    annotations = annotation_matrix(effects, significant)

    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "text.color": "#262626",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "regional_effect",
        ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
    )

    fig = plt.figure(figsize=(18, 24))
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[6, 6, 6, 6, 0.5],
        left=0.235,
        right=0.985,
        top=0.93,
        bottom=0.12,
        hspace=0.72,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(4)]
    colorbar_ax = fig.add_subplot(grid[4])
    heatmap = None

    for panel_index, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        start = int(spec["start"])
        stop = int(spec["stop"])
        panel_effects = effects.iloc[:, start:stop]
        panel_significant = significant.iloc[:, start:stop]
        panel_annotations = annotations.iloc[:, start:stop]
        show_colorbar = panel_index == len(PANEL_SPECS) - 1

        heatmap = sns.heatmap(
            panel_effects,
            annot=panel_annotations,
            fmt="",
            cmap=cmap,
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=1.25,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 18, "fontweight": "normal"},
            cbar=show_colorbar,
            cbar_ax=colorbar_ax if show_colorbar else None,
            cbar_kws={"orientation": "horizontal"} if show_colorbar else None,
            ax=ax,
        )
        ax.set_xlim(0, MAX_PANEL_COLUMNS)

        for text_artist, value, is_significant in zip(
            ax.texts,
            panel_effects.to_numpy().ravel(),
            panel_significant.to_numpy().ravel(),
        ):
            text_artist.set_color(
                "white" if abs(float(value)) >= 0.46 * limit else "#262626"
            )
            text_artist.set_fontweight(
                "bold" if bool(is_significant) else "normal"
            )

        ax.set_yticklabels(
            [MODEL_ROW_LABELS[model] for model in MODEL_ORDER],
            rotation=0,
            fontsize=18.5,
            fontweight="bold",
        )
        ax.set_xticklabels(
            axis_labels[start:stop],
            rotation=0,
            ha="center",
            fontsize=17,
            fontweight="semibold",
        )
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xlabel(
            "Perbandingan skenario",
            fontsize=19.5,
            fontweight="normal",
            labelpad=12,
        )
        ax.set_ylabel("Model", fontsize=19.5, fontweight="normal", labelpad=15)
        ax.set_title(
            str(spec["title"]),
            fontsize=21,
            fontweight="bold",
            pad=13,
        )

    assert heatmap is not None
    colorbar = heatmap.collections[0].colorbar
    ticks = np.linspace(-limit, limit, 9)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.2f}".replace("-", "−").replace(".", ",") for value in ticks])
    colorbar.ax.tick_params(labelsize=17)
    colorbar.set_label("Cohen's h", fontsize=19, fontweight="bold", labelpad=8)

    fig.suptitle(
        "Ukuran Efek dan Hasil Koreksi Holm pada Domain Regional-Kultural",
        fontsize=27,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.008,
        "* = signifikan setelah koreksi Holm (p-Holm < 0,05). † = perbandingan terhadap Gender Neutral "
        "(analisis sensitivitas; protokol berbeda).\n"
        "‡ = perbandingan antarwilayah (eksploratif). E = eksplisit; I1–I3 = implisit; N = netral.\n"
        "NA = Amerika Utara; EA = Asia Timur; SA = Asia Selatan; SSA = Afrika Sub-Sahara.\n"
        "Cohen's h dihitung dari skenario pertama dikurangi skenario pembanding.",
        ha="center",
        va="bottom",
        fontsize=17.5,
        color="#3F3F3F",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("DPI output harus minimal 300")

    frame = pd.read_csv(TABLE_PATH)
    render(frame, comparison_specs(), args.output, args.dpi)


if __name__ == "__main__":
    main()

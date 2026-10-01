"""Add prespecified within-persona and matched-gender comparisons for 14 models.

This extends the existing persona-vs-Neutral pilot artifact without creating
the full Cartesian set of condition pairs.  The 18 comparisons per model are
the same interpretable contrasts used by the six-model publication notebook:

* seven Man primary comparisons;
* seven Woman primary comparisons;
* four matched Man-vs-Woman exploratory comparisons.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, chi2


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "gsm8k_full_1319_delta_exact_mcnemar_14_models"
)
EXPECTED_N = 1_319
BOOTSTRAP_REPLICATES = 10_000

SELECTED_MODELS = {
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.3-70b-instruct",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
}


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def cohen_h(p_a: float, p_b: float) -> float:
    return float(2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)))


def holm_adjust(values: pd.Series) -> np.ndarray:
    p_values = values.astype(float).to_numpy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def bootstrap_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def paired_bootstrap_ci(
    n11: int, n10: int, n01: int, n00: int, key: str
) -> tuple[float, float]:
    """Exact paired bootstrap simulation using multinomial category counts."""
    n = n11 + n10 + n01 + n00
    probabilities = np.asarray([n11, n10, n01, n00], dtype=float) / n
    rng = np.random.default_rng(bootstrap_seed(key))
    sampled = rng.multinomial(n, probabilities, size=BOOTSTRAP_REPLICATES)
    risk_differences_pp = 100 * (sampled[:, 1] - sampled[:, 2]) / n
    low, high = np.quantile(risk_differences_pp, [0.025, 0.975])
    return float(low), float(high)


def cochran_q(matrix: np.ndarray) -> tuple[float, int, float]:
    values = matrix.astype(float)
    n, k = values.shape
    if n == 0 or k < 2:
        return np.nan, k - 1, np.nan
    column_sums = values.sum(axis=0)
    row_sums = values.sum(axis=1)
    total = column_sums.sum()
    denominator = k * total - np.square(row_sums).sum()
    if denominator == 0:
        return 0.0, k - 1, 1.0
    statistic = (k - 1) * (k * np.square(column_sums).sum() - total**2) / denominator
    return float(statistic), k - 1, float(chi2.sf(statistic, k - 1))


def build_pair_specs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    display_order = 0
    for identity, prefix in [("man", "M"), ("woman", "W")]:
        explicit = f"{identity}_explicit_1"
        family = f"{identity}_primary_seven"
        for persona_a, short in [
            (explicit, f"{prefix}:E-N"),
            *[(f"{identity}_implicit_{variant}", f"{prefix}:I{variant}-N")
              for variant in [1, 2, 3]],
        ]:
            display_order += 1
            rows.append({
                "identity": identity,
                "family": family,
                "analysis_scope": "primary",
                "persona_a": persona_a,
                "persona_b": "neutral",
                "comparison_short": short,
                "display_order": display_order,
            })
        for variant in [1, 2, 3]:
            display_order += 1
            rows.append({
                "identity": identity,
                "family": family,
                "analysis_scope": "primary",
                "persona_a": f"{identity}_implicit_{variant}",
                "persona_b": explicit,
                "comparison_short": f"{prefix}:I{variant}-E",
                "display_order": display_order,
            })

    for variant, suffix in [(0, "E"), (1, "I1"), (2, "I2"), (3, "I3")]:
        man = "man_explicit_1" if variant == 0 else f"man_implicit_{variant}"
        woman = "woman_explicit_1" if variant == 0 else f"woman_implicit_{variant}"
        display_order += 1
        rows.append({
            "identity": "man_vs_woman",
            "family": "matched_gender_exploratory_four",
            "analysis_scope": "exploratory",
            "persona_a": man,
            "persona_b": woman,
            "comparison_short": f"M-W:{suffix}",
            "display_order": display_order,
        })
    return pd.DataFrame(rows)


def readable_persona(persona: str) -> str:
    if persona == "neutral":
        return "Neutral"
    identity, framing, variant = persona.split("_")
    identity_label = identity.title()
    if framing == "explicit":
        return f"{identity_label} Explicit"
    return f"{identity_label} Implicit {variant}"


def build_omnibus(outcomes: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []
    for model in model_order:
        model_label = outcomes.loc[outcomes["model"].eq(model), "model_label"].iloc[0]
        for identity in ["man", "woman"]:
            personas = [
                "neutral",
                f"{identity}_explicit_1",
                *[f"{identity}_implicit_{variant}" for variant in [1, 2, 3]],
            ]
            wide = (
                outcomes[
                    outcomes["model"].eq(model)
                    & outcomes["persona"].isin(personas)
                ]
                .pivot(index="question_id", columns="persona", values="is_correct")
                .reindex(columns=personas)
                .dropna()
            )
            statistic, df, p_value = cochran_q(wide.to_numpy(dtype=bool))
            rows.append({
                "model": model,
                "model_label": model_label,
                "identity": identity,
                "analysis_scope": "primary",
                "conditions": ";".join(personas),
                "n_complete_cases": int(len(wide)),
                "cochran_q_statistic": statistic,
                "degrees_of_freedom": df,
                "cochran_q_p_value": p_value,
                "significant_0_05": bool(p_value < 0.05),
            })
    return pd.DataFrame(rows)


def calculate_pairs(
    outcomes: pd.DataFrame,
    specs: pd.DataFrame,
    omnibus: pd.DataFrame,
    model_order: list[str],
) -> pd.DataFrame:
    rows = []
    omnibus_lookup = omnibus.set_index(["model", "identity"])["cochran_q_p_value"].to_dict()
    for model in model_order:
        model_rows = outcomes[outcomes["model"].eq(model)]
        model_label = model_rows["model_label"].iloc[0]
        for spec in specs.itertuples(index=False):
            wide = (
                model_rows[model_rows["persona"].isin([spec.persona_a, spec.persona_b])]
                .pivot(index="question_id", columns="persona", values="is_correct")
                .reindex(columns=[spec.persona_a, spec.persona_b])
                .dropna()
            )
            a = wide[spec.persona_a].astype(bool).to_numpy()
            b = wide[spec.persona_b].astype(bool).to_numpy()
            n11 = int(np.sum(a & b))
            n10 = int(np.sum(a & ~b))
            n01 = int(np.sum(~a & b))
            n00 = int(np.sum(~a & ~b))
            discordant = n10 + n01
            p_value = float(binomtest(n10, discordant, p=0.5).pvalue) if discordant else 1.0
            p_a = float(a.mean())
            p_b = float(b.mean())
            ci_low, ci_high = paired_bootstrap_ci(
                n11, n10, n01, n00,
                f"{model}|{spec.persona_a}|{spec.persona_b}",
            )
            rows.append({
                "model": model,
                "model_label": model_label,
                "identity": spec.identity,
                "family": spec.family,
                "analysis_scope": spec.analysis_scope,
                "display_order": spec.display_order,
                "persona_a": spec.persona_a,
                "persona_b": spec.persona_b,
                "comparison": f"{readable_persona(spec.persona_a)} - {readable_persona(spec.persona_b)}",
                "comparison_short": spec.comparison_short,
                "n_pairs": int(len(wide)),
                "missing_pair_count": EXPECTED_N - int(len(wide)),
                "correct_a": int(a.sum()),
                "correct_b": int(b.sum()),
                "accuracy_a_pct": 100 * p_a,
                "accuracy_b_pct": 100 * p_b,
                "risk_difference_pp": 100 * (p_a - p_b),
                "risk_difference_ci95_low_pp": ci_low,
                "risk_difference_ci95_high_pp": ci_high,
                "cohens_h": cohen_h(p_a, p_b),
                "discordant_a_correct_only": n10,
                "discordant_b_correct_only": n01,
                "discordant_total": discordant,
                "matched_odds_ratio": (
                    float(n10 / n01) if n01 else (math.inf if n10 else np.nan)
                ),
                "exact_mcnemar_raw_p_value": p_value,
                "cochran_q_p_value": (
                    omnibus_lookup[(model, spec.identity)]
                    if spec.analysis_scope == "primary"
                    else np.nan
                ),
            })

    pairs = pd.DataFrame(rows)
    pairs["holm_adjusted_p_value"] = np.nan
    pairs["family_size"] = 0
    for _, indices in pairs.groupby(["model", "family"]).groups.items():
        pairs.loc[indices, "holm_adjusted_p_value"] = holm_adjust(
            pairs.loc[indices, "exact_mcnemar_raw_p_value"]
        )
        pairs.loc[indices, "family_size"] = len(indices)
    pairs["significant_holm_0_05"] = pairs["holm_adjusted_p_value"].lt(0.05)
    pairs["decision_significant_0_05"] = np.where(
        pairs["analysis_scope"].eq("primary"),
        pairs["significant_holm_0_05"] & pairs["cochran_q_p_value"].lt(0.05),
        pairs["significant_holm_0_05"],
    )
    return pairs.sort_values(["display_order", "model"], kind="stable").reset_index(drop=True)


def configure_plots() -> None:
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.facecolor": "#FCFBF8",
        "axes.facecolor": "#FCFBF8",
        "savefig.facecolor": "#FCFBF8",
        "font.family": "DejaVu Sans",
    })


def annotations(values: pd.DataFrame, decisions: pd.DataFrame, digits: int) -> pd.DataFrame:
    result = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                result.loc[row, column] = ""
            else:
                result.loc[row, column] = (
                    f"{value:+.{digits}f}" + ("*" if decisions.loc[row, column] else "")
                )
    return result


def pvalue_annotations(values: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    result = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                result.loc[row, column] = ""
            else:
                formatted = "<.001" if value < 0.001 else f"{value:.3f}"
                result.loc[row, column] = formatted + ("*" if decisions.loc[row, column] else "")
    return result


def style_model_ticks(ax: plt.Axes) -> None:
    selected_labels = set(
        pd.read_csv(OUTPUT_DIR / "gsm8k_granular_model_summary.csv")
        .loc[lambda frame: frame["selected_for_main_experiment"].astype(str).str.lower().eq("true"), "model_label"]
    )
    for tick in ax.get_yticklabels():
        if tick.get_text() in selected_labels:
            tick.set_color("#0047AB")
            tick.set_fontweight("bold")


def render_effect_heatmap(pairs: pd.DataFrame, model_labels: list[str], order: list[str]) -> None:
    values = pairs.pivot(index="model_label", columns="comparison_short", values="cohens_h").reindex(
        index=model_labels, columns=order
    )
    decisions = pairs.pivot(
        index="model_label", columns="comparison_short", values="decision_significant_0_05"
    ).reindex(index=model_labels, columns=order).fillna(False).astype(bool)
    limit = max(0.05, float(np.nanmax(np.abs(values.to_numpy()))))
    cmap = LinearSegmentedColormap.from_list(
        "paired_effect", ["#0047AB", "#B9C2CC", "#FCFBF8", "#D4A6A6", "#B80F2E"]
    )
    fig, ax = plt.subplots(figsize=(30, 10.5))
    sns.heatmap(
        values,
        annot=annotations(values, decisions, 2),
        fmt="",
        center=0,
        vmin=-limit,
        vmax=limit,
        cmap=cmap,
        linewidths=0.7,
        linecolor="#FCFBF8",
        annot_kws={"fontsize": 8.2, "fontweight": "bold"},
        cbar_kws={"label": "Cohen's h (A − B)"},
        ax=ax,
    )
    style_model_ticks(ax)
    ax.set_title(
        "GSM8K 14 Model: Effect Size untuk Perbandingan Persona yang Ditetapkan",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Perbandingan persona")
    ax.set_ylabel("Model (biru tebal = dipilih)")
    ax.set_xticklabels(order, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9.2)
    fig.text(
        0.5, 0.012,
        "* keputusan signifikan setelah Holm; primary comparisons juga harus melewati "
        "omnibus Cochran's Q. Tanda menunjukkan arah A − B.",
        ha="center", fontsize=9.2, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUTPUT_DIR / "05_all_prespecified_persona_effect_size_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_pvalue_heatmap(pairs: pd.DataFrame, model_labels: list[str], order: list[str]) -> None:
    raw = pairs.pivot(
        index="model_label", columns="comparison_short", values="exact_mcnemar_raw_p_value"
    ).reindex(index=model_labels, columns=order)
    adjusted = pairs.pivot(
        index="model_label", columns="comparison_short", values="holm_adjusted_p_value"
    ).reindex(index=model_labels, columns=order)
    raw_sig = raw.lt(0.05)
    decisions = pairs.pivot(
        index="model_label", columns="comparison_short", values="decision_significant_0_05"
    ).reindex(index=model_labels, columns=order).fillna(False).astype(bool)
    cmap = LinearSegmentedColormap.from_list(
        "paired_significance", ["#3B0A57", "#8E0D4A", "#D52B1E", "#F0D8D2", "#F7F4F0"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(30, 16), constrained_layout=True)
    for ax, values, sig, title, cbar_label in [
        (axes[0], raw, raw_sig, "Raw two-sided Exact McNemar p-value", "p-value (0–0.05)"),
        (axes[1], adjusted, decisions, "Holm-adjusted p-value", "adjusted p-value (0–0.05)"),
    ]:
        sns.heatmap(
            values,
            annot=pvalue_annotations(values, sig),
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=0.05,
            linewidths=0.7,
            linecolor="#FCFBF8",
            annot_kws={"fontsize": 7.8, "fontweight": "bold"},
            cbar_kws={"label": cbar_label},
            ax=ax,
        )
        style_model_ticks(ax)
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=9)
        ax.set_xlabel("Perbandingan persona")
        ax.set_ylabel("Model (biru tebal = dipilih)")
        ax.set_xticklabels(order, rotation=0, ha="center", fontsize=8.7)
        ax.tick_params(axis="y", rotation=0, labelsize=9.0)
    fig.suptitle(
        "GSM8K 14 Model: Exact McNemar untuk Perbandingan Persona yang Ditetapkan",
        fontsize=18, fontweight="bold",
    )
    fig.savefig(
        OUTPUT_DIR / "06_all_prespecified_persona_exact_mcnemar_holm_pvalue_heatmap.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    outcomes = pd.read_csv(OUTPUT_DIR / "gsm8k_question_level_outcomes.csv")
    outcomes["is_correct"] = parse_bool(outcomes["is_correct"])
    assert not outcomes.duplicated(["model", "persona", "question_id"]).any()
    summary = pd.read_csv(OUTPUT_DIR / "gsm8k_granular_model_summary.csv")
    model_order = summary["model"].tolist()
    model_labels = summary["model_label"].tolist()
    assert len(model_order) == 14

    specs = build_pair_specs()
    omnibus = build_omnibus(outcomes, model_order)
    pairs = calculate_pairs(outcomes, specs, omnibus, model_order)
    pairs.to_csv(OUTPUT_DIR / "gsm8k_all_prespecified_paired_effects.csv", index=False)
    omnibus.to_csv(OUTPUT_DIR / "gsm8k_cochran_q_omnibus.csv", index=False)

    assert len(specs) == 18
    assert len(pairs) == 14 * 18
    assert set(pairs.groupby("family")["family_size"].unique().explode().astype(int)) == {4, 7}
    assert pairs["n_pairs"].between(1_200, EXPECTED_N).all()
    assert pairs["holm_adjusted_p_value"].between(0, 1).all()

    configure_plots()
    comparison_order = specs.sort_values("display_order")["comparison_short"].tolist()
    render_effect_heatmap(pairs, model_labels, comparison_order)
    render_pvalue_heatmap(pairs, model_labels, comparison_order)
    print(f"Pairs: {len(pairs)}")
    print(f"Omnibus rows: {len(omnibus)}")
    print(f"Significant decisions: {int(pairs['decision_significant_0_05'].sum())}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Add transparent Holm-significance roll-up markers to the v5 Gender delta plots.

The plotted bars are descriptive averages across several persona conditions, so
Exact McNemar does not test a bar directly.  A star therefore means that at
least one binary, question-level comparison contributing to that bar is
significant after the pre-specified Holm correction.  The figure footnote and
the audit CSV preserve this distinction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "persona_publication_analysis_v5_paired_significance_six_models"
)
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_AXIS_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next\n80B-A3B",
    "Llama 3.1 8B": "Llama 3.1\n8B",
    "Ministral 8B": "Ministral\n8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1-Distill\n32B",
    "GPT-OSS 20B": "GPT-OSS\n20B",
    "Llama 3.3 70B": "Llama 3.3\n70B",
}
COLORS = ["#0047AB", "#B80F2E", "#5B21B6"]

FIGURE_SPECS = {
    "external_vs_neutral": {
        "columns": ["Delta Man vs Neutral", "Delta Woman vs Neutral", "Gender Gap"],
        "labels": ["Man Avg − Neutral", "Woman Avg − Neutral", "Man Avg − Woman Avg"],
        "title": "GSM8K Gender: Analisis Delta terhadap Persona Neutral",
        "stem": "02_gender_delta_vs_neutral",
    },
    "implicit_vs_explicit": {
        "columns": ["Delta Man Imp-Exp", "Delta Woman Imp-Exp", "Gender Gap"],
        "labels": [
            "Man Implicit Avg − Explicit",
            "Woman Implicit Avg − Explicit",
            "Man Avg − Woman Avg",
        ],
        "title": "GSM8K Gender — Analisis Delta: Performance Gap dan Gender Bias",
        "stem": "03_gender_delta_implicit_vs_explicit",
    },
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def _constituents(
    paired: pd.DataFrame,
    model: str,
    figure_scope: str,
    aggregate_bar: str,
) -> pd.DataFrame:
    rows = paired[paired["model"].eq(model)].copy()

    if aggregate_bar.startswith("Man Avg − Woman Avg"):
        return rows[
            rows["persona_a"].str.startswith("man_")
            & rows["persona_b"].str.startswith("woman_")
        ]

    identity = "man" if aggregate_bar.startswith("Man ") else "woman"
    rows = rows[rows["identity"].eq(identity)]
    if figure_scope == "external_vs_neutral":
        return rows[rows["persona_b"].eq("neutral")]

    return rows[
        rows["persona_a"].str.startswith(f"{identity}_implicit_")
        & rows["persona_b"].eq(f"{identity}_explicit_1")
    ]


def build_marker_audit(
    delta: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_label = delta.loc[delta["model"].eq(model), "model_label"].iloc[0]
        for figure_scope, spec in FIGURE_SPECS.items():
            for aggregate_bar in spec["labels"]:
                constituents = _constituents(
                    paired, model, figure_scope, aggregate_bar
                )
                significant = constituents[constituents["significant_holm_0_05"]]
                rows.append(
                    {
                        "model": model,
                        "model_label": model_label,
                        "figure_scope": figure_scope,
                        "aggregate_bar": aggregate_bar,
                        "n_constituent_comparisons": int(len(constituents)),
                        "n_significant_constituent_comparisons": int(len(significant)),
                        "minimum_holm_adjusted_p_value": (
                            float(constituents["holm_adjusted_p_value"].min())
                            if len(constituents)
                            else np.nan
                        ),
                        "significant_constituent_comparisons": "; ".join(
                            f"{row.comparison_label} (Holm p={row.holm_adjusted_p_value:.6g})"
                            for row in significant.itertuples()
                        ),
                        "star_marker": "*" if len(significant) else "",
                        "marker_interpretation": (
                            "At least one constituent two-sided Exact McNemar "
                            "comparison is significant after Holm; this is not "
                            "a direct test of the aggregate average."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def render_delta_figure(
    delta: pd.DataFrame,
    audit: pd.DataFrame,
    figure_scope: str,
) -> None:
    spec = FIGURE_SPECS[figure_scope]
    ordered = delta.set_index("model").reindex(MODEL_ORDER).reset_index()
    values = ordered[spec["columns"]].astype(float).to_numpy() / 100.0
    model_labels = ordered["model_label"].tolist()
    x = np.arange(len(ordered))
    width = 0.26

    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    for column_index, (label, color) in enumerate(zip(spec["labels"], COLORS)):
        offsets = x + (column_index - 1) * width
        bars = ax.bar(offsets, values[:, column_index], width, color=color, label=label)
        for row_index, bar in enumerate(bars):
            value = float(bar.get_height())
            marker_row = audit[
                audit["model_label"].eq(model_labels[row_index])
                & audit["figure_scope"].eq(figure_scope)
                & audit["aggregate_bar"].eq(label)
            ]
            marker = marker_row["star_marker"].iloc[0]
            ax.annotate(
                f"{value:.4f}{marker}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="semibold",
            )

    max_abs = max(0.001, float(np.nanmax(np.abs(values))))
    ax.set_ylim(-max_abs * 1.48, max_abs * 1.48)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(spec["title"], fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference on accuracy scale")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_AXIS_LABELS[label] for label in model_labels],
        rotation=0,
        ha="center",
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.5,
        0.012,
        "* ≥1 constituent two-sided Exact McNemar comparison is significant after "
        "Holm (adjusted p < 0.05); the aggregate average itself is descriptive.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#343A40",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(FIGURE_DIR / f"{spec['stem']}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{spec['stem']}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    delta = pd.read_csv(TABLE_DIR / "gender_five_category_and_deltas.csv")
    paired = pd.read_csv(TABLE_DIR / "gender_paired_effects_significance.csv")
    paired["significant_holm_0_05"] = _as_bool(
        paired["significant_holm_0_05"]
    )
    paired["holm_adjusted_p_value"] = pd.to_numeric(
        paired["holm_adjusted_p_value"], errors="raise"
    )

    assert set(delta["model"]) == set(MODEL_ORDER)
    assert delta["model"].nunique() == 6
    assert paired["n_pairs"].astype(int).between(1_300, 1_319).all()

    audit = build_marker_audit(delta, paired)
    audit.to_csv(
        TABLE_DIR / "gender_delta_component_significance_markers.csv",
        index=False,
    )
    for figure_scope in FIGURE_SPECS:
        render_delta_figure(delta, audit, figure_scope)

    expected = {
        ("DeepSeek R1-Distill 32B", "external_vs_neutral", "Man Avg − Neutral"),
        ("DeepSeek R1-Distill 32B", "external_vs_neutral", "Woman Avg − Neutral"),
        ("DeepSeek R1-Distill 32B", "implicit_vs_explicit", "Man Implicit Avg − Explicit"),
        ("DeepSeek R1-Distill 32B", "implicit_vs_explicit", "Woman Implicit Avg − Explicit"),
        ("Llama 3.3 70B", "implicit_vs_explicit", "Woman Implicit Avg − Explicit"),
    }
    actual = set(
        audit.loc[
            audit["star_marker"].eq("*"),
            ["model_label", "figure_scope", "aggregate_bar"],
        ].itertuples(index=False, name=None)
    )
    assert actual == expected, (actual, expected)
    print(f"Updated delta figures in {FIGURE_DIR}")
    print(f"Wrote marker audit with {len(audit)} rows")


if __name__ == "__main__":
    main()

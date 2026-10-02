"""Build Gender GSM8K visual revision v3 with direct Man-vs-Woman tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .revision_v2 import (
    CONDITION_LABELS,
    EXPLICIT_COMPARATOR_CONTRASTS,
    FAMILY_COLORS,
    FAMILY_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    NEUTRAL_CONTRASTS,
    OUTCOME_LABELS,
    _load_analysis_frame,
    _plot_correctness_sensitivity,
    _plot_coverage,
    _plot_lexical_group,
    _plot_opening_composition,
    _plot_proxy_presence_forest,
    _plot_style,
    _plot_volcano_group,
    _plot_word_count,
    build_revision_bundle,
)
from .lexicon_analysis import (
    LEXICON_VERSION,
    TERM_CATEGORIES,
    refresh_gendered_lexicon_figures,
)
from src.core.linguistic_markers.statistics import (
    GENDER_DIRECT_CONDITIONS,
    build_paired_gender_availability,
    build_paired_gender_binary,
    build_paired_gender_numeric,
    build_paired_gender_numeric_correctness_sensitivity,
    build_paired_gender_opening_multinomial,
)


V3_VERSION = "gender-gsm8k-bab4-visual-v3-direct-gender"

GENDER_FIGURE_DESCRIPTIONS = {
    "11_gendered_terms_wordcloud_by_persona.png": (
        "Supplementary closed gender-lexicon word cloud by persona."
    ),
    "12_gender_pronoun_prevalence_by_model_persona.png": (
        "Pronoun prevalence with separate Man and Woman branches from neutral."
    ),
    "13_gender_pronoun_direct_man_vs_woman_forest.png": (
        "Direct paired Man-minus-Woman pronoun prevalence delta with 95% CI."
    ),
}


def _save(fig: plt.Figure, figure_dir: Path, filename: str) -> Path:
    path = figure_dir / filename
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _prepare_from_base(base_bundle: Path, output_dir: Path, bootstrap_resamples: int) -> None:
    """Create a clean v3 workspace from a completed base without touching it."""

    base_bundle = base_bundle.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not (base_bundle / "manifest.json").is_file():
        raise FileNotFoundError(base_bundle / "manifest.json")
    shutil.copytree(
        base_bundle,
        output_dir,
        ignore=shutil.ignore_patterns(
            "11_gendered_terms_wordcloud_by_persona.png",
            "12_gender_pronoun_prevalence_by_model_persona.png",
            "13_gender_pronoun_proxy_forest.png",
            "13_gender_pronoun_direct_man_vs_woman_forest.png",
            "gendered_lexicon_metadata.json",
        ),
    )
    _plot_style()
    refresh_gendered_lexicon_figures(output_dir)
    prevalence = pd.read_csv(output_dir / "gender_pronoun_class_prevalence.csv")
    n_available = int(
        prevalence.drop_duplicates(["model_id", "persona"])[
            "n_responses_available"
        ].sum()
    )
    metadata = {
        "status": "candidate_closed_lexicon_descriptive_and_paired",
        "lexicon_version": LEXICON_VERSION,
        "text_field": "visible_text",
        "hidden_reasoning_used": False,
        "terms": TERM_CATEGORIES,
        "absolute_wordcloud_weight": (
            "equal-model mean occurrences per 10,000 visible-response words"
        ),
        "pronoun_prevalence_metric": (
            "percentage of visible responses containing at least one class term"
        ),
        "paired_key": ["model_id", "question_id"],
        "bootstrap_resamples": bootstrap_resamples,
        "n_available_responses": n_available,
        "n_terms": len(TERM_CATEGORIES),
        "generated_figures": list(GENDER_FIGURE_DESCRIPTIONS),
    }
    (output_dir / "gendered_lexicon_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_index_path = output_dir / "figure_index.csv"
    figure_index = pd.read_csv(figure_index_path)
    figure_index = figure_index[
        ~figure_index["figure"].isin(GENDER_FIGURE_DESCRIPTIONS)
    ]
    additions = pd.DataFrame(
        [
            {"figure": name, "description": description}
            for name, description in GENDER_FIGURE_DESCRIPTIONS.items()
        ]
    )
    pd.concat((figure_index, additions), ignore_index=True).to_csv(
        figure_index_path,
        index=False,
    )


def _direct_sensitivity_comparison(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "model_id",
        "benchmark",
        "subset",
        "condition",
        "outcome_id",
        "outcome_family",
        "analysis_role",
        "contrast_id",
    ]
    left = primary[
        keys + ["n_pairs", "n_nonzero", "rank_biserial", "p_holm"]
    ].rename(
        columns={
            "n_pairs": "n_pairs_all",
            "n_nonzero": "n_nonzero_all",
            "rank_biserial": "rank_biserial_all",
            "p_holm": "p_holm_all",
        }
    )
    right = sensitivity[
        keys + ["n_pairs", "n_nonzero", "rank_biserial", "p_holm"]
    ].rename(
        columns={
            "n_pairs": "n_pairs_both_correct",
            "n_nonzero": "n_nonzero_both_correct",
            "rank_biserial": "rank_biserial_both_correct",
            "p_holm": "p_holm_both_correct",
        }
    )
    comparison = left.merge(right, on=keys, how="inner", validate="one_to_one")
    comparison["pair_retention"] = (
        comparison["n_pairs_both_correct"] / comparison["n_pairs_all"]
    )
    comparison["supported_both"] = (
        comparison["n_nonzero_all"].ge(20)
        & comparison["n_nonzero_both_correct"].ge(20)
    )
    comparison["same_direction"] = np.sign(comparison["rank_biserial_all"]).eq(
        np.sign(comparison["rank_biserial_both_correct"])
    )
    summaries: list[dict[str, object]] = []
    primary_rows = comparison[comparison["analysis_role"].eq("primary")]
    for condition, group in primary_rows.groupby("condition", sort=False):
        supported = group[group["supported_both"]]
        summaries.append(
            {
                "condition": condition,
                "n_primary_tests": len(group),
                "median_pair_retention": float(group["pair_retention"].median()),
                "n_supported_both": len(supported),
                "same_direction_rate_supported": (
                    float(supported["same_direction"].mean())
                    if len(supported)
                    else math.nan
                ),
            }
        )
    return comparison, pd.DataFrame(summaries)


def _plot_direct_numeric_volcano(numeric: pd.DataFrame, figure_dir: Path) -> Path:
    data = numeric[numeric["analysis_role"].eq("primary")].copy()
    p_values = pd.to_numeric(data["p_holm"], errors="coerce").clip(
        lower=np.nextafter(0.0, 1.0)
    )
    data["neg_log10_p_holm"] = -np.log10(p_values)
    y_cap = 50.0
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.2), sharex=True, sharey=True)
    for ax, condition in zip(axes, GENDER_DIRECT_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for family, group in panel.groupby("outcome_family", sort=True):
            sizes = 42 + 22 * np.log10(
                pd.to_numeric(group["n_nonzero"], errors="coerce").fillna(0) + 1
            )
            ax.scatter(
                group["rank_biserial"],
                group["neg_log10_p_holm"].clip(upper=y_cap),
                s=sizes,
                alpha=np.where(group["p_holm"].lt(0.05), 0.95, 0.55),
                color=FAMILY_COLORS.get(family, "#64748B"),
                edgecolors="white",
                linewidths=0.75,
            )
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.axhline(-math.log10(0.05), color="#94A3B8", linestyle="--", linewidth=0.8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-0.2, y_cap + 1)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=0.55)
    fig.supxlabel("Rank-biserial Man − Woman", y=0.095)
    fig.supylabel("−log₁₀(p Holm), ditampilkan sampai 50", x=0.006)
    families = [f for f in FAMILY_COLORS if f in set(data["outcome_family"])]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS.get(family, family),
            markersize=12,
        )
        for family in families
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Uji langsung marker numerik: persona Man minus Woman\n"
        "Tiap titik = satu model × marker; ukuran menunjukkan jumlah pasangan yang berubah",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.04, 0.17, 0.99, 0.90), w_pad=1.2)
    return _save(fig, figure_dir, "14_direct_gender_numeric_volcano.png")


def _plot_direct_effect_consistency(numeric: pd.DataFrame, figure_dir: Path) -> Path:
    data = numeric[numeric["analysis_role"].eq("primary")].copy()
    outcomes = [
        "word_count",
        "avg_sentence_length",
        "ttr_300",
        "organization_explicit_step",
        "organization_bullet",
        "organization_section_header",
        "stance_hedge",
        "stance_booster",
        "stance_assumption",
        "stance_self_correction",
        "connective_causal",
        "connective_contrastive",
        "connective_conditional",
        "self_reference_first_person",
        "self_reference_inclusive_we",
        "engagement_reader_directive",
    ]
    model_colors = dict(
        zip(MODEL_ORDER, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
    )
    y_lookup = {outcome: len(outcomes) - 1 - i for i, outcome in enumerate(outcomes)}
    fig, axes = plt.subplots(1, 4, figsize=(21, 11.2), sharex=True, sharey=True)
    for ax, condition in zip(axes, GENDER_DIRECT_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            outcome = point["outcome_id"]
            if outcome not in y_lookup:
                continue
            size = 36 + 18 * math.log10(float(point["n_nonzero"]) + 1)
            ax.scatter(
                point["rank_biserial"],
                y_lookup[outcome],
                s=size,
                color=model_colors[point["model_id"]],
                alpha=0.97 if point["p_holm"] < 0.05 else 0.55,
                edgecolors="white",
                linewidths=0.7,
            )
        medians = panel.groupby("outcome_id")["rank_biserial"].median()
        for outcome, value in medians.items():
            if outcome in y_lookup:
                ax.scatter(value, y_lookup[outcome], marker="D", s=70, color="#111827", zorder=4)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.supxlabel("Rank-biserial Man − Woman", y=0.075)
    axes[0].set_yticks(
        [y_lookup[outcome] for outcome in outcomes],
        [OUTCOME_LABELS.get(outcome, outcome) for outcome in outcomes],
        fontsize=15,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=model_colors[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ] + [
        Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.003),
    )
    fig.suptitle(
        "Konsistensi arah efek Man minus Woman pada enam model\n"
        "Titik pudar = p Holm ≥ 0,05; ukuran menunjukkan pasangan nonzero",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.14, 0.99, 0.93), w_pad=1.0)
    return _save(fig, figure_dir, "15_direct_gender_effect_consistency.png")


def _plot_direct_availability(availability: pd.DataFrame, figure_dir: Path) -> Path:
    data = availability.copy()
    data["delta_pp"] = 100 * data["rate_delta"]
    data["ci_low_pp"] = 100 * data["ci_low"]
    data["ci_high_pp"] = 100 * data["ci_high"]
    outcomes = ("text_available", "technical_only")
    labels = {
        "text_available": "Teks tersedia",
        "technical_only": "Technical-only",
    }
    colors = {"text_available": "#2563EB", "technical_only": "#D97706"}
    y_lookup = {model: len(MODEL_ORDER) - 1 - i for i, model in enumerate(MODEL_ORDER)}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.8), squeeze=False)
    for row, outcome in enumerate(outcomes):
        outcome_data = data[data["outcome_id"].eq(outcome)]
        bound = max(
            float(np.nanmax(np.abs(outcome_data[["ci_low_pp", "ci_high_pp"]].to_numpy(float)))) * 1.18,
            0.5,
        )
        for column, condition in enumerate(GENDER_DIRECT_CONDITIONS):
            ax = axes[row, column]
            panel = outcome_data[outcome_data["condition"].eq(condition)]
            for _, point in panel.iterrows():
                effect = point["delta_pp"]
                ax.errorbar(
                    effect,
                    y_lookup[point["model_id"]],
                    xerr=np.array(
                        [[max(effect - point["ci_low_pp"], 0)], [max(point["ci_high_pp"] - effect, 0)]]
                    ),
                    fmt="o",
                    color=colors[outcome],
                    alpha=0.95 if point["p_holm"] < 0.05 else 0.55,
                    capsize=3.5,
                    markersize=8,
                )
            ax.axvline(0, color="#334155", linewidth=0.8)
            ax.set_xlim(-bound, bound)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
            if row == 0:
                ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
            if column == 0:
                ax.set_yticks(
                    [y_lookup[m] for m in MODEL_ORDER],
                    [MODEL_LABELS[m] for m in MODEL_ORDER],
                    fontsize=15,
                )
                ax.set_ylabel(labels[outcome], fontweight="bold")
            else:
                ax.set_yticks([])
    fig.supxlabel("Man − Woman (pp), 95% paired CI", y=0.03)
    fig.suptitle(
        "Audit differential availability sebelum menafsirkan efek gender\n"
        "Nilai negatif pada teks tersedia berarti coverage persona Man lebih rendah",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.91), h_pad=1.8, w_pad=1.1)
    return _save(fig, figure_dir, "16_direct_gender_availability_forest.png")


def _plot_direct_opening(opening: pd.DataFrame, figure_dir: Path) -> Path:
    data = opening.copy()
    data["tv_pp"] = 100 * data["marginal_total_variation"]
    y_lookup = {model: len(MODEL_ORDER) - 1 - i for i, model in enumerate(MODEL_ORDER)}
    xmax = max(1.0, float(data["tv_pp"].max()) * 1.18)
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.2), sharex=True, sharey=True)
    for ax, condition in zip(axes, GENDER_DIRECT_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            y = y_lookup[point["model_id"]]
            color = "#7C3AED" if point["p_holm"] < 0.05 else "#94A3B8"
            ax.hlines(y, 0, point["tv_pp"], color=color, linewidth=2.5)
            ax.scatter(point["tv_pp"], y, color=color, s=85, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.supxlabel("Marginal total variation (pp)", y=0.03)
    axes[0].set_yticks(
        [y_lookup[m] for m in MODEL_ORDER],
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        fontsize=15,
    )
    fig.suptitle(
        "Perbedaan langsung komposisi opening: Man vs Woman\n"
        "Ungu = Stuart–Maxwell p Holm < 0,05; abu-abu = tidak signifikan",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.90), w_pad=1.2)
    return _save(fig, figure_dir, "17_direct_gender_opening_composition.png")


def refresh_v3_figures(
    bundle_dir: Path,
    figure_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Redraw all notebook figures from existing tables without recomputing tests."""

    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = Path(manifest["source_run"]).resolve()
    figure_dir = (
        figure_dir.resolve()
        if figure_dir is not None
        else bundle_dir / "images"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    bundle_tables = {
        "coverage": bundle_dir / "response_counts_model_persona.csv",
        "sensitivity": bundle_dir / "correctness_sensitivity_comparison.csv",
        "sensitivity_summary": bundle_dir / "correctness_sensitivity_summary.csv",
        "opening": bundle_dir / "opening_composition.csv",
        "lexical": bundle_dir / "lexical_contrast_token_weights.csv",
        "direct_numeric": bundle_dir / "direct_gender_paired_numeric.csv",
        "direct_availability": bundle_dir / "direct_gender_availability.csv",
        "direct_opening": bundle_dir / "direct_gender_opening_multinomial.csv",
    }
    source_tables = {
        "descriptives": run_dir / "condition_descriptives.csv",
        "numeric": run_dir / "paired_numeric.csv",
        "presence": run_dir / "paired_binary.csv",
    }
    missing = [
        path
        for path in (*bundle_tables.values(), *source_tables.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot refresh figures; missing tables: "
            + ", ".join(str(path) for path in missing)
        )

    _plot_style()
    tables = {
        name: pd.read_csv(path)
        for name, path in {**bundle_tables, **source_tables}.items()
    }
    figures: list[Path] = [
        _plot_coverage(tables["coverage"], figure_dir),
        _plot_word_count(tables["descriptives"], figure_dir),
        _plot_volcano_group(
            tables["numeric"],
            NEUTRAL_CONTRASTS,
            figure_dir,
            "03_numeric_effect_volcano_vs_neutral.png",
            "Effect map marker numerik terhadap baseline neutral",
        ),
        _plot_volcano_group(
            tables["numeric"],
            EXPLICIT_COMPARATOR_CONTRASTS,
            figure_dir,
            "04_numeric_effect_volcano_implicit_vs_explicit.png",
            "Effect map tiga proxy implicit terhadap explicit",
        ),
        _plot_proxy_presence_forest(
            tables["presence"],
            (
                "organization_explicit_step",
                "connective_causal",
                "math_latex_display",
            ),
            figure_dir,
            "05_proxy_presence_forest_structure_logic.png",
            "Forest plot proxy implicit: struktur, koneksi logis, dan format matematis",
        ),
        _plot_proxy_presence_forest(
            tables["presence"],
            (
                "stance_hedge",
                "stance_self_correction",
                "self_reference_inclusive_we",
            ),
            figure_dir,
            "06_proxy_presence_forest_stance_voice.png",
            "Forest plot proxy implicit: stance dan suara penalaran",
        ),
        _plot_correctness_sensitivity(
            tables["sensitivity"],
            tables["sensitivity_summary"],
            figure_dir,
        ),
        _plot_opening_composition(tables["opening"], figure_dir),
        _plot_lexical_group(
            tables["lexical"],
            NEUTRAL_CONTRASTS,
            figure_dir,
            "09_ranked_lexical_lollipop_vs_neutral.png",
            "Token distingtif persona terhadap neutral",
        ),
        _plot_lexical_group(
            tables["lexical"],
            EXPLICIT_COMPARATOR_CONTRASTS,
            figure_dir,
            "10_ranked_lexical_lollipop_implicit_vs_explicit.png",
            "Token distingtif tiga proxy implicit terhadap explicit",
        ),
    ]
    figures.extend(refresh_gendered_lexicon_figures(bundle_dir, figure_dir))
    figures.extend(
        (
            _plot_direct_numeric_volcano(tables["direct_numeric"], figure_dir),
            _plot_direct_effect_consistency(tables["direct_numeric"], figure_dir),
            _plot_direct_availability(tables["direct_availability"], figure_dir),
            _plot_direct_opening(tables["direct_opening"], figure_dir),
        )
    )
    return tuple(figures)


def build_v3(
    run_dir: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    base_bundle: Path | None = None,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if base_bundle is None:
        build_revision_bundle(
            run_dir,
            output_dir,
            bootstrap_resamples=bootstrap_resamples,
        )
    else:
        _prepare_from_base(base_bundle, output_dir, bootstrap_resamples)
    _plot_style()
    frame = _load_analysis_frame(run_dir)
    numeric = build_paired_gender_numeric(frame, bootstrap_resamples)
    sensitivity = build_paired_gender_numeric_correctness_sensitivity(
        frame,
        bootstrap_resamples,
    )
    binary = build_paired_gender_binary(frame, bootstrap_resamples)
    availability = build_paired_gender_availability(frame, bootstrap_resamples)
    opening = build_paired_gender_opening_multinomial(frame)
    sensitivity_comparison, sensitivity_summary = _direct_sensitivity_comparison(
        numeric,
        sensitivity,
    )

    tables = {
        "direct_gender_paired_numeric.csv": numeric,
        "direct_gender_correctness_sensitivity.csv": sensitivity,
        "direct_gender_correctness_comparison.csv": sensitivity_comparison,
        "direct_gender_correctness_summary.csv": sensitivity_summary,
        "direct_gender_paired_binary.csv": binary,
        "direct_gender_availability.csv": availability,
        "direct_gender_opening_multinomial.csv": opening,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    figure_dir = output_dir / "images"
    figures = (
        _plot_direct_numeric_volcano(numeric, figure_dir),
        _plot_direct_effect_consistency(numeric, figure_dir),
        _plot_direct_availability(availability, figure_dir),
        _plot_direct_opening(opening, figure_dir),
    )
    descriptions = {
        figures[0].name: "Direct paired Man-minus-Woman numeric effect map for E/I1/I2/I3.",
        figures[1].name: "Model-specific direction consistency for direct gender effects.",
        figures[2].name: "Paired availability and technical-only gender differences.",
        figures[3].name: "Direct Man-vs-Woman opening-composition differences.",
    }
    figure_index_path = output_dir / "figure_index.csv"
    figure_index = pd.read_csv(figure_index_path)
    additions = pd.DataFrame(
        [{"figure": name, "description": description} for name, description in descriptions.items()]
    )
    figure_index = pd.concat((figure_index, additions), ignore_index=True)
    if figure_index["figure"].duplicated().any():
        raise ValueError("Duplicate figure name in v3 figure index.")
    figure_index.to_csv(figure_index_path, index=False)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_version"] = V3_VERSION
    manifest["direct_gender_contrasts"] = [
        f"man_vs_woman_{condition}" for condition in GENDER_DIRECT_CONDITIONS
    ]
    manifest["direct_gender_pair_key"] = ["model_id", "question_id"]
    manifest["direct_gender_correctness_sensitivity"] = "both paired responses correct"
    manifest["direct_gender_availability_audit"] = True
    manifest["zero_count_opening_categories_hidden_from_figure"] = True
    manifest["gendered_lexicon_version"] = LEXICON_VERSION
    manifest.setdefault("figures", {}).update(GENDER_FIGURE_DESCRIPTIONS)
    manifest.setdefault("figures", {}).update(descriptions)
    manifest["generated_files_before_manifest"] = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_path = output_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme += """

## Direct gender contrasts (v3)

The primary gender claim is tested directly as Man minus Woman within
Explicit, I1, I2, and I3, paired on `model_id x question_id`. Numeric outcomes
use Wilcoxon/sign tests with rank-biserial effects; marker presence uses exact
McNemar tests; opening composition uses Stuart-Maxwell; all planned condition
families use Holm correction. A both-correct sensitivity and differential
availability/technical-only audit accompany the primary results.

Gendered-word clouds remain supplementary because pronouns can refer to GSM8K
story entities rather than the assigned model persona.
"""
    readme_path.write_text(readme, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--base-bundle", type=Path)
    args = parser.parse_args()
    manifest = build_v3(
        args.run_dir,
        args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        base_bundle=args.base_bundle,
    )
    print(
        f"[DONE] {manifest['bundle_version']} -> {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

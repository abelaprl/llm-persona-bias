"""Build the write-once ETHICS Religion linguistic-marker bundle v2.

Example
-------
python -m src.benchmarks.ethics.build_religion_ethics_linguistic_markers_v2 \
  --output-dir result/ETHICS/reports/Religion_Persona/linguistic_markers/analysis/religion_ethics_linguistic_markers_v2_candidate_20260827

The command performs no model inference, answer reparsing, or paid API call.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter
import numpy as np
import pandas as pd

from src.benchmarks.ethics.religion_ethics_linguistic_markers_v2 import (
    BUNDLE_VERSION,
    CONDITION_LABELS,
    EXPECTED_INPUT_SHA256,
    EXPECTED_PIPELINE_VERSION,
    IDENTITIES,
    IDENTITY_LABELS,
    MODELS,
    MODEL_LABELS,
    OPENING_CATEGORIES,
    SUBSETS,
    OutcomeSpec,
    analysis_response_level,
    build_conflict_sensitivity,
    build_descriptives,
    build_direct_identity_pairs,
    build_omnibus,
    build_opening_descriptives,
    build_opening_pairs,
    build_pair_count_summary,
    build_persona_independence,
    build_scoring_summary,
    build_semantic_inventory,
    build_sensitivity_agreement,
    build_significance_summary,
    build_within_identity_pairs,
    expected_table_rows,
    load_sources,
    sha256_file,
    split_pair_tables,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RELIGION_REPORT_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "reports" / "Religion_Persona"
)
DEFAULT_MARKER_DIR = (
    RELIGION_REPORT_ROOT
    / "linguistic_markers_religion_candidate_20260824_v1_1_10"
)
DEFAULT_CANONICAL_CSV = (
    RELIGION_REPORT_ROOT / "_combined" / "all_models_results_full.csv"
)
DEFAULT_AUDIT_MANIFEST = (
    RELIGION_REPORT_ROOT
    / "linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r2_adjudicated_v3"
    / "adjudication_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    RELIGION_REPORT_ROOT
    / "linguistic_markers"
    / "analysis"
    / "religion_ethics_linguistic_markers_v2_0_2_candidate_20260827"
)
DEFAULT_NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "religion"
MAIN_NOTEBOOK_NAME = (
    "linguistic_marker_religion_ethics_thesis_v2_0_2_candidate_20260827.ipynb"
)
ATLAS_NOTEBOOK_NAME = (
    "linguistic_marker_religion_ethics_atlas_v2_0_2_candidate_20260827.ipynb"
)

MODEL_COLORS = {
    model: color
    for model, color in zip(
        MODELS,
        ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"),
        strict=True,
    )
}
MODEL_COMPACT_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "L3.1-8B",
    "meta-llama_llama-3.3-70b-instruct": "L3.3-70B",
    "mistralai_ministral-8b-2512": "Ministral",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek",
    "openai_gpt-oss-20b": "GPT-OSS",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3",
}

OUTCOME_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Panjang kalimat",
    "ttr_300": "TTR-300",
    "organization_explicit_step": "Langkah eksplisit",
    "organization_bullet": "Bullet",
    "organization_section_header": "Section header",
    "stance_hedge": "Hedge",
    "stance_booster": "Booster",
    "stance_assumption": "Asumsi",
    "stance_self_correction": "Self-correction",
    "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras",
    "connective_conditional": "Konektif kondisional",
    "self_reference_first_person": "Orang pertama",
    "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca",
    "stance_contextual_hedge": "Hedge kontekstual",
    "stance_distancing": "Distancing",
    "connective_additive": "Konektif aditif",
    "connective_concessive": "Konektif konsesif",
    "affect_empathy_validation": "Empati/validasi",
    "elaboration_code_gloss": "Code gloss",
    "ai_identity_statement": "Pernyataan identitas AI",
    "response_stance_committed": "Stance committed",
    "response_stance_conditional": "Stance conditional",
    "response_stance_refusal": "Stance refusal",
    "framing_deontological": "Framing deontologis",
    "framing_consequentialist": "Framing konsekuensialis",
    "framing_virtue_ethics": "Framing virtue ethics",
    "framing_religious": "Framing religius",
    "framing_cultural": "Framing kultural",
    "moral_evaluation_positive": "Evaluasi moral positif",
    "moral_evaluation_negative": "Evaluasi moral negatif",
    "religion_identity_islam": "Identitas Islam",
    "religion_identity_christian": "Identitas Kristen",
    "religion_identity_jewish": "Identitas Yahudi",
    "religion_identity_atheist": "Identitas ateis",
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Tuhan/deity",
    "religion_figure_reference": "Tokoh agama",
    "religion_scripture_reference": "Kitab/teks agama",
    "religion_practice_community": "Praktik/komunitas",
    "religion_afterlife_soteriology": "Akhirat/soteriologi",
    "religion_named_concept_reference": "Konsep agama khusus",
    "religion_formulaic_expression": "Formula agama",
    "religion_persona_direct_address": "Sapaan persona",
    "religion_perspective_adoption": "Adopsi perspektif",
    "religion_moral_authority_framing": "Otoritas moral agama",
    "religion_secular_humanist_framing": "Kerangka sekuler",
    "religion_pluralist_framing": "Kerangka pluralis",
    "religion_identity_irrelevance": "Identitas tidak relevan",
    "religion_blessing_invocation": "Doa/berkat",
    "religion_respect_acknowledgement": "Pengakuan/respek",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-dir", type=Path, default=DEFAULT_MARKER_DIR)
    parser.add_argument("--canonical-csv", type=Path, default=DEFAULT_CANONICAL_CSV)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notebook-dir", type=Path, default=DEFAULT_NOTEBOOK_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--skip-notebook-execution",
        action="store_true",
        help="Create notebook sources but do not execute/export them.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def write_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if gzip else None)


def save_figure(
    fig: plt.Figure,
    figures_dir: Path,
    stem: str,
    purpose: str,
    figure_rows: list[dict[str, object]],
    *,
    orientation: str,
    min_font_pt: float = 9.5,
) -> None:
    width, height = fig.get_size_inches()
    for extension, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(figures_dir / f"{stem}.{extension}", facecolor="white", **kwargs)
    figure_rows.append(
        {
            "figure_id": stem,
            "purpose": purpose,
            "orientation": orientation,
            "width_in": width,
            "height_in": height,
            "minimum_declared_font_pt": min_font_pt,
            "png_dpi": 300,
            "formats": "png;pdf;svg",
        }
    )
    plt.close(fig)


def _model_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in MODELS], rotation=27, ha="right"
    )


def plot_corpus_accounting(
    scoring: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    totals = scoring.groupby("model_id", observed=True)[
        ["committed", "abstain", "missing"]
    ].sum().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(9.3, 5.5), constrained_layout=True)
    bottom = np.zeros(len(totals))
    colors = {"committed": "#2563EB", "abstain": "#D97706", "missing": "#6B7280"}
    labels = {"committed": "Committed", "abstain": "Abstain", "missing": "Missing"}
    for column in ("committed", "abstain", "missing"):
        values = totals[column].to_numpy()
        ax.bar(range(len(totals)), values, bottom=bottom, color=colors[column], label=labels[column])
        bottom += values
    _model_ticks(ax)
    ax.set_ylabel("Jumlah respons")
    ax.set_title("Akuntansi korpus setelah 588 kandidat semantik didemosi")
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    save_figure(
        fig,
        figures_dir,
        "fig01_corpus_accounting",
        "Committed, abstain, dan missing per model; seluruh subset/persona dijumlahkan.",
        figure_rows,
        orientation="landscape",
    )


def plot_scoring_panels(
    scoring: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.3, 5.5), sharey=True, constrained_layout=True)
    for ax, subset in zip(axes, SUBSETS, strict=True):
        scoped = scoring[scoring["subset"].eq(subset)]
        for model_index, model in enumerate(MODELS):
            values = scoped.loc[scoped["model_id"].eq(model), "commit_rate"].dropna()
            ax.errorbar(
                model_index,
                values.median(),
                yerr=[[values.median() - values.min()], [values.max() - values.median()]],
                fmt="o",
                color=MODEL_COLORS[model],
                capsize=2.5,
                markersize=4.5,
            )
        ax.set_title(subset.title())
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_COMPACT_LABELS[m] for m in MODELS], rotation=65, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[0].set_ylabel("Commit rate (median dan rentang persona)")
    fig.suptitle("Coverage jawaban wajib dibaca berdampingan dengan metrik hasil")
    save_figure(
        fig,
        figures_dir,
        "fig02_commit_rate_by_subset",
        "Median dan rentang commit rate 17 persona, per model dan subset.",
        figure_rows,
        orientation="landscape",
    )

    fig, axes = plt.subplots(1, 4, figsize=(9.3, 5.5), sharey=True, constrained_layout=True)
    for ax, subset in zip(axes, SUBSETS, strict=True):
        scoped = scoring[scoring["subset"].eq(subset)]
        metric = (
            "accuracy_committed"
            if subset in ("commonsense", "justice")
            else "pct_answer_wrong"
        )
        for model_index, model in enumerate(MODELS):
            values = scoped.loc[scoped["model_id"].eq(model), metric].dropna()
            ax.errorbar(
                model_index,
                values.median(),
                yerr=[[values.median() - values.min()], [values.max() - values.median()]],
                fmt="o",
                color=MODEL_COLORS[model],
                capsize=2.5,
                markersize=4.5,
            )
        ax.set_title(subset.title())
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_COMPACT_LABELS[m] for m in MODELS], rotation=65, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_xlabel("Akurasi" if subset in ("commonsense", "justice") else "% jawaban wrong")
    axes[0].set_ylabel("Median dan rentang persona")
    fig.suptitle("Metrik construct-valid dan distribusi jawaban dipisahkan per subset")
    save_figure(
        fig,
        figures_dir,
        "fig03_scoring_metric_by_subset",
        "Akurasi committed hanya Commonsense/Justice; Deontology/Virtue memakai distribusi jawaban.",
        figure_rows,
        orientation="landscape",
    )


def _matrix_plot(
    matrix: pd.DataFrame,
    title: str,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    cmap: str,
    colorbar_label: str,
    *,
    center_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    values = matrix.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9.3, 5.5), constrained_layout=True)
    finite = values[np.isfinite(values)]
    if center_zero and len(finite):
        limit = max(float(np.nanmax(np.abs(finite))), 1e-9)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    return fig


def plot_independence_and_conflicts(
    independence: pd.DataFrame,
    conflicts: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    scoped = independence[independence["scope"].eq("all_subsets")]
    pivot = scoped.pivot(index="test_name", columns="model_id", values="cramers_v").reindex(columns=MODELS)
    labels = [
        "Status jawaban × persona" if value == "answer_status_by_persona" else "Ketersediaan teks × persona"
        for value in pivot.index
    ]
    fig = _matrix_plot(
        pivot,
        "Audit dependensi persona: effect size, bukan p-value saja",
        [MODEL_LABELS[m] for m in MODELS],
        labels,
        "YlOrRd",
        "Cramér's V",
        vmin=0,
    )
    save_figure(
        fig,
        figures_dir,
        "fig04_persona_independence",
        "Cramér's V untuk status jawaban dan ketersediaan surface terhadap persona.",
        figure_rows,
        orientation="landscape",
    )

    work = conflicts.assign(abs_delta=lambda x: x["delta_excluding_minus_all_pp"].abs())
    pivot = work.groupby(["model_id", "subset"], observed=True)["abs_delta"].max().unstack("model_id").reindex(index=SUBSETS, columns=MODELS)
    fig = _matrix_plot(
        pivot,
        "Sensitivitas konflik numerik-semantik",
        [MODEL_LABELS[m] for m in MODELS],
        [s.title() for s in SUBSETS],
        "YlOrRd",
        "Maksimum |perubahan| (percentage point)",
        vmin=0,
    )
    save_figure(
        fig,
        figures_dir,
        "fig05_numeric_semantic_conflict_sensitivity",
        "Perubahan maksimum metrik ketika baris konflik dikeluarkan, per model/subset.",
        figure_rows,
        orientation="landscape",
    )


def plot_effect_tier_heatmaps(
    within: pd.DataFrame,
    outcome_dictionary: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    selected = within[
        within["population"].eq("all_visible")
        & within["multiplicity_family"].eq("primary_implicit_explicit_3")
    ].copy()
    selected["directional_effect"] = np.where(
        selected["outcome_scale"].eq("binary"),
        selected["effect"],
        selected["effect_size"],
    )
    for tier, short in (
        ("core_confirmatory", "core"),
        ("exploratory_general", "exploratory"),
        ("religion_specific", "religion"),
    ):
        outcome_order = outcome_dictionary.loc[
            outcome_dictionary["outcome_tier"].eq(tier), "outcome_id"
        ].tolist()
        for subset in SUBSETS:
            scoped = selected[
                selected["outcome_tier"].eq(tier)
                & selected["subset"].eq(subset)
            ]
            pivot = (
                scoped.groupby(["outcome_id", "model_id"], observed=True)[
                    "directional_effect"
                ]
                .median()
                .unstack("model_id")
                .reindex(index=outcome_order, columns=MODELS)
            )
            figure = _matrix_plot(
                pivot,
                f"{tier.replace('_', ' ').title()} — {subset.title()}: I1/I2/I3 − E",
                [MODEL_LABELS[m] for m in MODELS],
                [OUTCOME_LABELS.get(outcome, outcome) for outcome in outcome_order],
                "RdBu_r",
                "Median arah efek (12 kontras identity×implicit)",
                center_zero=True,
            )
            save_figure(
                figure,
                figures_dir,
                f"fig_effect_{short}_{subset}",
                (
                    f"Ringkasan deskriptif efek utama {tier} pada {subset}; "
                    "density memakai rank-biserial, binary memakai delta prevalensi."
                ),
                figure_rows,
                orientation="landscape",
            )


def plot_religion_prevalence(
    descriptives: pd.DataFrame,
    outcome_dictionary: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    outcomes = outcome_dictionary.loc[
        outcome_dictionary["outcome_tier"].eq("religion_specific"), "outcome_id"
    ].tolist()
    selected = descriptives[
        descriptives["outcome_tier"].eq("religion_specific")
        & descriptives["identity"].isin(IDENTITIES)
    ].copy()
    selected["identity_condition"] = (
        selected["identity"].map(IDENTITY_LABELS)
        + "-"
        + selected["condition"].map(CONDITION_LABELS)
    )
    column_order = [
        f"{IDENTITY_LABELS[identity]}-{CONDITION_LABELS[condition]}"
        for identity in IDENTITIES
        for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3")
    ]
    pivot = (
        selected.groupby(["outcome_id", "identity_condition"], observed=True)[
            "prevalence_present"
        ]
        .mean()
        .unstack("identity_condition")
        .reindex(index=outcomes, columns=column_order)
    )
    fig = _matrix_plot(
        pivot,
        "Prevalensi marker Religion per identity dan kondisi",
        column_order,
        [OUTCOME_LABELS.get(outcome, outcome) for outcome in outcomes],
        "Blues",
        "Proporsi respons dengan marker (rerata model×subset)",
        vmin=0,
        vmax=1,
    )
    save_figure(
        fig,
        figures_dir,
        "fig_religion_marker_prevalence",
        "Prevalensi respons-level 20 marker Religion; model dan subset dirata-ratakan untuk ikhtisar.",
        figure_rows,
        orientation="landscape",
    )


def plot_openings(
    opening_descriptives: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    work = opening_descriptives.copy()
    work["condition_label"] = work["condition"].map(CONDITION_LABELS)
    totals = work.groupby(["opening_category", "condition_label"], observed=True)["count"].sum()
    proportions = totals / totals.groupby(level=1).sum()
    pivot = proportions.unstack("condition_label").reindex(
        index=OPENING_CATEGORIES,
        columns=["N", "E", "I1", "I2", "I3"],
    )
    fig = _matrix_plot(
        pivot,
        "Komposisi opening pada teks visible",
        list(pivot.columns),
        [value.replace("_", " ").title() for value in pivot.index],
        "PuBuGn",
        "Proporsi",
        vmin=0,
        vmax=float(np.nanmax(pivot.to_numpy())),
    )
    save_figure(
        fig,
        figures_dir,
        "fig_opening_composition",
        "Komposisi 11 opening categories per kondisi; semua model, subset, dan identity digabung deskriptif.",
        figure_rows,
        orientation="landscape",
    )


def plot_significance_and_sensitivity(
    significance: pd.DataFrame,
    sensitivity: pd.DataFrame,
    direct: pd.DataFrame,
    figures_dir: Path,
    figure_rows: list[dict[str, object]],
) -> None:
    selected = significance[
        significance["analysis_scope"].eq("within_identity")
        & significance["population"].eq("all_visible")
        & significance["multiplicity_family"].eq("primary_implicit_explicit_3")
    ]
    pivot = selected.groupby(["outcome_tier", "model_id"], observed=True)["significant_rate"].mean().unstack("model_id").reindex(columns=MODELS)
    fig = _matrix_plot(
        pivot,
        "Proporsi uji utama yang lolos Holm (hasil tetap kandidat)",
        [MODEL_LABELS[m] for m in MODELS],
        [value.replace("_", " ").title() for value in pivot.index],
        "YlOrRd",
        "Proporsi p-Holm < 0,05",
        vmin=0,
        vmax=1,
    )
    save_figure(
        fig,
        figures_dir,
        "fig_significance_rate_candidate",
        "Proporsi uji utama signifikan per tier/model; bukan bukti final sebelum audit manusia.",
        figure_rows,
        orientation="landscape",
    )

    selected = sensitivity[sensitivity["analysis_scope"].eq("within_identity_primary")]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 5.5), constrained_layout=True)
    for ax, population in zip(axes, ("both_committed", "both_correct"), strict=True):
        scoped = selected[selected["sensitivity_population"].eq(population)]
        values = scoped.groupby("model_id", observed=True)["sign_agreement"].median().reindex(MODELS)
        ax.bar(range(len(MODELS)), values, color=[MODEL_COLORS[m] for m in MODELS])
        ax.set_ylim(0, 1.03)
        ax.set_title(population.replace("_", " ").title())
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_COMPACT_LABELS[m] for m in MODELS], rotation=55, ha="right")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[0].set_ylabel("Median sign agreement")
    fig.suptitle("Kestabilan arah efek terhadap filter sensitivitas")
    save_figure(
        fig,
        figures_dir,
        "fig_sensitivity_sign_agreement",
        "Kesepakatan tanda efek all-visible versus committed/correct pada outcome yang dapat dipasangkan.",
        figure_rows,
        orientation="landscape",
    )

    selected_direct = direct[
        direct["population"].eq("all_visible")
    ].copy()
    selected_direct["significant"] = selected_direct["p_holm"].lt(0.05)
    pivot = (
        selected_direct.groupby(
            ["identity_contrast", "outcome_tier"], observed=True
        )["significant"]
        .mean()
        .unstack("outcome_tier")
    )
    fig = _matrix_plot(
        pivot,
        "Direct identity: proporsi hasil p-Holm < 0,05",
        [value.replace("_", " ").title() for value in pivot.columns],
        [value.replace("_vs_", " − ").title() for value in pivot.index],
        "YlOrRd",
        "Proporsi",
        vmin=0,
        vmax=1,
    )
    save_figure(
        fig,
        figures_dir,
        "fig_direct_identity_significance",
        "Ringkasan deskriptif keluarga 24 direct-identity comparisons per outcome.",
        figure_rows,
        orientation="landscape",
    )


def build_figures(
    tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=False)
    configure_matplotlib()
    rows: list[dict[str, object]] = []
    plot_corpus_accounting(tables["scoring_summary"], figures_dir, rows)
    plot_scoring_panels(tables["scoring_summary"], figures_dir, rows)
    plot_independence_and_conflicts(
        tables["persona_independence"], tables["conflict_sensitivity"], figures_dir, rows
    )
    plot_effect_tier_heatmaps(
        tables["paired_within"], tables["outcome_dictionary"], figures_dir, rows
    )
    plot_religion_prevalence(
        tables["descriptives"], tables["outcome_dictionary"], figures_dir, rows
    )
    plot_openings(tables["opening_descriptives"], figures_dir, rows)
    plot_significance_and_sensitivity(
        tables["significance_summary"],
        tables["sensitivity_agreement"],
        tables["paired_direct"],
        figures_dir,
        rows,
    )
    return pd.DataFrame(rows)


def build_human_gate(audit_manifest: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("canonical_source_hash", "automated", "pass", EXPECTED_INPUT_SHA256),
            ("marker_pipeline_version", "automated", "pass", EXPECTED_PIPELINE_VERSION),
            ("visible_surface_only", "automated", "pass", "hidden reasoning excluded"),
            ("single_registry_all_models_personas", "automated", "pass", "50 frozen markers"),
            (
                "ai_exhaustive_second_pass",
                "ai_review",
                "pass_nonfinal",
                str(audit_manifest.get("reviewer", "")),
            ),
            ("human_precision_reviewer_1", "human", "pending", "required"),
            ("human_precision_reviewer_2", "human", "pending", "required"),
            ("human_recall_reviewer_1", "human", "pending", "required"),
            ("human_recall_reviewer_2", "human", "pending", "required"),
            ("inter_rater_agreement", "human", "pending", "required"),
            ("disagreement_adjudication", "human", "pending", "required"),
            (
                "thesis_final_release",
                "release_gate",
                "blocked",
                "unblocked only after two independent human reviews",
            ),
        ],
        columns=["gate_id", "gate_type", "status", "evidence_or_requirement"],
    )


def build_pair_availability_audit(
    within: pd.DataFrame,
    direct: pd.DataFrame,
) -> pd.DataFrame:
    """Expose zero-pair outcomes instead of silently dropping or imputing them."""

    columns = [
        "population",
        "model_id",
        "subset",
        "outcome_id",
        "outcome_tier",
        "outcome_scale",
        "n_pairs",
        "ci_low",
        "ci_high",
    ]
    within_zero = within.loc[within["n_pairs"].eq(0), columns + [
        "identity",
        "contrast_id",
        "left_persona",
        "right_persona",
    ]].copy()
    within_zero.insert(0, "comparison_scope", "within_identity")
    direct_zero = direct.loc[direct["n_pairs"].eq(0), columns + [
        "identity_contrast",
        "condition",
        "contrast_id",
        "left_persona",
        "right_persona",
    ]].copy()
    direct_zero.insert(0, "comparison_scope", "direct_identity")
    result = pd.concat([within_zero, direct_zero], ignore_index=True, sort=False)
    result["known_surface_limitation"] = (
        result["model_id"].eq("openai_gpt-oss-20b")
        & result["outcome_id"].eq("ttr_300")
    )
    result["handling"] = np.where(
        result["known_surface_limitation"],
        "retain_row_with_n_pairs_0_and_inferential_fields_NA",
        "unexpected_requires_investigation",
    )
    result["interpretation"] = np.where(
        result["known_surface_limitation"],
        "GPT-OSS visible explanations in this cell have no paired finite TTR-300 values; do not impute.",
        "Unclassified availability loss.",
    )
    return result


def write_tables(
    output_dir: Path,
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    outcomes: Sequence[OutcomeSpec],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    table_dir = output_dir / "tables"
    dataset_dir = output_dir / "dataset"
    provenance_dir = output_dir / "provenance"
    table_dir.mkdir(parents=True, exist_ok=False)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    provenance_dir.mkdir(parents=True, exist_ok=False)

    analysis_dataset = analysis_response_level(frame, outcomes)
    print("[write] analysis-ready response-level dataset", flush=True)
    analysis_dataset.to_parquet(
        dataset_dir / "ethics_religion_marker_analysis_response_level.parquet",
        index=False,
        compression="zstd",
    )
    write_csv(
        analysis_dataset,
        dataset_dir / "ethics_religion_marker_analysis_response_level.csv.gz",
        gzip=True,
    )
    write_csv(registry, provenance_dir / "marker_registry_snapshot.csv")

    file_map = {
        "outcome_dictionary": "outcome_dictionary.csv",
        "scoring_summary": "scoring_summary.csv",
        "semantic_inventory": "semantic_unaudited_inventory.csv",
        "conflict_sensitivity": "numeric_semantic_conflict_sensitivity.csv",
        "persona_independence": "persona_independence_tests.csv",
        "descriptives": "marker_surface_descriptives.csv.gz",
        "paired_within": "paired_within_identity_all_scales.csv.gz",
        "paired_direct": "paired_direct_identity_all_scales.csv.gz",
        "omnibus": "omnibus_five_condition.csv.gz",
        "opening_descriptives": "opening_descriptives.csv",
        "opening_pairs": "opening_pairwise_stuart_maxwell.csv",
        "opening_transitions": "opening_transitions.csv.gz",
        "pair_count_summary": "pair_count_summary.csv",
        "pair_availability_audit": "pair_availability_audit.csv",
        "significance_summary": "significance_summary_candidate.csv",
        "sensitivity_agreement": "sensitivity_agreement.csv",
        "human_validation_gate": "human_validation_gate.csv",
    }
    gzip_keys = {
        "descriptives",
        "paired_within",
        "paired_direct",
        "omnibus",
        "opening_transitions",
    }
    index_rows: list[dict[str, object]] = []
    for key, filename in file_map.items():
        target = table_dir / filename
        write_csv(tables[key], target, gzip=key in gzip_keys)
        index_rows.append(
            {
                "table_id": key,
                "relative_path": target.relative_to(output_dir).as_posix(),
                "rows": len(tables[key]),
                "columns": len(tables[key].columns),
            }
        )

    for scope, pair_table in (
        ("within", tables["paired_within"]),
        ("direct", tables["paired_direct"]),
    ):
        for scale, split in split_pair_tables(pair_table).items():
            target = table_dir / f"paired_{scope}_{scale}.csv.gz"
            write_csv(split, target, gzip=True)
            index_rows.append(
                {
                    "table_id": f"paired_{scope}_{scale}",
                    "relative_path": target.relative_to(output_dir).as_posix(),
                    "rows": len(split),
                    "columns": len(split.columns),
                }
            )
    index = pd.DataFrame(index_rows)
    write_csv(index, output_dir / "table_index.csv")
    return index


def notebook_markdown_status() -> str:
    return """\
> **Status kandidat — belum boleh diperlakukan sebagai hasil final tesis.**
> Registry 50 marker masih menunggu dua reviewer manusia independen. Audit AI
> exhaustive membantu screening, tetapi tidak memenuhi gerbang validasi manusia.
> Seluruh angka inferensial di notebook ini disediakan agar analisis dapat
> direproduksi dan diaudit; interpretasi final ditahan sampai gerbang lolos.
"""


def create_main_notebook(bundle_dir: Path, source_path: Path) -> nbformat.NotebookNode:
    b = repr(str(bundle_dir))
    cells = [
        nbformat.v4.new_markdown_cell(
            "# ETHICS Religion — Linguistic Marker Tesis v2\n\n"
            + notebook_markdown_status()
        ),
        nbformat.v4.new_markdown_cell(
            "## Kontrak analisis\n\n"
            "Unit analisis adalah satu respons. Marker berasal dari penjelasan visible; "
            "`missing` dan `technical_only` adalah NA. Analisis utama memakai semua pasangan "
            "teks visible, termasuk abstain. Kontras utama adalah I1/I2/I3 − E; "
            "E/I1/I2/I3 − N adalah sensitivity."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n\n"
            f"BUNDLE = Path({b})\n"
            "TABLES = BUNDLE / 'tables'\n"
            "FIGURES = BUNDLE / 'figures'\n"
            "manifest = json.loads((BUNDLE / 'manifest.json').read_text(encoding='utf-8'))\n"
            "display({k: manifest[k] for k in ['bundle_version','result_status','total_responses','bootstrap_resamples','human_validation_complete']})"
        ),
        nbformat.v4.new_markdown_cell("## 1. Akuntansi data dan coverage"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / 'fig01_corpus_accounting.png'), width=1050))\n"
            "display(Image(filename=str(FIGURES / 'fig02_commit_rate_by_subset.png'), width=1050))"
        ),
        nbformat.v4.new_code_cell(
            "scoring = pd.read_csv(TABLES / 'scoring_summary.csv')\n"
            "summary = scoring.groupby('subset')[['total','committed','abstain','missing']].sum()\n"
            "summary['commit_rate'] = summary['committed'] / summary['total']\n"
            "display(summary)"
        ),
        nbformat.v4.new_markdown_cell(
            "## 2. Metrik ETHICS per subset\n\n"
            "Akurasi committed digunakan hanya untuk Commonsense dan Justice. "
            "Deontology dan Virtue dilaporkan sebagai distribusi stance jawaban."
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / 'fig03_scoring_metric_by_subset.png'), width=1050))"
        ),
        nbformat.v4.new_markdown_cell("## 3. Audit measurement error dan konflik"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / 'fig04_persona_independence.png'), width=1050))\n"
            "display(Image(filename=str(FIGURES / 'fig05_numeric_semantic_conflict_sensitivity.png'), width=1050))"
        ),
        nbformat.v4.new_markdown_cell(
            "## 4. Outcome inti: kontras implicit − explicit\n\n"
            "Heatmap berikut adalah ringkasan deskriptif median arah efek. "
            "Keputusan inferensial tetap berasal dari tabel berpasangan per model, subset, identity, dan contrast."
        ),
        nbformat.v4.new_code_cell(
            "for subset in ['commonsense','deontology','justice','virtue']:\n"
            "    display(Markdown(f'### {subset.title()}'))\n"
            "    display(Image(filename=str(FIGURES / f'fig_effect_core_{subset}.png'), width=1050))"
        ),
        nbformat.v4.new_markdown_cell("## 5. Marker khusus Religion"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / 'fig_religion_marker_prevalence.png'), width=1050))\n"
            "display(Image(filename=str(FIGURES / 'fig_significance_rate_candidate.png'), width=1050))"
        ),
        nbformat.v4.new_code_cell(
            "within = pd.read_csv(TABLES / 'paired_within_identity_all_scales.csv.gz')\n"
            "top = within[(within.population == 'all_visible') & (within.multiplicity_family == 'primary_implicit_explicit_3')]\n"
            "top = top.assign(abs_effect=top.effect_size.abs()).sort_values(['p_holm','abs_effect'], ascending=[True,False])\n"
            "display(top[['model_id','subset','identity','contrast_id','outcome_id','n_pairs','effect','effect_size_name','effect_size','ci_low','ci_high','p_raw','p_holm']].head(30))"
        ),
        nbformat.v4.new_markdown_cell("## 6. Opening dan sensitivitas"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(FIGURES / 'fig_opening_composition.png'), width=1050))\n"
            "display(Image(filename=str(FIGURES / 'fig_sensitivity_sign_agreement.png'), width=1050))\n"
            "display(Image(filename=str(FIGURES / 'fig_direct_identity_significance.png'), width=1050))"
        ),
        nbformat.v4.new_markdown_cell(
            "## 7. Gerbang rilis\n\n"
            "Bundle ini menjadi thesis-final hanya setelah precision dan recall audit "
            "diselesaikan oleh dua reviewer manusia, agreement dilaporkan, dan disagreement diadjudikasi."
        ),
        nbformat.v4.new_code_cell(
            "gate = pd.read_csv(TABLES / 'human_validation_gate.csv')\n"
            "display(gate)\n"
            "assert (gate.loc[gate.gate_id == 'thesis_final_release','status'].iloc[0] == 'blocked')"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    # nbformat 5.10 on Python 3.14/Windows can misclassify pathlib.Path as a
    # file-like object.  An explicit UTF-8 handle avoids that compatibility bug.
    with source_path.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(notebook, handle)
    return notebook


def create_atlas_notebook(bundle_dir: Path, source_path: Path) -> nbformat.NotebookNode:
    b = repr(str(bundle_dir))
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Atlas Lengkap — ETHICS Religion Linguistic Marker v2\n\n"
            + notebook_markdown_status()
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n"
            f"BUNDLE = Path({b})\n"
            "TABLES, FIGURES = BUNDLE / 'tables', BUNDLE / 'figures'\n"
            "table_index = pd.read_csv(BUNDLE / 'table_index.csv')\n"
            "figure_index = pd.read_csv(BUNDLE / 'figure_index.csv')\n"
            "display(table_index)\n"
            "display(figure_index)"
        ),
        nbformat.v4.new_markdown_cell("## Kamus outcome dan provenance"),
        nbformat.v4.new_code_cell(
            "dictionary = pd.read_csv(TABLES / 'outcome_dictionary.csv')\n"
            "display(dictionary.groupby(['outcome_tier','outcome_scale']).size().rename('outcomes'))\n"
            "display(dictionary)"
        ),
        nbformat.v4.new_markdown_cell("## Semua visual tesis-native"),
        nbformat.v4.new_code_cell(
            "for row in figure_index.itertuples(index=False):\n"
            "    display(Markdown(f'### {row.figure_id}'))\n"
            "    display(Markdown(row.purpose))\n"
            "    display(Image(filename=str(FIGURES / f'{row.figure_id}.png'), width=1050))"
        ),
        nbformat.v4.new_markdown_cell("## Jelajah hasil berpasangan"),
        nbformat.v4.new_code_cell(
            "within = pd.read_csv(TABLES / 'paired_within_identity_all_scales.csv.gz')\n"
            "direct = pd.read_csv(TABLES / 'paired_direct_identity_all_scales.csv.gz')\n"
            "display(within.groupby(['population','multiplicity_family','outcome_tier']).size().rename('rows'))\n"
            "display(direct.groupby(['population','outcome_tier']).size().rename('rows'))"
        ),
        nbformat.v4.new_code_cell(
            "def query_within(model=None, subset=None, identity=None, tier=None, significant_only=False):\n"
            "    result = within.copy()\n"
            "    for column, value in [('model_id',model),('subset',subset),('identity',identity),('outcome_tier',tier)]:\n"
            "        if value is not None:\n"
            "            result = result[result[column] == value]\n"
            "    if significant_only:\n"
            "        result = result[result.p_holm < 0.05]\n"
            "    return result.sort_values(['p_holm','outcome_id'])\n\n"
            "display(query_within(subset='commonsense', tier='core_confirmatory', significant_only=True).head(50))"
        ),
        nbformat.v4.new_markdown_cell("## Pair counts, omnibus, opening, dan sensitivitas"),
        nbformat.v4.new_code_cell(
            "for filename in ['pair_count_summary.csv','omnibus_five_condition.csv.gz','opening_pairwise_stuart_maxwell.csv','sensitivity_agreement.csv']:\n"
            "    frame = pd.read_csv(TABLES / filename)\n"
            "    display(Markdown(f'### {filename} — {len(frame):,} baris'))\n"
            "    display(frame.head(20))"
        ),
        nbformat.v4.new_markdown_cell("## Gerbang manusia"),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(TABLES / 'human_validation_gate.csv'))"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    with source_path.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(notebook, handle)
    return notebook


def execute_and_export_notebook(
    source_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path]:
    notebook = nbformat.read(source_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(repo_root)}},
    )
    executed = client.execute(cwd=str(repo_root))
    executed_path = output_dir / f"{source_path.stem}_executed.ipynb"
    with executed_path.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(executed, handle)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, resources = exporter.from_notebook_node(executed)
    writer = FilesWriter(build_directory=str(output_dir))
    html_name = f"{source_path.stem}_executed"
    writer.write(body, resources, notebook_name=html_name)
    return executed_path, output_dir / f"{html_name}.html"


def create_notebooks(
    bundle_dir: Path,
    notebook_dir: Path,
    execute: bool,
) -> list[Path]:
    notebook_dir.mkdir(parents=True, exist_ok=True)
    main_source = notebook_dir / MAIN_NOTEBOOK_NAME
    atlas_source = notebook_dir / ATLAS_NOTEBOOK_NAME
    for path in (main_source, atlas_source):
        if path.exists():
            raise FileExistsError(f"Notebook output is write-once and already exists: {path}")
    create_main_notebook(bundle_dir, main_source)
    create_atlas_notebook(bundle_dir, atlas_source)
    generated = [main_source, atlas_source]
    bundle_notebooks = bundle_dir / "notebooks"
    bundle_notebooks.mkdir(parents=True, exist_ok=False)
    shutil.copy2(main_source, bundle_notebooks / main_source.name)
    shutil.copy2(atlas_source, bundle_notebooks / atlas_source.name)
    if execute:
        for source in (main_source, atlas_source):
            executed, html = execute_and_export_notebook(source, bundle_notebooks, REPO_ROOT)
            generated.extend([executed, html])
    return generated


def write_readme(output_dir: Path, bootstrap_resamples: int) -> None:
    text = f"""# ETHICS Religion Linguistic Marker v2 — kandidat

Status: **belum thesis-final**. Registry marker masih menunggu dua reviewer
manusia independen. Audit AI exhaustive tidak menggantikan validasi manusia.

## Isi

- `dataset/`: dataset response-level analisis dalam CSV gzip dan Parquet;
- `tables/`: scoring, descriptives, paired inference, omnibus, opening,
  sensitivity, pair counts, dan gerbang validasi;
- `figures/`: setiap visual dalam PNG 300 dpi, PDF, dan SVG;
- `notebooks/`: salinan notebook source, executed notebook, dan HTML;
- `provenance/`: snapshot marker registry;
- `thesis/`: snippet integrasi LaTeX dan catatan batas interpretasi.

Kontras utama adalah I1/I2/I3 minus E (Holm dalam keluarga 3). Kontras terhadap
N adalah sensitivity (Holm dalam keluarga 4). Direct identity memakai enam
pasangan identity × empat kondisi (Holm dalam keluarga 24). Bootstrap memakai
{bootstrap_resamples:,} resample per baris hasil.

`missing` dan `technical_only` dipertahankan sebagai NA. Semua respons visible,
termasuk abstain, masuk analisis utama. Akurasi hanya dipakai sebagai konstruk
utama pada Commonsense dan Justice; Deontology/Virtue memakai distribusi stance.

Sebanyak 84 baris pasangan (`40` within dan `44` direct) memiliki `n_pairs=0`,
seluruhnya khusus outcome `ttr_300` pada GPT-OSS 20B. Baris itu sengaja tetap
ada dengan CI `NA`; tidak ada imputasi atau penghapusan diam-diam. Rinciannya
tersimpan pada `tables/pair_availability_audit.csv`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_thesis_material(output_dir: Path) -> None:
    thesis_dir = output_dir / "thesis"
    thesis_dir.mkdir(parents=True, exist_ok=False)
    snippet = r"""% Bundle kandidat: jangan aktifkan sebagai hasil final sebelum human gate lolos.
% Seluruh figure disediakan dalam PDF/SVG/PNG; PDF diprioritaskan untuk LaTeX.
\begin{landscape}
\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_effect_core_commonsense.pdf}
  \caption{Ringkasan arah efek linguistic marker inti pada ETHICS Commonsense.}
  \label{fig:ethics-religion-lm-core-commonsense}
\end{figure}
\end{landscape}

% Tabel inferensial lengkap tidak ditempel seluruhnya ke Bab IV. Pilih hanya
% hasil yang relevan setelah audit manusia, lalu arahkan tabel lengkap ke lampiran.
"""
    (thesis_dir / "integration_snippet_candidate.tex").write_text(
        snippet, encoding="utf-8"
    )
    limitation = """# Batas interpretasi wajib

1. Hasil masih kandidat karena dua audit manusia independen belum selesai.
2. AI second-pass bukan pengganti human validation.
3. Heatmap efek merangkum model/identity/contrast secara deskriptif; keputusan
   inferensial harus dibaca dari tabel response-paired lengkap.
4. Deontology dan Virtue tidak diinterpretasikan sebagai akurasi moral.
5. Marker Religion tidak boleh dijumlahkan menjadi satu skor religiositas.
6. Perbedaan marker tidak membuktikan kualitas reasoning atau keyakinan model.
"""
    (thesis_dir / "interpretation_guardrails.md").write_text(
        limitation, encoding="utf-8"
    )


def validate_in_memory(
    frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    figure_index: pd.DataFrame,
) -> dict[str, object]:
    expected = expected_table_rows()
    observed = {
        "scoring_summary": len(tables["scoring_summary"]),
        "descriptives": len(tables["descriptives"]),
        "within_pairs": len(tables["paired_within"]),
        "direct_pairs": len(tables["paired_direct"]),
        "omnibus": len(tables["omnibus"]),
        "opening_pairs": len(tables["opening_pairs"]),
    }
    row_contract = observed == expected
    within = tables["paired_within"]
    direct = tables["paired_direct"]
    correction_valid = bool(
        (within["p_holm"] + 1e-12 >= within["p_raw"]).all()
        and (direct["p_holm"] + 1e-12 >= direct["p_raw"]).all()
    )
    marker_within = within[within["registered_measurement"].ne("surface")]
    marker_direct = direct[direct["registered_measurement"].ne("surface")]
    raw_counts_present = bool(
        marker_within[
            ["raw_count_sum_left", "raw_count_sum_right"]
        ].notna().all().all()
        and marker_direct[
            ["raw_count_sum_left", "raw_count_sum_right"]
        ].notna().all().all()
    )
    zero_pairs = pd.concat(
        [
            within.loc[within["n_pairs"].eq(0), ["model_id", "outcome_id"]],
            direct.loc[direct["n_pairs"].eq(0), ["model_id", "outcome_id"]],
        ],
        ignore_index=True,
    )
    zero_pairs_are_known_ttr_limitation = bool(
        len(zero_pairs)
        and zero_pairs["model_id"].eq("openai_gpt-oss-20b").all()
        and zero_pairs["outcome_id"].eq("ttr_300").all()
    )
    positive_within = within["n_pairs"].gt(0)
    positive_direct = direct["n_pairs"].gt(0)
    ci_contract = bool(
        within.loc[positive_within, ["ci_low", "ci_high"]].notna().all().all()
        and direct.loc[positive_direct, ["ci_low", "ci_high"]].notna().all().all()
        and within.loc[~positive_within, ["ci_low", "ci_high"]].isna().all().all()
        and direct.loc[~positive_direct, ["ci_low", "ci_high"]].isna().all().all()
    )
    validations = {
        "source_rows_154428": len(frame) == 154_428,
        "unique_response_keys": not frame.duplicated(
            ["model_id", "persona", "question_id"]
        ).any(),
        "source_hash_pinned": True,
        "outcome_hierarchy_21_12_20": tables["outcome_dictionary"][
            "outcome_tier"
        ].value_counts().to_dict()
        == {
            "core_confirmatory": 21,
            "religion_specific": 20,
            "exploratory_general": 12,
        },
        "table_row_contract": row_contract,
        "zero_pair_rows_are_known_gpt_oss_ttr_limitation": zero_pairs_are_known_ttr_limitation,
        "all_non_ttr_pair_counts_positive": bool(
            within.loc[within["outcome_id"].ne("ttr_300"), "n_pairs"].gt(0).all()
            and direct.loc[direct["outcome_id"].ne("ttr_300"), "n_pairs"].gt(0).all()
        ),
        "holm_not_below_raw": correction_valid,
        "confidence_interval_na_matches_zero_pair_rows": ci_contract,
        "raw_counts_reported_for_markers": raw_counts_present,
        "subsets_not_pooled_in_inference": bool(
            within["subset"].nunique() == 4 and direct["subset"].nunique() == 4
        ),
        "figure_min_font_at_least_9_5_pt": bool(
            figure_index["minimum_declared_font_pt"].ge(9.5).all()
        ),
        "figure_width_thesis_native": bool(
            figure_index["width_in"].le(9.3 + 1e-9).all()
        ),
        "human_validation_complete": False,
    }
    required_automated = [
        key for key in validations if key != "human_validation_complete"
    ]
    if not all(validations[key] for key in required_automated):
        raise ValueError(
            f"Automated validation failed: {validations}; rows={observed}; expected={expected}"
        )
    return {
        **validations,
        "observed_table_rows": observed,
        "expected_table_rows": expected,
    }


def collect_hashes(output_dir: Path) -> dict[str, str]:
    files = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"]
    return {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(files)
    }


def build(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    marker_dir = args.marker_dir.resolve()
    canonical_csv = args.canonical_csv.resolve()
    audit_manifest_path = args.audit_manifest.resolve()
    notebook_dir = args.notebook_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output is write-once and already exists: {output_dir}")
    if args.bootstrap_resamples < 1_000:
        raise ValueError("At least 1,000 bootstrap resamples are required.")
    output_dir.mkdir(parents=True, exist_ok=False)

    print("[1/10] Load and validate frozen sources", flush=True)
    (
        frame,
        registry,
        outcome_dictionary,
        marker_manifest,
        audit_manifest,
        source_validation,
    ) = load_sources(marker_dir, canonical_csv, audit_manifest_path)
    outcomes = tuple(
        OutcomeSpec(
            outcome_id=row.outcome_id,
            outcome_tier=row.outcome_tier,
            outcome_family=row.outcome_family,
            value_col=row.value_col,
            raw_col=row.raw_col if isinstance(row.raw_col, str) else "",
            outcome_scale=row.outcome_scale,
            registered_measurement=row.registered_measurement,
            analysis_role=row.analysis_role,
        )
        for row in outcome_dictionary.itertuples(index=False)
    )

    tables: dict[str, pd.DataFrame] = {"outcome_dictionary": outcome_dictionary}
    print("[2/10] Build scoring, provenance, and independence audits", flush=True)
    tables["scoring_summary"] = build_scoring_summary(frame)
    tables["semantic_inventory"] = build_semantic_inventory(frame)
    tables["conflict_sensitivity"] = build_conflict_sensitivity(frame)
    tables["persona_independence"] = build_persona_independence(frame)

    print("[3/10] Build response-level descriptives", flush=True)
    tables["descriptives"] = build_descriptives(frame, outcomes)

    print(
        f"[4/10] Paired within-identity inference ({args.bootstrap_resamples:,} bootstrap resamples)",
        flush=True,
    )
    tables["paired_within"] = build_within_identity_pairs(
        frame, outcomes, args.bootstrap_resamples
    )

    print("[5/10] Paired direct-identity inference", flush=True)
    tables["paired_direct"] = build_direct_identity_pairs(
        frame, outcomes, args.bootstrap_resamples
    )

    print("[6/10] Five-condition omnibus and opening-composition inference", flush=True)
    tables["omnibus"] = build_omnibus(frame, outcomes)
    tables["opening_descriptives"] = build_opening_descriptives(frame)
    opening_pairs, opening_transitions = build_opening_pairs(frame)
    tables["opening_pairs"] = opening_pairs
    tables["opening_transitions"] = opening_transitions
    tables["pair_count_summary"] = build_pair_count_summary(
        tables["paired_within"], tables["paired_direct"], tables["opening_pairs"]
    )
    tables["pair_availability_audit"] = build_pair_availability_audit(
        tables["paired_within"], tables["paired_direct"]
    )
    tables["significance_summary"] = build_significance_summary(
        tables["paired_within"],
        tables["paired_direct"],
        tables["omnibus"],
        tables["opening_pairs"],
    )
    tables["sensitivity_agreement"] = build_sensitivity_agreement(
        tables["paired_within"], tables["paired_direct"]
    )
    tables["human_validation_gate"] = build_human_gate(audit_manifest)

    print("[7/10] Write analysis dataset and tables", flush=True)
    table_index = write_tables(
        output_dir, frame, registry, outcomes, tables
    )

    print("[8/10] Render thesis-native figures", flush=True)
    figure_index = build_figures(tables, output_dir / "figures")
    write_csv(figure_index, output_dir / "figure_index.csv")
    write_readme(output_dir, args.bootstrap_resamples)
    write_thesis_material(output_dir)

    print("[9/10] Automated validation", flush=True)
    validations = validate_in_memory(frame, tables, figure_index)
    (output_dir / "automated_validation.json").write_text(
        json.dumps(validations, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    provisional_manifest = {
        "bundle_version": BUNDLE_VERSION,
        "result_status": "candidate_blocked_pending_two_independent_human_reviews",
        "created_at": created_at,
        "analysis_builder": str(Path(__file__).resolve()),
        "analysis_builder_sha256": sha256_file(Path(__file__).resolve()),
        "core_engine": str(
            REPO_ROOT
            / "src"
            / "benchmarks"
            / "ethics"
            / "religion_ethics_linguistic_markers_v2.py"
        ),
        "core_engine_sha256": sha256_file(
            REPO_ROOT
            / "src"
            / "benchmarks"
            / "ethics"
            / "religion_ethics_linguistic_markers_v2.py"
        ),
        "canonical_csv": str(canonical_csv),
        "canonical_sha256": sha256_file(canonical_csv),
        "marker_dir": str(marker_dir),
        "marker_pipeline_version": marker_manifest["pipeline_version"],
        "marker_manifest_sha256": sha256_file(marker_dir / "manifest.json"),
        "audit_manifest": str(audit_manifest_path),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "audit_reviewer": audit_manifest.get("reviewer"),
        "human_review_status": audit_manifest.get("human_review_status"),
        "human_validation_complete": False,
        "total_responses": len(frame),
        "responses_text_available": int(frame["text_available"].sum()),
        "responses_text_missing": int(frame["text_status"].eq("missing").sum()),
        "responses_text_technical_only": int(
            frame["text_status"].eq("technical_only").sum()
        ),
        "responses_candidate_uncommitted": int(frame["candidate_uncommitted"].sum()),
        "models": len(MODELS),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": list(SUBSETS),
        "outcome_hierarchy": {
            "core_confirmatory": 21,
            "exploratory_general": 12,
            "religion_specific": 20,
            "opening_categories": 11,
        },
        "bootstrap_resamples": args.bootstrap_resamples,
        "analysis_populations": {
            "primary": "both responses have visible linguistic text; abstain retained",
            "committed_sensitivity": "both responses validated committed",
            "correctness_sensitivity": "both responses correct; Commonsense and Justice only",
        },
        "persona_independence_inference": {
            "default": "Pearson chi-square asymptotic",
            "sparse_expected_cells": "Pearson statistic with 10,000 fixed-margin Monte Carlo resamples",
            "effect_size": "Cramer's V",
        },
        "sensitivity_effect_metric": (
            "paired rank-biserial for numeric/density outcomes; paired prevalence delta for binary outcomes"
        ),
        "multiplicity_families": {
            "primary_within_identity": "Holm over I1/I2/I3 minus E (3)",
            "neutral_sensitivity": "Holm over E/I1/I2/I3 minus N (4)",
            "direct_identity": "Holm over 6 identity pairs x 4 conditions (24)",
            "cross_outcome_robustness": "Benjamini-Hochberg within contrast and outcome tier",
        },
        "known_surface_availability_limitations": {
            "zero_pair_rows_total": int(
                len(tables["pair_availability_audit"])
            ),
            "within_zero_pair_rows": int(
                tables["pair_availability_audit"]["comparison_scope"]
                .eq("within_identity")
                .sum()
            ),
            "direct_zero_pair_rows": int(
                tables["pair_availability_audit"]["comparison_scope"]
                .eq("direct_identity")
                .sum()
            ),
            "scope": "ttr_300 on openai_gpt-oss-20b only",
            "handling": "retain rows with n_pairs=0 and inferential CI=NA; never impute",
        },
        "hidden_reasoning_used": False,
        "ground_truth_used_for_marker_extraction": False,
        "marker_extraction_rerun": False,
        "answer_reparsing_run": False,
        "paid_api_calls": False,
        "source_validation": source_validation,
        "automated_validation": validations,
        "table_index_rows": len(table_index),
        "figure_index_rows": len(figure_index),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(provisional_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[10/10] Create and execute thesis + atlas notebooks", flush=True)
    notebook_paths = create_notebooks(
        output_dir,
        notebook_dir,
        execute=not args.skip_notebook_execution,
    )
    final_manifest = dict(provisional_manifest)
    final_manifest["notebooks"] = [str(path) for path in notebook_paths]
    final_manifest["generated_file_sha256"] = collect_hashes(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {output_dir}", flush=True)
    return output_dir


def main() -> int:
    args = parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

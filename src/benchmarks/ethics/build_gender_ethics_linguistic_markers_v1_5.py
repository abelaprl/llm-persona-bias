"""Build the integrated ETHICS Gender linguistic-marker candidate v1.5.

This builder extends the validated-structure v1.4 candidate without replacing
its 21-outcome core.  It adds the pieces needed to mirror the Gender GSM8K
analysis while preserving ETHICS-specific validity constraints:

* direct text availability and answer commitment;
* eleven registered gender markers, kept separate from general style markers;
* a response-only closed gender lexicon, reported as descriptive evidence;
* Friedman/Cochran-Q omnibus checks;
* two-reviewer audit templates covering every inferential marker.

No LLM inference, answer parsing, or marker extraction is run here.  The input
is the frozen v1.6 response-level marker bundle.  Every output is written to a
new directory and remains ``candidate`` until the human-audit finalizer opens
the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.benchmarks.ethics import (
    build_gender_ethics_direct_thesis_visuals_v1_2 as descriptive,
)
from src.benchmarks.ethics import (
    build_gender_ethics_linguistic_markers_v1_4 as core,
)
from src.core.linguistic_markers.gender_registry import (
    ETHICS_GENDER_MARKERS,
)
from src.core.linguistic_markers.registry import MarkerSpec
from src.core.linguistic_markers.statistics import (
    build_omnibus,
    build_paired_binary,
    build_paired_gender_availability,
    build_paired_gender_binary,
    build_paired_gender_numeric,
    build_paired_numeric,
)


BUNDLE_VERSION = "gender-ethics-linguistic-markers-v1.5.0-integrated-candidate"
BOOTSTRAP_RESAMPLES = 2_000
EXPECTED_PRECISION_ITEMS = 2_583
EXPECTED_RECALL_ITEMS = 162
EXPECTED_LEXICON_TERMS = 95

GENDER_MARKER_IDS = tuple(descriptive.GENDER_MARKERS)
AUDITED_MARKER_IDS = tuple((*core.MARKER_IDS, *GENDER_MARKER_IDS))


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--marker-dir",
        type=Path,
        default=(
            repo_root
            / "result/ETHICS/reports/Gender_Persona"
            / "linguistic_markers_gender_candidate_20260821_v1_6_0"
        ),
    )
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        default=(
            repo_root
            / "result/ETHICS/reports/Gender_Persona/_combined"
            / "parser_refinement_20260820/all_models_results_full.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_gender_markers() -> tuple[MarkerSpec, ...]:
    lookup = {marker.marker_id: marker for marker in ETHICS_GENDER_MARKERS}
    missing = sorted(set(GENDER_MARKER_IDS).difference(lookup))
    if missing:
        raise ValueError(f"Gender registry is missing markers: {missing}")
    selected = tuple(lookup[marker_id] for marker_id in GENDER_MARKER_IDS)
    if len(selected) != 11 or len({marker.marker_id for marker in selected}) != 11:
        raise ValueError("Expected eleven unique registered gender markers.")
    return selected


def _split_within(
    table: pd.DataFrame,
    *,
    prefix: str,
    pair_filter: str,
) -> dict[str, pd.DataFrame]:
    return {
        f"paired_{prefix}_vs_neutral_{pair_filter}.csv": core._tag_table(
            table[table["contrast_id"].isin(core.NEUTRAL_CONTRASTS)],
            analysis_family=f"{prefix}_vs_neutral",
            pair_filter=pair_filter,
        ),
        f"paired_{prefix}_vs_explicit_{pair_filter}.csv": core._tag_table(
            table[table["contrast_id"].isin(core.EXPLICIT_CONTRASTS)],
            analysis_family=f"{prefix}_vs_explicit",
            pair_filter=pair_filter,
        ),
    }


def _binary_sensitivity(
    primary: pd.DataFrame,
    committed: pd.DataFrame,
    *,
    keys: list[str],
    family: str,
) -> pd.DataFrame:
    left = primary[
        [*keys, "n_pairs", "prevalence_delta", "p_holm"]
    ].rename(
        columns={
            "n_pairs": "n_pairs_all_text",
            "prevalence_delta": "prevalence_delta_all_text",
            "p_holm": "p_holm_all_text",
        }
    )
    right = committed[
        [*keys, "n_pairs", "prevalence_delta", "p_holm"]
    ].rename(
        columns={
            "n_pairs": "n_pairs_committed",
            "prevalence_delta": "prevalence_delta_committed",
            "p_holm": "p_holm_committed",
        }
    )
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged.insert(0, "result_status", "candidate")
    merged.insert(1, "analysis_family", family)
    merged["pair_retention"] = (
        merged["n_pairs_committed"] / merged["n_pairs_all_text"].replace(0, np.nan)
    )
    merged["same_direction"] = np.sign(
        merged["prevalence_delta_all_text"]
    ).eq(np.sign(merged["prevalence_delta_committed"]))
    return merged


def _plot_availability(table: pd.DataFrame, image_dir: Path) -> Path:
    selected = table[table["outcome_id"].eq("text_available")].copy()
    selected["delta_pp"] = 100 * selected["rate_delta"]
    limit = max(0.25, float(selected["delta_pp"].abs().max()))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, sharey=True)
    for index, (ax, subset) in enumerate(zip(axes.flat, core.SUBSETS)):
        panel = selected[selected["subset"].eq(subset)].pivot(
            index="model_id", columns="condition", values="delta_pp"
        )
        panel = panel.reindex(index=core.MODEL_ORDER, columns=core.CONDITIONS)
        panel.index = [core.MODEL_LABELS[item] for item in panel.index]
        panel.columns = [core.CONDITION_LABELS[item] for item in panel.columns]
        sns.heatmap(
            panel,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=True,
            fmt="+.2f",
            linewidths=0.5,
            linecolor="white",
            cbar=index == 3,
            cbar_kws={"label": "Man − Woman (poin persentase)"} if index == 3 else None,
        )
        ax.set_title(core.SUBSET_LABELS[subset], fontweight="bold")
        ax.set_xlabel("Skenario")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Differential availability visible text: Man − Woman", fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.03, 0.94, 0.96), h_pad=1.7, w_pad=1.7)
    return descriptive._save(fig, image_dir, "01_direct_visible_text_availability.png")


def _make_audit_templates(
    marker_dir: Path,
    output_dir: Path,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    previous = core.MARKER_IDS
    try:
        core.MARKER_IDS = AUDITED_MARKER_IDS
        precision, recall, status = core._build_audit_sheets(
            marker_dir, output_dir, canonical
        )
    finally:
        core.MARKER_IDS = previous
    if len(precision) != EXPECTED_PRECISION_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_PRECISION_ITEMS:,} precision items, got {len(precision):,}."
        )
    if len(recall) != EXPECTED_RECALL_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_RECALL_ITEMS:,} recall items, got {len(recall):,}."
        )
    if set(precision["marker_id"]) != set(AUDITED_MARKER_IDS):
        raise ValueError("Precision templates do not cover every audited marker.")
    return precision, recall, status


def _write_inventory(marker_dir: Path, output_dir: Path) -> pd.DataFrame:
    registry = pd.read_csv(marker_dir / "marker_registry.csv")
    selected = registry[registry["marker_id"].isin(AUDITED_MARKER_IDS)].copy()
    if len(selected) != len(AUDITED_MARKER_IDS):
        raise ValueError("Integrated marker inventory does not contain 29 markers.")
    surface = pd.DataFrame(
        [
            {
                "marker_id": outcome,
                "registry_scope": "surface",
                "family": "surface",
                "measurement": "numeric",
                "analysis_role": "primary",
                "validation_status": "automated_surface_metric",
                "theoretical_source": "visible response surface",
                "operational_definition": "",
            }
            for outcome in core.SURFACE_OUTCOMES
        ]
    )
    inventory = pd.concat([surface, selected], ignore_index=True, sort=False)
    inventory.insert(0, "result_status", "candidate")
    inventory.to_csv(
        output_dir / "marker_inventory_integrated.csv", index=False, encoding="utf-8"
    )
    return inventory


def _build_supplementary(
    marker_dir: Path,
    canonical_csv: Path,
    output_dir: Path,
) -> tuple[list[str], list[str], dict[str, object]]:
    supplementary = output_dir / "supplementary_images"
    supplementary.mkdir(exist_ok=False)
    descriptive._style()
    frame, _, scenario_lookup, lexicon_coverage = descriptive._load_sources(
        marker_dir, canonical_csv
    )
    hits, presence, occurrences, term_validation = descriptive._build_term_hits(
        frame, scenario_lookup, lexicon_coverage
    )
    term_prevalence = descriptive._term_prevalence(frame, hits)
    direct_terms = descriptive._direct_term_table(frame, presence, occurrences)
    general_direct, gender_direct = descriptive._direct_marker_tables(frame)
    response_outcomes = descriptive._response_outcomes(frame)

    tables = {
        "gender_term_response_hits.csv.gz": hits,
        "gender_term_surface_reconstruction_validation.csv": term_validation,
        "gender_term_prevalence_by_model_identity_condition_subset.csv": term_prevalence,
        "gender_term_direct_man_woman_response_only.csv": direct_terms,
        "general_marker_direct_man_woman_descriptive.csv": general_direct,
        "gender_marker_direct_man_woman_by_subset_descriptive.csv": gender_direct,
        "response_outcomes_direct_man_woman_by_subset.csv": response_outcomes,
    }
    for filename, table in tables.items():
        compression = "gzip" if filename.endswith(".gz") else None
        table.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8",
            compression=compression,
        )

    selected_terms = descriptive._selected_terms(term_prevalence)
    rendered: list[Path] = []
    rendered.append(
        descriptive._plot_term_absolute(term_prevalence, selected_terms, supplementary)
    )
    term_limit = max(
        1.0,
        float(
            direct_terms[
                direct_terms["term"].isin(selected_terms)
                & ~direct_terms["subset"].eq("all")
            ]["prevalence_delta_pp"].abs().max()
        ),
    )
    for number, condition in zip(range(5, 9), descriptive.CONDITIONS):
        rendered.append(
            descriptive._plot_term_direct_heatmap(
                direct_terms,
                selected_terms,
                supplementary,
                condition,
                number,
                term_limit,
            )
        )
    gender_limit = max(
        1.0,
        float(
            gender_direct[~gender_direct["subset"].eq("all")][
                "prevalence_delta_pp"
            ].abs().max()
        ),
    )
    for number, condition in zip(range(13, 17), descriptive.CONDITIONS):
        rendered.append(
            descriptive._plot_marker_forest(
                gender_direct,
                supplementary,
                condition=condition,
                outcomes=descriptive.GENDER_MARKERS,
                labels=descriptive.GENDER_MARKER_LABELS,
                value="prevalence_delta_pp",
                xlabel="Man − Woman (poin persentase)",
                title=(
                    "Marker gender: perbandingan langsung Man dan Woman pada "
                    f"{descriptive.CONDITION_LABELS[condition]}"
                ),
                filename=f"{number:02d}_direct_gender_markers_{condition}.png",
                xlimit=gender_limit,
            )
        )
    rendered.append(descriptive._plot_commit_delta(response_outcomes, supplementary))

    validations = {
        "lexicon_term_count": int(term_validation["term"].nunique()),
        "lexicon_occurrences_match": bool(term_validation["occurrences_match"].all()),
        "lexicon_responses_match": bool(term_validation["responses_match"].all()),
        "direct_term_rows": len(direct_terms),
        "response_outcome_rows": len(response_outcomes),
        "selected_gender_terms": selected_terms,
        "supplementary_png_count": len(rendered),
        "supplementary_svg_count": sum(path.with_suffix(".svg").is_file() for path in rendered),
    }
    return list(tables), [path.name for path in rendered], validations


def build_bundle(
    marker_dir: Path,
    canonical_csv: Path,
    output_dir: Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> Path:
    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    output_dir = output_dir.resolve()
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")

    print("[1/10] Building the 21-outcome core bundle")
    core.build_bundle(marker_dir, canonical_csv, output_dir, bootstrap_resamples)
    core_markers = core._selected_markers()
    gender_markers = _selected_gender_markers()
    frame, canonical, marker_manifest, validations = core._load_and_validate(
        marker_dir,
        canonical_csv,
        markers=(*core_markers, *gender_markers),
    )
    density_gender = tuple(
        marker for marker in gender_markers if marker.measurement == "density"
    )
    committed = core._committed_frame(frame, (*core_markers, *gender_markers))

    print("[2/10] Computing gender-marker presence statistics")
    binary_all = build_paired_binary(
        frame, bootstrap_resamples, markers=gender_markers
    )
    binary_committed = build_paired_binary(
        committed, bootstrap_resamples, markers=gender_markers
    )
    binary_tables = {
        **_split_within(
            binary_all,
            prefix="gender_marker_presence",
            pair_filter="all_visible_text",
        ),
        **_split_within(
            binary_committed,
            prefix="gender_marker_presence",
            pair_filter="both_committed",
        ),
    }
    direct_binary_all = core._tag_table(
        build_paired_gender_binary(
            frame, bootstrap_resamples, markers=gender_markers
        ),
        analysis_family="gender_marker_presence_man_vs_woman",
        pair_filter="all_visible_text",
    )
    direct_binary_committed = core._tag_table(
        build_paired_gender_binary(
            committed, bootstrap_resamples, markers=gender_markers
        ),
        analysis_family="gender_marker_presence_man_vs_woman",
        pair_filter="both_committed",
    )
    binary_tables[
        "paired_gender_marker_presence_direct_man_woman_all_text.csv"
    ] = direct_binary_all
    binary_tables[
        "paired_gender_marker_presence_direct_man_woman_committed.csv"
    ] = direct_binary_committed
    for filename, table in binary_tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8")

    print("[3/10] Computing density statistics for density-defined gender markers")
    density_tables: dict[str, pd.DataFrame] = {}
    if density_gender:
        density_ids = {marker.marker_id for marker in density_gender}
        within_density_all = build_paired_numeric(
            frame, bootstrap_resamples, markers=density_gender
        )
        within_density_all = within_density_all[
            within_density_all["outcome_id"].isin(density_ids)
        ]
        within_density_all = core._attach_raw_count_summaries(
            within_density_all, frame, density_gender
        )
        within_density_committed = build_paired_numeric(
            committed, bootstrap_resamples, markers=density_gender
        )
        within_density_committed = within_density_committed[
            within_density_committed["outcome_id"].isin(density_ids)
        ]
        within_density_committed = core._attach_raw_count_summaries(
            within_density_committed, committed, density_gender
        )
        density_tables.update(
            _split_within(
                within_density_all,
                prefix="gender_marker_density",
                pair_filter="all_visible_text",
            )
        )
        density_tables.update(
            _split_within(
                within_density_committed,
                prefix="gender_marker_density",
                pair_filter="both_committed",
            )
        )
        direct_density_all = build_paired_gender_numeric(
            frame, bootstrap_resamples, markers=density_gender
        )
        direct_density_all = direct_density_all[
            direct_density_all["outcome_id"].isin(density_ids)
        ]
        direct_density_all = core._attach_raw_count_summaries(
            direct_density_all, frame, density_gender
        )
        direct_density_committed = build_paired_gender_numeric(
            committed, bootstrap_resamples, markers=density_gender
        )
        direct_density_committed = direct_density_committed[
            direct_density_committed["outcome_id"].isin(density_ids)
        ]
        direct_density_committed = core._attach_raw_count_summaries(
            direct_density_committed, committed, density_gender
        )
        density_tables[
            "paired_gender_marker_density_direct_man_woman_all_text.csv"
        ] = core._tag_table(
            direct_density_all,
            analysis_family="gender_marker_density_man_vs_woman",
            pair_filter="all_visible_text",
        )
        density_tables[
            "paired_gender_marker_density_direct_man_woman_committed.csv"
        ] = core._tag_table(
            direct_density_committed,
            analysis_family="gender_marker_density_man_vs_woman",
            pair_filter="both_committed",
        )
        for filename, table in density_tables.items():
            table.to_csv(output_dir / filename, index=False, encoding="utf-8")

    print("[4/10] Computing omnibus and differential-availability checks")
    omnibus_all = core._tag_table(
        build_omnibus(frame, markers=(*core_markers, *gender_markers)),
        analysis_family="omnibus_five_scenarios",
        pair_filter="all_visible_text",
    )
    omnibus_committed = core._tag_table(
        build_omnibus(committed, markers=(*core_markers, *gender_markers)),
        analysis_family="omnibus_five_scenarios",
        pair_filter="both_committed",
    )
    omnibus_all.to_csv(
        output_dir / "omnibus_five_scenarios_all_text.csv",
        index=False,
        encoding="utf-8",
    )
    omnibus_committed.to_csv(
        output_dir / "omnibus_five_scenarios_committed.csv",
        index=False,
        encoding="utf-8",
    )
    availability = core._tag_table(
        build_paired_gender_availability(frame, bootstrap_resamples),
        analysis_family="differential_text_availability",
        pair_filter="all_responses",
    )
    availability.to_csv(
        output_dir / "direct_gender_text_availability.csv",
        index=False,
        encoding="utf-8",
    )

    within_keys = [
        "model_id",
        "benchmark",
        "subset",
        "identity",
        "outcome_id",
        "contrast_id",
    ]
    direct_keys = [
        "model_id",
        "benchmark",
        "subset",
        "outcome_id",
        "contrast_id",
    ]
    binary_sensitivity = pd.concat(
        [
            _binary_sensitivity(
                binary_all,
                binary_committed,
                keys=within_keys,
                family="gender_marker_presence_within_identity",
            ),
            _binary_sensitivity(
                direct_binary_all,
                direct_binary_committed,
                keys=direct_keys,
                family="gender_marker_presence_man_vs_woman",
            ),
        ],
        ignore_index=True,
    )
    binary_sensitivity.to_csv(
        output_dir / "paired_gender_marker_presence_committed_sensitivity.csv",
        index=False,
        encoding="utf-8",
    )

    print("[5/10] Rebuilding the response-only closed gender lexicon")
    supplementary_tables, supplementary_figures, supplementary_validation = (
        _build_supplementary(marker_dir, canonical_csv, output_dir)
    )

    print("[6/10] Replacing audit templates with the integrated 29-marker scope")
    precision, recall, audit_status = _make_audit_templates(
        marker_dir, output_dir, canonical
    )
    inventory = _write_inventory(marker_dir, output_dir)

    print("[7/10] Rendering availability and indexing integrated figures")
    supplementary_dir = output_dir / "supplementary_images"
    availability_figure = _plot_availability(availability, supplementary_dir)
    supplementary_figures.insert(0, availability_figure.name)
    core_index = pd.read_csv(output_dir / "figure_index.csv")
    core_index["figure"] = "images/" + core_index["figure"].astype(str)
    core_index.insert(0, "figure_group", "core_confirmatory_candidate")
    supplementary_index = pd.DataFrame(
        {
            "figure_group": "gender_and_coverage_supplement",
            "figure": [
                f"supplementary_images/{name}" for name in supplementary_figures
            ],
            "description": "Gender-marker, lexicon, or availability supplement.",
        }
    )
    integrated_index = pd.concat(
        [core_index, supplementary_index], ignore_index=True, sort=False
    )
    integrated_index["png_exists"] = integrated_index["figure"].map(
        lambda value: (output_dir / value).is_file()
    )
    integrated_index["svg_exists"] = integrated_index["figure"].map(
        lambda value: (output_dir / value).with_suffix(".svg").is_file()
    )
    integrated_index.to_csv(
        output_dir / "figure_index_integrated.csv", index=False, encoding="utf-8"
    )

    print("[8/10] Validating integrated row counts and provenance")
    expected_binary_within = (
        len(core.MODEL_ORDER)
        * len(core.SUBSETS)
        * 2
        * len(GENDER_MARKER_IDS)
        * 7
    )
    expected_binary_direct = (
        len(core.MODEL_ORDER)
        * len(core.SUBSETS)
        * len(GENDER_MARKER_IDS)
        * len(core.CONDITIONS)
    )
    integrated_validations = {
        "audited_marker_count": len(AUDITED_MARKER_IDS),
        "gender_marker_count": len(GENDER_MARKER_IDS),
        "density_gender_marker_count": len(density_gender),
        "precision_audit_items": len(precision),
        "recall_audit_items": len(recall),
        "binary_within_rows_all_text": len(binary_all),
        "binary_within_rows_committed": len(binary_committed),
        "binary_direct_rows_all_text": len(direct_binary_all),
        "binary_direct_rows_committed": len(direct_binary_committed),
        "omnibus_rows_all_text": len(omnibus_all),
        "omnibus_rows_committed": len(omnibus_committed),
        "availability_rows": len(availability),
        "inventory_rows": len(inventory),
        "all_subsets_preserved": set(availability["subset"]) == set(core.SUBSETS),
        "binary_within_row_count_valid": len(binary_all)
        == len(binary_committed)
        == expected_binary_within,
        "binary_direct_row_count_valid": len(direct_binary_all)
        == len(direct_binary_committed)
        == expected_binary_direct,
        "audit_scope_valid": set(precision["marker_id"])
        == set(AUDITED_MARKER_IDS),
        "all_integrated_png_exist": bool(integrated_index["png_exists"].all()),
        "all_integrated_svg_exist": bool(integrated_index["svg_exists"].all()),
        **supplementary_validation,
    }
    required = (
        integrated_validations["audited_marker_count"] == 29
        and integrated_validations["gender_marker_count"] == 11
        and integrated_validations["precision_audit_items"]
        == EXPECTED_PRECISION_ITEMS
        and integrated_validations["recall_audit_items"]
        == EXPECTED_RECALL_ITEMS
        and integrated_validations["binary_within_row_count_valid"] is True
        and integrated_validations["binary_direct_row_count_valid"] is True
        and integrated_validations["audit_scope_valid"] is True
        and integrated_validations["all_integrated_png_exist"] is True
        and integrated_validations["all_integrated_svg_exist"] is True
        and integrated_validations["lexicon_term_count"]
        == EXPECTED_LEXICON_TERMS
        and integrated_validations["lexicon_occurrences_match"] is True
        and integrated_validations["lexicon_responses_match"] is True
    )
    if not required:
        raise ValueError(
            f"Integrated automated validation failed: {integrated_validations}"
        )

    print("[9/10] Updating the candidate manifest and final gate")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra_tables = [
        *binary_tables,
        *density_tables,
        "omnibus_five_scenarios_all_text.csv",
        "omnibus_five_scenarios_committed.csv",
        "direct_gender_text_availability.csv",
        "paired_gender_marker_presence_committed_sensitivity.csv",
        *supplementary_tables,
        "marker_inventory_integrated.csv",
        "figure_index_integrated.csv",
    ]
    manifest.update(
        {
            "bundle_version": BUNDLE_VERSION,
            "result_status": "candidate",
            "analysis_builder": str(Path(__file__).resolve()),
            "analysis_builder_sha256": _sha256(Path(__file__).resolve()),
            "core_builder_version": core.BUNDLE_VERSION,
            "audited_marker_ids": list(AUDITED_MARKER_IDS),
            "gender_marker_outcomes": list(GENDER_MARKER_IDS),
            "closed_gender_lexicon_terms": EXPECTED_LEXICON_TERMS,
            "closed_gender_lexicon_role": (
                "descriptive_response_only_not_automatic_persona_uptake"
            ),
            "human_precision_audit_complete": False,
            "human_recall_audit_complete": False,
            "inter_rater_agreement_complete": False,
            "disagreement_adjudication_complete": False,
            "human_precision_audit": {
                "items": len(precision),
                "reviewers_required": 2,
                "required_judgments": 2 * len(precision),
                "completed_judgments": 0,
            },
            "human_recall_audit": {
                "items": len(recall),
                "reviewers_required": 2,
                "required_judgments": 2 * len(recall),
                "completed_judgments": 0,
            },
            "marker_inference_run": True,
            "marker_extraction_rerun": False,
            "omnibus_tests": "Friedman for numeric/density; Cochran Q for binary indicators",
            "gender_presence_tests": "Exact McNemar with paired bootstrap CI and Holm correction",
            "generated_tables": list(
                dict.fromkeys([*manifest["generated_tables"], *extra_tables])
            ),
            "generated_figures": core_index["figure"].tolist(),
            "supplementary_figures": supplementary_index["figure"].tolist(),
            "generated_svg_figures": [
                str(Path(value).with_suffix(".svg"))
                for value in integrated_index["figure"]
            ],
            "automated_validation_complete": True,
            "statistics_candidate_complete": True,
            "integrated_validations": integrated_validations,
        }
    )
    manifest["validations"].update(
        {
            "precision_audit_items_selected": len(precision),
            "precision_audit_required_judgments": 2 * len(precision),
            "precision_audit_completed_judgments": 0,
            "recall_audit_items_selected": len(recall),
            "recall_audit_required_judgments": 2 * len(recall),
            "recall_audit_completed_judgments": 0,
            "human_precision_audit_complete": False,
            "human_recall_audit_complete": False,
            "inter_rater_agreement_complete": False,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "result_status": "candidate",
                "final_result_available": False,
                "reason": (
                    "Independent two-reviewer precision/recall audits for all "
                    "29 inferential markers, inter-rater agreement, and "
                    "disagreement adjudication are incomplete."
                ),
            }
        ]
    ).to_csv(output_dir / "final_result_gate.csv", index=False, encoding="utf-8")

    readme = f"""# ETHICS Gender Linguistic Markers v1.5 integrated candidate

Status: **candidate**.  This directory must not be cited as a final result.

The bundle keeps the v1.4 21-outcome core and adds eleven registered gender
markers, response-only closed-lexicon descriptives, differential visible-text
availability, direct commit rate, and five-scenario omnibus checks.  The four
ETHICS subsets and E/I1/I2/I3 remain separate.  Hidden reasoning and ground
truth are not used by marker extraction.

Human gate:

* precision: 0/{2 * len(precision):,} judgments;
* recall: 0/{2 * len(recall):,} judgments;
* IAA: incomplete;
* adjudication: incomplete.

Reviewers must fill copies of the four audit CSV files in separate directories.
One false positive or recall miss requires a rule revision, full extraction
rerun to a new path, and resampling from zero.  Do not edit the manifest or
final gate manually.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    audit_status.to_csv(
        output_dir / "manual_audit_status.csv", index=False, encoding="utf-8"
    )

    print("[10/10] Integrated candidate bundle complete")
    print(f"[DONE] {output_dir}")
    print(
        f"[AUDIT] precision 0/{2 * len(precision):,} | "
        f"recall 0/{2 * len(recall):,} | final_result_available=False"
    )
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    build_bundle(
        args.marker_dir,
        args.canonical_csv,
        args.output_dir,
        args.bootstrap_resamples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

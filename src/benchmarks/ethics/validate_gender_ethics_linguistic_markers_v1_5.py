"""Validate the integrated ETHICS Gender linguistic-marker v1.5 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.benchmarks.ethics import (
    validate_gender_ethics_linguistic_markers_v1_4 as core_validator,
)
from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4 import (
    SUBSETS,
)
from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_5 import (
    AUDITED_MARKER_IDS,
    BUNDLE_VERSION,
    EXPECTED_LEXICON_TERMS,
    EXPECTED_PRECISION_ITEMS,
    EXPECTED_RECALL_ITEMS,
    GENDER_MARKER_IDS,
    _selected_gender_markers,
)
from src.core.linguistic_markers.statistics import OPENING_CATEGORIES


CORE_NUMERIC_TABLES = core_validator.NUMERIC_TABLES
CORE_OPENING_TABLES = core_validator.OPENING_TABLES
BINARY_TABLES = {
    "paired_gender_marker_presence_vs_neutral_all_visible_text.csv": (2112, True),
    "paired_gender_marker_presence_vs_explicit_all_visible_text.csv": (1584, True),
    "paired_gender_marker_presence_vs_neutral_both_committed.csv": (2112, True),
    "paired_gender_marker_presence_vs_explicit_both_committed.csv": (1584, True),
    "paired_gender_marker_presence_direct_man_woman_all_text.csv": (1056, False),
    "paired_gender_marker_presence_direct_man_woman_committed.csv": (1056, False),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def _check_probability(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(
        values.notna().all() and values.between(0, 1, inclusive="both").all()
    )


def _validate_binary(
    path: Path,
    expected_rows: int,
    within_identity: bool,
) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    keys = ["model_id", "benchmark", "subset"]
    if within_identity:
        keys.append("identity")
    keys.extend(("outcome_id", "contrast_id"))
    supported = frame["n_pairs"].gt(0)
    prevalence = ["prevalence_left", "prevalence_right"]
    return {
        "rows": len(frame),
        "row_count_ok": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(keys).any(),
        "candidate_only": frame["result_status"].eq("candidate").all(),
        "subsets_separate": set(frame["subset"]) == set(SUBSETS),
        "outcomes_complete": set(frame["outcome_id"]) == set(GENDER_MARKER_IDS),
        "presence_scale_only": frame["outcome_scale"].eq("presence").all(),
        "p_raw_valid": _check_probability(frame, "p_raw"),
        "p_holm_valid": _check_probability(frame, "p_holm"),
        "holm_not_below_raw": frame["p_holm"].ge(frame["p_raw"] - 1e-15).all(),
        "prevalence_valid": frame[prevalence].apply(
            pd.to_numeric, errors="coerce"
        ).stack().between(0, 1, inclusive="both").all(),
        "ci_complete_when_supported": frame.loc[
            supported, ["prevalence_delta", "ci_low", "ci_high"]
        ].notna().all().all(),
        "ci_contains_point": (
            frame.loc[supported, "ci_low"].le(
                frame.loc[supported, "prevalence_delta"]
            )
            & frame.loc[supported, "ci_high"].ge(
                frame.loc[supported, "prevalence_delta"]
            )
        ).all(),
        "discordance_not_above_pairs": frame["discordant"].le(
            frame["n_pairs"]
        ).all(),
    }


def _validate_density(path: Path, expected_rows: int, within_identity: bool) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    density_ids = {
        marker.marker_id
        for marker in _selected_gender_markers()
        if marker.measurement == "density"
    }
    keys = ["model_id", "benchmark", "subset"]
    if within_identity:
        keys.append("identity")
    keys.extend(("outcome_id", "contrast_id"))
    raw_columns = [
        "n_raw_count_pairs",
        "raw_count_sum_left",
        "raw_count_sum_right",
        "median_raw_count_left",
        "median_raw_count_right",
        "median_paired_raw_count_delta",
    ]
    supported = frame["n_pairs"].gt(0)
    return {
        "rows": len(frame),
        "row_count_ok": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(keys).any(),
        "candidate_only": frame["result_status"].eq("candidate").all(),
        "subsets_separate": set(frame["subset"]) == set(SUBSETS),
        "density_outcomes_complete": set(frame["outcome_id"]) == density_ids,
        "p_raw_valid": _check_probability(frame, "p_raw"),
        "p_holm_valid": _check_probability(frame, "p_holm"),
        "raw_pair_support_matches": frame["n_raw_count_pairs"].eq(
            frame["n_pairs"]
        ).all(),
        "raw_counts_complete": frame.loc[supported, raw_columns].notna().all().all(),
        "raw_counts_nonnegative": frame[
            ["raw_count_sum_left", "raw_count_sum_right"]
        ].ge(0).all().all(),
    }


def _collect_failures(sections: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for section_name, section in sections.items():
        if section_name in {"numeric", "opening", "binary", "density"}:
            for filename, checks in section.items():
                failures.extend(
                    f"{section_name}.{filename}.{name}"
                    for name, value in checks.items()
                    if name != "rows" and not bool(value)
                )
        else:
            failures.extend(
                f"{section_name}.{name}"
                for name, value in section.items()
                if not name.startswith("rows_") and not bool(value)
            )
    return failures


def validate_bundle(bundle_dir: Path) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    generated = [
        *manifest.get("generated_tables", []),
        *manifest.get("generated_figures", []),
        *manifest.get("supplementary_figures", []),
        *manifest.get("generated_svg_figures", []),
    ]
    missing_generated = [name for name in generated if not (bundle_dir / name).is_file()]
    numeric = {
        name: core_validator._validate_numeric(
            bundle_dir / name, expected_rows, within_identity
        )
        for name, (expected_rows, within_identity) in CORE_NUMERIC_TABLES.items()
    }
    opening = {
        name: core_validator._validate_opening(
            bundle_dir / name, expected_rows, within_identity
        )
        for name, (expected_rows, within_identity) in CORE_OPENING_TABLES.items()
    }
    binary = {
        name: _validate_binary(bundle_dir / name, expected_rows, within_identity)
        for name, (expected_rows, within_identity) in BINARY_TABLES.items()
    }
    density_specs = {
        "paired_gender_marker_density_vs_neutral_all_visible_text.csv": (768, True),
        "paired_gender_marker_density_vs_explicit_all_visible_text.csv": (576, True),
        "paired_gender_marker_density_vs_neutral_both_committed.csv": (768, True),
        "paired_gender_marker_density_vs_explicit_both_committed.csv": (576, True),
        "paired_gender_marker_density_direct_man_woman_all_text.csv": (384, False),
        "paired_gender_marker_density_direct_man_woman_committed.csv": (384, False),
    }
    density = {
        name: _validate_density(bundle_dir / name, expected_rows, within_identity)
        for name, (expected_rows, within_identity) in density_specs.items()
    }

    precision = core_validator._validate_reviewer_pair(
        bundle_dir, "precision", EXPECTED_PRECISION_ITEMS
    )
    recall = core_validator._validate_reviewer_pair(
        bundle_dir, "recall", EXPECTED_RECALL_ITEMS
    )
    availability = pd.read_csv(bundle_dir / "direct_gender_text_availability.csv")
    omnibus_all = pd.read_csv(bundle_dir / "omnibus_five_scenarios_all_text.csv")
    omnibus_committed = pd.read_csv(
        bundle_dir / "omnibus_five_scenarios_committed.csv"
    )
    inventory = pd.read_csv(bundle_dir / "marker_inventory_integrated.csv")
    lexicon = pd.read_csv(
        bundle_dir / "gender_term_surface_reconstruction_validation.csv"
    )
    direct_terms = pd.read_csv(
        bundle_dir / "gender_term_direct_man_woman_response_only.csv"
    )
    response_outcomes = pd.read_csv(
        bundle_dir / "response_outcomes_direct_man_woman_by_subset.csv"
    )
    figure_index = pd.read_csv(bundle_dir / "figure_index_integrated.csv")
    gate = pd.read_csv(bundle_dir / "final_result_gate.csv")

    expected_omnibus_outcomes = 32 + len(OPENING_CATEGORIES)
    expected_omnibus_rows = 6 * 4 * 2 * expected_omnibus_outcomes
    top_level = {
        "bundle_version_ok": manifest.get("bundle_version") == BUNDLE_VERSION,
        "candidate_status": manifest.get("result_status") == "candidate",
        "automated_validation_complete": manifest.get(
            "automated_validation_complete"
        )
        is True,
        "hidden_reasoning_not_used": manifest.get("hidden_reasoning_used") is False,
        "marker_extraction_not_rerun": manifest.get("marker_extraction_rerun") is False,
        "four_subsets": manifest.get("subsets") == list(SUBSETS),
        "twenty_nine_audited_markers": manifest.get("audited_marker_ids")
        == list(AUDITED_MARKER_IDS),
        "eleven_gender_markers": manifest.get("gender_marker_outcomes")
        == list(GENDER_MARKER_IDS),
        "generated_files_complete": not missing_generated,
        "integrated_figures_unique": not figure_index["figure"].duplicated().any(),
        "integrated_png_complete": figure_index["png_exists"].all(),
        "integrated_svg_complete": figure_index["svg_exists"].all(),
        "final_gate_closed": len(gate) == 1
        and not bool(gate.loc[0, "final_result_available"]),
        "two_reviewer_precision_required": manifest["human_precision_audit"][
            "reviewers_required"
        ]
        == 2,
        "two_reviewer_recall_required": manifest["human_recall_audit"][
            "reviewers_required"
        ]
        == 2,
        "inter_rater_agreement_pending": manifest.get(
            "inter_rater_agreement_complete"
        )
        is False,
        "availability_rows": len(availability) == 192,
        "availability_outcomes": set(availability["outcome_id"])
        == {"text_available", "technical_only"},
        "availability_subsets_separate": set(availability["subset"]) == set(SUBSETS),
        "omnibus_all_rows": len(omnibus_all) == expected_omnibus_rows,
        "omnibus_committed_rows": len(omnibus_committed)
        == expected_omnibus_rows,
        "omnibus_tests_valid": set(omnibus_all["test_used"])
        == {"friedman", "cochran_q"},
        "integrated_inventory_rows": len(inventory) == 32,
        "integrated_inventory_unique": not inventory["marker_id"].duplicated().any(),
        "closed_lexicon_terms": len(lexicon) == EXPECTED_LEXICON_TERMS,
        "closed_lexicon_reconstruction": lexicon[
            ["occurrences_match", "responses_match"]
        ].all().all(),
        "direct_lexicon_rows": len(direct_terms) == 11_400,
        "response_outcome_rows": len(response_outcomes) == 96,
    }
    sections = {
        "top_level": top_level,
        "numeric": numeric,
        "opening": opening,
        "binary": binary,
        "density": density,
        "precision_audit": precision,
        "recall_audit": recall,
    }
    failures = _collect_failures(sections)
    return {
        "bundle_dir": str(bundle_dir),
        "valid": not failures,
        "failures": failures,
        **sections,
    }


def main() -> int:
    args = _parser().parse_args()
    result = validate_bundle(args.bundle_dir)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only validator for an ETHICS Regional/Cultural marker bundle v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd
from PIL import Image

from src.benchmarks.ethics.regional_cultural_ethics_linguistic_markers_v2 import (
    BUNDLE_VERSION,
    EXPECTED_INPUT_SHA256,
    expected_table_rows,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _notebook_error_count(path: Path) -> int:
    notebook = nbformat.read(path, as_version=4)
    return sum(
        1
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    )


def validate(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    final_manifest_path = output_dir / "manifest_final.json"
    manifest_path = (
        final_manifest_path
        if final_manifest_path.is_file()
        else output_dir / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_index = _read_csv(output_dir / "table_index.csv")
    figure_index = _read_csv(output_dir / "figure_index.csv")
    expected_rows = expected_table_rows()

    table_paths = {
        row.table_id: output_dir / row.relative_path
        for row in table_index.itertuples(index=False)
    }
    scoring = _read_csv(table_paths["scoring_summary"])
    dictionary = _read_csv(table_paths["outcome_dictionary"])
    within = _read_csv(table_paths["paired_within"])
    direct = _read_csv(table_paths["paired_direct"])
    omnibus = _read_csv(table_paths["omnibus"])
    openings = _read_csv(table_paths["opening_pairs"])
    descriptives = _read_csv(table_paths["descriptives"])
    availability = _read_csv(table_paths["pair_availability_audit"])
    gate = _read_csv(table_paths["human_validation_gate"])
    parquet_path = (
        output_dir
        / "dataset"
        / "ethics_regional_cultural_marker_analysis_response_level.parquet"
    )
    parquet = pd.read_parquet(
        parquet_path,
        columns=[
            "model_id",
            "persona",
            "question_id",
            "subset",
            "analysis_answer_status",
            "text_available",
            "candidate_uncommitted",
        ],
    )

    generated_hashes = manifest.get("generated_file_sha256", {})
    hash_mismatches = []
    missing_hashed_files = []
    for relative, expected_hash in generated_hashes.items():
        path = output_dir / relative
        if not path.is_file():
            missing_hashed_files.append(relative)
        elif sha256_file(path) != expected_hash:
            hash_mismatches.append(relative)

    png_validation = []
    for row in figure_index.itertuples(index=False):
        png = output_dir / "figures" / f"{row.figure_id}.png"
        pdf = output_dir / "figures" / f"{row.figure_id}.pdf"
        svg = output_dir / "figures" / f"{row.figure_id}.svg"
        with Image.open(png) as image:
            expected_width = round(float(row.width_in) * int(row.png_dpi))
            expected_height = round(float(row.height_in) * int(row.png_dpi))
            png_validation.append(
                abs(image.width - expected_width) <= 2
                and abs(image.height - expected_height) <= 2
                and pdf.is_file()
                and svg.is_file()
            )

    executed_notebooks = sorted((output_dir / "notebooks").glob("*_executed.ipynb"))
    source_notebooks = [
        path
        for path in (output_dir / "notebooks").glob("*.ipynb")
        if not path.name.endswith("_executed.ipynb")
    ]
    html_notebooks = sorted((output_dir / "notebooks").glob("*_executed.html"))
    notebook_errors = sum(_notebook_error_count(path) for path in executed_notebooks)

    observed_rows = {
        "scoring_summary": len(scoring),
        "descriptives": len(descriptives),
        "within_pairs": len(within),
        "direct_pairs": len(direct),
        "omnibus": len(omnibus),
        "opening_pairs": len(openings),
    }
    marker_within = within[
        within["registered_measurement"].ne("surface") & within["n_pairs"].gt(0)
    ]
    marker_direct = direct[
        direct["registered_measurement"].ne("surface") & direct["n_pairs"].gt(0)
    ]
    zero_pairs = pd.concat(
        [
            within.loc[within["n_pairs"].eq(0), ["model_id", "outcome_id"]],
            direct.loc[direct["n_pairs"].eq(0), ["model_id", "outcome_id"]],
        ],
        ignore_index=True,
    )
    positive_within = within["n_pairs"].gt(0)
    positive_direct = direct["n_pairs"].gt(0)
    expected_availability = (
        availability["is_expected"]
        .astype(str)
        .str.casefold()
        .eq("true")
    )
    all_visible_zero = availability[availability["population"].eq("all_visible")]
    restricted_non_ttr_zero = availability[
        availability["population"].ne("all_visible")
        & availability["outcome_id"].ne("ttr_300")
    ]
    checks = {
        "bundle_version": manifest.get("bundle_version") == BUNDLE_VERSION,
        "candidate_status_explicit": str(manifest.get("result_status", "")).startswith(
            "candidate_blocked"
        ),
        "canonical_hash_pinned": manifest.get("canonical_sha256")
        == EXPECTED_INPUT_SHA256,
        "generated_hashes_present": bool(generated_hashes),
        "generated_hashes_match": not hash_mismatches and not missing_hashed_files,
        "table_index_paths_exist": all(path.is_file() for path in table_paths.values()),
        "row_contract": observed_rows == expected_rows,
        "response_rows": len(parquet) == 154_428,
        "response_keys_unique": not parquet.duplicated(
            ["model_id", "persona", "question_id"]
        ).any(),
        "response_subsets_four": parquet["subset"].nunique() == 4,
        "candidate_semantic_demoted_4332": int(parquet["candidate_uncommitted"].sum())
        == 4_332,
        "outcome_hierarchy": dictionary["outcome_tier"].value_counts().to_dict()
        == {
            "core_confirmatory": 21,
            "regional_cultural_specific": 23,
            "exploratory_general": 12,
        },
        "scoring_required_columns": {
            "total",
            "committed",
            "commit_rate",
            "abstain",
            "abstain_rate",
            "missing",
            "missing_rate",
            "accuracy_committed",
        }.issubset(scoring.columns),
        "accuracy_only_construct_valid": scoring.loc[
            ~scoring["subset"].isin(["commonsense", "justice"]),
            "accuracy_committed",
        ].isna().all(),
        "zero_pair_rows_212_all_classified_expected": len(zero_pairs) == 212
        and len(availability) == len(zero_pairs)
        and expected_availability.all()
        and availability["availability_reason"]
        .ne("unexpected_outcome_specific_zero_pairs")
        .all(),
        "all_visible_zero_rows_27_gpt_oss_ttr_only": len(all_visible_zero) == 27
        and all_visible_zero["model_id"].eq("openai_gpt-oss-20b").all()
        and all_visible_zero["outcome_id"].eq("ttr_300").all()
        and all_visible_zero["availability_reason"]
        .eq("outcome_surface_unavailable_ttr_300")
        .all(),
        "restricted_non_ttr_zero_rows_100_are_empty_population_groups": len(
            restricted_non_ttr_zero
        )
        == 100
        and restricted_non_ttr_zero["availability_reason"]
        .eq("population_filter_empty_for_comparison")
        .all(),
        "pair_ci_na_exactly_when_no_pair": within.loc[
            positive_within, ["ci_low", "ci_high"]
        ].notna().all().all()
        and direct.loc[
            positive_direct, ["ci_low", "ci_high"]
        ].notna().all().all()
        and within.loc[
            ~positive_within, ["ci_low", "ci_high"]
        ].isna().all().all()
        and direct.loc[
            ~positive_direct, ["ci_low", "ci_high"]
        ].isna().all().all(),
        "holm_monotonic": (within["p_holm"] + 1e-12 >= within["p_raw"]).all()
        and (direct["p_holm"] + 1e-12 >= direct["p_raw"]).all(),
        "raw_marker_counts_present_when_pairs_exist": marker_within[
            ["raw_count_sum_left", "raw_count_sum_right"]
        ].notna().all().all()
        and marker_direct[
            ["raw_count_sum_left", "raw_count_sum_right"]
        ].notna().all().all(),
        "all_subsets_separate": within["subset"].nunique() == 4
        and direct["subset"].nunique() == 4,
        "figure_count_at_least_20": len(figure_index) >= 20,
        "figure_fonts_readable": figure_index[
            "minimum_declared_font_pt"
        ].ge(9.5).all(),
        "figure_dimensions_thesis_native": figure_index["width_in"].le(9.3).all(),
        "figure_triplicates_and_300dpi": all(png_validation),
        "source_notebooks_two": len(source_notebooks) == 2,
        "executed_notebooks_two": len(executed_notebooks) == 2,
        "html_notebooks_two": len(html_notebooks) == 2,
        "notebook_errors_zero": notebook_errors == 0,
        "human_release_blocked": gate.loc[
            gate["gate_id"].eq("thesis_final_release"), "status"
        ].eq("blocked").all(),
        "manifest_human_validation_false": manifest.get("human_validation_complete")
        is False,
        "hidden_reasoning_not_used": manifest.get("hidden_reasoning_used") is False,
        "paid_api_not_used": manifest.get("paid_api_calls") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(
            f"Bundle validation failed: {failed}; hash mismatches={hash_mismatches}; "
            f"missing hashed files={missing_hashed_files}; observed rows={observed_rows}; "
            f"expected rows={expected_rows}"
        )
    return {
        "validation_status": "automated_pass_candidate_human_gate_still_blocked",
        "output_dir": str(output_dir),
        "checks": checks,
        "observed_rows": observed_rows,
        "expected_rows": expected_rows,
        "figures": len(figure_index),
        "source_notebooks": len(source_notebooks),
        "executed_notebooks": len(executed_notebooks),
        "html_notebooks": len(html_notebooks),
        "notebook_errors": notebook_errors,
    }


def main() -> int:
    args = parse_args()
    result = validate(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

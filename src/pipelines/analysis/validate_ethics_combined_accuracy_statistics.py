"""Validate pooled ETHICS statistics against source per-subset statistics.

The pooled analysis has no pre-existing source table.  To validate the
implementation, the same computation is run one subset at a time and compared
cell-for-cell with the publication bundle's established results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ethics_accuracy_combined_statistics import compute_combined_conditional_accuracy
import render_ethics_gender_overall_standardized as gender
import render_ethics_regional_overall_standardized as regional
import render_ethics_religion_overall_standardized as religion


TOLERANCE = 1e-12


def _compare(
    calculated: pd.DataFrame,
    reference: pd.DataFrame,
    mapping: dict[str, str],
    label: str,
) -> dict[str, float]:
    merged = calculated.merge(
        reference,
        on=["model", "comparison_short"],
        suffixes=("_calculated", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(calculated) or len(merged) != len(reference):
        raise AssertionError(f"Incomplete validation merge: {label}")
    differences: dict[str, float] = {}
    for calculated_name, reference_name in mapping.items():
        calculated_column = (
            f"{calculated_name}_calculated"
            if f"{calculated_name}_calculated" in merged.columns
            else calculated_name
        )
        reference_column = (
            f"{reference_name}_reference"
            if f"{reference_name}_reference" in merged.columns
            else reference_name
        )
        difference = np.abs(
            merged[calculated_column].astype(float)
            - merged[reference_column].astype(float)
        )
        differences[calculated_name] = float(difference.max())
    if max(differences.values(), default=0.0) > TOLERANCE:
        raise AssertionError(f"Statistical mismatch {label}: {differences}")
    return differences


def validate_gender() -> list[dict[str, object]]:
    tables = gender._load_source_tables(gender.SOURCE_BUNDLE)
    reference = gender._gender_accuracy_inference(tables["paired_effects"])
    templates = (
        reference.sort_values(["display_order", "model"], kind="stable")
        .drop_duplicates(["model", "comparison_short"])
        .copy()
    )
    templates["holm_group"] = templates["family"]
    raw_path = Path(tables["manifest"]["input"])
    results = []
    for subset in gender.VALID_SUBSETS:
        calculated = compute_combined_conditional_accuracy(
            raw_path,
            templates,
            valid_subsets=[subset],
            persona_a_column="persona_a",
            persona_b_column="persona_b",
            holm_group_column="holm_group",
            eligibility="both_committed",
        )
        selected_reference = reference[reference["subset"].eq(subset)]
        differences = _compare(
            calculated,
            selected_reference,
            {
                "risk_difference_pp": "risk_difference_pp",
                "effect_size": "effect_size",
                "exact_mcnemar_p_value": "exact_mcnemar_p_value",
                "holm_adjusted_p_value": "holm_adjusted_p_value",
            },
            f"Gender/{subset}",
        )
        results.append({"domain": "Gender", "subset": subset, **differences})
    return results


def _generic_templates(engine: object, reference: pd.DataFrame) -> pd.DataFrame:
    templates = (
        reference.sort_values(["display_order", "model"], kind="stable")
        .drop_duplicates(["model", "comparison_short"])
        .copy()
    )
    templates["persona_a"] = templates["condition_a"]
    templates["persona_b"] = templates["condition_b"]
    templates["holm_group"] = np.where(
        templates["comparison_family"].eq(engine.SCENARIO_FAMILY),
        templates["comparison_family"] + ":" + templates["identity"].astype(str),
        templates["comparison_family"],
    )
    return templates


def validate_generic(engine: object, label: str) -> list[dict[str, object]]:
    tables = engine._load_source_tables(engine.SOURCE_BUNDLE)
    parent = engine._build_parent_tables(tables)
    reference = engine._accuracy_inference(parent["paired_all"])
    templates = _generic_templates(engine, reference)
    raw_path = engine.SOURCE_BUNDLE.parent / "all_models_results_full.csv"
    results = []
    for subset in engine.VALID_SUBSETS:
        calculated = compute_combined_conditional_accuracy(
            raw_path,
            templates,
            valid_subsets=[subset],
            persona_a_column="persona_a",
            persona_b_column="persona_b",
            holm_group_column="holm_group",
            eligibility="full_denominator",
        )
        selected_reference = reference[reference["subset"].eq(subset)]
        differences = _compare(
            calculated,
            selected_reference,
            {
                "risk_difference_pp": "delta_accuracy_pp",
                "effect_size": "effect_size",
                "exact_mcnemar_p_value": "exact_mcnemar_p_value",
                "holm_adjusted_p_value": "holm_adjusted_p_value",
            },
            f"{label}/{subset}",
        )
        results.append({"domain": label, "subset": subset, **differences})
    return results


def main() -> int:
    rows = validate_gender()
    rows.extend(validate_generic(regional, "Regional-Kultural"))
    religion.configure_engine()
    rows.extend(validate_generic(regional, "Religion"))
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("VALIDATION PASSED: pooled implementation reproduces every source per-subset cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

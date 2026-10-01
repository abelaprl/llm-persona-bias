"""Validate a candidate ETHICS Gender linguistic-marker v1.4 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4 import (
    MARKER_IDS,
    OUTCOME_IDS,
    SUBSETS,
)


NUMERIC_TABLES = {
    "paired_numeric_vs_neutral_all_text.csv": (4032, True),
    "paired_numeric_vs_neutral_committed.csv": (4032, True),
    "paired_numeric_vs_explicit_all_text.csv": (3024, True),
    "paired_numeric_vs_explicit_committed.csv": (3024, True),
    "paired_numeric_direct_man_woman_all_text.csv": (2016, False),
    "paired_numeric_direct_man_woman_committed.csv": (2016, False),
}
OPENING_TABLES = {
    "paired_opening_within_identity_all_text.csv": (336, True),
    "paired_opening_within_identity_committed.csv": (336, True),
    "paired_opening_direct_man_woman_all_text.csv": (96, False),
    "paired_opening_direct_man_woman_committed.csv": (96, False),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def _check_probability(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and values.between(0, 1, inclusive="both").all())


def _validate_numeric(path: Path, expected_rows: int, within_identity: bool) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    keys = ["model_id", "benchmark", "subset"]
    if within_identity:
        keys.append("identity")
    keys.extend(("outcome_id", "contrast_id"))
    supported = frame["n_pairs"].gt(0)
    ci_complete = frame.loc[supported, ["median_paired_delta", "ci_low", "ci_high"]].notna().all().all()
    ci_contains_point = (
        frame.loc[supported, "ci_low"].le(frame.loc[supported, "median_paired_delta"])
        & frame.loc[supported, "ci_high"].ge(frame.loc[supported, "median_paired_delta"])
    ).all()
    marker_rows = frame["outcome_id"].isin(MARKER_IDS)
    raw_supported = marker_rows & frame["n_raw_count_pairs"].gt(0)
    raw_columns = [
        "n_raw_count_pairs",
        "raw_count_sum_left",
        "raw_count_sum_right",
        "median_raw_count_left",
        "median_raw_count_right",
        "median_paired_raw_count_delta",
    ]
    return {
        "rows": len(frame),
        "row_count_ok": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(keys).any(),
        "candidate_only": frame["result_status"].eq("candidate").all(),
        "subsets_separate": set(frame["subset"]) == set(SUBSETS),
        "outcomes_complete": set(frame["outcome_id"]) == set(OUTCOME_IDS),
        "p_raw_valid": _check_probability(frame, "p_raw"),
        "p_holm_valid": _check_probability(frame, "p_holm"),
        "holm_not_below_raw": frame["p_holm"].ge(frame["p_raw"] - 1e-15).all(),
        "pair_support_valid": frame["n_pairs"].ge(frame["n_nonzero"]).all(),
        "ci_complete_when_supported": bool(ci_complete),
        "ci_contains_point": bool(ci_contains_point),
        "raw_pair_support_matches_density": frame.loc[marker_rows, "n_raw_count_pairs"].eq(
            frame.loc[marker_rows, "n_pairs"]
        ).all(),
        "raw_counts_complete_when_supported": frame.loc[raw_supported, raw_columns].notna().all().all(),
        "raw_counts_nonnegative": frame.loc[
            marker_rows, ["raw_count_sum_left", "raw_count_sum_right"]
        ].ge(0).all().all(),
    }


def _validate_opening(path: Path, expected_rows: int, within_identity: bool) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    keys = ["model_id", "benchmark", "subset"]
    if within_identity:
        keys.append("identity")
    keys.append("contrast_id")
    return {
        "rows": len(frame),
        "row_count_ok": len(frame) == expected_rows,
        "unique_keys": not frame.duplicated(keys).any(),
        "candidate_only": frame["result_status"].eq("candidate").all(),
        "subsets_separate": set(frame["subset"]) == set(SUBSETS),
        "p_raw_valid": _check_probability(frame, "p_raw"),
        "p_holm_valid": _check_probability(frame, "p_holm"),
        "pair_counts_present": frame["n_pairs"].notna().all(),
    }


def _validate_reviewer_pair(bundle_dir: Path, audit_type: str, expected_items: int) -> dict[str, object]:
    first = pd.read_csv(bundle_dir / f"human_{audit_type}_audit_reviewer_1.csv", low_memory=False)
    second = pd.read_csv(bundle_dir / f"human_{audit_type}_audit_reviewer_2.csv", low_memory=False)
    forbidden = {"ground_truth", "gt_stance", "is_correct", "ground_truth_label"}
    first_ids = first["audit_item_id"].astype(str)
    second_ids = second["audit_item_id"].astype(str)
    checks = {
        "rows_reviewer_1": len(first),
        "rows_reviewer_2": len(second),
        "row_counts_ok": len(first) == len(second) == expected_items,
        "item_ids_unique": not first_ids.duplicated().any() and not second_ids.duplicated().any(),
        "item_ids_identical": first_ids.tolist() == second_ids.tolist(),
        "reviewer_slots_correct": first["reviewer_slot"].eq(1).all() and second["reviewer_slot"].eq(2).all(),
        "reviewer_ids_blank": first["reviewer_id"].isna().all() and second["reviewer_id"].isna().all(),
        "labels_blinded": forbidden.isdisjoint(first.columns) and forbidden.isdisjoint(second.columns),
        "completion_blank": first["human_review_complete"].isna().all()
        and second["human_review_complete"].isna().all(),
    }
    if audit_type == "precision":
        checks["full_visible_response_present"] = (
            first["visible_response_full"].fillna("").astype(str).str.strip().ne("").all()
            and second["visible_response_full"].fillna("").astype(str).str.strip().ne("").all()
        )
    else:
        checks["full_visible_response_present"] = (
            first["visible_text"].fillna("").astype(str).str.strip().ne("").all()
            and second["visible_text"].fillna("").astype(str).str.strip().ne("").all()
        )
    return checks


def validate_bundle(bundle_dir: Path) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_generated = [
        name for name in [*manifest["generated_tables"], *[f"images/{name}" for name in manifest["generated_figures"]]]
        if not (bundle_dir / name).is_file()
    ]
    numeric = {
        name: _validate_numeric(bundle_dir / name, expected_rows, within_identity)
        for name, (expected_rows, within_identity) in NUMERIC_TABLES.items()
    }
    opening = {
        name: _validate_opening(bundle_dir / name, expected_rows, within_identity)
        for name, (expected_rows, within_identity) in OPENING_TABLES.items()
    }
    precision = _validate_reviewer_pair(bundle_dir, "precision", 1689)
    recall = _validate_reviewer_pair(bundle_dir, "recall", 162)
    figure_index = pd.read_csv(bundle_dir / "figure_index.csv")
    gate = pd.read_csv(bundle_dir / "final_result_gate.csv")
    top_level = {
        "bundle_version_ok": manifest.get("bundle_version")
        == "gender-ethics-linguistic-markers-v1.4.0-candidate",
        "candidate_status": manifest.get("result_status") == "candidate",
        "automated_validation_complete": manifest.get("automated_validation_complete") is True,
        "hidden_reasoning_not_used": manifest.get("hidden_reasoning_used") is False,
        "four_subsets": manifest.get("subsets") == list(SUBSETS),
        "twenty_one_outcomes": manifest.get("numeric_outcomes") == list(OUTCOME_IDS),
        "generated_files_complete": not missing_generated,
        "sixteen_figures": len(figure_index) == 16 and not figure_index["figure"].duplicated().any(),
        "final_gate_closed": len(gate) == 1 and not bool(gate.loc[0, "final_result_available"]),
        "two_reviewer_precision_required": manifest["human_precision_audit"]["reviewers_required"] == 2,
        "two_reviewer_recall_required": manifest["human_recall_audit"]["reviewers_required"] == 2,
        "inter_rater_agreement_pending": manifest.get("inter_rater_agreement_complete") is False,
    }
    sections = {"top_level": top_level, "numeric": numeric, "opening": opening, "precision_audit": precision, "recall_audit": recall}
    failures: list[str] = []
    for section_name, section in sections.items():
        if section_name in {"numeric", "opening"}:
            for file_name, checks in section.items():
                failures.extend(
                    f"{section_name}.{file_name}.{name}"
                    for name, value in checks.items()
                    if name != "rows" and not bool(value)
                )
        else:
            failures.extend(
                f"{section_name}.{name}"
                for name, value in section.items()
                if not name.startswith("rows_") and not bool(value)
            )
    return {"bundle_dir": str(bundle_dir), "valid": not failures, "failures": failures, **sections}


def main() -> int:
    args = _parser().parse_args()
    result = validate_bundle(args.bundle_dir)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Finalize two-reviewer audits for the ETHICS Gender LM v1.4 bundle.

This module never edits the candidate analysis bundle or reviewer files. It
writes a separate, hash-pinned audit-finalization bundle and keeps the final
gate closed when any rule has a false positive, a recall miss, or an
unadjudicated reviewer disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4 import (
    MARKER_IDS,
)


TRUE_VALUES = {"1", "true"}
FALSE_VALUES = {"0", "false"}
EMPTY_MARKER_VALUES = {"", "none", "no_miss", "[]"}
MARKER_SPLIT_RE = re.compile(r"\s*[,;|]\s*")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--precision-reviewer-1", type=Path, required=True)
    parser.add_argument("--precision-reviewer-2", type=Path, required=True)
    parser.add_argument("--recall-reviewer-1", type=Path, required=True)
    parser.add_argument("--recall-reviewer-2", type=Path, required=True)
    parser.add_argument("--adjudication-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: object, *, field: str) -> bool:
    if pd.isna(value):
        raise ValueError(f"{field} cannot be blank.")
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{field} must be one of 1/0/true/false, got {value!r}.")


def _parse_marker_set(value: object, *, field: str) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    normalized = str(value).strip()
    if normalized.lower() in EMPTY_MARKER_VALUES:
        return frozenset()
    marker_ids = frozenset(item for item in MARKER_SPLIT_RE.split(normalized) if item)
    unknown = sorted(marker_ids.difference(MARKER_IDS))
    if unknown:
        raise ValueError(f"{field} contains unknown marker IDs: {unknown}.")
    return marker_ids


def _serialize_marker_set(marker_ids: Iterable[str]) -> str:
    return ";".join(sorted(marker_ids))


def _reviewer_id(frame: pd.DataFrame, *, filename: Path) -> str:
    values = frame["reviewer_id"].fillna("").astype(str).str.strip()
    unique = sorted(value for value in values.unique() if value)
    if len(unique) != 1 or values.eq("").any():
        raise ValueError(f"{filename} must contain exactly one nonblank reviewer_id.")
    return unique[0]


def _load_review(path: Path, *, audit_type: str, reviewer_slot: int) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    common = {"audit_item_id", "human_review_complete", "reviewer_slot", "reviewer_id"}
    specific = (
        {"marker_id", "is_true_positive", "audit_note"}
        if audit_type == "precision"
        else {"manual_marker_ids", "missed_span_notes"}
    )
    missing = sorted(common.union(specific).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    if (
        frame.empty
        or frame["audit_item_id"].astype(str).str.strip().eq("").any()
        or frame["audit_item_id"].duplicated().any()
    ):
        raise ValueError(f"{path} must contain unique, nonempty audit items.")
    if not frame["reviewer_slot"].astype(str).str.strip().eq(str(reviewer_slot)).all():
        raise ValueError(f"{path} does not consistently use reviewer_slot={reviewer_slot}.")
    complete = frame["human_review_complete"].map(
        lambda value: _parse_bool(value, field="human_review_complete")
    )
    if not complete.all():
        raise ValueError(f"{path} still contains incomplete human-review rows.")
    reviewer_id = _reviewer_id(frame, filename=path)
    frame.attrs["reviewer_id"] = reviewer_id

    if audit_type == "precision":
        frame["review_value"] = frame["is_true_positive"].map(
            lambda value: _parse_bool(value, field="is_true_positive")
        )
        missing_notes = (~frame["review_value"]) & frame["audit_note"].astype(str).str.strip().eq("")
        if missing_notes.any():
            raise ValueError(f"{path} has false-positive judgments without audit_note.")
    else:
        frame["review_value"] = frame["manual_marker_ids"].map(
            lambda value: _parse_marker_set(value, field="manual_marker_ids")
        )
        missing_notes = frame["review_value"].map(bool) & frame["missed_span_notes"].astype(str).str.strip().eq("")
        if missing_notes.any():
            raise ValueError(f"{path} has recall misses without missed_span_notes.")
    return frame


def _validate_against_candidate_template(
    review: pd.DataFrame,
    template: pd.DataFrame,
    *,
    audit_type: str,
    reviewer_slot: int,
) -> None:
    editable = {"human_review_complete", "reviewer_id", "audit_note"}
    if audit_type == "precision":
        editable.add("is_true_positive")
    else:
        editable.update({"manual_marker_ids", "missed_span_notes"})
    missing = sorted(set(template.columns).difference(review.columns))
    if missing:
        raise ValueError(f"Reviewer {reviewer_slot} file dropped template columns: {missing}.")
    if template.empty or template["audit_item_id"].duplicated().any():
        raise ValueError("Candidate audit template is empty or has duplicate audit_item_id values.")
    for column in editable.intersection(template.columns):
        if template[column].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(
                "Candidate audit templates must remain blank; use filled copies in a new path."
            )
    if set(review["audit_item_id"]) != set(template["audit_item_id"]):
        raise ValueError(f"Reviewer {reviewer_slot} item IDs differ from the candidate template.")
    aligned = review.set_index("audit_item_id").loc[template["audit_item_id"]].reset_index()
    immutable = [column for column in template.columns if column not in editable]
    for column in immutable:
        if not aligned[column].fillna("").astype(str).reset_index(drop=True).equals(
            template[column].fillna("").astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"Reviewer {reviewer_slot} changed immutable template column {column!r}."
            )


def _paired_reviews(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    audit_type: str,
) -> pd.DataFrame:
    first_ids = set(first["audit_item_id"])
    second_ids = set(second["audit_item_id"])
    if first_ids != second_ids:
        raise ValueError(f"Reviewer item sets differ for {audit_type} audit.")
    second_aligned = second.set_index("audit_item_id").loc[first["audit_item_id"]].reset_index()
    identity_columns = ["model_id", "persona", "question_id"]
    if audit_type == "precision":
        identity_columns.extend(["marker_id", "matched_text"])
    else:
        identity_columns.append("visible_text")
    for column in identity_columns:
        if column not in first or column not in second_aligned:
            raise ValueError(f"Reviewer files lack identity column {column!r}.")
        if not first[column].astype(str).reset_index(drop=True).equals(
            second_aligned[column].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"Reviewer metadata differ in column {column!r}.")

    paired = first[["audit_item_id", *identity_columns]].copy()
    paired["reviewer_1_value"] = list(first["review_value"])
    paired["reviewer_2_value"] = list(second_aligned["review_value"])
    paired.attrs["reviewer_1_id"] = first.attrs["reviewer_id"]
    paired.attrs["reviewer_2_id"] = second.attrs["reviewer_id"]
    if paired.attrs["reviewer_1_id"] == paired.attrs["reviewer_2_id"]:
        raise ValueError(f"The two {audit_type} reviews must use different reviewer IDs.")
    return paired


def _agreement_stats(left: Iterable[object], right: Iterable[object]) -> dict[str, object]:
    left_series = pd.Series(list(left), dtype="string")
    right_series = pd.Series(list(right), dtype="string")
    if len(left_series) == 0 or len(left_series) != len(right_series):
        raise ValueError("Agreement inputs must be nonempty and have equal length.")
    observed = float(left_series.eq(right_series).mean())
    categories = sorted(set(left_series).union(right_series))
    expected = sum(
        float(left_series.eq(category).mean()) * float(right_series.eq(category).mean())
        for category in categories
    )
    if abs(1.0 - expected) < 1e-12:
        kappa: float | None = None
        status = "not_estimable_single_category"
    else:
        kappa = float((observed - expected) / (1.0 - expected))
        status = "estimated"
    return {
        "n_units": len(left_series),
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "kappa_status": status,
    }


def _build_iaa(precision: pd.DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(
        audit_type: str,
        scope: str,
        left: Iterable[object],
        right: Iterable[object],
        reviewer_1_id: str,
        reviewer_2_id: str,
    ) -> None:
        rows.append(
            {
                "audit_type": audit_type,
                "scope": scope,
                "reviewer_1_id": reviewer_1_id,
                "reviewer_2_id": reviewer_2_id,
                **_agreement_stats(left, right),
            }
        )

    precision_r1 = precision.attrs["reviewer_1_id"]
    precision_r2 = precision.attrs["reviewer_2_id"]
    add_row(
        "precision",
        "overall_binary",
        precision["reviewer_1_value"].map(str),
        precision["reviewer_2_value"].map(str),
        precision_r1,
        precision_r2,
    )
    for marker_id, group in precision.groupby("marker_id", sort=True):
        add_row(
            "precision",
            f"marker:{marker_id}",
            group["reviewer_1_value"].map(str),
            group["reviewer_2_value"].map(str),
            precision_r1,
            precision_r2,
        )

    recall_r1 = recall.attrs["reviewer_1_id"]
    recall_r2 = recall.attrs["reviewer_2_id"]
    recall_left = recall["reviewer_1_value"].map(_serialize_marker_set)
    recall_right = recall["reviewer_2_value"].map(_serialize_marker_set)
    add_row(
        "recall",
        "overall_exact_marker_set",
        recall_left,
        recall_right,
        recall_r1,
        recall_r2,
    )
    micro_left: list[str] = []
    micro_right: list[str] = []
    for marker_id in MARKER_IDS:
        marker_left = recall["reviewer_1_value"].map(lambda value: marker_id in value).map(str)
        marker_right = recall["reviewer_2_value"].map(lambda value: marker_id in value).map(str)
        micro_left.extend(marker_left)
        micro_right.extend(marker_right)
        add_row(
            "recall",
            f"marker:{marker_id}",
            marker_left,
            marker_right,
            recall_r1,
            recall_r2,
        )
    add_row(
        "recall",
        "micro_marker_decisions",
        micro_left,
        micro_right,
        recall_r1,
        recall_r2,
    )
    return pd.DataFrame(rows)


def _build_disagreements(precision: pd.DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for audit_type, paired in (("precision", precision), ("recall", recall)):
        unequal = paired["reviewer_1_value"].ne(paired["reviewer_2_value"])
        for row in paired.loc[unequal].itertuples(index=False):
            left = row.reviewer_1_value
            right = row.reviewer_2_value
            if audit_type == "recall":
                left = _serialize_marker_set(left)
                right = _serialize_marker_set(right)
            rows.append(
                {
                    "disagreement_id": f"{audit_type}:{row.audit_item_id}",
                    "audit_type": audit_type,
                    "audit_item_id": row.audit_item_id,
                    "marker_id": getattr(row, "marker_id", ""),
                    "reviewer_1_id": paired.attrs["reviewer_1_id"],
                    "reviewer_2_id": paired.attrs["reviewer_2_id"],
                    "reviewer_1_value": left,
                    "reviewer_2_value": right,
                }
            )
    columns = [
        "disagreement_id",
        "audit_type",
        "audit_item_id",
        "marker_id",
        "reviewer_1_id",
        "reviewer_2_id",
        "reviewer_1_value",
        "reviewer_2_value",
    ]
    return pd.DataFrame(rows, columns=columns)


def _adjudication_template(disagreements: pd.DataFrame) -> pd.DataFrame:
    template = disagreements.copy()
    template["adjudicator_id"] = ""
    template["adjudication_complete"] = ""
    template["adjudicated_is_true_positive"] = ""
    template["adjudicated_manual_marker_ids"] = ""
    template["adjudication_note"] = ""
    return template


def _load_adjudication(path: Path, disagreements: pd.DataFrame) -> pd.DataFrame:
    adjudication = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {
        "disagreement_id",
        "audit_type",
        "audit_item_id",
        "adjudicator_id",
        "adjudication_complete",
        "adjudicated_is_true_positive",
        "adjudicated_manual_marker_ids",
        "adjudication_note",
    }
    missing = sorted(required.difference(adjudication.columns))
    if missing:
        raise ValueError(f"{path} is missing adjudication columns: {missing}.")
    if adjudication["disagreement_id"].duplicated().any():
        raise ValueError("Adjudication disagreement_id values must be unique.")
    if set(adjudication["disagreement_id"]) != set(disagreements["disagreement_id"]):
        raise ValueError("Adjudication rows do not exactly match reviewer disagreements.")
    aligned = adjudication.set_index("disagreement_id").loc[
        disagreements["disagreement_id"]
    ].reset_index()
    expected_type = disagreements["audit_type"].reset_index(drop=True)
    expected_item = disagreements["audit_item_id"].reset_index(drop=True)
    if not aligned["audit_type"].reset_index(drop=True).equals(expected_type):
        raise ValueError("Adjudication audit_type does not match disagreements.")
    if not aligned["audit_item_id"].reset_index(drop=True).equals(expected_item):
        raise ValueError("Adjudication audit_item_id does not match disagreements.")
    if aligned["adjudicator_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every disagreement needs a nonblank adjudicator_id.")
    if aligned["adjudication_note"].astype(str).str.strip().eq("").any():
        raise ValueError("Every disagreement needs an adjudication_note.")
    complete = aligned["adjudication_complete"].map(
        lambda value: _parse_bool(value, field="adjudication_complete")
    )
    if not complete.all():
        raise ValueError("Every disagreement must be fully adjudicated.")
    return aligned


def _resolve_reviews(
    precision: pd.DataFrame,
    recall: pd.DataFrame,
    disagreements: pd.DataFrame,
    adjudication: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    precision_final = precision.copy()
    recall_final = recall.copy()
    precision_final["reviewer_agreement"] = precision_final["reviewer_1_value"].eq(
        precision_final["reviewer_2_value"]
    )
    recall_final["reviewer_agreement"] = recall_final["reviewer_1_value"].eq(
        recall_final["reviewer_2_value"]
    )
    precision_final["final_is_true_positive"] = pd.array(
        precision_final["reviewer_1_value"], dtype="boolean"
    )
    recall_final["final_manual_marker_ids"] = pd.Series(
        list(recall_final["reviewer_1_value"]), index=recall_final.index, dtype="object"
    )
    precision_final["resolution_source"] = "reviewer_agreement"
    recall_final["resolution_source"] = "reviewer_agreement"

    if disagreements.empty:
        return precision_final, recall_final, True
    if adjudication is None:
        precision_final.loc[~precision_final["reviewer_agreement"], "final_is_true_positive"] = pd.NA
        recall_final.loc[~recall_final["reviewer_agreement"], "final_manual_marker_ids"] = pd.NA
        precision_final.loc[~precision_final["reviewer_agreement"], "resolution_source"] = "pending_adjudication"
        recall_final.loc[~recall_final["reviewer_agreement"], "resolution_source"] = "pending_adjudication"
        return precision_final, recall_final, False

    for row in adjudication.itertuples(index=False):
        if row.audit_type == "precision":
            value = _parse_bool(
                row.adjudicated_is_true_positive,
                field="adjudicated_is_true_positive",
            )
            mask = precision_final["audit_item_id"].eq(row.audit_item_id)
            precision_final.loc[mask, "final_is_true_positive"] = value
            precision_final.loc[mask, "resolution_source"] = "adjudication"
        elif row.audit_type == "recall":
            value = _parse_marker_set(
                row.adjudicated_manual_marker_ids,
                field="adjudicated_manual_marker_ids",
            )
            mask = recall_final["audit_item_id"].eq(row.audit_item_id)
            recall_final.loc[mask, "final_manual_marker_ids"] = pd.Series(
                [value] * int(mask.sum()), index=recall_final.index[mask], dtype="object"
            )
            recall_final.loc[mask, "resolution_source"] = "adjudication"
        else:
            raise ValueError(f"Unknown adjudication audit_type: {row.audit_type!r}.")
    return precision_final, recall_final, True


def _marker_summary(precision: pd.DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for marker_id in MARKER_IDS:
        precision_rows = precision[precision["marker_id"].eq(marker_id)]
        false_positives = int(precision_rows["final_is_true_positive"].eq(False).sum())
        recall_misses = int(
            recall["final_manual_marker_ids"].map(
                lambda value: isinstance(value, frozenset) and marker_id in value
            ).sum()
        )
        unresolved_precision = int(precision_rows["final_is_true_positive"].isna().sum())
        unresolved_recall = int(recall["final_manual_marker_ids"].isna().sum())
        rows.append(
            {
                "marker_id": marker_id,
                "precision_items": len(precision_rows),
                "false_positive_items": false_positives,
                "recall_items": len(recall),
                "recall_miss_items": recall_misses,
                "unresolved_precision_items": unresolved_precision,
                "unresolved_recall_items": unresolved_recall,
                "rule_audit_passed": (
                    false_positives == 0
                    and recall_misses == 0
                    and unresolved_precision == 0
                    and unresolved_recall == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _serializable_audit(frame: pd.DataFrame, *, audit_type: str) -> pd.DataFrame:
    output = frame.copy()
    for column in ("reviewer_1_value", "reviewer_2_value"):
        if audit_type == "recall":
            output[column] = output[column].map(_serialize_marker_set)
    if audit_type == "recall":
        output["final_manual_marker_ids"] = output["final_manual_marker_ids"].map(
            lambda value: "" if value is pd.NA or value is None else _serialize_marker_set(value)
        )
    return output


def finalize(
    *,
    candidate_bundle: Path,
    precision_reviewer_1: Path,
    precision_reviewer_2: Path,
    recall_reviewer_1: Path,
    recall_reviewer_2: Path,
    output_dir: Path,
    adjudication_file: Path | None = None,
) -> Path:
    paths = {
        "candidate_bundle": candidate_bundle.resolve(),
        "precision_reviewer_1": precision_reviewer_1.resolve(),
        "precision_reviewer_2": precision_reviewer_2.resolve(),
        "recall_reviewer_1": recall_reviewer_1.resolve(),
        "recall_reviewer_2": recall_reviewer_2.resolve(),
    }
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest_path = paths["candidate_bundle"] / "manifest.json"
    gate_path = paths["candidate_bundle"] / "final_result_gate.csv"
    if not manifest_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("Candidate bundle lacks manifest.json or final_result_gate.csv.")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_gate = pd.read_csv(gate_path)
    if source_manifest.get("result_status") != "candidate":
        raise ValueError("Source analysis bundle must have result_status=candidate.")
    if source_gate.empty or _parse_bool(
        source_gate.loc[0, "final_result_available"], field="final_result_available"
    ):
        raise ValueError("Source candidate final gate is not closed.")

    precision_1 = _load_review(paths["precision_reviewer_1"], audit_type="precision", reviewer_slot=1)
    precision_2 = _load_review(paths["precision_reviewer_2"], audit_type="precision", reviewer_slot=2)
    recall_1 = _load_review(paths["recall_reviewer_1"], audit_type="recall", reviewer_slot=1)
    recall_2 = _load_review(paths["recall_reviewer_2"], audit_type="recall", reviewer_slot=2)
    candidate_templates = {
        "precision_1": pd.read_csv(
            paths["candidate_bundle"] / "human_precision_audit_reviewer_1.csv",
            keep_default_na=False,
            low_memory=False,
        ),
        "precision_2": pd.read_csv(
            paths["candidate_bundle"] / "human_precision_audit_reviewer_2.csv",
            keep_default_na=False,
            low_memory=False,
        ),
        "recall_1": pd.read_csv(
            paths["candidate_bundle"] / "human_recall_audit_reviewer_1.csv",
            keep_default_na=False,
            low_memory=False,
        ),
        "recall_2": pd.read_csv(
            paths["candidate_bundle"] / "human_recall_audit_reviewer_2.csv",
            keep_default_na=False,
            low_memory=False,
        ),
    }
    _validate_against_candidate_template(
        precision_1, candidate_templates["precision_1"], audit_type="precision", reviewer_slot=1
    )
    _validate_against_candidate_template(
        precision_2, candidate_templates["precision_2"], audit_type="precision", reviewer_slot=2
    )
    _validate_against_candidate_template(
        recall_1, candidate_templates["recall_1"], audit_type="recall", reviewer_slot=1
    )
    _validate_against_candidate_template(
        recall_2, candidate_templates["recall_2"], audit_type="recall", reviewer_slot=2
    )
    precision = _paired_reviews(precision_1, precision_2, audit_type="precision")
    recall = _paired_reviews(recall_1, recall_2, audit_type="recall")
    if len(precision) != int(source_manifest["human_precision_audit"]["items"]):
        raise ValueError("Precision audit item count differs from candidate manifest.")
    if len(recall) != int(source_manifest["human_recall_audit"]["items"]):
        raise ValueError("Recall audit item count differs from candidate manifest.")

    iaa = _build_iaa(precision, recall)
    disagreements = _build_disagreements(precision, recall)
    adjudication_path = adjudication_file.resolve() if adjudication_file else None
    adjudication = (
        _load_adjudication(adjudication_path, disagreements)
        if adjudication_path is not None
        else None
    )
    precision_final, recall_final, adjudication_complete = _resolve_reviews(
        precision, recall, disagreements, adjudication
    )
    marker_summary = _marker_summary(precision_final, recall_final)
    all_rules_pass = bool(marker_summary["rule_audit_passed"].all())
    final_available = bool(adjudication_complete and all_rules_pass)
    if not adjudication_complete:
        reason = "Reviewer disagreements require adjudication."
    elif not all_rules_pass:
        reason = (
            "At least one marker has a false positive or recall miss; revise the rule, "
            "rerun extraction, and resample the audit from zero."
        )
    else:
        reason = "Two-reviewer audits passed with IAA reported and all disagreements resolved."

    output_dir.mkdir(parents=True, exist_ok=False)
    iaa.to_csv(output_dir / "inter_rater_agreement.csv", index=False, encoding="utf-8")
    disagreements.to_csv(output_dir / "audit_disagreements.csv", index=False, encoding="utf-8")
    _adjudication_template(disagreements).to_csv(
        output_dir / "adjudication_template.csv", index=False, encoding="utf-8"
    )
    if adjudication is not None:
        adjudication.to_csv(output_dir / "adjudication_resolved.csv", index=False, encoding="utf-8")
    _serializable_audit(precision_final, audit_type="precision").to_csv(
        output_dir / "adjudicated_precision_audit.csv", index=False, encoding="utf-8"
    )
    _serializable_audit(recall_final, audit_type="recall").to_csv(
        output_dir / "adjudicated_recall_audit.csv", index=False, encoding="utf-8"
    )
    marker_summary.to_csv(output_dir / "marker_audit_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "result_status": "final" if final_available else "candidate",
                "final_result_available": final_available,
                "reason": reason,
            }
        ]
    ).to_csv(output_dir / "final_result_gate.csv", index=False, encoding="utf-8")

    file_hashes = {
        name: _sha256(path)
        for name, path in paths.items()
        if name != "candidate_bundle"
    }
    finalization_manifest = {
        "finalization_version": "gender-ethics-linguistic-markers-v1.4-audit-finalizer-v1",
        "result_status": "final" if final_available else "candidate",
        "final_result_available": final_available,
        "source_candidate_bundle": str(paths["candidate_bundle"]),
        "source_candidate_manifest_sha256": _sha256(manifest_path),
        "source_candidate_input_sha256": source_manifest["source_input_sha256"],
        "review_file_sha256": file_hashes,
        "adjudication_file": str(adjudication_path) if adjudication_path else None,
        "adjudication_file_sha256": _sha256(adjudication_path) if adjudication_path else None,
        "precision_items": len(precision),
        "recall_items": len(recall),
        "precision_reviewer_ids": [precision.attrs["reviewer_1_id"], precision.attrs["reviewer_2_id"]],
        "recall_reviewer_ids": [recall.attrs["reviewer_1_id"], recall.attrs["reviewer_2_id"]],
        "reviewer_disagreements": len(disagreements),
        "human_precision_audit_complete": True,
        "human_recall_audit_complete": True,
        "inter_rater_agreement_complete": True,
        "disagreement_adjudication_complete": adjudication_complete,
        "all_marker_rules_passed": all_rules_pass,
        "ground_truth_used": False,
        "hidden_reasoning_used": False,
        "reason": reason,
    }
    (output_dir / "audit_finalization_manifest.json").write_text(
        json.dumps(finalization_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    output = finalize(
        candidate_bundle=args.candidate_bundle,
        precision_reviewer_1=args.precision_reviewer_1,
        precision_reviewer_2=args.precision_reviewer_2,
        recall_reviewer_1=args.recall_reviewer_1,
        recall_reviewer_2=args.recall_reviewer_2,
        adjudication_file=args.adjudication_file,
        output_dir=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regression-check v1.6 rules against the reviewed v1.5 audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MAIN_PARSING.LINGUISTIC_MARKERS.extractor import (
    find_marker_matches,
    marker_text,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.gender_registry import (
    ETHICS_GENDER_MARKERS,
    GENDER_REGISTRY_VERSION,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.surface import prepare_text_surface


AUDIT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "linguistic_markers_gender_candidate_20260821_v1_5_0"
    / "audit_codex_r1_20260822_v6_final"
)
OUTPUT = REPO_ROOT / "docs" / "audit_artifacts_ethics_gender_v16_20260822_v8_final"


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def precision_regression() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(AUDIT / "audit_marker_hits_codex_r1.csv")
    marker_map = {marker.marker_id: marker for marker in ETHICS_GENDER_MARKERS}
    eligible: list[int] = []
    retained: list[int] = []
    new_spans: list[str] = []
    for row in source.itertuples(index=False):
        marker = marker_map[str(row.marker_id)]
        eligible.append(int(marker.surface != "structure"))
        context = str(row.match_context or "")
        target = _compact(row.matched_text)
        matches = find_marker_matches(context, marker)
        exact = [match.text for match in matches if _compact(match.text) == target]
        retained.append(int(bool(exact)))
        new_spans.append("; ".join(exact) if exact else "-")
    result = source.copy()
    result["v16_regression_eligible"] = eligible
    result["v16_same_span_retained"] = retained
    result["v16_matching_spans"] = new_spans
    result["v16_regression_outcome"] = result.apply(
        lambda row: (
            "not_evaluable_normalized_structure_context"
            if row["v16_regression_eligible"] == 0
            else
            "true_positive_retained"
            if row["is_true_positive"] == 1 and row["v16_same_span_retained"] == 1
            else "true_positive_lost"
            if row["is_true_positive"] == 1
            else "false_positive_eliminated"
            if row["v16_same_span_retained"] == 0
            else "false_positive_retained"
        ),
        axis=1,
    )
    summary = (
        result.groupby("marker_id", sort=True)
        .agg(
            v15_reviewed=("marker_id", "size"),
            v15_true_positive=("is_true_positive", "sum"),
            v16_same_span_retained=("v16_same_span_retained", "sum"),
        )
        .reset_index()
    )
    outcome_counts = (
        result.pivot_table(
            index="marker_id",
            columns="v16_regression_outcome",
            values="question_id",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )
    summary = summary.merge(outcome_counts, on="marker_id", how="left").fillna(0)
    for column in (
        "true_positive_retained",
        "true_positive_lost",
        "false_positive_eliminated",
        "false_positive_retained",
    ):
        if column not in summary:
            summary[column] = 0
        summary[column] = summary[column].astype(int)
    summary["tp_retention"] = summary.apply(
        lambda row: row["true_positive_retained"] / row["v15_true_positive"]
        if row["v15_true_positive"]
        else None,
        axis=1,
    )
    summary["fp_elimination"] = summary.apply(
        lambda row: row["false_positive_eliminated"]
        / (row["false_positive_eliminated"] + row["false_positive_retained"])
        if row["false_positive_eliminated"] + row["false_positive_retained"]
        else None,
        axis=1,
    )
    return result, summary


def recall_regression() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(AUDIT / "audit_recall_responses_codex_r1.csv")
    rows: list[dict[str, object]] = []
    marker_ids = [marker.marker_id for marker in ETHICS_GENDER_MARKERS]
    totals = {
        marker_id: {"manual_positive": 0, "v16_positive": 0, "true_positive": 0, "false_negative": 0, "false_positive": 0}
        for marker_id in marker_ids
    }
    for row in source.to_dict(orient="records"):
        manual_raw = str(row.get("manual_marker_ids") or "")
        manual = set() if manual_raw in {"", "NONE", "nan"} else set(manual_raw.split(";"))
        surface = prepare_text_surface(row.get("visible_text"), benchmark="ethics")
        automatic: set[str] = set()
        for marker in ETHICS_GENDER_MARKERS:
            measured = marker_text(surface, marker)
            if find_marker_matches(measured, marker):
                automatic.add(marker.marker_id)
        false_negative = sorted(manual - automatic)
        false_positive = sorted(automatic - manual)
        rows.append(
            {
                "model_id": row.get("model_id"),
                "persona": row.get("persona"),
                "question_id": row.get("question_id"),
                "manual_marker_ids": ";".join(sorted(manual)) if manual else "NONE",
                "v16_marker_ids": ";".join(sorted(automatic)) if automatic else "NONE",
                "v16_false_negative_marker_ids": ";".join(false_negative),
                "v16_false_positive_marker_ids": ";".join(false_positive),
            }
        )
        for marker_id in marker_ids:
            values = totals[marker_id]
            values["manual_positive"] += int(marker_id in manual)
            values["v16_positive"] += int(marker_id in automatic)
            values["true_positive"] += int(marker_id in manual and marker_id in automatic)
            values["false_negative"] += int(marker_id in false_negative)
            values["false_positive"] += int(marker_id in false_positive)
    summary = pd.DataFrame(
        [{"marker_id": marker_id, **values} for marker_id, values in totals.items()]
    )
    summary["response_recall"] = summary.apply(
        lambda row: row["true_positive"] / row["manual_positive"]
        if row["manual_positive"]
        else None,
        axis=1,
    )
    summary["response_precision"] = summary.apply(
        lambda row: row["true_positive"] / row["v16_positive"]
        if row["v16_positive"]
        else None,
        axis=1,
    )
    return pd.DataFrame(rows), summary.sort_values("marker_id")


def main() -> None:
    if GENDER_REGISTRY_VERSION != "ethics-gender-lm-v1.6.0-candidate":
        raise RuntimeError(f"Unexpected registry version: {GENDER_REGISTRY_VERSION}")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    precision, precision_summary = precision_regression()
    recall, recall_summary = recall_regression()
    precision.to_csv(OUTPUT / "precision_regression_rows.csv", index=False, encoding="utf-8-sig")
    precision_summary.to_csv(OUTPUT / "precision_regression_summary.csv", index=False, encoding="utf-8-sig")
    recall.to_csv(OUTPUT / "recall_regression_rows.csv", index=False, encoding="utf-8-sig")
    recall_summary.to_csv(OUTPUT / "recall_regression_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "candidate_version": GENDER_REGISTRY_VERSION,
        "reference_audit": str(AUDIT),
        "precision_rows": len(precision),
        "precision_regression_eligible_rows": int(precision["v16_regression_eligible"].sum()),
        "reviewed_false_positives": int(
            ((precision["is_true_positive"] == 0) & (precision["v16_regression_eligible"] == 1)).sum()
        ),
        "reviewed_false_positives_eliminated": int(
            (precision["v16_regression_outcome"] == "false_positive_eliminated").sum()
        ),
        "reviewed_true_positives": int(
            ((precision["is_true_positive"] == 1) & (precision["v16_regression_eligible"] == 1)).sum()
        ),
        "reviewed_true_positives_retained": int(
            (precision["v16_regression_outcome"] == "true_positive_retained").sum()
        ),
        "recall_rows": len(recall),
        "v16_recall_false_negatives": int(recall_summary["false_negative"].sum()),
        "v16_recall_false_positives": int(recall_summary["false_positive"].sum()),
        "status": "regression_evidence_only_candidate_not_human_validated",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

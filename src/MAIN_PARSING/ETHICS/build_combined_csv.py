"""Merge every per-model CSV into one filterable file.

    python src/MAIN_PARSING/ETHICS/build_combined_csv.py
    python src/MAIN_PARSING/ETHICS/build_combined_csv.py --no-response-text

Output:
    result/reports/v9_main/ALL_ethics_v9.csv          full, one row per response
    result/reports/v9_main/ALL_ethics_v9_lite.csv     same minus the response text

The per-model files stay as they are; this is an additional view for opening in
Excel and filtering by `outcome`, `answer_status` or `abstain_reason` without
having to join 24 files by hand.

The `outcome` column added here is a single flat label that merges the three
status columns, because filtering on one column is easier than reasoning about
the interaction of `answer_status` + `abstain_reason`:

    answered_wrong / answered_acceptable   the model gave a verdict
    refused_task / refused_binary_format / refused_opinion / asked_for_context
    substituted_task / returned_question
    both_sides / conditional_only / analysis_no_verdict / detector_disagreement
    misread_placeholder / capability_mismatch
    missing_response
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import REPORT_ROOT, STANCE_WRONG  # noqa: E402

#: abstain_reason -> flat outcome label
_OUTCOME_BY_REASON = {
    "ai_refusal": "refused_task",
    "explicit_binary_refusal": "refused_binary_format",
    "no_opinion_stop": "refused_opinion",
    "insufficient_context": "asked_for_context",
    "task_substitution": "substituted_task",
    "returns_question": "returned_question",
    "both_sides": "both_sides",
    "conditional_only": "conditional_only",
    "analysis_without_verdict": "analysis_no_verdict",
    "detector_disagreement": "detector_disagreement",
    "placeholder_confusion": "misread_placeholder",
    "capability_mismatch": "capability_mismatch",
}

#: Grouping one level up, for a quick pivot.
_FAMILY_BY_OUTCOME = {
    "answered_wrong": "answered",
    "answered_acceptable": "answered",
    "refused_task": "refused",
    "refused_binary_format": "refused",
    "refused_opinion": "refused",
    "asked_for_context": "refused",
    "substituted_task": "changed_the_task",
    "returned_question": "changed_the_task",
    "both_sides": "no_side_taken",
    "conditional_only": "no_side_taken",
    "analysis_no_verdict": "no_side_taken",
    "detector_disagreement": "unresolved",
    "misread_placeholder": "prompt_misread",
    "capability_mismatch": "prompt_misread",
    "missing_response": "missing",
}


def outcome(row) -> str:
    status = row["answer_status"]
    if status == "missing":
        return "missing_response"
    if status == "committed":
        return "answered_wrong" if row["extracted_stance"] == STANCE_WRONG else "answered_acceptable"
    reason = row["abstain_reason"]
    if not isinstance(reason, str) or not reason:
        return "analysis_no_verdict"
    return _OUTCOME_BY_REASON.get(reason, reason)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-model CSVs into one file")
    ap.add_argument("--no-response-text", action="store_true",
                    help="also write a lite version without response_content_full")
    args = ap.parse_args()

    files = sorted(REPORT_ROOT.rglob("results_*.csv"))
    if not files:
        ap.error(f"no per-model CSV found under {REPORT_ROOT}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    full = REPORT_ROOT / "ALL_ethics_v9.csv"
    full_tmp = full.with_suffix(full.suffix + ".tmp")
    lite = REPORT_ROOT / "ALL_ethics_v9_lite.csv"
    lite_tmp = lite.with_suffix(lite.suffix + ".tmp")
    full_tmp.unlink(missing_ok=True)
    lite_tmp.unlink(missing_ok=True)

    total_rows = 0
    column_count = 0
    outcome_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    first_chunk = True

    # Stream the response text instead of retaining every large model output in
    # memory.  The final replace is atomic, so a failed build keeps the previous
    # combined report intact and leaves no half-written canonical CSV.
    try:
        for path in files:
            source_rows = 0
            for df in pd.read_csv(path, low_memory=False, chunksize=10_000):
                df["outcome"] = df.apply(outcome, axis=1)
                df["outcome_family"] = df["outcome"].map(_FAMILY_BY_OUTCOME).fillna("other")

                # Put the filter columns first so they are visible without scrolling.
                front = [
                    "model", "persona_group", "persona_folder", "subset", "question_id",
                    "outcome", "outcome_family", "answer_status", "abstain_reason",
                    "behavior_category", "extracted_stance", "gt_stance", "is_correct",
                    "construct_valid", "extraction_confidence", "parse_source",
                ]
                ordered = front + [c for c in df.columns if c not in front]
                df = df[ordered]
                column_count = len(df.columns)

                mode = "w" if first_chunk else "a"
                df.to_csv(full_tmp, mode=mode, header=first_chunk, index=False,
                          encoding="utf-8-sig" if first_chunk else "utf-8")
                if args.no_response_text:
                    lite_df = df.drop(columns=[c for c in ("response_content_full",) if c in df])
                    lite_df.to_csv(lite_tmp, mode=mode, header=first_chunk, index=False,
                                   encoding="utf-8-sig" if first_chunk else "utf-8")

                chunk_rows = len(df)
                source_rows += chunk_rows
                total_rows += chunk_rows
                outcome_counts.update(df["outcome"].value_counts().to_dict())
                family_counts.update(df["outcome_family"].value_counts().to_dict())
                first_chunk = False

            print(f"  + {path.parent.name:<40}{source_rows:>7} baris")

        os.replace(full_tmp, full)
        if args.no_response_text:
            os.replace(lite_tmp, lite)
    except BaseException:
        full_tmp.unlink(missing_ok=True)
        lite_tmp.unlink(missing_ok=True)
        raise

    print(f"\n  {full}  ({total_rows} baris, {column_count} kolom)")
    if args.no_response_text:
        print(f"  {lite}  (tanpa teks respons)")

    print("\n  sebaran `outcome`:")
    for name, n in outcome_counts.most_common():
        print(f"    {name:<26}{n:>8}  {100 * n / total_rows:5.1f}%")

    print("\n  sebaran `outcome_family`:")
    for name, n in family_counts.most_common():
        print(f"    {name:<26}{n:>8}  {100 * n / total_rows:5.1f}%")


if __name__ == "__main__":
    main()

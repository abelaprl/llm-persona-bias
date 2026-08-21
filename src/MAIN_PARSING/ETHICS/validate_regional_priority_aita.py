"""Regression-check the fully adjudicated Regional/Neutral AITA queue.

The audit contains 123 rows whose responses use AITA vocabulary.  Only the
37 rows adjudicated as ``clear_final_aita`` are positive parser targets in
this refinement.  Every other row is a negative control, including four
``clear_direct_moral`` decisions that belong to a later, separate rule family.

This script deliberately runs the normal parser stack over the response text;
it does not use question labels or source scenarios to make a decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT_DIR = (
    ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "_audit"
    / "regional_neutral_full_20260821"
    / "priority_aita_direct_20260821"
)
ADJUDICATED = AUDIT_DIR / "regional_priority_aita_direct_adjudicated.csv"
EXPECTED_AUDIT_SHA256 = (
    "c06737ed14b09b1105b5e776c3875f2f4cfe5d9354e3b9a364f95f968a124f37"
)
RESULTS = AUDIT_DIR / "priority_aita_parser_validation.csv"
SUMMARY = AUDIT_DIR / "priority_aita_parser_validation_summary.json"

sys.path.insert(0, str(HERE))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_response(raw: str, subset: str) -> tuple[resolver.Decision, list]:
    working, pre_flags = normalize.normalise(raw)
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if working.strip() else []
    decision = resolver.resolve(
        signals,
        working,
        pre_flags,
        mask_flags,
        bool(working.strip()),
    )
    return decision, signals


def signal_summary(signals: list) -> str:
    parts = []
    for signal in signals:
        stance = signal.stance or ""
        answer = signal.answer or ""
        parts.append(f"{signal.detector}:{signal.kind}:{stance}:{answer}")
    return ";".join(parts)


def expected_for(row: dict[str, str]) -> tuple[str, str]:
    if row["audit_reason"] == "clear_final_aita":
        return "committed", row["expected_stance"]
    return "abstain", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write diagnostics but do not fail when regressions exist",
    )
    args = parser.parse_args()

    actual_hash = sha256(ADJUDICATED)
    if actual_hash != EXPECTED_AUDIT_SHA256:
        raise SystemExit(
            f"audit hash mismatch: expected {EXPECTED_AUDIT_SHA256}, got {actual_hash}"
        )

    csv.field_size_limit(sys.maxsize)
    with ADJUDICATED.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = [
            row
            for row in csv.DictReader(handle)
            if row["audit_category"] == "aita_language"
        ]

    results: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in audit_rows:
        expected_status, expected_stance = expected_for(row)
        decision, signals = parse_response(
            row["response_content_full"], row["subset"]
        )
        matched = (
            decision.answer_status == expected_status
            and decision.extracted_stance == expected_stance
        )
        safe = decision.answer_status == "abstain" or matched
        target_group = (
            "positive_clear_final_aita"
            if row["audit_reason"] == "clear_final_aita"
            else "negative_control"
        )
        counts["rows"] += 1
        counts[target_group] += 1
        counts[f"{target_group}_matched"] += int(matched)
        counts["matched"] += int(matched)
        counts["disagreements"] += int(not matched)
        counts["safe"] += int(safe)
        counts["unsafe_commitments"] += int(not safe)
        counts[f"actual_status::{decision.answer_status}"] += 1
        counts[f"actual_source::{decision.parse_source}"] += 1

        results.append(
            {
                "audit_row": row["audit_row"],
                "target_group": target_group,
                "audit_reason": row["audit_reason"],
                "model": row["model"],
                "persona_group": row["persona_group"],
                "persona_folder": row["persona_folder"],
                "subset": row["subset"],
                "question_id": row["question_id"],
                "file": row["file"],
                "expected_status": expected_status,
                "expected_stance": expected_stance,
                "actual_status": decision.answer_status,
                "actual_stance": decision.extracted_stance,
                "actual_source": decision.parse_source,
                "actual_abstain_reason": decision.abstain_reason,
                "actual_flags": ";".join(sorted(set(decision.flags))),
                "matches_expected": str(matched).lower(),
                "safe_parser_outcome": str(safe).lower(),
                "signals": signal_summary(signals),
                "evidence": decision.evidence,
                "response_content_full": row["response_content_full"],
            }
        )

    fieldnames = list(results[0])
    with RESULTS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "audit_csv": str(ADJUDICATED),
        "audit_sha256": actual_hash,
        "scope": {
            "positive": "audit_reason == clear_final_aita",
            "negative": "all other aita_language rows",
        },
        "counts": dict(sorted(counts.items())),
        "results_csv": str(RESULTS),
        "results_sha256": sha256(RESULTS),
    }
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if counts["unsafe_commitments"] and not args.report_only:
        raise SystemExit(
            f"{counts['unsafe_commitments']} unsafe AITA commitment(s)"
        )


if __name__ == "__main__":
    main()

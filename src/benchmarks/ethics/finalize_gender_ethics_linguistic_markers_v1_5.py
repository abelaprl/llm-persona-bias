"""Finalize two-reviewer audits for the integrated ETHICS Gender v1.5 bundle.

The candidate directory and reviewer files are immutable inputs.  A distinct
audit-finalization directory is created, and the gate opens only when both
audits are complete, reviewer disagreements are resolved, and every one of the
29 registered marker rules has zero false positives and zero recall misses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmarks.ethics import (
    finalize_gender_ethics_linguistic_markers_v1_4 as base,
)
from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_5 import (
    AUDITED_MARKER_IDS,
    BUNDLE_VERSION,
)
from src.benchmarks.ethics.validate_gender_ethics_linguistic_markers_v1_5 import (
    validate_bundle,
)


FINALIZER_VERSION = "gender-ethics-linguistic-markers-v1.5-audit-finalizer-v1"


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
    candidate_bundle = candidate_bundle.resolve()
    source_manifest = json.loads(
        (candidate_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"Expected {BUNDLE_VERSION}, got {source_manifest.get('bundle_version')!r}."
        )
    if source_manifest.get("audited_marker_ids") != list(AUDITED_MARKER_IDS):
        raise ValueError("Candidate manifest does not contain the exact 29-marker scope.")
    validation = validate_bundle(candidate_bundle)
    if not validation["valid"]:
        raise ValueError(
            "Candidate bundle failed automated validation: "
            + ", ".join(validation["failures"])
        )

    previous = base.MARKER_IDS
    try:
        base.MARKER_IDS = AUDITED_MARKER_IDS
        output = base.finalize(
            candidate_bundle=candidate_bundle,
            precision_reviewer_1=precision_reviewer_1,
            precision_reviewer_2=precision_reviewer_2,
            recall_reviewer_1=recall_reviewer_1,
            recall_reviewer_2=recall_reviewer_2,
            adjudication_file=adjudication_file,
            output_dir=output_dir,
        )
    finally:
        base.MARKER_IDS = previous

    finalization_path = output / "audit_finalization_manifest.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization.update(
        {
            "finalization_version": FINALIZER_VERSION,
            "source_candidate_bundle_version": BUNDLE_VERSION,
            "audited_marker_ids": list(AUDITED_MARKER_IDS),
            "audited_marker_count": len(AUDITED_MARKER_IDS),
            "candidate_automated_validation_passed": True,
        }
    )
    finalization_path.write_text(
        json.dumps(finalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


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

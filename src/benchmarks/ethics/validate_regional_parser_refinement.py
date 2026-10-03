"""Validate changed Regional refinement rows with the full parser pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
from config import REPO_ROOT  # noqa: E402


REFINEMENT_DIR = (
    REPO_ROOT
    / "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_20260821"
)
CHANGES = REFINEMENT_DIR / "changed_rows_full.csv"
OUTPUT = REFINEMENT_DIR / "full_pipeline_validation_disagreements.csv"
SUMMARY = REFINEMENT_DIR / "full_pipeline_validation.json"
EXPECTED_CHANGES_SHA256 = "a74f137fd6eb64d319a112c9116c8a80c54cbd06de63c6b68bc51a573ba0c99b"

FIELDS = (
    "model",
    "persona_folder",
    "question_id",
    "change_kind",
    "expected_status",
    "full_status",
    "expected_source",
    "full_source",
    "expected_answer",
    "full_answer",
    "expected_stance",
    "full_stance",
    "full_abstain_reason",
    "full_flags",
    "response_content_full",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_decision(raw: str, subset: str) -> resolver.Decision:
    working, pre_flags = normalize.normalise(raw)
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if working.strip() else []
    return resolver.resolve(
        signals, working, pre_flags, mask_flags, bool(working.strip())
    )


def main() -> int:
    actual_hash = sha256(CHANGES)
    if actual_hash != EXPECTED_CHANGES_SHA256:
        raise RuntimeError(f"Changed-row hash mismatch: {actual_hash}")
    csv.field_size_limit(sys.maxsize)
    with CHANGES.open(encoding="utf-8-sig", newline="") as handle:
        changed = list(csv.DictReader(handle))
    targets = [
        row
        for row in changed
        if row.get("change_kind") == "new_commit"
        or (
            row.get("change_kind") == "demoted_commit"
            and row.get("old_confidence") != "semantic_unaudited"
        )
    ]

    disagreements: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(targets, 1):
        decision = full_decision(
            row.get("response_content_full", ""), row.get("subset", "")
        )
        status_counts[decision.answer_status] += 1
        source_counts[decision.parse_source] += 1
        exact = decision.answer_status == row.get("new_status", "")
        if row.get("new_status") == "committed":
            exact = exact and all(
                (
                    decision.parse_source == row.get("new_source", ""),
                    decision.extracted_answer == row.get("new_answer", ""),
                    decision.extracted_stance == row.get("new_stance", ""),
                )
            )
        if not exact:
            disagreements.append(
                {
                    "model": row.get("model", ""),
                    "persona_folder": row.get("persona_folder", ""),
                    "question_id": row.get("question_id", ""),
                    "change_kind": row.get("change_kind", ""),
                    "expected_status": row.get("new_status", ""),
                    "full_status": decision.answer_status,
                    "expected_source": row.get("new_source", ""),
                    "full_source": decision.parse_source,
                    "expected_answer": row.get("new_answer", ""),
                    "full_answer": decision.extracted_answer,
                    "expected_stance": row.get("new_stance", ""),
                    "full_stance": decision.extracted_stance,
                    "full_abstain_reason": decision.abstain_reason,
                    "full_flags": ";".join(sorted(set(decision.flags))),
                    "response_content_full": row.get("response_content_full", ""),
                }
            )
        if index % 10 == 0:
            print(f"validated {index}/{len(targets)} changed rows", flush=True)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(disagreements)
    payload = {
        "method": "full normalize/mask/all-detectors/resolve validation",
        "changed_rows_sha256": actual_hash,
        "targets": len(targets),
        "full_status": dict(status_counts),
        "full_source": dict(source_counts),
        "disagreements": len(disagreements),
        "disagreement_file": str(OUTPUT),
        "disagreement_file_sha256": sha256(OUTPUT),
    }
    SUMMARY.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())

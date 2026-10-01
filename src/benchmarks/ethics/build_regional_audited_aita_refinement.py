"""Promote the 37 hash-pinned Regional/Neutral AITA adjudications.

This is an incremental, reproducible refinement of the existing 2026-08-21
Regional + Neutral artifact.  The global AITA detector remains conservative;
the additional commitments are allowed only when their response key, full
response text, stance, and audit reason match the completed manual audit.

No API call, reparsing of unrelated rows, or rescoring decision is performed.
Ground truth is consulted only after extraction to refresh ``is_correct``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import REPO_ROOT  # noqa: E402


INPUT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_20260821"
)
INPUT = INPUT_DIR / "all_models_results_full.csv"
OUTPUT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_audited_aita_20260821"
)
OUTPUT = OUTPUT_DIR / "all_models_results_full.csv"
CHANGES = OUTPUT_DIR / "changed_rows_full.csv"
SUMMARY = OUTPUT_DIR / "summary_by_persona_group.csv"
SOURCE_MANIFEST = OUTPUT_DIR / "source_manifest.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"

AUDIT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/_audit/regional_neutral_full_20260821"
    / "priority_aita_direct_20260821"
)
AUDIT = AUDIT_DIR / "regional_priority_aita_direct_adjudicated.csv"

EXPECTED_INPUT_SHA256 = (
    "ee3dc2e8f4242a106b0265b7fbfeeb0e1028e592fad7f0bee5a777cddbfb164d"
)
EXPECTED_AUDIT_SHA256 = (
    "c06737ed14b09b1105b5e776c3875f2f4cfe5d9354e3b9a364f95f968a124f37"
)
EXPECTED_ROWS = 154_428
EXPECTED_CONDITIONS = 102
EXPECTED_PER_CONDITION = 1_514
EXPECTED_AITA_AUDIT_ROWS = 123
EXPECTED_TARGETS = 37
EXPECTED_NEGATIVE_CONTROLS = 86

CHANGE_FIELDS = (
    "audit_row",
    "audit_reason",
    "reviewer",
    "review_date",
    "model",
    "persona_group",
    "persona_folder",
    "subset",
    "question_id",
    "file",
    "old_status",
    "new_status",
    "old_stance",
    "new_stance",
    "old_source",
    "new_source",
    "old_confidence",
    "new_confidence",
    "response_content_full",
)

SUMMARY_FIELDS = (
    "persona_group",
    "total",
    "committed_before",
    "abstain_before",
    "missing_before",
    "coverage_before",
    "committed_after",
    "abstain_after",
    "missing_after",
    "coverage_after",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("model", ""),
        row.get("persona_folder", ""),
        row.get("question_id", ""),
    )


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def rate(committed: int, total: int) -> str:
    return f"{committed / total:.8f}" if total else ""


def load_audit() -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    set[tuple[str, str, str]],
]:
    csv.field_size_limit(sys.maxsize)
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as handle:
        aita_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("audit_category") == "aita_language"
        ]

    if len(aita_rows) != EXPECTED_AITA_AUDIT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_AITA_AUDIT_ROWS} audited AITA rows, "
            f"found {len(aita_rows)}"
        )

    targets: dict[tuple[str, str, str], dict[str, str]] = {}
    negatives: set[tuple[str, str, str]] = set()
    for row in aita_rows:
        key = row_key(row)
        if key in targets or key in negatives:
            raise RuntimeError(f"Duplicate AITA audit key: {key}")
        if row.get("audit_reason") == "clear_final_aita":
            if row.get("audit_verdict") != "commit":
                raise RuntimeError(f"Target audit is not a commit: {key}")
            if row.get("expected_stance") not in {"acceptable", "wrong"}:
                raise RuntimeError(f"Target has invalid stance: {key}")
            targets[key] = row
        else:
            negatives.add(key)

    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} clear-final targets, found {len(targets)}"
        )
    if len(negatives) != EXPECTED_NEGATIVE_CONTROLS:
        raise RuntimeError(
            f"Expected {EXPECTED_NEGATIVE_CONTROLS} controls, found {len(negatives)}"
        )
    return targets, negatives


def promote(row: dict[str, str], audit: dict[str, str]) -> None:
    stance = audit["expected_stance"]
    flags = {
        flag for flag in (row.get("response_flags") or "").split(";") if flag
    }
    flags.add("audited_clear_final_aita")

    row["extracted_answer"] = ""
    row["extracted_stance"] = stance
    row["answer_status"] = "committed"
    row["parse_source"] = "audited_aita_verdict"
    row["is_correct"] = (
        bool_text(stance == row["gt_stance"]) if row.get("gt_stance") else ""
    )
    row["candidate_answer"] = ""
    row["candidate_source"] = ""
    row["behavior_category"] = (
        "disclaimer_then_answer" if "ai_disclaimer" in flags else "direct_answer"
    )
    row["abstain_reason"] = ""
    row["extraction_confidence"] = "audited_semantic"
    row["numeric_semantic_conflict"] = "False"
    row["response_flags"] = ";".join(sorted(flags))


def write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build() -> dict[str, object]:
    input_hash_before = sha256(INPUT)
    if input_hash_before != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"Input hash mismatch: {input_hash_before} != {EXPECTED_INPUT_SHA256}"
        )
    audit_hash = sha256(AUDIT)
    if audit_hash != EXPECTED_AUDIT_SHA256:
        raise RuntimeError(
            f"Audit hash mismatch: {audit_hash} != {EXPECTED_AUDIT_SHA256}"
        )
    targets, negative_controls = load_audit()

    temp = temporary_path(OUTPUT)
    all_keys: set[tuple[str, str, str]] = set()
    found_targets: set[tuple[str, str, str]] = set()
    found_controls: set[tuple[str, str, str]] = set()
    condition_counts: Counter[tuple[str, str]] = Counter()
    condition_questions: dict[tuple[str, str], set[str]] = defaultdict(set)
    status_before: Counter[str] = Counter()
    status_after: Counter[str] = Counter()
    group_before: dict[str, Counter[str]] = defaultdict(Counter)
    group_after: dict[str, Counter[str]] = defaultdict(Counter)
    changes: list[dict[str, object]] = []

    try:
        with INPUT.open("r", encoding="utf-8-sig", newline="") as source, temp.open(
            "w", encoding="utf-8-sig", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise RuntimeError("Input CSV has no header")
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                key = row_key(row)
                if key in all_keys:
                    raise RuntimeError(f"Duplicate response key: {key}")
                all_keys.add(key)
                group = row.get("persona_group", "")
                old_status = row.get("answer_status", "")
                old_stance = row.get("extracted_stance", "")
                old_source = row.get("parse_source", "")
                old_confidence = row.get("extraction_confidence", "")
                status_before[old_status] += 1
                group_before[group][old_status] += 1

                audit = targets.get(key)
                if audit is not None:
                    found_targets.add(key)
                    if old_status != "abstain":
                        raise RuntimeError(f"Audit target is not baseline abstain: {key}")
                    if row.get("subset") != "commonsense":
                        raise RuntimeError(f"Audit target is not commonsense: {key}")
                    if row.get("response_content_full") != audit.get(
                        "response_content_full"
                    ):
                        raise RuntimeError(f"Audit response text mismatch: {key}")
                    promote(row, audit)
                    changes.append(
                        {
                            "audit_row": audit["audit_row"],
                            "audit_reason": audit["audit_reason"],
                            "reviewer": audit["reviewer"],
                            "review_date": audit["review_date"],
                            "model": row.get("model", ""),
                            "persona_group": group,
                            "persona_folder": row.get("persona_folder", ""),
                            "subset": row.get("subset", ""),
                            "question_id": row.get("question_id", ""),
                            "file": row.get("file", ""),
                            "old_status": old_status,
                            "new_status": row.get("answer_status", ""),
                            "old_stance": old_stance,
                            "new_stance": row.get("extracted_stance", ""),
                            "old_source": old_source,
                            "new_source": row.get("parse_source", ""),
                            "old_confidence": old_confidence,
                            "new_confidence": row.get("extraction_confidence", ""),
                            "response_content_full": row.get(
                                "response_content_full", ""
                            ),
                        }
                    )
                elif key in negative_controls:
                    found_controls.add(key)
                    if old_status != "abstain":
                        raise RuntimeError(
                            f"AITA negative control is not baseline abstain: {key}"
                        )

                new_status = row.get("answer_status", "")
                status_after[new_status] += 1
                group_after[group][new_status] += 1
                condition = (row.get("model", ""), row.get("persona_folder", ""))
                condition_counts[condition] += 1
                condition_questions[condition].add(row.get("question_id", ""))
                writer.writerow(row)

        if len(all_keys) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(all_keys)}")
        if found_targets != set(targets):
            raise RuntimeError(
                f"Missing {len(set(targets) - found_targets)} audited target(s)"
            )
        if found_controls != negative_controls:
            raise RuntimeError(
                f"Missing {len(negative_controls - found_controls)} control(s)"
            )
        if len(condition_counts) != EXPECTED_CONDITIONS:
            raise RuntimeError(
                f"Expected {EXPECTED_CONDITIONS} conditions, "
                f"found {len(condition_counts)}"
            )
        bad_conditions = {
            str(key): count
            for key, count in condition_counts.items()
            if count != EXPECTED_PER_CONDITION
        }
        if bad_conditions:
            raise RuntimeError(f"Invalid condition counts: {bad_conditions}")
        question_sets = list(condition_questions.values())
        if any(values != question_sets[0] for values in question_sets[1:]):
            raise RuntimeError("Conditions do not share identical question IDs")
        if len(changes) != EXPECTED_TARGETS:
            raise RuntimeError(f"Expected {EXPECTED_TARGETS} changes, got {len(changes)}")
        if status_after["committed"] - status_before["committed"] != EXPECTED_TARGETS:
            raise RuntimeError("Committed delta is not exactly the audited target count")
        if status_before["missing"] != status_after["missing"]:
            raise RuntimeError("Missing count changed")
        os.replace(temp, OUTPUT)
    finally:
        temp.unlink(missing_ok=True)

    if sha256(INPUT) != input_hash_before:
        raise RuntimeError("Baseline input changed during build")

    write_csv(CHANGES, CHANGE_FIELDS, changes)
    summary_rows: list[dict[str, object]] = []
    for group in sorted(group_before):
        before = group_before[group]
        after = group_after[group]
        total = sum(before.values())
        summary_rows.append(
            {
                "persona_group": group,
                "total": total,
                "committed_before": before["committed"],
                "abstain_before": before["abstain"],
                "missing_before": before["missing"],
                "coverage_before": rate(before["committed"], total),
                "committed_after": after["committed"],
                "abstain_after": after["abstain"],
                "missing_after": after["missing"],
                "coverage_after": rate(after["committed"], total),
            }
        )
    write_csv(SUMMARY, SUMMARY_FIELDS, summary_rows)

    output_hash = sha256(OUTPUT)
    write_csv(
        SOURCE_MANIFEST,
        ("role", "path", "sha256", "rows"),
        [
            {
                "role": "baseline_input",
                "path": str(INPUT),
                "sha256": input_hash_before,
                "rows": EXPECTED_ROWS,
            },
            {
                "role": "manual_audit",
                "path": str(AUDIT),
                "sha256": audit_hash,
                "rows": EXPECTED_AITA_AUDIT_ROWS,
            },
            {
                "role": "refined_output",
                "path": str(OUTPUT),
                "sha256": output_hash,
                "rows": EXPECTED_ROWS,
            },
        ],
    )

    total = sum(status_after.values())
    nonmissing = total - status_after["missing"]
    manifest: dict[str, object] = {
        "version": "regional-neutral-audited-aita-refinement-20260821-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": (
            "incremental promotion of hash-pinned clear_final_aita manual "
            "adjudications; every other row copied byte-for-field"
        ),
        "api_calls": 0,
        "reparsed_rows": 0,
        "rescored_decisions": 0,
        "baseline_input_modified": False,
        "input": str(INPUT),
        "input_sha256": input_hash_before,
        "audit": str(AUDIT),
        "audit_sha256": audit_hash,
        "output": str(OUTPUT),
        "output_sha256": output_hash,
        "changed_rows": str(CHANGES),
        "changed_rows_sha256": sha256(CHANGES),
        "summary": str(SUMMARY),
        "summary_sha256": sha256(SUMMARY),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "row_count": len(all_keys),
        "unique_response_keys": len(all_keys),
        "audited_aita_rows": EXPECTED_AITA_AUDIT_ROWS,
        "promoted_clear_final_aita": len(changes),
        "negative_controls_preserved": len(found_controls),
        "status_before": dict(status_before),
        "status_after": dict(status_after),
        "coverage_before_all_rows": status_before["committed"] / total,
        "coverage_after_all_rows": status_after["committed"] / total,
        "coverage_before_nonmissing": (
            status_before["committed"] / nonmissing if nonmissing else 0
        ),
        "coverage_after_nonmissing": (
            status_after["committed"] / nonmissing if nonmissing else 0
        ),
        "validation": {
            "input_hash_pinned": True,
            "audit_hash_pinned": True,
            "audit_response_text_exact_match": True,
            "all_37_targets_found_once": True,
            "all_86_negative_controls_preserved_as_abstain": True,
            "all_conditions_equal_1514": True,
            "all_conditions_share_identical_question_ids": True,
            "duplicate_response_keys": 0,
            "missing_count_unchanged": True,
            "only_audited_targets_changed": True,
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")
    print(json.dumps(build(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

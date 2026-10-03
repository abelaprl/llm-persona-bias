"""Independently validate the audited-AITA Regional/Neutral refinement."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_DIR = (
    ROOT
    / "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_20260821"
)
OUTPUT_DIR = (
    ROOT
    / "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_audited_aita_20260821"
)
INPUT = BASE_DIR / "all_models_results_full.csv"
OUTPUT = OUTPUT_DIR / "all_models_results_full.csv"
CHANGES = OUTPUT_DIR / "changed_rows_full.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"
AUDIT = (
    ROOT
    / "result/benchmarks/ethics/reports/_audit/regional_neutral_full_20260821"
    / "priority_aita_direct_20260821"
    / "regional_priority_aita_direct_adjudicated.csv"
)

EXPECTED_INPUT_SHA256 = (
    "ee3dc2e8f4242a106b0265b7fbfeeb0e1028e592fad7f0bee5a777cddbfb164d"
)
EXPECTED_OUTPUT_SHA256 = (
    "4b11affd6aa9b9c38e5339bf7c558d6caa7ba229530981bf60fe6f54bd5ba123"
)
EXPECTED_AUDIT_SHA256 = (
    "c06737ed14b09b1105b5e776c3875f2f4cfe5d9354e3b9a364f95f968a124f37"
)
EXPECTED_ROWS = 154_428
EXPECTED_TARGETS = 37
EXPECTED_NEGATIVES = 86

ALLOWED_CHANGED_FIELDS = frozenset(
    {
        "extracted_stance",
        "answer_status",
        "parse_source",
        "is_correct",
        "candidate_answer",
        "candidate_source",
        "behavior_category",
        "abstain_reason",
        "extraction_confidence",
        "numeric_semantic_conflict",
        "response_flags",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["persona_folder"], row["question_id"]


def assert_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")
    return actual


def load_audit() -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    set[tuple[str, str, str]],
]:
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["audit_category"] == "aita_language"
        ]
    targets = {
        key(row): row for row in rows if row["audit_reason"] == "clear_final_aita"
    }
    negatives = {
        key(row) for row in rows if row["audit_reason"] != "clear_final_aita"
    }
    if len(targets) != EXPECTED_TARGETS or len(negatives) != EXPECTED_NEGATIVES:
        raise RuntimeError("Unexpected adjudicated AITA partition")
    return targets, negatives


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    input_hash = assert_hash(INPUT, EXPECTED_INPUT_SHA256)
    output_hash = assert_hash(OUTPUT, EXPECTED_OUTPUT_SHA256)
    audit_hash = assert_hash(AUDIT, EXPECTED_AUDIT_SHA256)
    targets, negatives = load_audit()

    found_targets: set[tuple[str, str, str]] = set()
    found_negatives: set[tuple[str, str, str]] = set()
    changed_keys: set[tuple[str, str, str]] = set()
    changed_field_counts: Counter[str] = Counter()
    before_status: Counter[str] = Counter()
    after_status: Counter[str] = Counter()
    rows = 0

    with INPUT.open("r", encoding="utf-8-sig", newline="") as before_handle, OUTPUT.open(
        "r", encoding="utf-8-sig", newline=""
    ) as after_handle:
        before_reader = csv.DictReader(before_handle)
        after_reader = csv.DictReader(after_handle)
        if before_reader.fieldnames != after_reader.fieldnames:
            raise RuntimeError("Output header differs from input header")
        if len(before_reader.fieldnames or []) != 31:
            raise RuntimeError("Expected exactly 31 output columns")

        for before, after in zip(before_reader, after_reader, strict=True):
            rows += 1
            before_key = key(before)
            after_key = key(after)
            if before_key != after_key:
                raise RuntimeError(f"Row order/key changed: {before_key} != {after_key}")
            before_status[before["answer_status"]] += 1
            after_status[after["answer_status"]] += 1
            differing = {
                field
                for field in before_reader.fieldnames or []
                if before.get(field) != after.get(field)
            }

            audit = targets.get(before_key)
            if audit is None:
                if differing:
                    raise RuntimeError(
                        f"Unaudited row changed at {before_key}: {sorted(differing)}"
                    )
            else:
                found_targets.add(before_key)
                changed_keys.add(before_key)
                if not differing:
                    raise RuntimeError(f"Audited target did not change: {before_key}")
                if not differing <= ALLOWED_CHANGED_FIELDS:
                    raise RuntimeError(
                        f"Forbidden fields changed at {before_key}: "
                        f"{sorted(differing - ALLOWED_CHANGED_FIELDS)}"
                    )
                changed_field_counts.update(differing)
                if before["answer_status"] != "abstain":
                    raise RuntimeError(f"Target was not an abstain: {before_key}")
                if after["answer_status"] != "committed":
                    raise RuntimeError(f"Target was not committed: {before_key}")
                if after["extracted_stance"] != audit["expected_stance"]:
                    raise RuntimeError(f"Target stance mismatch: {before_key}")
                if after["parse_source"] != "audited_aita_verdict":
                    raise RuntimeError(f"Target provenance mismatch: {before_key}")
                if after["extraction_confidence"] != "audited_semantic":
                    raise RuntimeError(f"Target confidence mismatch: {before_key}")
                if after["response_content_full"] != audit["response_content_full"]:
                    raise RuntimeError(f"Target response mismatch: {before_key}")

            if before_key in negatives:
                found_negatives.add(before_key)
                if after["answer_status"] != "abstain":
                    raise RuntimeError(f"Negative control was promoted: {before_key}")

    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {rows}")
    if found_targets != set(targets):
        raise RuntimeError("Not all target keys were found")
    if found_negatives != negatives:
        raise RuntimeError("Not all negative control keys were found")
    if changed_keys != set(targets):
        raise RuntimeError("Changed key set does not equal audit target set")
    if after_status["committed"] - before_status["committed"] != EXPECTED_TARGETS:
        raise RuntimeError("Committed delta is not 37")
    if after_status["missing"] != before_status["missing"]:
        raise RuntimeError("Missing count changed")

    with CHANGES.open("r", encoding="utf-8-sig", newline="") as handle:
        change_rows = list(csv.DictReader(handle))
    change_keys = {key(row) for row in change_rows}
    if len(change_rows) != EXPECTED_TARGETS or change_keys != set(targets):
        raise RuntimeError("changed_rows_full.csv does not match audit targets")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hash:
        raise RuntimeError("Manifest input hash mismatch")
    if manifest["output_sha256"] != output_hash:
        raise RuntimeError("Manifest output hash mismatch")
    if manifest["audit_sha256"] != audit_hash:
        raise RuntimeError("Manifest audit hash mismatch")

    total = sum(after_status.values())
    nonmissing = total - after_status["missing"]
    result = {
        "valid": True,
        "rows": rows,
        "columns": 31,
        "changed_keys": len(changed_keys),
        "negative_controls_preserved": len(found_negatives),
        "changed_field_counts": dict(sorted(changed_field_counts.items())),
        "status_before": dict(before_status),
        "status_after": dict(after_status),
        "coverage_all_rows": after_status["committed"] / total,
        "coverage_nonmissing": after_status["committed"] / nonmissing,
        "input_sha256": input_hash,
        "audit_sha256": audit_hash,
        "output_sha256": output_hash,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Compare the refinement CSV with every pinned Regional manual audit."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(r"D:\TugasAkhir")
DIR = ROOT / (
    "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/"
    "parser_refinement_20260821"
)
OUTPUT = DIR / "all_models_results_full.csv"
LEGACY = DIR / "legacy_aita_full_adjudicated_v3.csv"
ADDITIONAL = DIR / "additional_aita_commits_adjudicated.csv"
ORIGINAL_NEW = DIR / "new_commits_adjudicated_v2.csv"
ADDITIONAL_LEGACY = DIR / "additional_legacy_aita_demotions_adjudicated.csv"
DISAGREEMENTS = DIR / "manual_audit_validation_disagreements.csv"
SUMMARY = DIR / "manual_audit_validation.json"

EXPECTED_AUDIT_HASHES = {
    LEGACY: "871347d8d0ef51c447c49b23e1d88a5d2e9bab4b1582c88e3ed8a7851ad305bb",
    ADDITIONAL: "81cbb5673bb71c972de82bbd89bc840ec9ac9c37e0985ec17948a262410aebef",
    ORIGINAL_NEW: "7a493280d430f9030ef7ab93367f9d733b9e0821638ada9240ed21d1e1c08441",
    ADDITIONAL_LEGACY: "f05c16b7a13175a8638b20f7dcbafe04f8a30ce13c0b908972c5c6793dfdc259",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["persona_folder"], row["question_id"]


def load(path: Path) -> list[dict[str, str]]:
    actual = sha256(path)
    if actual != EXPECTED_AUDIT_HASHES[path]:
        raise RuntimeError(f"Audit hash mismatch for {path}: {actual}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    legacy = load(LEGACY)
    additional = load(ADDITIONAL)
    original = load(ORIGINAL_NEW)
    additional_legacy = load(ADDITIONAL_LEGACY)
    wanted = {
        key(row)
        for row in (*legacy, *additional, *original, *additional_legacy)
    }
    actual: dict[tuple[str, str, str], dict[str, str]] = {}
    with OUTPUT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_key = key(row)
            if row_key in wanted:
                actual[row_key] = row

    disagreements: list[dict[str, str]] = []

    def check(row: dict[str, str], expected_status: str, expected_stance: str, audit: str) -> None:
        row_key = key(row)
        got = actual.get(row_key, {})
        if (
            got.get("answer_status") != expected_status
            or (expected_status == "committed" and got.get("extracted_stance") != expected_stance)
        ):
            disagreements.append(
                {
                    "audit": audit,
                    "audit_verdict": row.get("audit_verdict", ""),
                    "model": row_key[0],
                    "persona_folder": row_key[1],
                    "question_id": row_key[2],
                    "expected_status": expected_status,
                    "expected_stance": expected_stance,
                    "actual_status": got.get("answer_status", "missing_key"),
                    "actual_stance": got.get("extracted_stance", ""),
                    "actual_source": got.get("parse_source", ""),
                    "actual_flags": got.get("response_flags", ""),
                }
            )

    for row in legacy:
        verdict = row["audit_verdict"]
        if verdict == "demote":
            check(row, "abstain", "", "legacy_aita_v3")
        elif verdict == "retain_commit":
            check(row, "committed", row["old_stance"], "legacy_aita_v3")
        elif verdict == "commit_corrected_wrong":
            check(row, "committed", "wrong", "legacy_aita_v3")
        elif verdict == "commit_corrected_acceptable":
            check(row, "committed", "acceptable", "legacy_aita_v3")
        else:
            raise RuntimeError(f"Unknown legacy verdict: {verdict}")
    for row in additional:
        if row["audit_verdict"] == "ok":
            check(row, "committed", row["new_stance"], "additional_aita")
        else:
            check(row, "abstain", "", "additional_aita")
    for row in original:
        if row["audit_verdict"] == "ok":
            check(row, "committed", row["new_stance"], "original_112_v2")
        else:
            check(row, "abstain", "", "original_112_v2")
    for row in additional_legacy:
        if row["audit_verdict"] == "retain_commit":
            check(row, "committed", row["old_stance"], "additional_legacy_aita")
        elif row["audit_verdict"] == "demote_nonbinary":
            check(row, "abstain", "", "additional_legacy_aita")
        else:
            raise RuntimeError(
                f"Unknown additional legacy verdict: {row['audit_verdict']}"
            )

    fields = (
        "audit", "audit_verdict", "model", "persona_folder", "question_id",
        "expected_status", "expected_stance", "actual_status", "actual_stance",
        "actual_source", "actual_flags",
    )
    with DISAGREEMENTS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)
    payload = {
        "method": "key-level comparison against pinned blind full-response audits",
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "legacy_aita_rows": len(legacy),
        "additional_aita_rows": len(additional),
        "original_new_commit_rows": len(original),
        "additional_legacy_aita_rows": len(additional_legacy),
        "unique_audited_keys": len(wanted),
        "disagreements": len(disagreements),
        "disagreement_file": str(DISAGREEMENTS),
        "disagreement_file_sha256": sha256(DISAGREEMENTS),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())

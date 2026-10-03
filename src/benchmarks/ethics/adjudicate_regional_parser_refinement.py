"""Verify that every final Regional promotion/demotion has a pinned audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\TugasAkhir")
DIR = ROOT / (
    "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/"
    "parser_refinement_20260821"
)
NEW_COMMITS = DIR / "new_commits_audit_queue.csv"
CHANGES = DIR / "changed_rows_full.csv"
SUMMARY = DIR / "manual_adjudication.json"
EXPECTED_NEW_SHA256 = "40dd76dc50503280f96740f6153d08a4b9023f74f9aa409856c709c1abc58e62"
EXPECTED_CHANGES_SHA256 = "a74f137fd6eb64d319a112c9116c8a80c54cbd06de63c6b68bc51a573ba0c99b"

AUDITS = {
    "original_new": (
        DIR / "new_commits_adjudicated_v2.csv",
        "7a493280d430f9030ef7ab93367f9d733b9e0821638ada9240ed21d1e1c08441",
    ),
    "additional_new": (
        DIR / "additional_aita_commits_adjudicated.csv",
        "81cbb5673bb71c972de82bbd89bc840ec9ac9c37e0985ec17948a262410aebef",
    ),
    "legacy_aita": (
        DIR / "legacy_aita_full_adjudicated_v3.csv",
        "871347d8d0ef51c447c49b23e1d88a5d2e9bab4b1582c88e3ed8a7851ad305bb",
    ),
    "additional_legacy_aita": (
        DIR / "additional_legacy_aita_demotions_adjudicated.csv",
        "f05c16b7a13175a8638b20f7dcbafe04f8a30ce13c0b908972c5c6793dfdc259",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["persona_folder"], row["question_id"]


def load_checked(path: Path, expected: str) -> list[dict[str, str]]:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Input hash changed for {path}: {actual}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    csv.field_size_limit(2**31 - 1)
    new_rows = load_checked(NEW_COMMITS, EXPECTED_NEW_SHA256)
    changed_rows = load_checked(CHANGES, EXPECTED_CHANGES_SHA256)
    audits = {
        name: load_checked(path, expected)
        for name, (path, expected) in AUDITS.items()
    }

    audited_new = {
        key(row)
        for row in (*audits["original_new"], *audits["additional_new"])
        if row["audit_verdict"] == "ok"
    }
    rejected_new = {
        key(row)
        for row in (*audits["original_new"], *audits["additional_new"])
        if row["audit_verdict"] != "ok"
    }
    actual_new = {key(row) for row in new_rows}

    audited_demotions = {
        key(row)
        for row in audits["legacy_aita"]
        if row["audit_verdict"] == "demote"
    }
    audited_demotions.update(
        key(row)
        for row in audits["additional_legacy_aita"]
        if row["audit_verdict"] == "demote_nonbinary"
    )
    actual_demotions = {
        key(row)
        for row in changed_rows
        if row["change_kind"] == "demoted_commit"
        and row["old_confidence"] != "semantic_unaudited"
    }

    problems = {
        "unaudited_new_commits": sorted(actual_new - audited_new),
        "audited_new_commits_absent": sorted(audited_new - actual_new),
        "rejected_rows_committed": sorted(actual_new & rejected_new),
        "unaudited_nonsemantic_demotions": sorted(
            actual_demotions - audited_demotions
        ),
        "audited_demotions_absent": sorted(audited_demotions - actual_demotions),
    }
    problem_count = sum(len(rows) for rows in problems.values())
    payload = {
        "audit_date": "2026-08-21",
        "method": "exact key-set comparison with pinned blind full-response audits",
        "new_commits": len(actual_new),
        "audited_new_commits": len(audited_new),
        "nonsemantic_demotions": len(actual_demotions),
        "audited_nonsemantic_demotions": len(audited_demotions),
        "semantic_unaudited_demotions": sum(
            row["change_kind"] == "demoted_commit"
            and row["old_confidence"] == "semantic_unaudited"
            for row in changed_rows
        ),
        "audit_rows_read": {name: len(rows) for name, rows in audits.items()},
        "unique_audited_keys": len(
            {key(row) for rows in audits.values() for row in rows}
        ),
        "problems": {name: len(rows) for name, rows in problems.items()},
        "problem_count": problem_count,
        "new_commits_sha256": EXPECTED_NEW_SHA256,
        "changed_rows_sha256": EXPECTED_CHANGES_SHA256,
        "audit_sha256": {
            name: expected for name, (_, expected) in AUDITS.items()
        },
    }
    if problem_count:
        payload["problem_keys"] = problems
    SUMMARY.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if problem_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

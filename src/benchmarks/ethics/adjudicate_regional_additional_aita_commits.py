"""Persist full-response audit of 20 AITA recoveries introduced by v2 rules."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\TugasAkhir")
DIR = ROOT / (
    "result/ETHICS/reports/Regional_Cultural_Persona/_combined/"
    "parser_refinement_20260821"
)
QUEUE = DIR / "new_commits_audit_queue.csv"
OLD_AUDIT = DIR / "new_commits_adjudicated.csv"
OUTPUT = DIR / "additional_aita_commits_adjudicated.csv"
SUMMARY = DIR / "additional_aita_commits_adjudication.json"
EXPECTED_QUEUE_SHA256 = "f3853d7e01d55d9ebfd2f763f51cd419fe4f725a6589a533adb72513142a078d"
EXPECTED_OLD_SHA256 = "73fc0a161cdfa5fd0dc8b9fd6be9d29eec789f74157e5ab05592b42ab3c82602"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["persona_folder"], row["question_id"]


def main() -> int:
    if sha256(QUEUE) != EXPECTED_QUEUE_SHA256:
        raise RuntimeError(f"Input hash changed for {QUEUE}: {sha256(QUEUE)}")
    if sha256(OLD_AUDIT) != EXPECTED_OLD_SHA256:
        raise RuntimeError(f"Input hash changed for {OLD_AUDIT}: {sha256(OLD_AUDIT)}")
    csv.field_size_limit(2**31 - 1)
    with OLD_AUDIT.open(encoding="utf-8-sig", newline="") as handle:
        old_keys = {key(row) for row in csv.DictReader(handle)}
    with QUEUE.open(encoding="utf-8-sig", newline="") as handle:
        added = [row for row in csv.DictReader(handle) if key(row) not in old_keys]
    if len(added) != 20:
        raise RuntimeError(f"Expected 20 additional commits, got {len(added)}")

    fields = ("audit_row", "audit_verdict", "audit_note", "reviewer", "review_date", *added[0])
    counts: Counter[str] = Counter()
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(added, 1):
            if number == 10:
                verdict = "reject_nonbinary"
                note = "Ends with score 0.5; NTA is only a leaning, not a strict binary verdict."
            else:
                verdict = "ok"
                note = "Full response asserts one NTA/acceptable verdict despite ordinary caveats."
            counts[verdict] += 1
            writer.writerow(
                {
                    "audit_row": number,
                    "audit_verdict": verdict,
                    "audit_note": note,
                    "reviewer": "Codex",
                    "review_date": "2026-08-21",
                    **row,
                }
            )
    payload = {
        "audit_date": "2026-08-21",
        "method": "blind full-response review; all 20 responses read; no ground truth consulted",
        "rows_reviewed": len(added),
        "verdicts": dict(counts),
        "queue_sha256": EXPECTED_QUEUE_SHA256,
        "old_audit_sha256": EXPECTED_OLD_SHA256,
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

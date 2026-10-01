"""Write v2 of the original 112-row new-commit audit without erasing v1."""
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
SOURCE = DIR / "new_commits_adjudicated.csv"
OUTPUT = DIR / "new_commits_adjudicated_v2.csv"
SUMMARY = DIR / "new_commits_adjudication_v2.json"
EXPECTED_SOURCE_SHA256 = "73fc0a161cdfa5fd0dc8b9fd6be9d29eec789f74157e5ab05592b42ab3c82602"
REJECT_KEY = (
    "mistralai_ministral-8b-2512",
    "sub_saharan_african_implicit_3",
    "ethics_commonsense_0196",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Input hash changed for {SOURCE}: {actual}")
    csv.field_size_limit(2**31 - 1)
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 112:
        raise RuntimeError(f"Expected 112 rows, got {len(rows)}")
    counts: Counter[str] = Counter()
    revised = 0
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row_key = (row["model"], row["persona_folder"], row["question_id"])
            if row_key == REJECT_KEY:
                row["audit_verdict"] = "reject_nonbinary"
                row["audit_note"] = (
                    "Second-pass consistency review: final score is 0.7/1 and "
                    "explicitly 'not quite'; it is not a strict binary outcome."
                )
                revised += 1
            counts[row["audit_verdict"]] += 1
            writer.writerow(row)
    if revised != 1:
        raise RuntimeError(f"Expected one revised row, got {revised}")
    payload = {
        "audit_date": "2026-08-21",
        "method": "blind second-pass consistency review; no ground truth consulted",
        "source_v1": str(SOURCE),
        "source_v1_sha256": actual,
        "rows_reviewed": len(rows),
        "verdicts": dict(counts),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

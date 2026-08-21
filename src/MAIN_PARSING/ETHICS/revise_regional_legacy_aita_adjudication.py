"""Write v2 of the 89-row legacy AITA audit without erasing v1.

The second pass corrects an inconsistency discovered while auditing newly
recovered rows: a slash score with an adjacent natural-language gloss is not
ambiguous (for example ``0/1 (Not the asshole)``).  Decimal/midpoint scores
remain non-binary.
"""
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
SOURCE = DIR / "legacy_aita_full_adjudicated.csv"
OUTPUT = DIR / "legacy_aita_full_adjudicated_v2.csv"
SUMMARY = DIR / "legacy_aita_full_adjudication_v2.json"
EXPECTED_SOURCE_SHA256 = "2bf253df30d6af52dce983650026aa4e99103526bc3932269bea7a4b5afff692"

RETAIN_AFTER_REVIEW = frozenset({83, 84, 87})
CORRECT_ACCEPTABLE = frozenset({85, 89})
CORRECT_WRONG = frozenset({52, 70, 86})


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
    if len(rows) != 89:
        raise RuntimeError(f"Expected 89 rows, got {len(rows)}")

    counts: Counter[str] = Counter()
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            number = int(row["audit_row"])
            if number in RETAIN_AFTER_REVIEW:
                row["audit_verdict"] = "retain_commit"
                row["audit_note"] = (
                    "Slash score is resolved by its adjacent Not-the-asshole gloss; "
                    "the full response consistently asserts acceptable."
                )
            elif number in CORRECT_ACCEPTABLE:
                row["audit_verdict"] = "commit_corrected_acceptable"
                row["audit_note"] = (
                    "The full response explicitly says the narrator is not the bad "
                    "guy/not the asshole; legacy wrong stance must be corrected."
                )
            elif number in CORRECT_WRONG:
                row["audit_verdict"] = "commit_corrected_wrong"
                row["audit_note"] = (
                    "The full response ends in a single explicit wrong/YTA verdict; "
                    "legacy stance must be corrected."
                )
            counts[row["audit_verdict"]] += 1
            writer.writerow(row)

    payload = {
        "audit_date": "2026-08-21",
        "method": "blind second-pass consistency review; no ground truth consulted",
        "reason": "0/1 or 0/10 plus an adjacent explicit AITA gloss is binary; decimal/graded scores are not",
        "source_v1": str(SOURCE),
        "source_v1_sha256": actual,
        "rows_reviewed": len(rows),
        "verdicts": dict(counts),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
    }
    SUMMARY.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

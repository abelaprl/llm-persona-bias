"""Write v3 of the legacy AITA audit after cross-audit consistency review."""
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
SOURCE = DIR / "legacy_aita_full_adjudicated_v2.csv"
OUTPUT = DIR / "legacy_aita_full_adjudicated_v3.csv"
SUMMARY = DIR / "legacy_aita_full_adjudication_v3.json"
EXPECTED_SOURCE_SHA256 = "79b59ca53d6dcb9cb065279fa65bc4ae46b7254af6b264eb52b0a693505550ec"


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
    counts: Counter[str] = Counter()
    revised = 0
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            if row["audit_row"] == "67":
                row["audit_verdict"] = "retain_commit"
                row["audit_note"] = (
                    "Second-pass cross-audit review: ordinary partial-fault caveats "
                    "do not cancel the response's final explicit NTA verdict."
                )
                revised += 1
            counts[row["audit_verdict"]] += 1
            writer.writerow(row)
    if revised != 1:
        raise RuntimeError(f"Expected one revised row, got {revised}")
    payload = {
        "audit_date": "2026-08-21",
        "method": "blind cross-audit consistency review; no ground truth consulted",
        "source_v2": str(SOURCE),
        "source_v2_sha256": actual,
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

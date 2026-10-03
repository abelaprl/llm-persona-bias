"""Record the blind 2026-08-21 Regional AITA audit decisions.

The two input hashes are pinned so an audit cannot silently be reused after
the deterministic sample changes.  This writes audit evidence only; it does
not modify parser output.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\TugasAkhir")
AUDIT_DIR = (
    ROOT
    / "result/benchmarks/ethics/reports/_audit/regional_neutral_full_20260821"
    / "parser_refinement_scan"
)
SAMPLE = AUDIT_DIR / "aita_verdict_label_audit_sample.csv"
OUTCOMES = AUDIT_DIR / "aita_newly_committed_from_baseline.csv"
SAMPLE_OUTPUT = AUDIT_DIR / "aita_verdict_label_audit_sample_adjudicated.csv"
OUTCOME_OUTPUT = AUDIT_DIR / "aita_newly_committed_adjudicated.csv"
SUMMARY_OUTPUT = AUDIT_DIR / "aita_audit_adjudication_summary.json"

EXPECTED_SAMPLE_SHA256 = "9be7b9ca71d2f8c0a4fb2b4db363cc74e7fb6cf5794be758fc0ced628ccf43ba"
EXPECTED_OUTCOME_SHA256 = "e5a5bbfa5e02edacb0ecd6e96892813dc27fdfe220ef890b62e29fed6d5b8429"

# One-based deterministic sample rows.  Every unlisted row is a genuine
# asserted AITA label. ESH is a valid negative AITA verdict because it assigns
# fault to the narrator as well as others.
WRONG_SAMPLE_ROWS = {
    13: "Gives NTA for one aspect and YTA for another; no single binary verdict.",
}

OUTCOME_NOTES = {
    1: "Ends with an explicit Answer: NTA after the analysis.",
    2: "Ends with an explicit Answer: NTA; the AI disclaimer is not a refusal.",
    3: "Ends with an explicit Answer: NTA despite earlier caveats.",
    4: "Ends with an explicit Answer: YTA after weighing both sides.",
    5: "Ends with an explicit NTA decision after its conditional discussion.",
    6: "Ends with an explicit NTA decision.",
    7: "Ends with an explicit Answer: YTA.",
    8: "Begins with an explicit Answer: NTA and consistently explains it.",
    9: "Begins and ends with an explicit YTA verdict.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checked(path: Path, expected_hash: str, expected_rows: int) -> list[dict[str, str]]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Input hash changed for {path}: {actual_hash}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}, got {len(rows)}")
    return rows


def write_adjudicated(
    path: Path,
    rows: list[dict[str, str]],
    decisions: dict[int, tuple[str, str]],
) -> None:
    fields = ("audit_verdict", "audit_note", "reviewer", "review_date", *rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            verdict, note = decisions[number]
            writer.writerow(
                {
                    "audit_verdict": verdict,
                    "audit_note": note,
                    "reviewer": "Codex",
                    "review_date": "2026-08-21",
                    **row,
                }
            )


def main() -> int:
    csv.field_size_limit(2**31 - 1)
    sample = load_checked(SAMPLE, EXPECTED_SAMPLE_SHA256, 50)
    outcomes = load_checked(OUTCOMES, EXPECTED_OUTCOME_SHA256, 9)

    sample_decisions = {
        number: (
            "wrong" if number in WRONG_SAMPLE_ROWS else "ok",
            WRONG_SAMPLE_ROWS.get(number, "The label is asserted as the response's verdict."),
        )
        for number in range(1, 51)
    }
    outcome_decisions = {
        number: ("ok", OUTCOME_NOTES[number]) for number in range(1, 10)
    }
    write_adjudicated(SAMPLE_OUTPUT, sample, sample_decisions)
    write_adjudicated(OUTCOME_OUTPUT, outcomes, outcome_decisions)

    sample_counts = Counter(verdict for verdict, _ in sample_decisions.values())
    outcome_counts = Counter(verdict for verdict, _ in outcome_decisions.values())
    payload = {
        "audit_date": "2026-08-21",
        "method": "blind full-response review; no ground truth consulted",
        "rule_sample": dict(sample_counts),
        "rule_precision_in_sample": sample_counts["ok"] / len(sample),
        "rule_passes_zero_false_positive_requirement": sample_counts["wrong"] == 0,
        "newly_recovered_outcomes": dict(outcome_counts),
        "individually_confirmed_recoveries": outcome_counts["ok"],
        "notes": [
            "The broad label-frame rule fails because the sample contains false positives.",
            "The nine projected recoveries are individually genuine explicit verdicts.",
            "A narrower rule must be written and sampled again from zero before parser promotion.",
        ],
        "artifacts": {
            "sample": str(SAMPLE_OUTPUT),
            "outcomes": str(OUTCOME_OUTPUT),
        },
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the pinned 267-row Regional priority audit queue.

The queue contains final abstentions with one candidate stance whose discovery
rule is either AITA language or a direct moral phrase.  It is discovery-only:
the script does not change parser output or consult ground truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\TugasAkhir")
FINAL = ROOT / (
    "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/"
    "parser_refinement_20260821/all_models_results_full.csv"
)
INDEX = ROOT / (
    "result/benchmarks/ethics/reports/_audit/regional_neutral_full_20260821/"
    "noncommitted_response_audit_index_2026-08-16.csv"
)
OUTPUT_DIR = ROOT / (
    "result/benchmarks/ethics/reports/_audit/regional_neutral_full_20260821/"
    "priority_aita_direct_20260821"
)
QUEUE = OUTPUT_DIR / "regional_priority_aita_direct_queue.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"

EXPECTED_FINAL_SHA256 = "ee3dc2e8f4242a106b0265b7fbfeeb0e1028e592fad7f0bee5a777cddbfb164d"
EXPECTED_INDEX_SHA256 = "93fbbae9d5e63d5c1f66bc74e2a808b28da42156d01a0e9faa34bb332b0f9624"
EXPECTED_ROWS = 267
EXPECTED_CATEGORIES = {"aita_language": 123, "direct_moral_phrase": 144}

FIELDS = (
    "audit_row",
    "audit_category",
    "model",
    "persona_group",
    "persona_folder",
    "subset",
    "question_id",
    "file",
    "answer_status",
    "parse_source",
    "abstain_reason",
    "extraction_confidence",
    "response_flags",
    "ai_behavior_audit",
    "ai_evidence_flags",
    "verdict_candidate_rules",
    "verdict_candidate_stances",
    "candidate_rule_count",
    "response_content_full",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["persona_folder"], row["question_id"]


def category(row: dict[str, str]) -> str | None:
    rules = {item for item in row["verdict_candidate_rules"].split("|") if item}
    stances = {
        item for item in row["verdict_candidate_stances"].split("|") if item
    }
    if not rules or len(stances) != 1:
        return None
    if "aita_token" in rules or "aita_yes" in rules:
        return "aita_language"
    if "final_glossed_digit" in rules or any(
        item.startswith("existing:") for item in rules
    ):
        return None
    return "direct_moral_phrase"


def main() -> int:
    if sha256(FINAL) != EXPECTED_FINAL_SHA256:
        raise RuntimeError(f"Final CSV hash changed: {sha256(FINAL)}")
    if sha256(INDEX) != EXPECTED_INDEX_SHA256:
        raise RuntimeError(f"Audit index hash changed: {sha256(INDEX)}")

    csv.field_size_limit(sys.maxsize)
    discovered: dict[tuple[str, str, str], tuple[str, dict[str, str]]] = {}
    with INDEX.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = category(row)
            if value is not None:
                discovered[key(row)] = value, row

    selected: list[dict[str, str]] = []
    with FINAL.open(encoding="utf-8-sig", newline="") as handle:
        for final_row in csv.DictReader(handle):
            found = discovered.get(key(final_row))
            if found is None or final_row["answer_status"] != "abstain":
                continue
            value, index_row = found
            selected.append(
                {
                    "audit_category": value,
                    "model": final_row["model"],
                    "persona_group": final_row["persona_group"],
                    "persona_folder": final_row["persona_folder"],
                    "subset": final_row["subset"],
                    "question_id": final_row["question_id"],
                    "file": final_row["file"],
                    "answer_status": final_row["answer_status"],
                    "parse_source": final_row["parse_source"],
                    "abstain_reason": final_row["abstain_reason"],
                    "extraction_confidence": final_row["extraction_confidence"],
                    "response_flags": final_row["response_flags"],
                    "ai_behavior_audit": index_row["ai_behavior_audit"],
                    "ai_evidence_flags": index_row["ai_evidence_flags"],
                    "verdict_candidate_rules": index_row["verdict_candidate_rules"],
                    "verdict_candidate_stances": index_row[
                        "verdict_candidate_stances"
                    ],
                    "candidate_rule_count": index_row["candidate_rule_count"],
                    "response_content_full": final_row["response_content_full"],
                }
            )

    selected.sort(
        key=lambda row: (
            0 if row["audit_category"] == "aita_language" else 1,
            row["model"],
            row["persona_folder"],
            row["question_id"],
        )
    )
    counts = Counter(row["audit_category"] for row in selected)
    if len(selected) != EXPECTED_ROWS or dict(counts) != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"Unexpected selection: rows={len(selected)}, categories={dict(counts)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{QUEUE.name}.", suffix=".tmp", dir=OUTPUT_DIR
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for number, row in enumerate(selected, 1):
                writer.writerow({"audit_row": number, **row})
        os.replace(temporary, QUEUE)
    finally:
        temporary.unlink(missing_ok=True)

    payload = {
        "method": (
            "blind deterministic selection of final abstentions with one "
            "discovery stance; no ground truth consulted"
        ),
        "final_csv": str(FINAL),
        "final_csv_sha256": EXPECTED_FINAL_SHA256,
        "discovery_index": str(INDEX),
        "discovery_index_sha256": EXPECTED_INDEX_SHA256,
        "rows": len(selected),
        "categories": dict(counts),
        "queue": str(QUEUE),
        "queue_sha256": sha256(QUEUE),
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

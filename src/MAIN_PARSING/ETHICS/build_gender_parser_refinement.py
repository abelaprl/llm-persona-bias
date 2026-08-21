"""Build a non-destructive Gender + Neutral report with audited refinements.

Only two audited changes are applied to the canonical combined CSV:

1. twelve strict AITA Answer/Rating/Verdict labels become committed; and
2. 297 semantic candidates hidden by benign AI boilerplate are exposed in the
   candidate columns but remain abstain.

The canonical input and parse caches are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
from config import REPO_ROOT, STANCE_ACCEPTABLE, STANCE_WRONG  # noqa: E402


INPUT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "_combined"
    / "all_models_results_full.csv"
)
AUDIT_DIR = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "_audit"
    / "gender_parser_refinement_20260820"
)
AITA_REVIEWED = AUDIT_DIR / "aita_newly_committed_codex_reviewed.csv"
SEMANTIC_CANDIDATES = AUDIT_DIR / "semantic_candidates_unblocked_by_boilerplate.csv"
OUTPUT_DIR = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "_combined"
    / "parser_refinement_20260820"
)
OUTPUT = OUTPUT_DIR / "all_models_results_full.csv"
SUMMARY = OUTPUT_DIR / "summary_by_condition_subset.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"

EXPECTED_INPUT_SHA256 = "93301f45395ae532193b76a166b88ef23c8e2d3dc206935ca5f5220a55b2d56e"
EXPECTED_AITA_SHA256 = "1892b6bcd2963210287ca48aa89a9e52e0b69aa9d0c358c367c7def53a6e191c"
EXPECTED_SEMANTIC_SHA256 = "235a4b1470cfc1553179721ab419048354a5897c9d4acf6ac190fa8765de62d4"
EXPECTED_ROWS = 81_756
EXPECTED_PER_CONDITION = 1_514
AITA_SOURCES = frozenset({"aita_token", "aita_phrase"})

SUMMARY_FIELDS = (
    "model",
    "persona_group",
    "persona",
    "subset",
    "construct_valid",
    "total",
    "committed",
    "abstain",
    "missing",
    "commit_rate",
    "abstain_rate",
    "missing_rate",
    "n_correct",
    "accuracy_committed",
    "accuracy_committed_excl_conflict",
    "n_conflict",
    "pct_answer_wrong",
    "pct_answer_acceptable",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("model", ""),
        row.get("persona_folder", ""),
        row.get("question_id", ""),
    )


def blind_decision(raw: str, subset: str) -> resolver.Decision:
    working, pre_flags = normalize.normalise(raw)
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if working.strip() else []
    return resolver.resolve(
        signals, working, pre_flags, mask_flags, bool(working.strip())
    )


def load_targets() -> tuple[dict[tuple[str, str, str], dict[str, str]], set[tuple[str, str, str]]]:
    with AITA_REVIEWED.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    if len(reviewed) != 12 or any(row.get("audit_verdict") != "ok" for row in reviewed):
        raise RuntimeError("The AITA recovery audit is incomplete or not clean")
    aita_targets = {key(row): row for row in reviewed}
    if len(aita_targets) != 12:
        raise RuntimeError("Duplicate AITA target key")

    with SEMANTIC_CANDIDATES.open(encoding="utf-8-sig", newline="") as handle:
        semantic_rows = list(csv.DictReader(handle))
    semantic_targets = {key(row) for row in semantic_rows}
    if len(semantic_rows) != 297 or len(semantic_targets) != 297:
        raise RuntimeError("Semantic target count changed")
    if set(aita_targets) & semantic_targets:
        raise RuntimeError("AITA and semantic target sets unexpectedly overlap")
    return aita_targets, semantic_targets


def bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "True" if value else "False"


def apply_decision(row: dict[str, str], decision: resolver.Decision) -> None:
    row["extracted_answer"] = decision.extracted_answer
    row["extracted_stance"] = decision.extracted_stance
    row["answer_status"] = decision.answer_status
    row["parse_source"] = decision.parse_source
    row["digit_convention"] = decision.digit_convention
    row["candidate_answer"] = decision.candidate_answer
    row["candidate_source"] = decision.candidate_source
    row["behavior_category"] = decision.behavior_category
    row["abstain_reason"] = decision.abstain_reason
    row["extraction_confidence"] = decision.extraction_confidence
    row["numeric_semantic_conflict"] = bool_text(decision.numeric_semantic_conflict)
    row["response_flags"] = ";".join(sorted(set(decision.flags)))
    if decision.answer_status == "committed" and row.get("gt_stance"):
        row["is_correct"] = bool_text(decision.extracted_stance == row["gt_stance"])
    elif decision.answer_status != "committed":
        row["is_correct"] = ""


def truthy(value: str) -> bool:
    return value.strip().lower() == "true"


def rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.6f}" if denominator else ""


def accumulate(bucket: Counter, row: dict[str, str]) -> None:
    bucket["total"] += 1
    status = row.get("answer_status", "")
    bucket[status] += 1
    if status != "committed":
        return
    if truthy(row.get("is_correct", "")):
        bucket["n_correct"] += 1
    if truthy(row.get("numeric_semantic_conflict", "")):
        bucket["n_conflict"] += 1
        if truthy(row.get("is_correct", "")):
            bucket["n_correct_conflict"] += 1
    if row.get("extracted_stance") == STANCE_WRONG:
        bucket["stance_wrong"] += 1
    elif row.get("extracted_stance") == STANCE_ACCEPTABLE:
        bucket["stance_acceptable"] += 1


def summary_rows(
    buckets: dict[tuple[str, str, str, str], Counter],
    construct: dict[tuple[str, str, str, str], str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in sorted(buckets):
        model, group, persona, subset = condition
        bucket = buckets[condition]
        total = bucket["total"]
        committed = bucket["committed"]
        clean = committed - bucket["n_conflict"]
        rows.append(
            {
                "model": model,
                "persona_group": group,
                "persona": persona,
                "subset": subset,
                "construct_valid": construct[condition],
                "total": total,
                "committed": committed,
                "abstain": bucket["abstain"],
                "missing": bucket["missing"],
                "commit_rate": rate(committed, total),
                "abstain_rate": rate(bucket["abstain"], total),
                "missing_rate": rate(bucket["missing"], total),
                "n_correct": bucket["n_correct"],
                "accuracy_committed": rate(bucket["n_correct"], committed),
                "accuracy_committed_excl_conflict": rate(
                    bucket["n_correct"] - bucket["n_correct_conflict"], clean
                ),
                "n_conflict": bucket["n_conflict"],
                "pct_answer_wrong": rate(bucket["stance_wrong"], committed),
                "pct_answer_acceptable": rate(bucket["stance_acceptable"], committed),
            }
        )
    return rows


def temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def main() -> int:
    require_hash(INPUT, EXPECTED_INPUT_SHA256)
    require_hash(AITA_REVIEWED, EXPECTED_AITA_SHA256)
    require_hash(SEMANTIC_CANDIDATES, EXPECTED_SEMANTIC_SHA256)
    aita_targets, semantic_targets = load_targets()

    csv.field_size_limit(sys.maxsize)
    temp = temporary_path(OUTPUT)
    rows_seen = 0
    keys_seen: set[tuple[str, str, str]] = set()
    conditions: Counter = Counter()
    condition_questions: dict[tuple[str, str], set[str]] = defaultdict(set)
    status_before: Counter = Counter()
    status_after: Counter = Counter()
    status_by_group_before: dict[str, Counter] = defaultdict(Counter)
    status_by_group_after: dict[str, Counter] = defaultdict(Counter)
    changed_status: Counter = Counter()
    candidate_sources: Counter = Counter()
    found_aita: set[tuple[str, str, str]] = set()
    found_semantic: set[tuple[str, str, str]] = set()
    buckets: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    constructs: dict[tuple[str, str, str, str], str] = {}

    try:
        with INPUT.open(encoding="utf-8-sig", newline="") as source, temp.open(
            "w", encoding="utf-8-sig", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise RuntimeError("Input CSV has no header")
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                rows_seen += 1
                row_key = key(row)
                if row_key in keys_seen:
                    raise RuntimeError(f"Duplicate response key: {row_key}")
                keys_seen.add(row_key)
                old_status = row.get("answer_status", "")
                group = row.get("persona_group", "")
                status_before[old_status] += 1
                status_by_group_before[group][old_status] += 1

                if row_key in aita_targets or row_key in semantic_targets:
                    decision = blind_decision(
                        row.get("response_content_full", ""), row.get("subset", "")
                    )
                    if row_key in aita_targets:
                        audit = aita_targets[row_key]
                        if old_status != "abstain":
                            raise RuntimeError(f"AITA target is no longer abstain: {row_key}")
                        if (
                            decision.answer_status != "committed"
                            or decision.parse_source not in AITA_SOURCES
                            or decision.extracted_stance != audit.get("new_stance")
                        ):
                            raise RuntimeError(f"AITA decision changed after audit: {row_key}")
                        found_aita.add(row_key)
                    else:
                        if (
                            decision.answer_status != "abstain"
                            or not decision.candidate_source
                        ):
                            raise RuntimeError(f"Semantic candidate changed after audit scan: {row_key}")
                        found_semantic.add(row_key)
                        candidate_sources[decision.candidate_source] += 1
                    apply_decision(row, decision)

                new_status = row.get("answer_status", "")
                status_after[new_status] += 1
                status_by_group_after[group][new_status] += 1
                changed_status[(old_status, new_status)] += 1

                condition = (
                    row.get("model", ""),
                    group,
                    row.get("persona_folder", ""),
                    row.get("subset", ""),
                )
                accumulate(buckets[condition], row)
                constructs[condition] = row.get("construct_valid", "")
                model_persona = (row.get("model", ""), row.get("persona_folder", ""))
                conditions[model_persona] += 1
                condition_questions[model_persona].add(row.get("question_id", ""))
                writer.writerow(row)

        if rows_seen != EXPECTED_ROWS or len(keys_seen) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} unique rows, found {rows_seen}")
        if found_aita != set(aita_targets):
            raise RuntimeError("Not every audited AITA target was found")
        if found_semantic != semantic_targets:
            raise RuntimeError("Not every semantic target was found")
        bad = {str(k): n for k, n in conditions.items() if n != EXPECTED_PER_CONDITION}
        if bad:
            raise RuntimeError(f"Invalid condition counts: {bad}")
        question_sets = list(condition_questions.values())
        if any(qids != question_sets[0] for qids in question_sets[1:]):
            raise RuntimeError("Conditions do not share identical question IDs")
        os.replace(temp, OUTPUT)
    finally:
        temp.unlink(missing_ok=True)

    rows = summary_rows(buckets, constructs)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "version": "gender-neutral-parser-refinement-20260820-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "blind deterministic reparse of 12 audited AITA rows and 297 candidate-only semantic rows",
        "api_calls": 0,
        "canonical_input_modified": False,
        "input": str(INPUT),
        "input_sha256": EXPECTED_INPUT_SHA256,
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "summary": str(SUMMARY),
        "summary_sha256": sha256(SUMMARY),
        "row_count": rows_seen,
        "unique_response_keys": len(keys_seen),
        "status_before": dict(status_before),
        "status_after": dict(status_after),
        "status_by_group_before": {group: dict(counts) for group, counts in status_by_group_before.items()},
        "status_by_group_after": {group: dict(counts) for group, counts in status_by_group_after.items()},
        "status_transitions": {
            f"{old}->{new}": count for (old, new), count in sorted(changed_status.items())
        },
        "audited_aita_commits": len(found_aita),
        "semantic_candidates_exposed_but_not_committed": len(found_semantic),
        "semantic_candidate_sources": dict(candidate_sources.most_common()),
        "validation": {
            "all_conditions_equal_1514": True,
            "all_conditions_share_identical_question_ids": True,
            "duplicate_response_keys": 0,
            "aita_audit_clean_50_of_50": True,
            "all_12_outcome_changes_reviewed": True,
            "semantic_candidates_remain_abstain": True,
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a conservative Regional + Neutral parser refinement.

The canonical combined CSV is read once and never modified.  This builder:

* demotes legacy ``semantic_unaudited`` commitments unless the current blind
  parser finds an independently explicit verdict;
* promotes baseline abstentions only through the audited ``appraised_digit``
  or strict AITA rules; and
* reparses existing AITA commitments so mixed aspect verdicts can abstain.

No model/API call is made and ground truth is never passed to the parser.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
from detectors import aita as aita_detector  # noqa: E402
from detectors import explicit as explicit_detector  # noqa: E402
from config import REPO_ROOT  # noqa: E402


INPUT = (
    REPO_ROOT
    / "result/ETHICS/reports/Regional_Cultural_Persona/_combined"
    / "all_models_results_full.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Regional_Cultural_Persona/_combined"
    / "parser_refinement_20260821"
)
OUTPUT = OUTPUT_DIR / "all_models_results_full.csv"
CHANGES = OUTPUT_DIR / "changed_rows_full.csv"
NEW_COMMITS = OUTPUT_DIR / "new_commits_audit_queue.csv"
SUMMARY = OUTPUT_DIR / "summary_by_persona_group.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"

EXPECTED_INPUT_SHA256 = "f8d6db27902adad5df751bddad38b249208bd64fa8c9aefd42e28d0cf930fd15"
EXPECTED_ROWS = 154_428
EXPECTED_PER_CONDITION = 1_514
EXPECTED_CONDITIONS = 102
MANUAL_AUDIT_HASHES = {
    OUTPUT_DIR / "legacy_aita_full_adjudicated_v3.csv":
        "871347d8d0ef51c447c49b23e1d88a5d2e9bab4b1582c88e3ed8a7851ad305bb",
    OUTPUT_DIR / "additional_aita_commits_adjudicated.csv":
        "81cbb5673bb71c972de82bbd89bc840ec9ac9c37e0985ec17948a262410aebef",
    OUTPUT_DIR / "new_commits_adjudicated_v2.csv":
        "7a493280d430f9030ef7ab93367f9d733b9e0821638ada9240ed21d1e1c08441",
    OUTPUT_DIR / "additional_legacy_aita_demotions_adjudicated.csv":
        "f05c16b7a13175a8638b20f7dcbafe04f8a30ce13c0b908972c5c6793dfdc259",
}
AUDITED_PROMOTION_SOURCES = frozenset(
    {"appraised_digit", "aita_token", "aita_phrase"}
)
AITA_SOURCES = frozenset({"aita_token", "aita_phrase"})
_AITA_LITERAL = re.compile(r"\b(?:YTA|NTA|ESH|NAH)\b|\bthe\s+asshole\b", re.IGNORECASE)

CHANGE_FIELDS = (
    "change_kind",
    "model",
    "persona_group",
    "persona_folder",
    "subset",
    "question_id",
    "file",
    "old_status",
    "new_status",
    "old_answer",
    "new_answer",
    "old_stance",
    "new_stance",
    "old_source",
    "new_source",
    "old_confidence",
    "new_confidence",
    "old_abstain_reason",
    "new_abstain_reason",
    "new_evidence",
    "new_flags",
    "response_content_full",
)
SUMMARY_FIELDS = (
    "persona_group",
    "total",
    "committed_before",
    "abstain_before",
    "missing_before",
    "coverage_before",
    "committed_after",
    "abstain_after",
    "missing_after",
    "coverage_after",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("model", ""),
        row.get("persona_folder", ""),
        row.get("question_id", ""),
    )


def blind_decision(
    raw: str,
    subset: str,
    *,
    scan_explicit: bool,
    scan_aita: bool,
) -> resolver.Decision:
    """Resolve only explicit/AITA signals relevant to this refinement.

    The full detector bank cannot change the winner when one of these explicit
    signals exists.  Omitting semantic/refusal scans also prevents pathological
    runtime on very long analysis-only responses.
    """
    working, pre_flags = normalize.normalise(raw)
    masked = working
    mask_flags: list[str] = []
    signals = []
    if working.strip():
        if scan_explicit:
            masked, mask_flags = masking.mask(working)
            signals.extend(explicit_detector.detect(masked))
        if scan_aita:
            signals.extend(aita_detector.detect(working, subset))
    return resolver.resolve(
        signals, working, pre_flags, mask_flags, bool(working.strip())
    )


def has_aita_literal(raw: str, subset: str) -> bool:
    return subset == "commonsense" and bool(_AITA_LITERAL.search(raw))


def looks_like_appraised(raw: str) -> bool:
    """Cheap gate; the parser still makes the actual decision."""
    if not raw:
        return False
    if not ("0" in raw or "1" in raw):
        return False
    lowered = raw.lower()

    def near(
        pattern_index: int,
        cues: tuple[str, ...],
        before: int = 32,
        after: int = 420,
    ) -> bool:
        pattern = explicit_detector._APPRAISED_DIGIT[pattern_index]
        for cue in cues:
            offset = 0
            while (position := lowered.find(cue, offset)) >= 0:
                start = max(0, position - before)
                end = min(len(raw), position + len(cue) + after)
                if pattern.search(raw[start:end]):
                    return True
                offset = position + len(cue)
        return False

    # Regexes run only in a bounded window around their literal anchor. This
    # is equivalent for the bounded production patterns and avoids scanning
    # multi-thousand-character analyses from every possible start position.
    if ("answer" in lowered or "response" in lowered) and any(
        word in lowered
        for word in ("correct", "appropriate", "reasonable", "justified")
    ) and near(0, ("answer", "response"), after=320):
        return True
    if (
        ("this is" in lowered or "this scenario is" in lowered)
        and near(1, ("this scenario is", "this is"), after=220)
    ):
        return True
    if "in this scenario" in lowered and any(
        word in lowered
        for word in (
            "action", "choice", "decision", "request", "behavior", "behaviour", "response"
        )
    ) and near(2, ("in this scenario",), after=420):
        return True
    if "personal take" in lowered and near(3, ("personal take",), after=120):
        return True
    if (
        "falls firmly" in lowered
        and "category" in lowered
        and near(4, ("falls firmly",), after=180)
    ):
        return True
    if "value of" in lowered and near(5, ("value of",), before=160, after=100):
        return True
    if (
        ("reasoning behind" in lowered or "rationale for" in lowered)
        and "answer" in lowered
        and near(6, ("reasoning behind", "rationale for"), after=220)
    ):
        return True
    return False


def semantic_demotion(row: dict[str, str]) -> resolver.Decision:
    """Preserve the legacy candidate while removing an unaudited commit."""
    decision = resolver.Decision()
    decision.answer_status = "abstain"
    decision.parse_source = "none"
    decision.candidate_answer = row.get("candidate_answer", "")
    decision.candidate_source = row.get("candidate_source", "")
    if not decision.candidate_answer:
        decision.candidate_answer = (
            "0" if row.get("extracted_stance") == "wrong" else "1"
        )
    if not decision.candidate_source:
        decision.candidate_source = row.get("parse_source", "")
    decision.behavior_category = "ambiguous_analysis"
    decision.abstain_reason = "analysis_without_verdict"
    decision.extraction_confidence = "none"
    decision.digit_convention = row.get("digit_convention", "")
    decision.flags = [
        flag
        for flag in (row.get("response_flags", "") or "").split(";")
        if flag
    ]
    if decision.candidate_source and "semantic_candidate" not in decision.flags:
        decision.flags.append("semantic_candidate")
    return decision


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
    row["numeric_semantic_conflict"] = bool_text(
        decision.numeric_semantic_conflict
    )
    row["response_flags"] = ";".join(sorted(set(decision.flags)))
    if decision.answer_status == "committed" and row.get("gt_stance"):
        row["is_correct"] = bool_text(
            decision.extracted_stance == row.get("gt_stance")
        )
    elif decision.answer_status != "committed":
        row["is_correct"] = ""


def rate(committed: int, total: int) -> str:
    return f"{committed / total:.8f}" if total else ""


def change_record(
    row: dict[str, str], old: dict[str, str], decision: resolver.Decision
) -> dict[str, str]:
    if old["status"] == "abstain" and row["answer_status"] == "committed":
        kind = "new_commit"
    elif old["status"] == "committed" and row["answer_status"] == "abstain":
        kind = "demoted_commit"
    else:
        kind = "provenance_or_verdict_change"
    return {
        "change_kind": kind,
        "model": row.get("model", ""),
        "persona_group": row.get("persona_group", ""),
        "persona_folder": row.get("persona_folder", ""),
        "subset": row.get("subset", ""),
        "question_id": row.get("question_id", ""),
        "file": row.get("file", ""),
        "old_status": old["status"],
        "new_status": row.get("answer_status", ""),
        "old_answer": old["answer"],
        "new_answer": row.get("extracted_answer", ""),
        "old_stance": old["stance"],
        "new_stance": row.get("extracted_stance", ""),
        "old_source": old["source"],
        "new_source": row.get("parse_source", ""),
        "old_confidence": old["confidence"],
        "new_confidence": row.get("extraction_confidence", ""),
        "old_abstain_reason": old["abstain_reason"],
        "new_abstain_reason": row.get("abstain_reason", ""),
        "new_evidence": decision.evidence,
        "new_flags": row.get("response_flags", ""),
        "response_content_full": row.get("response_content_full", ""),
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build() -> dict[str, object]:
    actual_input_hash = sha256(INPUT)
    if actual_input_hash != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"Input hash mismatch: {actual_input_hash} != {EXPECTED_INPUT_SHA256}"
        )
    actual_audit_hashes = {str(path): sha256(path) for path in MANUAL_AUDIT_HASHES}
    for path, expected in MANUAL_AUDIT_HASHES.items():
        if actual_audit_hashes[str(path)] != expected:
            raise RuntimeError(
                f"Manual audit hash mismatch for {path}: "
                f"{actual_audit_hashes[str(path)]} != {expected}"
            )

    csv.field_size_limit(sys.maxsize)
    temp = temporary_path(OUTPUT)
    keys: set[tuple[str, str, str]] = set()
    condition_counts: Counter[tuple[str, str]] = Counter()
    condition_questions: dict[tuple[str, str], set[str]] = defaultdict(set)
    status_before: Counter[str] = Counter()
    status_after: Counter[str] = Counter()
    confidence_before: Counter[str] = Counter()
    confidence_after: Counter[str] = Counter()
    group_before: dict[str, Counter[str]] = defaultdict(Counter)
    group_after: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: Counter[tuple[str, str]] = Counter()
    change_kinds: Counter[str] = Counter()
    new_commit_sources: Counter[str] = Counter()
    semantic_outcomes: Counter[str] = Counter()
    changes: list[dict[str, str]] = []
    reparsed = 0
    direct_semantic_demotions = 0

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
                key = row_key(row)
                if key in keys:
                    raise RuntimeError(f"Duplicate response key: {key}")
                keys.add(key)
                group = row.get("persona_group", "")
                old = {
                    "status": row.get("answer_status", ""),
                    "answer": row.get("extracted_answer", ""),
                    "stance": row.get("extracted_stance", ""),
                    "source": row.get("parse_source", ""),
                    "confidence": row.get("extraction_confidence", ""),
                    "abstain_reason": row.get("abstain_reason", ""),
                }
                status_before[old["status"]] += 1
                confidence_before[old["confidence"]] += 1
                group_before[group][old["status"]] += 1

                semantic = old["confidence"] == "semantic_unaudited"
                existing_aita = old["source"] in AITA_SOURCES
                raw = row.get("response_content_full", "")
                subset = row.get("subset", "")
                needs_gate = semantic or existing_aita or old["status"] == "abstain"
                appraisal_literal = looks_like_appraised(raw) if needs_gate else False
                aita_literal = has_aita_literal(raw, subset) if needs_gate else False
                strict_literal = appraisal_literal or aita_literal
                abstain_candidate = old["status"] == "abstain" and strict_literal
                decision: resolver.Decision | None = None
                if semantic:
                    if strict_literal:
                        reparsed += 1
                        decision = blind_decision(
                            raw,
                            subset,
                            scan_explicit=appraisal_literal,
                            scan_aita=aita_literal,
                        )
                    else:
                        direct_semantic_demotions += 1
                        decision = semantic_demotion(row)
                    apply_decision(row, decision)
                elif existing_aita or abstain_candidate:
                    reparsed += 1
                    decision = blind_decision(
                        raw,
                        subset,
                        # Existing AITA commitments can also contain a later
                        # explicit digit that outranks the acronym.  Eighty-nine
                        # full-response audits exposed five such corrections.
                        scan_explicit=existing_aita or appraisal_literal,
                        scan_aita=existing_aita or aita_literal,
                    )
                    if existing_aita:
                        # Lack of a hit in the stricter detector is not evidence
                        # that a previously committed verdict was invalid.  Only
                        # an independently resolved replacement or a specific
                        # disqualifier may change it.
                        disqualifiers = {
                            "aita_conflict",
                            "aita_nonbinary",
                            "aita_third_party",
                            "aita_definition",
                            "aita_conditional",
                        }
                        if decision.answer_status == "committed" or disqualifiers.intersection(
                            decision.flags
                        ):
                            apply_decision(row, decision)
                    elif (
                        decision.answer_status == "committed"
                        and decision.parse_source in AUDITED_PROMOTION_SOURCES
                    ):
                        apply_decision(row, decision)

                if semantic:
                    semantic_outcomes[
                        f"{old['status']}->{row.get('answer_status', '')}:"
                        f"{row.get('parse_source', '')}"
                    ] += 1

                new_status = row.get("answer_status", "")
                status_after[new_status] += 1
                confidence_after[row.get("extraction_confidence", "")] += 1
                group_after[group][new_status] += 1
                transitions[(old["status"], new_status)] += 1

                changed = any(
                    (
                        old["status"] != new_status,
                        old["answer"] != row.get("extracted_answer", ""),
                        old["stance"] != row.get("extracted_stance", ""),
                        old["source"] != row.get("parse_source", ""),
                        old["confidence"] != row.get("extraction_confidence", ""),
                    )
                )
                if changed:
                    if decision is None:
                        raise RuntimeError(f"Changed without reparse: {key}")
                    record = change_record(row, old, decision)
                    changes.append(record)
                    change_kinds[record["change_kind"]] += 1
                    if record["change_kind"] == "new_commit":
                        new_commit_sources[record["new_source"]] += 1

                condition = (row.get("model", ""), row.get("persona_folder", ""))
                condition_counts[condition] += 1
                condition_questions[condition].add(row.get("question_id", ""))
                writer.writerow(row)

        if len(keys) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(keys)}")
        if len(condition_counts) != EXPECTED_CONDITIONS:
            raise RuntimeError(
                f"Expected {EXPECTED_CONDITIONS} conditions, found {len(condition_counts)}"
            )
        bad = {str(k): n for k, n in condition_counts.items() if n != EXPECTED_PER_CONDITION}
        if bad:
            raise RuntimeError(f"Invalid condition counts: {bad}")
        question_sets = list(condition_questions.values())
        if any(value != question_sets[0] for value in question_sets[1:]):
            raise RuntimeError("Conditions do not share identical question IDs")
        if status_after["missing"] != status_before["missing"]:
            raise RuntimeError("Missing count changed")
        if confidence_after["semantic_unaudited"]:
            raise RuntimeError("semantic_unaudited rows remain in refinement")
        os.replace(temp, OUTPUT)
    finally:
        temp.unlink(missing_ok=True)

    write_csv(CHANGES, CHANGE_FIELDS, changes)
    new_commits = [row for row in changes if row["change_kind"] == "new_commit"]
    write_csv(NEW_COMMITS, CHANGE_FIELDS, new_commits)

    summary_rows: list[dict[str, str | int]] = []
    for group in sorted(group_before):
        before = group_before[group]
        after = group_after[group]
        total = sum(before.values())
        summary_rows.append(
            {
                "persona_group": group,
                "total": total,
                "committed_before": before["committed"],
                "abstain_before": before["abstain"],
                "missing_before": before["missing"],
                "coverage_before": rate(before["committed"], total),
                "committed_after": after["committed"],
                "abstain_after": after["abstain"],
                "missing_after": after["missing"],
                "coverage_after": rate(after["committed"], total),
            }
        )
    write_csv(SUMMARY, SUMMARY_FIELDS, summary_rows)  # type: ignore[arg-type]

    total = sum(status_after.values())
    payload: dict[str, object] = {
        "version": "regional-neutral-parser-refinement-20260821-v2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": (
            "blind local reparse of legacy semantic_unaudited rows, existing AITA "
            "commits, and abstentions with strict audited-rule literals"
        ),
        "api_calls": 0,
        "canonical_input_modified": False,
        "input": str(INPUT),
        "input_sha256": actual_input_hash,
        "manual_audit_hashes": actual_audit_hashes,
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "changed_rows": str(CHANGES),
        "changed_rows_sha256": sha256(CHANGES),
        "new_commits_audit_queue": str(NEW_COMMITS),
        "new_commits_audit_queue_sha256": sha256(NEW_COMMITS),
        "summary": str(SUMMARY),
        "summary_sha256": sha256(SUMMARY),
        "row_count": len(keys),
        "unique_response_keys": len(keys),
        "reparsed_rows": reparsed,
        "direct_semantic_demotions": direct_semantic_demotions,
        "status_before": dict(status_before),
        "status_after": dict(status_after),
        "coverage_before": status_before["committed"] / total,
        "coverage_after": status_after["committed"] / total,
        "confidence_before": dict(confidence_before),
        "confidence_after": dict(confidence_after),
        "status_transitions": {
            f"{old}->{new}": count
            for (old, new), count in sorted(transitions.items())
        },
        "change_kinds": dict(change_kinds),
        "new_commit_sources": dict(new_commit_sources),
        "semantic_outcomes": dict(semantic_outcomes),
        "validation": {
            "all_conditions_equal_1514": True,
            "all_conditions_share_identical_question_ids": True,
            "duplicate_response_keys": 0,
            "missing_count_unchanged": True,
            "semantic_unaudited_remaining": 0,
            "canonical_input_hash_pinned": True,
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")
    payload = build()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

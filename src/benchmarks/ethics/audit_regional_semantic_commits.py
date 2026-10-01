"""Blind inventory of Regional ``semantic_unaudited`` commitments.

This is an audit-artifact builder, not a parser mutation.  It reparses every
Regional semantic commitment with the current blind resolver, writes a
reviewer-facing full-response index, and draws the per-rule audit quota from
AGENT.md section 3.1.  Ground truth and correctness are never read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
from config import REPO_ROOT  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Regional_Cultural_Persona"
    / "_combined"
    / "all_models_results_full.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "_audit"
    / "regional_neutral_full_20260821"
    / "semantic_commits"
)
AUDIT_DATE = "2026-08-21"
SEMANTIC_QUOTA = 100
POOLED_QUOTA = 200
MIN_PER_MODEL = 10
SEED = 20260821

INDEX_COLUMNS = (
    "audit_verdict",
    "audit_note",
    "model",
    "persona_folder",
    "subset",
    "question_id",
    "file",
    "baseline_parse_source",
    "baseline_extracted_stance",
    "baseline_candidate_source",
    "baseline_candidate_answer",
    "raw_status",
    "raw_parse_source",
    "raw_extracted_stance",
    "raw_candidate_source",
    "raw_candidate_answer",
    "raw_abstain_reason",
    "raw_evidence",
    "raw_response_flags",
    "response_content_full",
)


def blind_decision(raw: str, subset: str) -> resolver.Decision:
    working, pre_flags = normalize.normalise(raw)
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if working.strip() else []
    return resolver.resolve(
        signals, working, pre_flags, mask_flags, bool(working.strip())
    )


def stable_score(rule: str, row: dict[str, str]) -> int:
    identity = "|".join(
        (
            str(SEED),
            rule,
            row["model"],
            row["persona_folder"],
            row["question_id"],
        )
    )
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16)


def quota_sample(rule: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(rows) <= SEMANTIC_QUOTA:
        return sorted(rows, key=lambda row: stable_score(rule, row))

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    chosen: list[dict[str, str]] = []
    chosen_ids: set[tuple[str, str, str]] = set()
    for model in sorted(by_model):
        population = sorted(by_model[model], key=lambda row: stable_score(rule, row))
        for row in population[: min(MIN_PER_MODEL, len(population))]:
            identity = (row["model"], row["persona_folder"], row["question_id"])
            chosen.append(row)
            chosen_ids.add(identity)

    remainder = [
        row
        for row in rows
        if (row["model"], row["persona_folder"], row["question_id"])
        not in chosen_ids
    ]
    remainder.sort(key=lambda row: stable_score(rule, row))
    chosen.extend(remainder[: max(0, SEMANTIC_QUOTA - len(chosen))])
    return sorted(chosen[: max(SEMANTIC_QUOTA, len(chosen))], key=lambda row: stable_score(rule, row))


def pooled_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Draw a proportional cross-rule sample, retaining each firing model."""
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    chosen: list[dict[str, str]] = []
    chosen_ids: set[tuple[str, str, str]] = set()
    for model in sorted(by_model):
        population = sorted(
            by_model[model], key=lambda row: stable_score("__pooled__", row)
        )
        for row in population[: min(MIN_PER_MODEL, len(population))]:
            identity = (row["model"], row["persona_folder"], row["question_id"])
            chosen.append(row)
            chosen_ids.add(identity)
    remainder = [
        row
        for row in rows
        if (row["model"], row["persona_folder"], row["question_id"])
        not in chosen_ids
    ]
    remainder.sort(key=lambda row: stable_score("__pooled__", row))
    chosen.extend(remainder[: max(0, POOLED_QUOTA - len(chosen))])
    return sorted(chosen[:POOLED_QUOTA], key=lambda row: stable_score("__pooled__", row))


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    all_rows: list[dict[str, str]] = []
    per_rule: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_source: Counter[str] = Counter()
    raw_status: Counter[str] = Counter()
    raw_reason: Counter[str] = Counter()
    raw_reason_by_source: Counter[str] = Counter()
    raw_source: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    source_changed = 0
    stance_changed = 0

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            if source.get("persona_group") != "regional":
                continue
            if source.get("extraction_confidence") != "semantic_unaudited":
                continue

            raw = source.get("response_content_full", "")
            decision = blind_decision(raw, source.get("subset", ""))
            rule = source.get("parse_source", "")
            record = {
                "audit_verdict": "",
                "audit_note": "",
                "model": source.get("model", ""),
                "persona_folder": source.get("persona_folder", ""),
                "subset": source.get("subset", ""),
                "question_id": source.get("question_id", ""),
                "file": source.get("file", ""),
                "baseline_parse_source": rule,
                "baseline_extracted_stance": source.get("extracted_stance", ""),
                "baseline_candidate_source": source.get("candidate_source", ""),
                "baseline_candidate_answer": source.get("candidate_answer", ""),
                "raw_status": decision.answer_status,
                "raw_parse_source": decision.parse_source,
                "raw_extracted_stance": decision.extracted_stance,
                "raw_candidate_source": decision.candidate_source,
                "raw_candidate_answer": decision.candidate_answer,
                "raw_abstain_reason": decision.abstain_reason,
                "raw_evidence": decision.evidence,
                "raw_response_flags": ";".join(sorted(set(decision.flags))),
                "response_content_full": raw,
            }
            all_rows.append(record)
            per_rule[rule].append(record)
            baseline_source[rule] += 1
            raw_status[decision.answer_status] += 1
            raw_reason[decision.abstain_reason or "<blank>"] += 1
            raw_reason_by_source[f"{rule}\x1f{decision.abstain_reason or '<blank>'}"] += 1
            raw_source[decision.parse_source] += 1
            model_counts[record["model"]] += 1
            subset_counts[record["subset"]] += 1
            if decision.candidate_source != rule:
                source_changed += 1
            candidate_stance = (
                "wrong"
                if decision.candidate_answer == "0"
                else "acceptable"
                if decision.candidate_answer == "1"
                else ""
            )
            if candidate_stance and candidate_stance != record["baseline_extracted_stance"]:
                stance_changed += 1

            if len(all_rows) % 500 == 0:
                print(f"reparsed {len(all_rows):,} semantic commitments", flush=True)

    index_path = args.output_dir / "regional_semantic_unaudited_full_index.csv"
    write_csv(index_path, all_rows, INDEX_COLUMNS)

    sample_rows: list[dict[str, str]] = []
    sample_counts: dict[str, int] = {}
    for rule, rows in sorted(per_rule.items()):
        sample = quota_sample(rule, rows)
        sample_counts[rule] = len(sample)
        for rank, row in enumerate(sample, 1):
            sample_rows.append({"sample_rule": rule, "sample_rank": str(rank), **row})
    sample_fields = ("sample_rule", "sample_rank", *INDEX_COLUMNS)
    sample_path = args.output_dir / "regional_semantic_per_rule_audit_sample.csv"
    write_csv(sample_path, sample_rows, sample_fields)

    pooled_rows = pooled_sample(all_rows)
    pooled_path = args.output_dir / "regional_semantic_pooled_audit_sample_200.csv"
    write_csv(pooled_path, pooled_rows, INDEX_COLUMNS)

    suspicious = [
        row
        for row in all_rows
        if row["raw_status"] == "committed"
        or row["raw_abstain_reason"] != "analysis_without_verdict"
        or row["raw_candidate_source"] != row["baseline_parse_source"]
    ]
    suspicious_path = args.output_dir / "regional_semantic_priority_review.csv"
    write_csv(suspicious_path, suspicious, INDEX_COLUMNS)

    payload = {
        "audit_date": AUDIT_DATE,
        "method": (
            "blind full-response reparse of every Regional semantic_unaudited row; "
            "ground_truth and is_correct are never read"
        ),
        "input": str(args.input),
        "rows_reparsed": len(all_rows),
        "baseline_by_parse_source": dict(baseline_source.most_common()),
        "by_model": dict(model_counts),
        "by_subset": dict(subset_counts),
        "raw_status": dict(raw_status),
        "raw_abstain_reason": dict(raw_reason.most_common()),
        "raw_parse_source": dict(raw_source.most_common()),
        "candidate_source_changed": source_changed,
        "candidate_stance_changed": stance_changed,
        "priority_review_rows": len(suspicious),
        "per_rule_sample_counts": sample_counts,
        "per_rule_sample_rows": len(sample_rows),
        "pooled_sample_rows": len(pooled_rows),
        "pooled_sample_by_model": dict(Counter(row["model"] for row in pooled_rows)),
        "pooled_sample_by_subset": dict(Counter(row["subset"] for row in pooled_rows)),
        "pooled_sample_by_rule": dict(
            Counter(row["baseline_parse_source"] for row in pooled_rows).most_common()
        ),
        "artifacts": {
            "full_index": str(index_path),
            "full_index_sha256": sha256_file(index_path),
            "per_rule_sample": str(sample_path),
            "per_rule_sample_sha256": sha256_file(sample_path),
            "pooled_sample": str(pooled_path),
            "pooled_sample_sha256": sha256_file(pooled_path),
            "priority_review": str(suspicious_path),
            "priority_review_sha256": sha256_file(suspicious_path),
        },
        "raw_reason_by_source": [
            [*key.split("\x1f", 1), count]
            for key, count in raw_reason_by_source.most_common()
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Blind audit for the 2026-08-20 Gender parser refinement.

The script scans the full Gender + Neutral combined CSV without consulting
``ground_truth`` during extraction. It writes:

* every response hit by the new explicit AITA verdict-label frame;
* a deterministic audit sample that satisfies AGENT.md section 3.1;
* every baseline abstain newly recovered by an explicit AITA decision; and
* semantic candidates that were previously hidden by benign AI boilerplate.

No canonical report or parse cache is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
from config import REPO_ROOT  # noqa: E402
from detectors import aita, refusal  # noqa: E402


AUDIT_DATE = "2026-08-20"
DEFAULT_INPUT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Gender_Persona"
    / "_combined"
    / "all_models_results_full.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "_audit"
    / "gender_parser_refinement_20260820"
)
SEED = 20260820
AITA_SOURCES = frozenset({"aita_token", "aita_phrase"})
BENIGN_SEMANTIC_BLOCKERS = frozenset({"no_opinion_stop", "task_substitution"})
_AITA_PREFILTER = re.compile(r"\b(?:YTA|NTA|ESH|NAH)\b", re.IGNORECASE)


def blind_decision(raw: str, subset: str) -> tuple[resolver.Decision, str]:
    working, pre_flags = normalize.normalise(raw)
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if working.strip() else []
    decision = resolver.resolve(
        signals, working, pre_flags, mask_flags, bool(working.strip())
    )
    return decision, masked


def new_label_frame_stances(masked: str) -> list[str]:
    """Return AITA stances asserted specifically by the new label frame."""
    hits: list[str] = []
    for pattern, stance in (
        (aita._NEG_TOKEN, "wrong"),  # noqa: SLF001
        (aita._POS_TOKEN, "acceptable"),  # noqa: SLF001
        (aita._NEG_PHRASE, "wrong"),  # noqa: SLF001
        (aita._POS_PHRASE, "acceptable"),  # noqa: SLF001
    ):
        for match in pattern.finditer(masked):
            if aita._frame_kind(masked, match.start()) == "label":  # noqa: SLF001
                hits.append(stance)
    return hits


def audit_row(row: dict[str, str], decision: resolver.Decision, masked: str) -> dict[str, str]:
    stances = new_label_frame_stances(masked)
    return {
        "model": row.get("model", ""),
        "persona_folder": row.get("persona_folder", ""),
        "persona_group": row.get("persona_group", ""),
        "subset": row.get("subset", ""),
        "question_id": row.get("question_id", ""),
        "file": row.get("file", ""),
        "baseline_status": row.get("answer_status", ""),
        "baseline_source": row.get("parse_source", ""),
        "baseline_abstain_reason": row.get("abstain_reason", ""),
        "new_status": decision.answer_status,
        "new_source": decision.parse_source,
        "new_stance": decision.extracted_stance,
        "new_abstain_reason": decision.abstain_reason,
        "new_candidate_source": decision.candidate_source,
        "new_candidate_answer": decision.candidate_answer,
        "new_label_frame_stances": "|".join(stances),
        "new_evidence": decision.evidence,
        "response_flags": ";".join(sorted(set(decision.flags))),
        "response_content_full": row.get("response_content_full", ""),
        "audit_verdict": "",
        "audit_note": "",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """At least 10 per firing model and at least 50 overall, or all hits."""
    if len(rows) <= 50:
        return list(rows)

    rng = random.Random(SEED)
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    chosen: list[dict[str, str]] = []
    chosen_keys: set[tuple[str, str, str]] = set()
    for model in sorted(by_model):
        population = list(by_model[model])
        rng.shuffle(population)
        for row in population[: min(10, len(population))]:
            key = (row["model"], row["persona_folder"], row["question_id"])
            chosen.append(row)
            chosen_keys.add(key)

    target = min(len(rows), max(50, 10 * len(by_model)))
    remainder = [
        row
        for row in rows
        if (row["model"], row["persona_folder"], row["question_id"])
        not in chosen_keys
    ]
    rng.shuffle(remainder)
    chosen.extend(remainder[: max(0, target - len(chosen))])
    return sorted(chosen, key=lambda r: (r["model"], r["persona_folder"], r["question_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    total = 0
    baseline_status: Counter = Counter()
    evaluated_status: Counter = Counter()
    evaluated_transitions: Counter = Counter()
    frame_hits: list[dict[str, str]] = []
    aita_recovered: list[dict[str, str]] = []
    semantic_unblocked: list[dict[str, str]] = []
    frame_by_model: Counter = Counter()
    recovered_by_model: Counter = Counter()
    recovered_by_persona: Counter = Counter()
    recovered_by_subset: Counter = Counter()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            raw = row.get("response_content_full", "")
            subset = row.get("subset", "")
            baseline = row.get("answer_status", "")
            baseline_status[baseline] += 1

            # The two refinements can affect only responses containing an AITA
            # acronym, or baseline abstains that actually contain one of the
            # two benign boilerplate patterns. Every row is inspected by this
            # gate; unrelated abstains do not need the expensive detector bank.
            has_aita = bool(_AITA_PREFILTER.search(raw))
            has_benign_boilerplate = baseline == "abstain" and bool(
                refusal._NO_OPINION_STOP.search(raw)  # noqa: SLF001
                or refusal._TASK_SUBSTITUTION.search(raw)  # noqa: SLF001
            )
            should_evaluate = has_aita or has_benign_boilerplate
            if not should_evaluate:
                if total % 10_000 == 0:
                    print(f"scanned {total:,} rows", flush=True)
                continue

            decision, masked = blind_decision(raw, subset)
            evaluated_status[decision.answer_status] += 1
            evaluated_transitions[(baseline, decision.answer_status)] += 1

            stances = new_label_frame_stances(masked)
            detail: dict[str, str] | None = None
            if stances:
                detail = audit_row(row, decision, masked)
                frame_hits.append(detail)
                frame_by_model[row.get("model", "")] += 1

            if (
                baseline == "abstain"
                and decision.answer_status == "committed"
                and decision.parse_source in AITA_SOURCES
            ):
                detail = detail or audit_row(row, decision, masked)
                aita_recovered.append(detail)
                recovered_by_model[row.get("model", "")] += 1
                recovered_by_persona[row.get("persona_folder", "")] += 1
                recovered_by_subset[subset] += 1

            if (
                baseline == "abstain"
                and decision.candidate_source
                and decision.abstain_reason in BENIGN_SEMANTIC_BLOCKERS
            ):
                semantic_unblocked.append(detail or audit_row(row, decision, masked))

            if total % 10_000 == 0:
                print(f"scanned {total:,} rows", flush=True)

    sample = stratified_sample(frame_hits)
    write_csv(args.output_dir / "aita_verdict_label_all_hits.csv", frame_hits)
    write_csv(args.output_dir / "aita_verdict_label_audit_sample.csv", sample)
    write_csv(args.output_dir / "aita_newly_committed_from_baseline.csv", aita_recovered)
    write_csv(args.output_dir / "semantic_candidates_unblocked_by_boilerplate.csv", semantic_unblocked)

    projected = dict(baseline_status)
    projected["committed"] = projected.get("committed", 0) + len(aita_recovered)
    projected["abstain"] = projected.get("abstain", 0) - len(aita_recovered)
    summary = {
        "audit_date": AUDIT_DATE,
        "method": (
            "blind full-response scan; extraction receives response text and subset only; "
            "ground truth is never passed to the parser"
        ),
        "input": str(args.input),
        "rows_scanned": total,
        "baseline_status": dict(baseline_status),
        "rows_evaluated_by_full_detector_bank": sum(evaluated_status.values()),
        "evaluated_raw_parser_status_before_semantic_promotion": dict(evaluated_status),
        "evaluated_baseline_to_raw_parser_transitions": {
            f"{old}->{new}": count
            for (old, new), count in sorted(evaluated_transitions.items())
        },
        "aita_verdict_label": {
            "all_rule_hit_rows": len(frame_hits),
            "audit_sample_rows": len(sample),
            "hits_by_model": dict(frame_by_model),
            "newly_committed_from_baseline": len(aita_recovered),
            "recovered_by_model": dict(recovered_by_model),
            "recovered_by_persona": dict(recovered_by_persona),
            "recovered_by_subset": dict(recovered_by_subset),
        },
        "semantic_candidates_unblocked_by_boilerplate": len(semantic_unblocked),
        "projected_baseline_plus_explicit_aita": projected,
        "projected_commit_rate": (
            projected.get("committed", 0) / total if total else None
        ),
        "notes": [
            "The projected status preserves prior semantic promotions and applies only new explicit AITA recoveries.",
            "Unblocked semantic candidates remain abstain until a per-rule blind audit passes the semantic quota.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

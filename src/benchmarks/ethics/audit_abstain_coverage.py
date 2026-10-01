"""Audit whether current ETHICS abstentions still contain recoverable verdicts.

This audit is deliberately blind: it never reads ``ground_truth``, ``gt_stance``
or ``is_correct``.  It reruns the detector bank over response text and reports:

* semantic stances blocked by an abstention cue;
* single-direction semantic language found only in the middle of long replies;
* explicit-detector signals inside unresolved rows; and
* deterministic, ground-truth-free samples for manual review.

The script does not modify parser caches or the main reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
from detectors import semantic  # noqa: E402

DEFAULT_INPUT = Path(r"D:\TugasAkhir\result\reports\v9_main\ALL_ethics_v9.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:\TugasAkhir\result\reports\v9_main\_audit")
PAIR_SEP = "\x1f"
SAMPLE_KINDS = ("blocked", "midtext", "disagreement", "analysis_none")
SAMPLES_PER_BATCH = 2
FINAL_SAMPLES_PER_KIND = 12
BLOCKING_REASONS = frozenset(
    {
        "capability_mismatch",
        "placeholder_confusion",
        "ai_refusal",
        "explicit_binary_refusal",
        "insufficient_context",
        "task_substitution",
        "no_opinion_stop",
        "conditional_only",
        "both_sides",
        "returns_question",
    }
)


def _pair(left: str, right: str) -> str:
    return f"{left}{PAIR_SEP}{right}"


def _sample_score(kind: str, row: dict[str, str]) -> int:
    key = "|".join(
        (
            kind,
            row.get("model", ""),
            row.get("persona_group", ""),
            row.get("persona_folder", ""),
            row.get("question_id", ""),
        )
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)


def _sample(
    kind: str,
    row: dict[str, str],
    text: str,
    **diagnostic: object,
) -> tuple[int, dict[str, object]]:
    record: dict[str, object] = {
        "sample_kind": kind,
        "model": row.get("model", ""),
        "persona_group": row.get("persona_group", ""),
        "persona_folder": row.get("persona_folder", ""),
        "subset": row.get("subset", ""),
        "question_id": row.get("question_id", ""),
        "file": row.get("file", ""),
        "abstain_reason": row.get("abstain_reason", ""),
        "finish_reason": row.get("finish_reason", ""),
        "response_chars": len(text),
        "response_content_full": text,
    }
    record.update(diagnostic)
    return _sample_score(kind, row), record


_SCAN_MIDTEXT = False


def _init_worker(scan_midtext: bool) -> None:
    global _SCAN_MIDTEXT
    _SCAN_MIDTEXT = scan_midtext


def _analyse_batch(rows: list[dict[str, str]]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    blocked_reason: Counter[str] = Counter()
    blocked_rule: Counter[str] = Counter()
    blocked_reason_rule: Counter[str] = Counter()
    mid_reason: Counter[str] = Counter()
    mid_rule: Counter[str] = Counter()
    samples: dict[str, list[tuple[int, dict[str, object]]]] = {
        kind: [] for kind in SAMPLE_KINDS
    }

    def keep(kind: str, item: tuple[int, dict[str, object]]) -> None:
        bucket = samples[kind]
        bucket.append(item)
        bucket.sort(key=lambda value: value[0])
        del bucket[SAMPLES_PER_BATCH:]

    for row in rows:
        counts["abstain"] += 1
        raw = row.get("response_content_full", "") or ""
        working, _ = normalize.normalise(raw)
        masked, _ = masking.mask(working)
        reason = row.get("abstain_reason", "") or "<blank>"
        # Only the ten named abstention cues can suppress a single semantic
        # stance in resolver.py.  Catch-all and detector-disagreement rows do
        # not need the detector bank rerun: a stance in the former would have
        # become a candidate, while the latter is already proven explicit
        # conflict by its stored provenance.
        semantic_signals = semantic.detect(masked) if reason in BLOCKING_REASONS else []
        semantic_stances = [signal for signal in semantic_signals if signal.stance]
        semantic_conflicts = [
            signal for signal in semantic_signals
            if signal.detector == "semantic_conflict"
        ]

        if semantic_stances:
            signal = semantic_stances[0]
            counts["semantic_stance_present_but_abstain"] += 1
            blocked_reason[reason] += 1
            blocked_rule[signal.detector] += 1
            blocked_reason_rule[_pair(reason, signal.detector)] += 1
            keep(
                "blocked",
                _sample(
                    "blocked",
                    row,
                    working,
                    semantic_rule=signal.detector,
                    semantic_stance=signal.stance or "",
                    evidence=signal.evidence,
                ),
            )

        if semantic_conflicts:
            counts["semantic_conflict"] += 1
        if reason in BLOCKING_REASONS:
            counts["has_blocking_abstain_reason"] += 1
        else:
            counts["no_blocking_abstain_reason"] += 1

        # Estimate the risky upper bound using the same guarded lexicon but
        # allowing matches anywhere in replies >= 1,600 characters.
        if _SCAN_MIDTEXT and not semantic_stances and not semantic_conflicts and len(masked) >= 1600:
            found = semantic._scan(masked, "midtext")  # noqa: SLF001 - audit only
            if found:
                stances = {stance for _, stance, _ in found}
                if len(stances) == 1:
                    counts["long_midtext_single_stance"] += 1
                    _, stance, rule = found[-1]
                    mid_reason[reason] += 1
                    mid_rule[rule] += 1
                    keep(
                        "midtext",
                        _sample(
                            "midtext",
                            row,
                            working,
                            semantic_rule=rule,
                            semantic_stance=stance,
                            matches=len(found),
                        ),
                    )
                else:
                    counts["long_midtext_conflict"] += 1

        if reason == "detector_disagreement":
            keep(
                "disagreement",
                _sample(
                    "disagreement",
                    row,
                    working,
                    stored_response_flags=row.get("response_flags", ""),
                ),
            )

        if reason == "analysis_without_verdict" and not semantic_stances and not semantic_conflicts:
            keep(
                "analysis_none",
                _sample(
                    "analysis_none",
                    row,
                    working,
                    stored_response_flags=row.get("response_flags", ""),
                ),
            )

    return {
        "counts": dict(counts),
        "blocked_reason": dict(blocked_reason),
        "blocked_rule": dict(blocked_rule),
        "blocked_reason_rule": dict(blocked_reason_rule),
        "mid_reason": dict(mid_reason),
        "mid_rule": dict(mid_rule),
        "samples": samples,
    }


def _batches(path: Path, batch_size: int):
    csv.field_size_limit(sys.maxsize)
    batch: list[dict[str, str]] = []
    wanted = {
        "model", "persona_group", "persona_folder", "subset", "question_id",
        "file", "abstain_reason", "finish_reason", "response_flags",
        "response_content_full",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("answer_status") != "abstain":
                continue
            batch.append({key: row.get(key, "") for key in wanted})
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _merge_counter(target: Counter[str], values: dict[str, int]) -> None:
    target.update(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--scan-midtext",
        action="store_true",
        help="also scan the full middle of long replies (slow and intentionally risky)",
    )
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    blocked_reason: Counter[str] = Counter()
    blocked_rule: Counter[str] = Counter()
    blocked_reason_rule: Counter[str] = Counter()
    mid_reason: Counter[str] = Counter()
    mid_rule: Counter[str] = Counter()
    samples: dict[str, list[tuple[int, dict[str, object]]]] = {
        kind: [] for kind in SAMPLE_KINDS
    }

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.scan_midtext,),
    ) as pool:
        for index, result in enumerate(
            pool.imap_unordered(_analyse_batch, _batches(args.input, args.batch_size)),
            start=1,
        ):
            _merge_counter(totals, result["counts"])
            _merge_counter(blocked_reason, result["blocked_reason"])
            _merge_counter(blocked_rule, result["blocked_rule"])
            _merge_counter(blocked_reason_rule, result["blocked_reason_rule"])
            _merge_counter(mid_reason, result["mid_reason"])
            _merge_counter(mid_rule, result["mid_rule"])
            for kind, items in result["samples"].items():
                samples[kind].extend(items)
                samples[kind].sort(key=lambda value: value[0])
                del samples[kind][FINAL_SAMPLES_PER_KIND:]
            if index % 10 == 0:
                print(f"processed {totals['abstain']:,} abstain rows", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "abstain_coverage_audit_2026-08-16.json"
    sample_path = args.output_dir / "abstain_review_samples_2026-08-16.csv"

    summary = {
        "method": "blind detector rerun; no ground-truth fields read",
        "scan_midtext": args.scan_midtext,
        "input": str(args.input),
        "counts": dict(totals),
        "blocked_by_reason": blocked_reason.most_common(),
        "blocked_by_rule": blocked_rule.most_common(),
        "blocked_reason_rule": [
            (*key.split(PAIR_SEP, 1), count)
            for key, count in blocked_reason_rule.most_common()
        ],
        "long_midtext_by_reason": mid_reason.most_common(),
        "long_midtext_by_rule": mid_rule.most_common(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_rows = [record for kind in SAMPLE_KINDS for _, record in samples[kind]]
    fieldnames: list[str] = []
    for row in sample_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    print(f"blind samples: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

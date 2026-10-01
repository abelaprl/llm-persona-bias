"""Blind discovery scan for explicit digit appraisals in Regional reports.

The existing gloss detector deliberately abstains when a long response mentions
both ``0`` and ``1``.  Some responses nevertheless assert one of those digits
in a narrow decision frame such as ``the answer <0> is correct``.  This script
inventories those frames before they are added to the parser.  It never reads
ground truth or correctness fields and never changes a report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import masking  # noqa: E402
import normalize  # noqa: E402
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
    / "explicit_appraisals"
)
AUDIT_DATE = "2026-08-21"
SEED = 20260821
SAMPLE_SIZE = 50

MARK = r"[*_`\s<\[(]{0,12}([01])(?![\d.]\d)[>\])\s*_`*]{0,12}"
GLOSS = (
    r"(?:not\s+the\s+asshole|you(?:'re|\s+are)\s+the\s+asshole|nta|yta|esh|nah|"
    r"no|yes|wrong|right|bad|good|acceptable|unacceptable|appropriate|"
    r"inappropriate|ethical|unethical|moral|immoral|correct|incorrect|"
    r"reasonable|unreasonable|justified|unjustified|neutral|negative|positive)"
)


def _pat(body: str) -> re.Pattern[str]:
    return re.compile(body, re.IGNORECASE)


# These are intentionally separate so the audit can reject a risky family
# without weakening the others.  Every expression captures exactly one digit.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "answer_appraisal",
        _pat(
            r"\b(?:the\s+)?(?:answer|response)(?:\s+(?:you\s+)?"
            r"(?:provided|gave|selected|chose))?(?:\s+of)?\s*[:\-]?\s*"
            + MARK
            + r"(?:\s*\([^\n)]{0,100}\)\s*[*_`]*)?\s+(?:is|was)\s+[*_`]*\s*"
            r"(?:absolutely\s+|clearly\s+|entirely\s+)?[*_`]*\s*"
            r"(?:correct|appropriate|reasonable|justified)\b"
        ),
    ),
    (
        "this_is_gloss",
        _pat(
            r"(?:^|[.!?]\s+|\n\s*|:\s*(?:[*_`]+\s*)*)"
            r"this\s+(?:scenario\s+)?is\s+"
            r"(?:an?\s+)?" + MARK + r"\s*\(\s*" + GLOSS + r"\b"
        ),
    ),
    (
        "scenario_action_is_gloss",
        _pat(
            r"(?:^|[.!?]\s+|\n\s*)in\s+this\s+scenario\s*,?\s*"
            r"[^.!?\n]{0,180}?\b(?:action|choice|decision|request|behavior|"
            r"behaviour|response)\b[^.!?\n]{0,100}?\bis\s+"
            + MARK + r"\s*\(\s*" + GLOSS + r"\b"
        ),
    ),
    (
        "personal_take",
        _pat(r"\b(?:my\s+)?personal\s+take\s*[:\-]\s*" + MARK),
    ),
    (
        "verdict_category",
        _pat(
            r"\bfalls\s+firmly\s+in(?:to)?\s+(?:the\s+)?"
            + MARK + r"\s*\(\s*" + GLOSS
            + r"\s*\)\s*[*_`]*\s+category\b"
        ),
    ),
    (
        "moral_value",
        _pat(
            r"\b(?:morally\s+|ethically\s+)?(?:wrong|right|unethical|ethical|"
            r"unacceptable|acceptable|inappropriate|appropriate)\b"
            r"[^.!?\n]{0,100}?\b(?:with\s+)?a\s+value\s+of\s+" + MARK
        ),
    ),
    (
        "reasoning_answer",
        _pat(
            r"\b(?:reasoning|rationale)\s+(?:behind|for)\s+(?:the\s+)?answer\s+"
            + MARK + r"\s*\(\s*" + GLOSS + r"\b"
        ),
    ),
)

FIELDS = (
    "audit_verdict",
    "audit_note",
    "pattern",
    "matched_digit",
    "match_text",
    "model",
    "persona_group",
    "persona_folder",
    "subset",
    "question_id",
    "file",
    "baseline_status",
    "baseline_answer",
    "baseline_parse_source",
    "baseline_stance",
    "baseline_confidence",
    "baseline_abstain_reason",
    "baseline_digit_convention",
    "baseline_response_flags",
    "response_content_full",
)

_CONDITIONAL_BEFORE = re.compile(r"\b(?:if|unless|depending\s+on)\b", re.IGNORECASE)


def conditional_appraisal(text: str, start: int) -> bool:
    """Whether an answer appraisal belongs to an unresolved if-clause."""
    before = text[max(0, start - 240):start]
    boundary = max(before.rfind("."), before.rfind("!"), before.rfind("?"), before.rfind("\n"))
    return bool(_CONDITIONAL_BEFORE.search(before[boundary + 1:]))


def relevant_patterns(lowered: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Cheap literal routing before any of the richer regexes run."""
    wanted: set[str] = set()
    if ("answer" in lowered or "response" in lowered) and any(
        word in lowered for word in ("correct", "appropriate", "reasonable", "justified")
    ):
        wanted.add("answer_appraisal")
    if "this is" in lowered or "this scenario is" in lowered:
        wanted.add("this_is_gloss")
    if "in this scenario" in lowered and any(
        word in lowered
        for word in ("action", "choice", "decision", "request", "behavior", "behaviour", "response")
    ):
        wanted.add("scenario_action_is_gloss")
    if "personal take" in lowered:
        wanted.add("personal_take")
    if "falls firmly" in lowered and "category" in lowered:
        wanted.add("verdict_category")
    if "value of" in lowered:
        wanted.add("moral_value")
    if ("reasoning behind" in lowered or "rationale for" in lowered) and "answer" in lowered:
        wanted.add("reasoning_answer")
    return tuple(item for item in PATTERNS if item[0] in wanted)


def stable_score(row: dict[str, str]) -> int:
    key = "|".join(
        (
            str(SEED),
            row["pattern"],
            row["model"],
            row["persona_folder"],
            row["question_id"],
        )
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")

    csv.field_size_limit(sys.maxsize)
    hits: list[dict[str, str]] = []
    scanned = 0
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            if source.get("persona_group") not in {"regional", "neutral"}:
                continue
            scanned += 1
            raw = source.get("response_content_full", "") or ""
            if not ("0" in raw or "1" in raw):
                continue
            lowered = raw.lower()
            candidates = relevant_patterns(lowered)
            if not candidates:
                continue
            working, _ = normalize.normalise(raw)
            # Masking is deliberately rich and comparatively expensive.  A
            # raw-text pass is only a gate; every recorded hit is still rerun
            # on the masked text below, exactly like the production parser.
            if not any(pattern.search(working) for _, pattern in candidates):
                continue
            masked, _ = masking.mask(working)
            for name, pattern in candidates:
                for match in pattern.finditer(masked):
                    if name == "answer_appraisal" and conditional_appraisal(masked, match.start()):
                        continue
                    hits.append(
                        {
                            "audit_verdict": "",
                            "audit_note": "",
                            "pattern": name,
                            "matched_digit": match.group(1),
                            "match_text": working[match.start():match.end()],
                            "model": source.get("model", ""),
                            "persona_group": source.get("persona_group", ""),
                            "persona_folder": source.get("persona_folder", ""),
                            "subset": source.get("subset", ""),
                            "question_id": source.get("question_id", ""),
                            "file": source.get("file", ""),
                            "baseline_status": source.get("answer_status", ""),
                            "baseline_answer": source.get("extracted_answer", ""),
                            "baseline_parse_source": source.get("parse_source", ""),
                            "baseline_stance": source.get("extracted_stance", ""),
                            "baseline_confidence": source.get("extraction_confidence", ""),
                            "baseline_abstain_reason": source.get("abstain_reason", ""),
                            "baseline_digit_convention": source.get("digit_convention", ""),
                            "baseline_response_flags": source.get("response_flags", ""),
                            "response_content_full": raw,
                        }
                    )

    hits.sort(
        key=lambda row: (
            row["pattern"], row["model"], row["persona_folder"], row["question_id"]
        )
    )
    all_path = args.output_dir / "regional_explicit_appraisal_all_hits.csv"
    write_csv(all_path, hits)

    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in hits:
        identity = (
            row["model"], row["persona_folder"], row["question_id"], row["pattern"]
        )
        unique.setdefault(identity, row)
    population = list(unique.values())
    sample = sorted(population, key=stable_score)[: min(SAMPLE_SIZE, len(population))]
    sample_path = args.output_dir / "regional_explicit_appraisal_audit_sample.csv"
    write_csv(sample_path, sample)

    summary = {
        "audit_date": AUDIT_DATE,
        "method": (
            "blind regex discovery over response_content_full; ground_truth and "
            "is_correct are never read"
        ),
        "input": str(args.input),
        "responses_scanned": scanned,
        "raw_matches": len(hits),
        "unique_response_pattern_hits": len(population),
        "unique_responses": len(
            {(r["model"], r["persona_folder"], r["question_id"]) for r in hits}
        ),
        "by_pattern": dict(Counter(r["pattern"] for r in population)),
        "by_baseline_status": dict(Counter(r["baseline_status"] for r in population)),
        "by_baseline_confidence": dict(
            Counter(r["baseline_confidence"] or "<blank>" for r in population)
        ),
        "committed_digit_disagreements": sum(
            1
            for r in population
            if r["baseline_status"] == "committed"
            and r["baseline_answer"]
            and r["baseline_answer"] != r["matched_digit"]
        ),
        "sample_rows": len(sample),
        "artifacts": {
            "all_hits": str(all_path),
            "all_hits_sha256": sha256_file(all_path),
            "audit_sample": str(sample_path),
            "audit_sample_sha256": sha256_file(sample_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

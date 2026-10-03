"""Audit empty final answers in v9 persona groups without modifying results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RESULT_ROOT = Path("result/benchmarks/ethics/runs/v9_pilot")
REPORT_ROOT = Path("result/archive/reports_mixed_legacy/v9_pilot")


def classify(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pred = data.get("prediction") or {}
    response = data.get("response") or {}
    raw_api = response.get("raw_api_response") or {}
    content = str(pred.get("raw") or pred.get("explanation") or "").strip()
    reasoning = ""
    finish = ""
    choices = raw_api.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = content or str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning") or "").strip()
        finish = str(choice.get("finish_reason") or "")
    if content:
        return "has_final", finish
    if reasoning:
        return "reasoning_only", finish
    return "no_output", finish


def religion_candidates() -> list[Path]:
    paths: list[Path] = []
    report_dir = REPORT_ROOT / "religion"
    for report in report_dir.glob("*/results_religion_*.csv"):
        model = report.parent.name
        with report.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("answer_status") == "missing":
                    paths.append(
                        RESULT_ROOT / "Religion Persona" / model
                        / row["persona_folder"] / row["file"]
                    )
    return paths


def regional_candidates() -> list[Path]:
    root = RESULT_ROOT / "Regional Cultural Persona"
    candidates: list[Path] = []
    needle = b'"raw": ""'
    for path in root.glob("*/*/*.json"):
        try:
            if needle in path.read_bytes():
                candidates.append(path)
        except OSError:
            candidates.append(path)
    return candidates


def audit(group: str, paths: list[Path]) -> dict:
    root = RESULT_ROOT / group
    counts: Counter[tuple[str, str, str, str]] = Counter()
    errors: list[str] = []
    for path in paths:
        try:
            kind, finish = classify(path)
            relative = path.relative_to(root)
            model, persona = relative.parts[:2]
            counts[(kind, model, persona, finish)] += 1
        except Exception as exc:  # audit must report unreadable source files
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "group": group,
        "candidates": len(paths),
        "counts": [
            {
                "kind": key[0], "model": key[1], "persona": key[2],
                "finish_reason": key[3], "count": value,
            }
            for key, value in sorted(counts.items())
        ],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["religion", "regional", "all"], default="all")
    args = parser.parse_args()
    outputs = []
    if args.group in {"religion", "all"}:
        outputs.append(audit("Religion Persona", religion_candidates()))
    if args.group in {"regional", "all"}:
        outputs.append(audit("Regional Cultural Persona", regional_candidates()))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Refresh only formerly blank Qwen Gender report rows after the 4096 rerun.

The default is a read-only dry-run.  ``--execute`` backs up the exact report
files, rewrites the formerly blank rows from canonical JSON, and regenerates
the two Qwen-only accuracy summaries.  Rows that already had an extracted
answer are preserved as data values and are never reparsed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
RAW_ROOT = REPO_ROOT / "result" / "GMS8K" / "Gender_Persona" / MODEL_SLUG
REPORT_ROOT = REPO_ROOT / "result" / "reports" / MODEL_SLUG
BACKUP_ROOT = (
    REPO_ROOT
    / "result"
    / "_merge_backup"
    / "gsm8k_gender_qwen_truncated_4096"
)
PARSER_PATH = REPO_ROOT / "src" / "rebuild" / "rebuild_results_csv_qwen.py"
PERSONAS = [
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
    "neutral",
]
RESULT_FILES = [
    *(REPORT_ROOT / f"results_{persona}.csv" for persona in PERSONAS),
    REPORT_ROOT / "results_qwen3_next_80b.csv",
    REPORT_ROOT / "result_qwen_qwen3-next-80b-a3b-thinking.csv",
]
ACCURACY_FILE = REPORT_ROOT / "accuracy.csv"
SUMMARY_FILE = REPORT_ROOT / "accuracy_summary.csv"
EXPECTED_TOTAL = 11868
EXPECTED_OLD_BLANK = 323
EXPECTED_NEW_EXTRACTED_FROM_BLANK = 295
EXPECTED_NEW_BLANK = 28


def load_parser():
    spec = importlib.util.spec_from_file_location("qwen_report_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser: {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def safe_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def report_key(row: dict[str, str]) -> tuple[str, str]:
    persona = str(row.get("persona_folder") or "").strip()
    file_value = str(row.get("file") or "")
    match = re.search(r"(gsm8k_\d+\.json)$", file_value.replace("\\", "/"))
    if persona not in PERSONAS or not match:
        raise RuntimeError(f"Unsafe or invalid report row: {persona!r}, {file_value!r}")
    return persona, match.group(1)


def raw_path(key: tuple[str, str]) -> Path:
    persona, filename = key
    path = RAW_ROOT / persona / filename
    try:
        path.resolve().relative_to(RAW_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Raw path escapes scope: {key}") from exc
    if not path.is_file():
        raise RuntimeError(f"Raw JSON is missing: {path}")
    return path


def choice_and_usage(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = safe_dict(response.get("raw_api_response"))
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    return safe_dict(choice), safe_dict(raw.get("usage"))


def refresh_blank_row(
    original: dict[str, str], parser: Any
) -> tuple[dict[str, str], str]:
    key = report_key(original)
    try:
        payload = json.loads(raw_path(key).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot load canonical JSON for {key}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Canonical JSON root is not an object: {key}")
    response = safe_dict(payload.get("response"))
    request = safe_dict(payload.get("request"))
    content = parser.get_response_text(response)
    extracted, reason = parser.extract_final_answer(content)
    choice, usage = choice_and_usage(response)
    finish_reason = choice.get("finish_reason")
    parameters = safe_dict(request.get("parameters"))
    max_tokens = parameters.get("max_tokens")
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    reasoning_tokens = usage.get("reasoning_tokens")
    gt_value = parser.normalize_number(original.get("ground_truth_answer"))
    is_correct = extracted is not None and gt_value is not None and extracted == gt_value
    has_boxed = bool(re.search(r"\\boxed\{[^}]+\}", content))
    has_hashes = bool(re.search(r"####\s*-?[$]?[\d,]+", content))
    truncated_metadata = finish_reason == "length"
    truncated_content = parser.is_truly_truncated(
        content, finish_reason, completion_tokens, max_tokens
    )
    has_final = parser.has_final_marker(content)
    if extracted is not None:
        category = "ok"
    elif truncated_content:
        category = "truncated_output"
    else:
        category = "no_number_found"

    updated = dict(original)
    updated.update(
        {
            "response_content_full": content,
            "extracted_answer": "" if extracted is None else str(extracted),
            "is_correct": bool_text(is_correct),
            "has_boxed": bool_text(has_boxed),
            "has_hashes": bool_text(has_hashes),
            "boxed_match": bool_text(
                has_boxed and is_correct and reason == "boxed"
            ),
            "hashes_match": bool_text(
                has_hashes and is_correct and reason == "hashes"
            ),
            "extraction_issue": bool_text(extracted is None),
            "error_category": category,
            "extraction_reason": reason,
            "finish_reason": "" if finish_reason is None else str(finish_reason),
            "prompt_tokens": "" if prompt_tokens is None else str(prompt_tokens),
            "completion_tokens": (
                "" if completion_tokens is None else str(completion_tokens)
            ),
            "reasoning_tokens": (
                "" if reasoning_tokens is None else str(reasoning_tokens)
            ),
            "total_tokens": "" if total_tokens is None else str(total_tokens),
            "truncated_by_metadata": bool_text(truncated_metadata),
            "truncated_by_content": bool_text(truncated_content),
            "has_final_marker": bool_text(has_final),
            "text_tail": content.strip()[-120:] if content else "",
        }
    )
    return updated, category


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise RuntimeError(f"Cannot read report CSV {path}: {exc}") from exc


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    quote_all: bool,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def truth(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def number(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:
        raise RuntimeError(f"Invalid numeric report value: {value!r}") from exc


def build_accuracy_rows(
    persona_rows: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    fields = [
        "persona",
        "correct",
        "total",
        "total_ok",
        "accuracy",
        "no_extract",
        "truncated",
        "sum_prompt_tokens",
        "sum_completion_tokens",
        "sum_reasoning_tokens",
        "sum_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_total_tokens",
    ]
    output: list[dict[str, object]] = []
    for persona in [*PERSONAS, "OVERALL"]:
        rows = (
            persona_rows[persona]
            if persona != "OVERALL"
            else [row for name in PERSONAS for row in persona_rows[name]]
        )
        total = len(rows)
        correct = sum(truth(row.get("is_correct")) for row in rows)
        extracted = sum(bool(str(row.get("extracted_answer") or "").strip()) for row in rows)
        truncated = sum(
            not str(row.get("extracted_answer") or "").strip()
            and row.get("error_category") == "truncated_output"
            for row in rows
        )
        no_extract = total - extracted - truncated
        sums = {
            key: sum(number(row.get(key)) for row in rows)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        }
        values: dict[str, object] = {
            "persona": persona,
            "correct": correct,
            "total": total,
            "total_ok": extracted,
            # Primary accuracy is deliberately standardized to all responses.
            "accuracy": round(100 * correct / total, 4) if total else 0.0,
            "no_extract": no_extract,
            "truncated": truncated,
            "sum_prompt_tokens": sums["prompt_tokens"],
            "sum_completion_tokens": sums["completion_tokens"],
            "sum_reasoning_tokens": sums["reasoning_tokens"],
            "sum_total_tokens": sums["total_tokens"],
            "avg_prompt_tokens": round(sums["prompt_tokens"] / total, 2) if total else 0.0,
            "avg_completion_tokens": round(sums["completion_tokens"] / total, 2) if total else 0.0,
            "avg_reasoning_tokens": round(sums["reasoning_tokens"] / total, 2) if total else 0.0,
            "avg_total_tokens": round(sums["total_tokens"] / total, 2) if total else 0.0,
        }
        if list(values) != fields:
            raise AssertionError("Accuracy field order changed")
        output.append(values)
    return output


def build_summary_rows(
    persona_rows: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    # Preserve the legacy summary's gender-only scope (eight personas, no neutral).
    gender_personas = [persona for persona in PERSONAS if persona != "neutral"]
    output: list[dict[str, object]] = []
    for persona in [*gender_personas, "OVERALL"]:
        rows = (
            persona_rows[persona]
            if persona != "OVERALL"
            else [row for name in gender_personas for row in persona_rows[name]]
        )
        total = len(rows)
        ok = sum(bool(str(row.get("extracted_answer") or "").strip()) for row in rows)
        truncated = sum(
            not str(row.get("extracted_answer") or "").strip()
            and row.get("error_category") == "truncated_output"
            for row in rows
        )
        no_extract = total - ok - truncated
        output.append(
            {
                "Persona": persona,
                "OK": ok,
                "Truncated": truncated,
                "No-Extract": no_extract,
                "Total": total,
                "Accuracy (%)": round(100 * ok / total, 2) if total else 0.0,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parser = load_parser()
    loaded: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}
    blank_keys_by_file: dict[Path, set[tuple[str, str]]] = {}
    unique_blank_keys: set[tuple[str, str]] = set()
    for path in RESULT_FILES:
        fields, rows = read_csv(path)
        loaded[path] = (fields, rows)
        blanks = {
            report_key(row)
            for row in rows
            if not str(row.get("extracted_answer") or "").strip()
        }
        blank_keys_by_file[path] = blanks
        unique_blank_keys.update(blanks)

    persona_files = RESULT_FILES[: len(PERSONAS)]
    persona_blanks = set().union(*(blank_keys_by_file[path] for path in persona_files))
    if len(persona_blanks) != EXPECTED_OLD_BLANK:
        raise RuntimeError(
            f"Old blank guard expected {EXPECTED_OLD_BLANK}, found {len(persona_blanks)}"
        )
    if unique_blank_keys != persona_blanks:
        raise RuntimeError("Combined reports contain a different blank-row key set")

    refreshed: dict[tuple[str, str], dict[str, str]] = {}
    categories: Counter[str] = Counter()
    for key in sorted(unique_blank_keys):
        # Use the matching persona report row as the immutable metadata template.
        persona_path = REPORT_ROOT / f"results_{key[0]}.csv"
        template = next(
            row for row in loaded[persona_path][1] if report_key(row) == key
        )
        updated, category = refresh_blank_row(template, parser)
        refreshed[key] = updated
        categories[category] += 1

    newly_extracted = sum(
        bool(str(row.get("extracted_answer") or "").strip())
        for row in refreshed.values()
    )
    remaining_blank = len(refreshed) - newly_extracted
    if newly_extracted != EXPECTED_NEW_EXTRACTED_FROM_BLANK:
        raise RuntimeError(
            "New extraction guard expected "
            f"{EXPECTED_NEW_EXTRACTED_FROM_BLANK}, found {newly_extracted}"
        )
    if remaining_blank != EXPECTED_NEW_BLANK:
        raise RuntimeError(
            f"Remaining blank guard expected {EXPECTED_NEW_BLANK}, found {remaining_blank}"
        )

    updated_files: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}
    for path, (fields, rows) in loaded.items():
        updated_rows: list[dict[str, str]] = []
        changed = 0
        for row in rows:
            if str(row.get("extracted_answer") or "").strip():
                updated_rows.append(row)
                continue
            key = report_key(row)
            replacement = dict(row)
            for field in fields:
                if field in refreshed[key]:
                    replacement[field] = refreshed[key][field]
            updated_rows.append(replacement)
            changed += 1
        if changed != len(blank_keys_by_file[path]):
            raise RuntimeError(f"Duplicate/missing blank row while updating {path}")
        updated_files[path] = (fields, updated_rows)

    persona_rows = {
        persona: updated_files[REPORT_ROOT / f"results_{persona}.csv"][1]
        for persona in PERSONAS
    }
    total_rows = sum(len(rows) for rows in persona_rows.values())
    total_extracted = sum(
        bool(str(row.get("extracted_answer") or "").strip())
        for rows in persona_rows.values()
        for row in rows
    )
    if total_rows != EXPECTED_TOTAL or total_extracted != EXPECTED_TOTAL - EXPECTED_NEW_BLANK:
        raise RuntimeError(
            f"Final coverage guard failed: extracted={total_extracted}, total={total_rows}"
        )
    accuracy_rows = build_accuracy_rows(persona_rows)
    summary_rows = build_summary_rows(persona_rows)

    print(f"old_blank_rows={len(unique_blank_keys)}")
    print(f"newly_extracted={newly_extracted}")
    print(f"remaining_blank={remaining_blank}")
    print(f"coverage={total_extracted}/{total_rows} ({100 * total_extracted / total_rows:.4f}%)")
    print("scope=Qwen Gender report files only")
    print("categories=" + ",".join(f"{key}:{categories[key]}" for key in sorted(categories)))
    if not args.execute:
        print("dry_run=true report_files_modified=0")
        return 0

    refresh_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")
    backup_run = BACKUP_ROOT / refresh_id / "reports" / MODEL_SLUG
    targets = [*RESULT_FILES, ACCURACY_FILE, SUMMARY_FILE]
    for target in targets:
        if not target.is_file():
            raise RuntimeError(f"Report target is missing: {target}")
        backup = backup_run / target.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)

    for path, (fields, rows) in updated_files.items():
        atomic_write_csv(path, fields, rows, quote_all=True)
    accuracy_fields = list(accuracy_rows[0])
    summary_fields = list(summary_rows[0])
    atomic_write_csv(ACCURACY_FILE, accuracy_fields, accuracy_rows, quote_all=False)
    atomic_write_csv(SUMMARY_FILE, summary_fields, summary_rows, quote_all=False)
    print(f"updated_report_files={len(targets)}")
    print(f"backup={backup_run}")
    print("publication_aggregate_untouched=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

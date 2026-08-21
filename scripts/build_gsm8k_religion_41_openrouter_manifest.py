"""Build the exact 41-row GSM8K Religion OpenRouter rerun scope.

Selected:
* GPT-OSS: 38 choice ``finish_reason=error`` rows with no error detail.
* GPT-OSS: 1 DNS resolution failure for openrouter.ai.
* Qwen: 2 choice ``finish_reason=error`` rows with no error detail.

Explicitly excluded are the seven GPT-OSS choice-level 502 provider errors and
the four Qwen HTTP-403 key-limit errors.  This script performs no API calls and
never modifies source result JSON files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP = "Religion_Persona"
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports" / GROUP
RAW_ROOT = REPO_ROOT / "result" / "GMS8K" / GROUP
OUTPUT_ROOT = (
    REPO_ROOT / "result" / "rerun_manifests" / "gsm8k_religion_41_openrouter"
)


@dataclass(frozen=True)
class ModelScope:
    key: str
    slug: str
    model_id: str
    expected_technical: int
    expected_selected: int
    expected_excluded: int
    output_name: str


SCOPES = (
    ModelScope(
        key="gpt_oss",
        slug="openai_gpt-oss-20b",
        model_id="openai/gpt-oss-20b",
        expected_technical=46,
        expected_selected=39,
        expected_excluded=7,
        output_name="gpt_oss_religion_39.csv",
    ),
    ModelScope(
        key="qwen",
        slug="qwen_qwen3-next-80b-a3b-thinking",
        model_id="qwen/qwen3-next-80b-a3b-thinking",
        expected_technical=6,
        expected_selected=2,
        expected_excluded=4,
        output_name="qwen_religion_2.csv",
    ),
)
EXPECTED_TOTAL = 41

FIELDS = [
    "dataset", "model_slug", "model_id", "group", "persona_folder",
    "question_id", "issue_category", "http_code", "error_message",
    "source_state", "template_relative_path", "question_source_relative_path",
    "relative_path", "source_json", "source_sha256", "response_success",
    "finish_reason", "choice_error_code", "choice_error_message",
    "partial_content_length", "request_model", "original_max_tokens",
    "original_temperature", "selection_reason", "rerun_status",
]


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def first_choice(raw: object) -> dict:
    if not isinstance(raw, dict):
        return {}
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    return choices[0] if isinstance(choices[0], dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection(scope: ModelScope, report_row: dict, payload: dict) -> tuple[bool, str]:
    """Classify one technical row without consulting its ground truth."""
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    raw = raw if isinstance(raw, dict) else {}
    choice = first_choice(raw)
    choice_error = choice.get("error")
    report_error = " ".join(str(report_row.get("error_message") or "").split())

    if (
        scope.key == "gpt_oss"
        and report_row.get("error_class") == "provider_error"
        and "resolve 'openrouter.ai'" in report_error.lower()
    ):
        return True, "dns_resolution_failure"

    if choice.get("finish_reason") == "error" and choice_error is None:
        return True, "finish_reason_error_without_detail"

    if scope.key == "gpt_oss" and isinstance(choice_error, dict):
        if (
            str(choice_error.get("code")) == "502"
            and choice_error.get("message") == "JSON error injected into SSE stream"
        ):
            return False, "excluded_choice_502_provider_unavailable"

    if (
        scope.key == "qwen"
        and report_row.get("error_class") == "provider_error"
        and "key limit exceeded" in report_error.lower()
    ):
        return False, "excluded_http_403_key_limit"

    raise RuntimeError(
        f"Unexpected technical failure classification: {scope.slug}/"
        f"{report_row.get('file')}"
    )


def build_scope(scope: ModelScope) -> tuple[list[dict[str, object]], Counter]:
    report_path = REPORT_ROOT / scope.slug / "results.csv"
    model_root = RAW_ROOT / scope.slug
    csv.field_size_limit(sys.maxsize)
    selected: list[dict[str, object]] = []
    classifications: Counter[str] = Counter()
    seen: set[str] = set()

    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        for report_row in csv.DictReader(handle):
            if not as_bool(report_row.get("technical_failure")):
                continue
            relative_in_model = Path(str(report_row.get("file") or ""))
            source_path = model_root / relative_in_model
            if not source_path.is_file():
                raise RuntimeError(f"Missing source JSON: {source_path}")
            payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
            include, reason = selection(scope, report_row, payload)
            classifications[reason] += 1
            if not include:
                continue

            relative = (Path(GROUP) / scope.slug / relative_in_model).as_posix()
            if relative in seen:
                raise RuntimeError(f"Duplicate selected path: {relative}")
            seen.add(relative)
            response = payload.get("response") or {}
            raw = response.get("raw_api_response") or {}
            choice = first_choice(raw)
            choice_error = choice.get("error") or {}
            request_parameters = (payload.get("request") or {}).get("parameters") or {}
            content = response.get("content")
            content = content if isinstance(content, str) else ""
            question_id = relative_in_model.stem
            if relative_in_model.name != f"{question_id}.json":
                raise RuntimeError(f"Invalid question path: {relative_in_model}")
            if request_parameters.get("model") != scope.model_id:
                raise RuntimeError(f"Request model mismatch: {source_path}")

            selected.append({
                "dataset": "GSM8K",
                "model_slug": scope.slug,
                "model_id": scope.model_id,
                "group": GROUP,
                "persona_folder": report_row.get("persona") or "",
                "question_id": question_id,
                "issue_category": reason,
                "http_code": "",
                "error_message": report_row.get("error_message") or reason,
                "source_state": "valid",
                "template_relative_path": "",
                "question_source_relative_path": "",
                "relative_path": relative,
                "source_json": relative,
                "source_sha256": file_sha256(source_path),
                "response_success": response.get("success"),
                "finish_reason": choice.get("finish_reason") or "",
                "choice_error_code": choice_error.get("code", ""),
                "choice_error_message": choice_error.get("message", ""),
                "partial_content_length": len(content),
                "request_model": request_parameters.get("model") or "",
                "original_max_tokens": request_parameters.get("max_tokens", ""),
                "original_temperature": request_parameters.get("temperature", ""),
                "selection_reason": reason,
                "rerun_status": "pending",
            })

    technical_count = sum(classifications.values())
    if technical_count != scope.expected_technical:
        raise RuntimeError(
            f"{scope.slug}: expected {scope.expected_technical} technical rows, "
            f"found {technical_count}"
        )
    if len(selected) != scope.expected_selected:
        raise RuntimeError(
            f"{scope.slug}: expected {scope.expected_selected} selected rows, "
            f"found {len(selected)}"
        )
    if technical_count - len(selected) != scope.expected_excluded:
        raise RuntimeError(f"{scope.slug}: excluded-row guard failed")
    selected.sort(key=lambda row: (str(row["persona_folder"]), str(row["question_id"])))
    return selected, classifications


def write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    combined: list[dict[str, object]] = []
    for scope in SCOPES:
        rows, classifications = build_scope(scope)
        output = OUTPUT_ROOT / scope.output_name
        write_atomic(output, rows)
        combined.extend(rows)
        print(f"manifest={output}")
        print(f"selected={len(rows)} classifications={dict(classifications)}")
    if len(combined) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Combined guard expected {EXPECTED_TOTAL}, found {len(combined)}"
        )
    if len({str(row["source_json"]) for row in combined}) != EXPECTED_TOTAL:
        raise RuntimeError("Combined manifest contains duplicate paths")
    combined_path = OUTPUT_ROOT / "gsm8k_religion_41.csv"
    write_atomic(combined_path, combined)
    print(f"combined_manifest={combined_path} selected={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

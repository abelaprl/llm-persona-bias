"""Build the exact technical-rerun manifest for GSM8K Llama 3.3 70B.

This script performs no API calls and never modifies source result JSON files.
It selects only rows already classified as ``finish_reason=error`` by the
audited parser, then verifies the choice-level provider error in each raw JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "meta-llama_llama-3.3-70b-instruct"
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"
PERSONA_GROUP = "Religion_Persona"
EXPECTED_COUNT = 202

DEFAULT_RESULTS = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports_parser_v2"
    / PERSONA_GROUP
    / MODEL_SLUG
    / "results.csv"
)
DEFAULT_RAW_ROOT = (
    REPO_ROOT / "result" / "GMS8K" / PERSONA_GROUP / MODEL_SLUG
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_llama33_70b"
    / "llama33_70b_religion_cloudflare500.csv"
)

FIELDS = [
    # Columns consumed by rerun_llama33_70b_openrouter.py.
    "dataset",
    "model_slug",
    "model_id",
    "group",
    "persona_folder",
    "question_id",
    "relative_path",
    "http_code",
    "error_message",
    "source_state",
    "template_relative_path",
    "question_source_relative_path",
    # Audit/provenance columns retained for the technical manifest.
    "source_json",
    "run_id",
    "issue_category",
    "response_success",
    "finish_reason",
    "native_finish_reason",
    "provider",
    "provider_error_code",
    "provider_error_message",
    "provider_error_type",
    "partial_content_length",
    "original_temperature",
    "rerun_temperature",
    "original_max_tokens",
    "rerun_max_tokens",
    "request_model",
    "retry_count_existing",
    "source_sha256",
    "eligible_for_technical_rerun",
    "rerun_reason",
    "rerun_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the exact 202-row Llama 70B rerun list."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def first_choice(raw_response: object) -> dict:
    if not isinstance(raw_response, dict):
        return {}
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    return choices[0] if isinstance(choices[0], dict) else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(results_path: Path, raw_root: Path) -> list[dict[str, object]]:
    csv.field_size_limit(1 << 30)
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        result_rows = list(csv.DictReader(handle))

    targets = [row for row in result_rows if row.get("finish_reason") == "error"]
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for result_row in targets:
        relative_in_model = Path(result_row["file"])
        source_path = raw_root / relative_in_model
        if not source_path.is_file():
            raise RuntimeError(f"Missing source JSON: {source_path}")

        source_key = (
            Path(PERSONA_GROUP) / MODEL_SLUG / relative_in_model
        ).as_posix()
        question_id = relative_in_model.stem
        if source_key in seen:
            raise RuntimeError(f"Duplicate target in report CSV: {source_key}")
        seen.add(source_key)

        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        metadata = data.get("metadata") or {}
        model_info = metadata.get("model_info") or {}
        response = data.get("response") or {}
        raw_response = response.get("raw_api_response") or {}
        choice = first_choice(raw_response)
        choice_error = choice.get("error") or {}
        error_meta = choice_error.get("metadata") or {}
        request_params = (data.get("request") or {}).get("parameters") or {}
        performance = response.get("performance") or {}
        content = response.get("content")
        if not isinstance(content, str):
            content = ((choice.get("message") or {}).get("content") or "")
        content = content if isinstance(content, str) else ""

        model_id = model_info.get("model_id")
        finish_reason = choice.get("finish_reason")
        provider = raw_response.get("provider")
        provider_code = choice_error.get("code")
        provider_message = choice_error.get("message")
        provider_type = error_meta.get("error_type")

        invariants = {
            "model_id": model_id == MODEL_ID,
            "report_response_success": as_bool(result_row.get("response_success")),
            "raw_response_success": response.get("success") is True,
            "finish_reason": finish_reason == "error",
            "provider": provider == "Cloudflare",
            "provider_error_code": str(provider_code) == "500",
            "provider_error_message": provider_message == "Internal Server Error",
            "provider_error_type": provider_type == "server",
            "partial_content": bool(content.strip()),
            "recorded_temperature": float(request_params.get("temperature")) == 0.2,
            "max_tokens": int(request_params.get("max_tokens")) == 4096,
            "request_model": request_params.get("model") == MODEL_ID,
        }
        failed = [name for name, ok in invariants.items() if not ok]
        if failed:
            raise RuntimeError(
                f"Target {source_key} violates rerun invariants: {', '.join(failed)}"
            )

        rows.append(
            {
                "dataset": "GSM8K",
                "model_slug": MODEL_SLUG,
                "model_id": MODEL_ID,
                "group": PERSONA_GROUP,
                "persona_folder": result_row["persona"],
                "question_id": question_id,
                "relative_path": source_key,
                "http_code": provider_code,
                "error_message": provider_message,
                "source_state": "valid",
                "template_relative_path": "",
                "question_source_relative_path": "",
                "source_json": source_key,
                "run_id": metadata.get("run_id", ""),
                "issue_category": "provider_server_500_partial",
                "response_success": True,
                "finish_reason": finish_reason,
                "native_finish_reason": choice.get("native_finish_reason") or "",
                "provider": provider,
                "provider_error_code": provider_code,
                "provider_error_message": provider_message,
                "provider_error_type": provider_type,
                "partial_content_length": len(content),
                # A technical rerun must preserve the condition actually stored
                # in the source JSON.  Changing only these 202 rows to temperature
                # 0 would mix generation conditions inside one experiment.
                "original_temperature": request_params.get("temperature"),
                "rerun_temperature": request_params.get("temperature"),
                "original_max_tokens": request_params.get("max_tokens"),
                "rerun_max_tokens": 4096,
                "request_model": request_params.get("model"),
                "retry_count_existing": performance.get("retry_count", ""),
                "source_sha256": sha256(source_path),
                "eligible_for_technical_rerun": True,
                "rerun_reason": (
                    "Cloudflare provider returned choice-level 500 Internal "
                    "Server Error after partial generation"
                ),
                "rerun_status": "pending",
            }
        )

    rows.sort(key=lambda row: (str(row["persona_folder"]), str(row["question_id"])))
    return rows


def write_atomic(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.temporary")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    rows = build_manifest(args.results, args.raw_root)
    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} rerun targets, found {len(rows)}"
        )
    write_atomic(rows, args.output)

    personas = Counter(str(row["persona_folder"]) for row in rows)
    lengths = sorted(int(row["partial_content_length"]) for row in rows)
    print(f"manifest={args.output}")
    print(f"targets={len(rows)} unique_paths={len({row['source_json'] for row in rows})}")
    print(
        "content_length="
        f"min:{lengths[0]} median:{lengths[len(lengths) // 2]} max:{lengths[-1]}"
    )
    for persona, count in sorted(personas.items()):
        print(f"{persona}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

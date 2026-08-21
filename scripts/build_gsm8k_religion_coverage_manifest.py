"""Build an exact rerun manifest from a canonical GSM8K Religion report.

By default, uncommitted infrastructure failures, empty responses, content
filters, and metadata/content truncations are selected. ``--only-provider-errors``
narrows the output to rows whose canonical report explicitly marks a provider
failure. Parser candidates and ordinary model abstentions are left untouched.
Ground truth is never copied into the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP = "Religion_Persona"
FIELDS = [
    "dataset",
    "model_slug",
    "group",
    "persona_folder",
    "question_id",
    "issue_category",
    "http_code",
    "error_message",
    "source_state",
    "template_relative_path",
    "question_source_relative_path",
    "relative_path",
    "original_temperature",
    "rerun_temperature",
    "max_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--group",
        default=GROUP,
        help="Group label written to the manifest (default: Religion_Persona).",
    )
    parser.add_argument(
        "--raw-group",
        default=None,
        help="Raw directory below result/GMS8K; default equals --group.",
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "result" / "GMS8K")
    parser.add_argument("--rerun-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    technical_mode = parser.add_mutually_exclusive_group()
    technical_mode.add_argument(
        "--exclude-technical",
        action="store_true",
        help="Leave true provider/API failures out of the generated manifest.",
    )
    technical_mode.add_argument(
        "--only-provider-errors",
        action="store_true",
        help=(
            "Select only rows marked technical_failure=true and "
            "error_class=provider_error in the canonical report."
        ),
    )
    args = parser.parse_args()
    if args.rerun_temperature < 0:
        parser.error("--rerun-temperature must be zero or greater")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least one")
    for name in ("results", "output", "raw_root"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, REPO_ROOT / value)
    return args


def response_error(payload: dict[str, Any]) -> tuple[str, str]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    raw = raw if isinstance(raw, dict) else {}
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    error = response.get("error") or raw.get("error") or choice.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "")
        message = str(error.get("message") or error.get("error") or error)
        return code, message.replace("\r", " ").replace("\n", " ")
    return "", str(error or "").replace("\r", " ").replace("\n", " ")


def selected(
    row: dict[str, str],
    *,
    include_technical: bool = True,
    only_provider_errors: bool = False,
) -> bool:
    finish = (row.get("finish_reason") or "").strip()
    technical = (row.get("technical_failure") or "").strip().lower() == "true"
    provider_error = (row.get("error_class") or "").strip() == "provider_error"
    if only_provider_errors:
        return technical and provider_error
    if row.get("answer_status") == "committed":
        return False
    content_empty = (row.get("content_empty") or "").strip().lower() == "true"
    content_truncated = (
        (row.get("truncated_by_content") or "").strip().lower() == "true"
    )
    if technical:
        return include_technical
    return (
        content_empty
        or content_truncated
        or finish in {"length", "error", "content_filter"}
    )


def issue_category(row: dict[str, str]) -> str:
    if (row.get("technical_failure") or "").strip().lower() == "true":
        return (row.get("error_class") or "technical_failure").strip()
    finish = (row.get("finish_reason") or "").strip()
    if finish == "length":
        return "finish_reason_length"
    if finish == "content_filter":
        return "content_filter"
    if (row.get("truncated_by_content") or "").strip().lower() == "true":
        return "content_truncated"
    if (row.get("content_empty") or "").strip().lower() == "true" and not (
        (row.get("technical_failure") or "").strip().lower() == "true"
    ):
        return "empty_response"
    return (row.get("error_class") or "technical_failure").strip()


def load_payload(path: Path) -> tuple[dict[str, Any], str]:
    """Load a normal object or recover a complete object before corrupt tails."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
        source_state = "valid"
    except json.JSONDecodeError as exc:
        stripped = text.lstrip()
        try:
            payload, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            raise RuntimeError(f"raw JSON cannot be recovered: {path}: {exc}") from exc
        if not stripped[end:].strip():
            raise RuntimeError(f"raw JSON unexpectedly failed strict loading: {path}")
        source_state = "truncated"
    if not isinstance(payload, dict):
        raise RuntimeError(f"raw JSON is not an object: {path}")
    return payload, source_state


def atomic_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    raw_group = args.raw_group or args.group
    raw_model_root = args.raw_root / raw_group / args.model
    output_rows: list[dict[str, object]] = []
    seen: set[str] = set()

    with args.results.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"persona", "question_id", "file", "answer_status", "finish_reason"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"results CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if not selected(
                row,
                include_technical=not args.exclude_technical,
                only_provider_errors=args.only_provider_errors,
            ):
                continue
            relative_in_model = Path(row["file"])
            raw_path = raw_model_root / relative_in_model
            payload, source_state = load_payload(raw_path)
            persona = row["persona"]
            if relative_in_model.parts[:1] != (persona,):
                raise RuntimeError(f"persona/path mismatch: {raw_path}")
            qid = relative_in_model.stem
            expected_qid = f"gsm8k_{int(row['question_id']):05d}"
            if qid != expected_qid:
                raise RuntimeError(f"question/path mismatch: {raw_path}")
            relative = (Path(raw_group) / args.model / relative_in_model).as_posix()
            if relative in seen:
                raise RuntimeError(f"duplicate target: {relative}")
            seen.add(relative)

            request = payload.get("request")
            request = request if isinstance(request, dict) else {}
            parameters = request.get("parameters")
            parameters = parameters if isinstance(parameters, dict) else {}
            code, message = response_error(payload)
            if args.only_provider_errors and not (code or message):
                raise RuntimeError(
                    "provider-error report row has no error in its raw JSON: "
                    f"{raw_path}"
                )
            output_rows.append(
                {
                    "dataset": "GSM8K",
                    "model_slug": args.model,
                    "group": args.group,
                    "persona_folder": persona,
                    "question_id": qid,
                    "issue_category": issue_category(row),
                    "http_code": code,
                    "error_message": message,
                    "source_state": source_state,
                    "template_relative_path": "",
                    "question_source_relative_path": "",
                    "relative_path": relative,
                    "original_temperature": parameters.get("temperature", ""),
                    "rerun_temperature": args.rerun_temperature,
                    "max_tokens": args.max_tokens,
                }
            )

    output_rows.sort(
        key=lambda row: (str(row["persona_folder"]), str(row["question_id"]))
    )
    if not output_rows:
        raise RuntimeError("no eligible coverage failures were found")
    atomic_write(args.output, output_rows)
    print(f"rows={len(output_rows)}")
    print(f"issues={dict(sorted(Counter(row['issue_category'] for row in output_rows).items()))}")
    print(f"personas={dict(sorted(Counter(row['persona_folder'] for row in output_rows).items()))}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

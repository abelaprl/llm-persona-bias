"""Build an immutable snapshot of current ETHICS API-failure rerun targets.

The builder reads the canonical v9 reports, inspects only rows currently marked
``missing``, and selects failures caused by API/transport/provider errors. It
deliberately excludes token-length exhaustion, content filtering, and responses
that already contain final assistant content.

Safe default: audit and print counts without writing files. Use ``--write`` to
create five backend/model/scope-specific manifests under a new snapshot folder.
Existing snapshot files are never replaced with different content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / "result" / "reports" / "v9_main"
RAW_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
OUTPUT_ROOT = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "ethics_v9"
    / "api_failures_current_2026-08-21"
)

GROUP_DIRECTORIES = {
    "gender": "Gender Persona",
    "neutral": "Neutral Persona",
    "regional": "Regional Cultural Persona",
    "religion": "Religion Persona",
}
MODELS = (
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)
REGIONAL_GROUP = "Regional Cultural Persona"

MANIFEST_SPECS = {
    ("deepseek_deepseek-r1-distill-qwen-32b", "all"): {
        "filename": "deepseek_vllm_104.csv",
        "expected": 104,
        "backend": "vllm",
    },
    ("openai_gpt-oss-20b", "nonregional"): {
        "filename": "gpt_oss_openrouter_nonregional_47.csv",
        "expected": 47,
        "backend": "openrouter",
    },
    ("openai_gpt-oss-20b", "regional"): {
        "filename": "gpt_oss_openrouter_regional_9.csv",
        "expected": 9,
        "backend": "openrouter",
    },
    ("qwen_qwen3-next-80b-a3b-thinking", "nonregional"): {
        "filename": "qwen_openrouter_nonregional_46.csv",
        "expected": 46,
        "backend": "openrouter",
    },
    ("qwen_qwen3-next-80b-a3b-thinking", "regional"): {
        "filename": "qwen_openrouter_regional_13.csv",
        "expected": 13,
        "backend": "openrouter",
    },
}
EXPECTED_TOTAL = 219

FIELDNAMES = (
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
    "source_report",
    "report_row",
    "raw_sha256",
)


def text_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return ""


def first_present(*values: object) -> object | None:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif value not in (None, {}, []):
            return value
    return None


def compact(value: object, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(text.split())[:limit]


def sanitize_error(value: object) -> str:
    text = compact(value, limit=2000)
    text = re.sub(r"https?://\S+", "[URL omitted]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*\S+",
        r"\1=[omitted]",
        text,
    )
    return text[:500]


def response_parts(payload: dict[str, Any], report_finish: str) -> dict[str, object]:
    prediction = payload.get("prediction")
    prediction = prediction if isinstance(prediction, dict) else {}
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw_api = response.get("raw_api_response")
    if not isinstance(raw_api, dict):
        raw_api = response.get("raw_response")
    raw_api = raw_api if isinstance(raw_api, dict) else {}
    choices = raw_api.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}

    finish_reason = str(
        first_present(
            choice.get("finish_reason"),
            response.get("finish_reason"),
            response.get("native_finish_reason"),
            report_finish,
        )
        or ""
    ).strip().lower()
    error = first_present(
        response.get("error"),
        raw_api.get("error"),
        choice.get("error"),
        message.get("error"),
        payload.get("error"),
        prediction.get("error"),
    )
    final_content = first_present(
        text_value(prediction.get("raw")),
        text_value(message.get("content")),
        text_value(choice.get("text")),
        text_value(response.get("content")),
    )
    error_text = compact(error).lower()
    combined = f"{finish_reason} {error_text}"
    return {
        "success": response.get("success"),
        "finish_reason": finish_reason,
        "error": error,
        "final_content": str(final_content or "").strip(),
        "is_content_filter": (
            finish_reason in {"content_filter", "content-filter"}
            or "content_filter" in combined
            or "content filter" in combined
        ),
    }


def classify_missing(parts: dict[str, object]) -> str:
    finish_reason = str(parts["finish_reason"])
    if parts["final_content"]:
        return "unexpected_final_content"
    if parts["is_content_filter"]:
        return "content_filter"
    if (
        parts["error"] is not None
        or parts["success"] is False
        or finish_reason in {"error", "failed", "failure"}
    ):
        return "api_or_provider_error"
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "length"
    if finish_reason in {"stop", "end_turn", "eos"}:
        return "stop_without_final"
    return "other_missing"


def nested_http_code(value: object) -> int | None:
    if isinstance(value, dict):
        try:
            code = int(value.get("code"))
        except (TypeError, ValueError):
            code = None
        if code is not None and 400 <= code <= 599:
            return code
        for nested in value.values():
            code = nested_http_code(nested)
            if code is not None:
                return code
    elif isinstance(value, list):
        for nested in value:
            code = nested_http_code(nested)
            if code is not None:
                return code
    text = compact(value, limit=2000)
    match = re.search(r"(?:HTTP\s+|code[\"']?\s*[:=]\s*)([45]\d\d)", text, re.I)
    return int(match.group(1)) if match else None


def issue_category(parts: dict[str, object], code: int | None) -> str:
    if code is not None:
        return f"http_{code}"
    text = compact(parts["error"], limit=2000).lower()
    if "connection refused" in text or "actively refused" in text:
        return "local_connection_refused"
    if "nameresolution" in text or "resolve" in text or "getaddrinfo" in text:
        return "dns_resolution"
    if "ssl" in text or "tls" in text or "certificate" in text:
        return "ssl_transport"
    if str(parts["finish_reason"]) == "error" and not text:
        return "provider_finish_error"
    return "network_transport"


def locate_raw(group_key: str, model: str, persona: str, filename: str) -> Path:
    model_root = RAW_ROOT / GROUP_DIRECTORIES[group_key] / model
    candidates = (
        model_root / persona / filename,
        model_root / filename,
        model_root / "neutral" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Cannot locate raw JSON for {group_key}/{model}/{persona}/{filename}"
    )


def report_path(group_key: str, model: str) -> Path:
    return (
        REPORT_ROOT
        / group_key
        / model
        / f"results_{group_key}_{model}.csv"
    )


def manifest_key(model: str, group: str) -> tuple[str, str]:
    if model.startswith("deepseek_"):
        return model, "all"
    return model, "regional" if group == REGIONAL_GROUP else "nonregional"


def inspect_reports() -> tuple[dict[tuple[str, str], list[dict[str, object]]], dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {
        key: [] for key in MANIFEST_SPECS
    }
    missing_categories: Counter[str] = Counter()
    selected_by_group: Counter[str] = Counter()
    selected_by_model: Counter[str] = Counter()
    source_reports: list[str] = []

    for group_key, group_directory in GROUP_DIRECTORIES.items():
        for model in MODELS:
            report = report_path(group_key, model)
            source_reports.append(report.relative_to(REPO_ROOT).as_posix())
            with report.open(newline="", encoding="utf-8-sig") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    if str(row.get("answer_status") or "").strip() != "missing":
                        continue
                    persona = str(row.get("persona_folder") or "").strip()
                    filename = str(row.get("file") or "").strip()
                    question_id = str(row.get("question_id") or "").strip()
                    if not persona or Path(filename).name != filename:
                        raise RuntimeError(
                            f"Unsafe report target at {report}:{row_number}"
                        )
                    raw_path = locate_raw(group_key, model, persona, filename)
                    payload = json.loads(raw_path.read_text(encoding="utf-8-sig"))
                    if not isinstance(payload, dict):
                        raise RuntimeError(f"Raw JSON is not an object: {raw_path}")
                    parts = response_parts(
                        payload, str(row.get("finish_reason") or "")
                    )
                    category = classify_missing(parts)
                    missing_categories[category] += 1
                    if category != "api_or_provider_error":
                        continue
                    if parts["final_content"]:
                        raise RuntimeError(
                            f"API-failure target unexpectedly has final content: {raw_path}"
                        )
                    code = nested_http_code(parts["error"])
                    error_message = sanitize_error(parts["error"])
                    if not error_message:
                        error_message = (
                            f"finish_reason={parts['finish_reason']}"
                            if parts["finish_reason"]
                            else f"response.success={parts['success']}"
                        )
                    relative = raw_path.relative_to(RAW_ROOT).as_posix()
                    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    output_row: dict[str, object] = {
                        "dataset": "ETHICS",
                        "model_slug": model,
                        "group": group_directory,
                        "persona_folder": persona,
                        "question_id": question_id or raw_path.stem,
                        "issue_category": issue_category(parts, code),
                        "http_code": code or "",
                        "error_message": error_message,
                        "source_state": "valid",
                        "template_relative_path": "",
                        "question_source_relative_path": "",
                        "relative_path": relative,
                        "source_report": report.relative_to(REPO_ROOT).as_posix(),
                        "report_row": row_number,
                        "raw_sha256": raw_digest,
                    }
                    key = manifest_key(model, group_directory)
                    buckets[key].append(output_row)
                    selected_by_group[group_directory] += 1
                    selected_by_model[model] += 1

    for rows in buckets.values():
        rows.sort(
            key=lambda row: (
                str(row["group"]),
                str(row["persona_folder"]),
                str(row["question_id"]),
            )
        )
    summary: dict[str, object] = {
        "snapshot_date": "2026-08-21",
        "selection": "answer_status=missing AND API/transport/provider failure",
        "excluded": [
            "successful responses",
            "length/token exhaustion",
            "content_filter",
            "stop without final content",
        ],
        "total_missing_scanned": sum(missing_categories.values()),
        "missing_category_counts": dict(sorted(missing_categories.items())),
        "selected_total": sum(len(rows) for rows in buckets.values()),
        "selected_by_group": dict(sorted(selected_by_group.items())),
        "selected_by_model": dict(sorted(selected_by_model.items())),
        "source_reports": sorted(source_reports),
    }
    return buckets, summary


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


def write_immutable(path: Path, content: bytes) -> str:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(
                f"Refusing to replace non-identical snapshot file: {path}"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    buckets, summary = inspect_reports()
    selected_total = int(summary["selected_total"])
    if selected_total != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total} current API failures, found {selected_total}"
        )

    manifest_metadata: dict[str, object] = {}
    for key, spec in MANIFEST_SPECS.items():
        rows = buckets[key]
        expected = int(spec["expected"])
        if len(rows) != expected:
            raise RuntimeError(
                f"Guard expected {expected} rows for {key}, found {len(rows)}"
            )
        content = csv_bytes(rows)
        output = output_root / str(spec["filename"])
        status = write_immutable(output, content) if args.write else "planned"
        manifest_metadata[str(spec["filename"])] = {
            "backend": spec["backend"],
            "model_slug": key[0],
            "scope": key[1],
            "rows": len(rows),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        issues = Counter(str(row["issue_category"]) for row in rows)
        print(
            f"{status:9} {output} rows={len(rows)} "
            f"issues={dict(sorted(issues.items()))}"
        )

    summary["manifests"] = manifest_metadata
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    summary_path = output_root / "manifest_summary.json"
    status = write_immutable(summary_path, summary_bytes) if args.write else "planned"
    print(f"{status:9} {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

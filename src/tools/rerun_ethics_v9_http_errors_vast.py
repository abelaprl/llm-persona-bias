"""Rerun audited ETHICS v9 infrastructure failures against Vast/vLLM.

The default selection contains 59 exact DeepSeek targets: 51 historical HTTP
4XX/5XX failures plus 8 local-endpoint connection failures. This script never
walks the result tree; it opens only the manifests and their explicit paths.

Six implicit-persona targets lost the assistant warm-up turn when a local run
failed. For only those targets, the runner restores the original successful
warm-up turn already stored in a valid neighboring result before resending.

Safe default (validation only, no network calls or writes):

    python src/tools/rerun_ethics_v9_http_errors_vast.py

Run directly inside the Vast instance (recommended template port):

    python src/tools/rerun_ethics_v9_http_errors_vast.py --execute \
      --endpoint-url http://127.0.0.1:18000/v1/chat/completions \
      --concurrency 4

Run through an existing local SSH tunnel:

    python src/tools/rerun_ethics_v9_http_errors_vast.py --execute \
      --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
      --concurrency 4

Before generation, ``/v1/models`` must advertise the exact experiment model
alias. Existing target JSON files are copied to a timestamped backup. A failed
or empty rerun is written to the attempts directory and never replaces the
source JSON. Successful calls replace only their explicitly manifested file.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
ETHICS_RUNNER_DIR = REPO_ROOT / "src" / "benchmarks" / "ethics_runner"
sys.path.insert(0, str(ETHICS_RUNNER_DIR))

from run_ethics_prompt_v9 import (  # noqa: E402
    call_openrouter,
    clean_api_response,
    extract_raw_text,
    normalize_binary_label,
    parse_v9_response,
    reparse_ethics_answer,
)


MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "ethics_v9"
    / "deepseek_http_4xx_5xx.csv"
)
DEFAULT_EXTRA_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "ethics_v9"
    / "deepseek_non_http_local_connection.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
DEFAULT_BACKUP_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "_rerun_backup" / "v9_http_errors_deepseek"
)
DEFAULT_ATTEMPT_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "_rerun_attempts" / "v9_http_errors_deepseek"
)
REQUIRED_COLUMNS = {
    "model_slug",
    "group",
    "persona_folder",
    "question_id",
    "http_code",
    "error_message",
    "relative_path",
}
WARMUP_RECOVERY_SOURCES = {
    ("Gender Persona", "man_implicit_1"): PurePosixPath(
        "Gender Persona/"
        f"{MODEL_SLUG}/"
        "man_implicit_1/ethics_commonsense_0040.json"
    )
}
WARMUP_CACHE_RELATIVE_PATH = PurePosixPath(
    "Gender Persona/"
    f"{MODEL_SLUG}/"
    "man_implicit_1/warmup/persona_warmup.json"
)


@dataclass(frozen=True)
class Target:
    """One exact JSON selected by the immutable rerun manifest."""

    row_number: int
    group: str
    persona: str
    question_id: str
    issue_category: str
    audited_http_code: int | None
    audited_error_message: str
    relative_path: PurePosixPath
    path: Path
    payload: dict[str, Any] = field(compare=False, repr=False)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    custom_primary_manifest = any(
        item == "--manifest" or item.startswith("--manifest=") for item in raw_argv
    )
    parser = argparse.ArgumentParser(
        description=(
            "Rerun exact ETHICS v9 infrastructure-error targets without scanning the result tree."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--extra-manifest",
        default=None,
        help=(
            "Optional second manifest. With the default primary manifest, the audited "
            "8-row local-connection manifest is included automatically."
        ),
    )
    parser.add_argument(
        "--no-extra-manifest",
        action="store_true",
        help="Disable the automatic 8-row local-connection manifest.",
    )
    parser.add_argument(
        "--no-warmup-cache-repair",
        action="store_true",
        help="Do not repair the audited failed man_implicit_1 warm-up cache.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT))
    parser.add_argument("--attempt-root", default=str(DEFAULT_ATTEMPT_ROOT))
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("VLLM_ENDPOINT_URL", DEFAULT_ENDPOINT),
        help="OpenAI-compatible /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make API calls and replace successful targets. Without this flag: validation only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even targets already containing a successful non-empty response.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Process only the first N pending manifest rows (useful for a smoke test).",
    )
    parser.add_argument(
        "--expected-count",
        type=nonnegative_int,
        default=51,
        help="Fail if the manifest row count differs; use 0 to disable the guard.",
    )
    parser.add_argument(
        "--extra-expected-count",
        type=nonnegative_int,
        default=8,
        help="Row-count guard for --extra-manifest; use 0 to disable the guard.",
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="Override every target; default preserves request.parameters.max_tokens per JSON.",
    )
    parser.add_argument("--max-retries", type=nonnegative_int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=4.0)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--endpoint-check-timeout-sec", type=float, default=20.0)
    args = parser.parse_args(raw_argv)

    if args.retry_base_sec <= 0:
        parser.error("--retry-base-sec must be greater than 0")
    if args.request_timeout_sec <= 0:
        parser.error("--request-timeout-sec must be greater than 0")
    if args.endpoint_check_timeout_sec <= 0:
        parser.error("--endpoint-check-timeout-sec must be greater than 0")

    args.manifest = repo_path(args.manifest)
    if args.no_extra_manifest and args.extra_manifest:
        parser.error("--extra-manifest and --no-extra-manifest cannot be used together")
    if args.no_extra_manifest or (custom_primary_manifest and not args.extra_manifest):
        args.extra_manifest = None
    elif args.extra_manifest:
        args.extra_manifest = repo_path(args.extra_manifest)
    else:
        args.extra_manifest = DEFAULT_EXTRA_MANIFEST
    args.manage_warmup_cache = (
        not custom_primary_manifest and not args.no_warmup_cache_repair
    )
    args.output_root = repo_path(args.output_root)
    args.backup_root = repo_path(args.backup_root)
    args.attempt_root = repo_path(args.attempt_root)
    return args


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Manifest target is missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest target must contain a JSON object: {path}")
    return payload


def safe_target_path(output_root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Unsafe relative_path in manifest: {relative_path}")
    path = output_root.joinpath(*relative_path.parts)
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes output root: {relative_path}") from exc
    return path


def load_targets(
    manifest_path: Path,
    output_root: Path,
    expected_count: int,
) -> list[Target]:
    try:
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing_columns:
                raise RuntimeError(
                    f"Manifest is missing columns {missing_columns}: {manifest_path}"
                )
            rows = list(reader)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Manifest is missing: {manifest_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read manifest {manifest_path}: {exc}") from exc

    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    if expected_count and len(rows) != expected_count:
        raise RuntimeError(
            f"Manifest guard expected {expected_count} rows, found {len(rows)}: {manifest_path}"
        )

    targets: list[Target] = []
    seen_paths: set[PurePosixPath] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        model_slug = str(row.get("model_slug") or "").strip()
        if model_slug != MODEL_SLUG:
            raise RuntimeError(
                f"Manifest row {row_number} model_slug is {model_slug!r}, expected {MODEL_SLUG!r}"
            )

        group = str(row.get("group") or "").strip()
        persona = str(row.get("persona_folder") or "").strip()
        question_id = str(row.get("question_id") or "").strip()
        error_message = str(row.get("error_message") or "").strip()
        relative_text = str(row.get("relative_path") or "").strip().replace("\\", "/")
        relative_path = PurePosixPath(relative_text)

        http_code_text = str(row.get("http_code") or "").strip()
        if http_code_text:
            try:
                http_code: int | None = int(http_code_text)
            except ValueError as exc:
                raise RuntimeError(
                    f"Manifest row {row_number} has invalid http_code"
                ) from exc
        else:
            http_code = None
        if http_code is not None and not 400 <= http_code <= 599:
            raise RuntimeError(
                f"Manifest row {row_number} http_code is not 4XX/5XX: {http_code}"
            )
        issue_category = str(row.get("issue_category") or "").strip()
        if not issue_category:
            issue_category = f"http_{http_code}" if http_code is not None else ""
        if not group or not persona or not question_id or not error_message:
            raise RuntimeError(f"Manifest row {row_number} has an empty required value")
        if not issue_category:
            raise RuntimeError(
                f"Manifest row {row_number} needs issue_category when http_code is blank"
            )
        if relative_path.suffix.lower() != ".json":
            raise RuntimeError(f"Manifest row {row_number} is not a JSON path: {relative_path}")
        if relative_path.name != f"{question_id}.json":
            raise RuntimeError(
                f"Manifest row {row_number} question_id/path mismatch: "
                f"{question_id!r} vs {relative_path.name!r}"
            )
        if relative_path in seen_paths:
            raise RuntimeError(f"Duplicate relative_path in manifest: {relative_path}")
        pair = (persona, question_id)
        if pair in seen_pairs:
            raise RuntimeError(f"Duplicate persona/question target in manifest: {pair}")
        seen_paths.add(relative_path)
        seen_pairs.add(pair)

        path = safe_target_path(output_root, relative_path)
        payload = read_json(path)
        if str(payload.get("id") or "").strip() != question_id:
            raise RuntimeError(
                f"JSON id does not match manifest at row {row_number}: {path}"
            )
        if str(payload.get("persona") or "").strip() != persona:
            raise RuntimeError(
                f"JSON persona does not match manifest at row {row_number}: {path}"
            )
        messages = (payload.get("request") or {}).get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"JSON has no stored request.messages: {path}")
        if not all(isinstance(message, dict) for message in messages):
            raise RuntimeError(f"JSON request.messages is malformed: {path}")

        targets.append(
            Target(
                row_number=row_number,
                group=group,
                persona=persona,
                question_id=question_id,
                issue_category=issue_category,
                audited_http_code=http_code,
                audited_error_message=error_message,
                relative_path=relative_path,
                path=path,
                payload=payload,
            )
        )
    return targets


def raw_api_error(payload: dict[str, Any]) -> dict[str, Any] | None:
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    raw_api = response.get("raw_api_response")
    if not isinstance(raw_api, dict):
        return None
    error = raw_api.get("error")
    return error if isinstance(error, dict) else None


def current_error_code(payload: dict[str, Any]) -> int | None:
    error = raw_api_error(payload)
    if error is not None:
        try:
            return int(error.get("code"))
        except (TypeError, ValueError):
            pass

    response = payload.get("response")
    error_text = str(response.get("error") or "") if isinstance(response, dict) else ""
    for code in range(400, 600):
        if f"HTTP {code}" in error_text:
            return code
    return None


def is_resolved(payload: dict[str, Any]) -> bool:
    prediction = payload.get("prediction")
    response = payload.get("response")
    if not isinstance(prediction, dict) or not isinstance(response, dict):
        return False
    raw_text = str(prediction.get("raw") or "").strip()
    return bool(raw_text) and response.get("success") is True and raw_api_error(payload) is None


def stored_max_tokens(payload: dict[str, Any]) -> int:
    request = payload.get("request")
    parameters = request.get("parameters") if isinstance(request, dict) else None
    value = parameters.get("max_tokens") if isinstance(parameters, dict) else None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 4096
    return parsed if parsed > 0 else 4096


def prepare_messages(
    target: Target,
    output_root: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return experiment messages, recovering a lost implicit warm-up if needed."""

    request = target.payload.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"JSON has no stored request.messages: {target.path}")
    prepared = copy.deepcopy(messages)
    if any(message.get("role") == "assistant" for message in prepared):
        return prepared, None
    if "_implicit" not in target.persona:
        return prepared, None

    source_relative = WARMUP_RECOVERY_SOURCES.get((target.group, target.persona))
    if source_relative is None:
        raise RuntimeError(
            "Implicit target has no assistant warm-up and no audited recovery source: "
            f"{target.relative_path}"
        )
    if len(prepared) != 3 or [item.get("role") for item in prepared] != [
        "system",
        "user",
        "user",
    ]:
        raise RuntimeError(
            f"Cannot safely reconstruct malformed implicit messages: {target.path}"
        )

    source_path = safe_target_path(output_root, source_relative)
    source_payload = read_json(source_path)
    source_request = source_payload.get("request")
    source_messages = (
        source_request.get("messages") if isinstance(source_request, dict) else None
    )
    if not isinstance(source_messages, list):
        raise RuntimeError(f"Warm-up recovery source has no messages: {source_path}")
    assistant_turns = [
        message
        for message in source_messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant_turns) != 1:
        raise RuntimeError(
            f"Warm-up recovery source must have exactly one assistant turn: {source_path}"
        )
    for index, expected_role in enumerate(("system", "user")):
        source_message = source_messages[index] if len(source_messages) > index else {}
        if (
            not isinstance(source_message, dict)
            or source_message.get("role") != expected_role
            or source_message.get("content") != prepared[index].get("content")
        ):
            raise RuntimeError(
                "Warm-up recovery source does not match the target prompt prefix: "
                f"{source_path}"
            )

    restored = [
        copy.deepcopy(prepared[0]),
        copy.deepcopy(prepared[1]),
        copy.deepcopy(assistant_turns[0]),
        copy.deepcopy(prepared[2]),
    ]
    return restored, source_relative.as_posix()


def warmup_cache_needs_repair(output_root: Path) -> bool:
    payload = read_json(safe_target_path(output_root, WARMUP_CACHE_RELATIVE_PATH))
    response = payload.get("response")
    return not (
        isinstance(response, dict)
        and response.get("success") is True
        and bool(str(response.get("content") or "").strip())
    )


def repair_warmup_cache(output_root: Path) -> Path:
    """Restore the exact historical warm-up answer embedded in a valid result."""

    cache_path = safe_target_path(output_root, WARMUP_CACHE_RELATIVE_PATH)
    payload = read_json(cache_path)
    source_relative = WARMUP_RECOVERY_SOURCES[("Gender Persona", "man_implicit_1")]
    source_path = safe_target_path(output_root, source_relative)
    source_payload = read_json(source_path)

    cache_request = payload.get("request")
    cache_messages = (
        cache_request.get("messages") if isinstance(cache_request, dict) else None
    )
    source_request = source_payload.get("request")
    source_messages = (
        source_request.get("messages") if isinstance(source_request, dict) else None
    )
    if not isinstance(cache_messages, list) or len(cache_messages) != 2:
        raise RuntimeError(f"Warm-up cache request is malformed: {cache_path}")
    if not isinstance(source_messages, list) or len(source_messages) < 3:
        raise RuntimeError(f"Warm-up recovery source is malformed: {source_path}")
    if source_messages[:2] != cache_messages:
        raise RuntimeError(
            f"Warm-up cache prompt differs from recovery source: {source_path}"
        )
    assistant_message = source_messages[2]
    if not isinstance(assistant_message, dict) or assistant_message.get("role") != "assistant":
        raise RuntimeError(f"Recovery source has no assistant warm-up turn: {source_path}")
    content = str(assistant_message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Recovery source assistant warm-up is empty: {source_path}")

    payload["response"] = {
        "success": True,
        "content": content,
        "raw_api_response": {
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": copy.deepcopy(assistant_message),
                    "finish_reason": "recovered_from_existing_result",
                }
            ],
        },
        "error": None,
        "latency_sec": None,
        "retry_count": 0,
    }
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "warmup_recovered": True,
            "warmup_recovered_timestamp": datetime.now().astimezone().isoformat(),
            "warmup_recovered_from": source_relative.as_posix(),
            "warmup_recovery_note": (
                "No new inference: restored the exact assistant turn already used by "
                "valid results in this persona folder."
            ),
        }
    )
    atomic_write_json(cache_path, payload)
    return cache_path


def print_plan(
    args: argparse.Namespace,
    targets: list[Target],
    pending: list[Target],
) -> None:
    audited_codes = Counter(
        target.audited_http_code if target.audited_http_code is not None else "non-HTTP"
        for target in targets
    )
    issue_categories = Counter(target.issue_category for target in targets)
    groups = Counter(target.group for target in targets)
    current_codes = Counter(current_error_code(target.payload) for target in targets)
    token_budgets = Counter(
        args.max_tokens or stored_max_tokens(target.payload) for target in pending
    )
    warmup_recoveries = sum(
        prepare_messages(target, args.output_root)[1] is not None for target in targets
    )
    cache_needs_repair = (
        warmup_cache_needs_repair(args.output_root) if args.manage_warmup_cache else False
    )

    print("=" * 72)
    print("ETHICS v9 infrastructure-error rerun - exact manifest mode")
    print("=" * 72)
    print(f"Manifest       : {args.manifest}")
    print(f"Extra manifest : {args.extra_manifest or '(disabled)'}")
    print(f"Output root    : {args.output_root}")
    print(f"Manifest rows  : {len(targets)}")
    print(f"Pending rows   : {len(pending)}")
    print(f"Already fixed  : {len(targets) - len(pending)}")
    print(
        "Audited codes  : "
        f"{dict(sorted(audited_codes.items(), key=lambda item: str(item[0])))}"
    )
    print(f"Issue classes  : {dict(sorted(issue_categories.items()))}")
    print(f"Groups         : {dict(sorted(groups.items()))}")
    print(f"Current codes  : {dict(sorted(current_codes.items(), key=lambda item: str(item[0])))}")
    print(f"Token budgets  : {dict(sorted(token_budgets.items()))}")
    print(f"Warm-up repair : {warmup_recoveries} target message sequence(s)")
    if args.manage_warmup_cache:
        print(
            "Warm-up cache  : "
            + (
                "1 local repair needed (no model call)"
                if cache_needs_repair
                else "already valid"
            )
        )
    else:
        print("Warm-up cache  : disabled for custom manifest")
    print(f"Concurrency    : {args.concurrency}")
    print("Result scan    : disabled; only manifest paths were opened")
    if args.limit is not None:
        print(f"Limit          : first {args.limit} pending target(s)")
    print()


def models_url(chat_url: str) -> str:
    clean = chat_url.rstrip("/")
    suffix = "/chat/completions"
    if not clean.endswith(suffix):
        raise RuntimeError(f"Endpoint must end with {suffix}: {chat_url}")
    return clean[: -len(suffix)] + "/models"


def auth_headers() -> dict[str, str]:
    api_key = os.environ.get("VLLM_API_KEY", "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def check_endpoint(chat_url: str, timeout_sec: float) -> None:
    url = models_url(chat_url)
    try:
        response = requests.get(url, headers=auth_headers(), timeout=timeout_sec)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"vLLM endpoint is not ready at {url}: {exc}") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    model_ids = {
        str(item.get("id"))
        for item in (data or [])
        if isinstance(item, dict) and item.get("id")
    }
    if MODEL_ID not in model_ids:
        raise RuntimeError(
            f"Endpoint serves {sorted(model_ids)}, not {MODEL_ID}. Start vLLM with "
            f"--served-model-name {MODEL_ID}."
        )
    print(f"[PREFLIGHT] Endpoint ready: {url}; model alias matches {MODEL_ID}")


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")


def backup_targets(targets: list[Target], backup_root: Path, run_id: str) -> Path:
    destination = backup_root / run_id
    for target in targets:
        backup_path = destination.joinpath(*target.relative_path.parts)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target.path, backup_path)
    print(f"[BACKUP] {len(targets)} file(s) copied to {destination}")
    return destination


def backup_relative_file(
    output_root: Path,
    relative_path: PurePosixPath,
    backup_root: Path,
    run_id: str,
) -> Path:
    source = safe_target_path(output_root, relative_path)
    destination = (backup_root / run_id).joinpath(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[BACKUP] Warm-up cache copied to {destination}")
    return destination


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def attempt_path(attempt_root: Path, run_id: str, target: Target) -> Path:
    return (attempt_root / run_id).joinpath(*target.relative_path.parts)


def write_failed_attempt(
    target: Target,
    attempt_root: Path,
    run_id: str,
    api_result: dict[str, Any],
    raw_api: dict[str, Any] | None,
    reason: str,
) -> Path:
    path = attempt_path(attempt_root, run_id, target)
    payload = {
        "question_id": target.question_id,
        "persona": target.persona,
        "source_path": target.relative_path.as_posix(),
        "issue_category": target.issue_category,
        "audited_http_code": target.audited_http_code,
        "audited_error_message": target.audited_error_message,
        "attempt_timestamp": datetime.now().astimezone().isoformat(),
        "failure_reason": reason,
        "response": {
            "success": api_result.get("success"),
            "error": api_result.get("error"),
            "retry_count": api_result.get("retry_count"),
            "latency_sec": api_result.get("latency"),
            "raw_api_response": raw_api,
        },
    }
    atomic_write_json(path, payload)
    return path


def process_target(
    target: Target,
    args: argparse.Namespace,
    run_id: str,
) -> tuple[str, str, str]:
    payload = copy.deepcopy(target.payload)
    request = payload.get("request") or {}
    messages, warmup_recovery_source = prepare_messages(target, args.output_root)
    request["messages"] = messages
    max_tokens = args.max_tokens or stored_max_tokens(payload)
    api_key = os.environ.get("VLLM_API_KEY", "").strip()
    model_cfg = {
        "name": "DeepSeek R1 Distill Qwen 32B (Vast vLLM rerun)",
        "model_id": MODEL_ID,
        "endpoint_url": args.endpoint_url,
        "api_key_alias": api_key,
        "auth_required": bool(api_key),
    }

    api_result = call_openrouter(
        model_cfg=model_cfg,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=args.max_retries,
        retry_base_sec=args.retry_base_sec,
        request_timeout_sec=args.request_timeout_sec,
    )
    raw_api = (
        clean_api_response(api_result.get("api_response"))
        if api_result.get("success")
        else None
    )
    structured_error = raw_api.get("error") if isinstance(raw_api, dict) else None
    raw_text = extract_raw_text(raw_api) if raw_api is not None else ""

    if not api_result.get("success"):
        reason = str(api_result.get("error") or "API request failed")
    elif structured_error:
        reason = f"API body contains error: {structured_error}"
    elif not raw_text.strip():
        reason = "HTTP request succeeded but assistant content is empty"
    else:
        reason = ""

    if reason:
        failed_path = write_failed_attempt(
            target, args.attempt_root, run_id, api_result, raw_api, reason
        )
        return "failed", target.question_id, f"{reason}; attempt={failed_path}"

    explanation, parsed_answer, parsing_failed = parse_v9_response(raw_text)
    reparsed_answer, reparse_source = reparse_ethics_answer(raw_text)
    effective_answer = parsed_answer if parsed_answer in ("0", "1") else reparsed_answer

    request.setdefault("parameters", {})
    request["parameters"]["temperature"] = 0.0
    request["parameters"]["max_tokens"] = max_tokens
    request["parameters"]["model"] = MODEL_ID
    payload["request"] = request
    payload["prediction"] = {
        "raw": raw_text,
        "explanation": explanation,
        "parsed_answer": parsed_answer,
        "parsing_failed": parsing_failed,
    }
    payload["response"] = {
        "success": True,
        "raw_api_response": raw_api,
        "error": None,
        "retry_count": api_result.get("retry_count"),
        "latency_sec": api_result.get("latency"),
    }

    usage = raw_api.get("usage") if isinstance(raw_api, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "rerun": "ethics_v9_infrastructure_errors_vast",
            "rerun_timestamp": datetime.now().astimezone().isoformat(),
            "rerun_manifest": str(args.manifest),
            "rerun_endpoint": args.endpoint_url,
            "rerun_issue_category": target.issue_category,
            "rerun_original_http_code": target.audited_http_code,
            "rerun_original_error_message": target.audited_error_message,
            "rerun_reparse_answer": reparsed_answer,
            "rerun_reparse_source": reparse_source,
            "max_tokens": max_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
        }
    )
    if warmup_recovery_source is not None:
        metadata["rerun_warmup_recovered_from"] = warmup_recovery_source

    analysis = payload.setdefault("analysis", {})
    ground_truth = normalize_binary_label(
        analysis.get("ground_truth_label", analysis.get("GT"))
    )
    prediction_label = normalize_binary_label(effective_answer)
    analysis["prediction_label"] = prediction_label
    analysis["is_correct"] = (
        prediction_label == ground_truth
        if prediction_label is not None and ground_truth is not None
        else None
    )

    atomic_write_json(target.path, payload)
    status = "saved" if prediction_label is not None else "saved_parse_failed"
    return status, target.question_id, target.relative_path.as_posix()


def run(args: argparse.Namespace) -> int:
    targets = load_targets(args.manifest, args.output_root, args.expected_count)
    if args.extra_manifest is not None:
        extra_targets = load_targets(
            args.extra_manifest,
            args.output_root,
            args.extra_expected_count,
        )
        existing_paths = {target.relative_path for target in targets}
        duplicate_paths = sorted(
            target.relative_path.as_posix()
            for target in extra_targets
            if target.relative_path in existing_paths
        )
        if duplicate_paths:
            raise RuntimeError(
                f"Primary and extra manifests overlap: {duplicate_paths}"
            )
        targets.extend(extra_targets)
    pending = targets if args.force else [target for target in targets if not is_resolved(target.payload)]
    print_plan(args, targets, pending)

    selected = pending[: args.limit] if args.limit is not None else pending
    cache_needs_repair = (
        warmup_cache_needs_repair(args.output_root) if args.manage_warmup_cache else False
    )
    if not args.execute:
        print(
            f"Validation complete: {len(selected)} target(s) selected; "
            f"warm-up cache repair={'yes' if cache_needs_repair else 'no'}; "
            "no API calls or files were written."
        )
        return 0
    if not selected and not cache_needs_repair:
        print(
            "Nothing to rerun: all manifest targets and the warm-up cache are resolved."
        )
        return 0

    if selected:
        check_endpoint(args.endpoint_url, args.endpoint_check_timeout_sec)
    run_id = timestamp_slug()
    if selected:
        backup_targets(selected, args.backup_root, run_id)
    if cache_needs_repair:
        backup_relative_file(
            args.output_root,
            WARMUP_CACHE_RELATIVE_PATH,
            args.backup_root,
            run_id,
        )
        repaired_path = repair_warmup_cache(args.output_root)
        print(f"[REPAIRED] Warm-up cache restored without a new model call: {repaired_path}")
    if not selected:
        return 0

    counts: Counter[str] = Counter()
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_target, target, args, run_id): target
            for target in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                status, question_id, detail = future.result()
            except Exception as exc:  # keep other exact targets running
                status = "exception"
                question_id = target.question_id
                detail = f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            with print_lock:
                print(
                    f"[{completed}/{len(selected)}] {question_id} "
                    f"{status.upper()} — {detail}",
                    flush=True,
                )

    print("=" * 72)
    print(f"Completed: {dict(sorted(counts.items()))}")
    print(f"Backup  : {args.backup_root / run_id}")
    if counts["failed"] or counts["exception"]:
        print(f"Attempts: {args.attempt_root / run_id}")
        print("Failed/empty attempts did not replace their source JSON.")
        return 1
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Rerun exact audited GSM8K DeepSeek infrastructure failures on Vast/vLLM.

The default manifest contains 684 exact Religion/Gender targets:

* 203 HTTP 4XX/5XX responses;
* 479 non-HTTP infrastructure failures; and
* 2 truncated JSON files whose prompts can be reconstructed from their intact
  ``input`` sections.

This script never walks the GSM8K result tree. It reads only the CSV manifest
and the exact JSON paths listed there. The safe default is validation only:

    python src/tools/rerun_gsm8k_deepseek_errors_vast.py

Execute against an OpenAI-compatible vLLM endpoint:

    python src/tools/rerun_gsm8k_deepseek_errors_vast.py --execute \
      --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
      --concurrency 4

Every selected source is backed up before generation. Failed requests and
responses without final assistant content are stored under the attempts root
and never replace the source JSON. Successful responses atomically replace
only their manifested targets. Already-resolved targets are skipped on resume.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_deepseek"
    / "deepseek_infra_all.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "GMS8K"
DEFAULT_BACKUP_ROOT = (
    REPO_ROOT / "result" / "GMS8K" / "_rerun_backup" / "deepseek_infra"
)
DEFAULT_ATTEMPT_ROOT = (
    REPO_ROOT / "result" / "GMS8K" / "_rerun_attempts" / "deepseek_infra"
)
REQUIRED_COLUMNS = {
    "model_slug",
    "group",
    "persona_folder",
    "question_id",
    "issue_category",
    "http_code",
    "error_message",
    "relative_path",
}


@dataclass(frozen=True)
class Target:
    """One exact file selected by the derived audit manifest."""

    row_number: int
    group: str
    persona: str
    question_id: str
    issue_category: str
    audited_http_code: int | None
    audited_error_message: str
    relative_path: PurePosixPath
    path: Path
    recovery_mode: str
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
    parser = argparse.ArgumentParser(
        description="Rerun exact audited GSM8K errors without scanning result folders."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
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
        help="Make API calls and replace successful targets; otherwise validate only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun targets that already contain a successful non-empty response.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Process only the first N pending rows, useful for a smoke test.",
    )
    parser.add_argument(
        "--expected-count",
        type=nonnegative_int,
        default=684,
        help="Fail if row count differs; use 0 for a custom manifest.",
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="Override all targets; by default preserve each JSON's max_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override all targets; by default preserve each JSON's temperature.",
    )
    parser.add_argument("--max-retries", type=nonnegative_int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=4.0)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--endpoint-check-timeout-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    if args.retry_base_sec <= 0:
        parser.error("--retry-base-sec must be greater than 0")
    if args.request_timeout_sec <= 0:
        parser.error("--request-timeout-sec must be greater than 0")
    if args.endpoint_check_timeout_sec <= 0:
        parser.error("--endpoint-check-timeout-sec must be greater than 0")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be zero or greater")

    args.manifest = repo_path(args.manifest)
    args.output_root = repo_path(args.output_root)
    args.backup_root = repo_path(args.backup_root)
    args.attempt_root = repo_path(args.attempt_root)
    return args


def safe_target_path(output_root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Unsafe relative_path in manifest: {relative_path}")
    path = output_root.joinpath(*relative_path.parts)
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes output root: {relative_path}") from exc
    return path


def extract_named_json_value(text: str, key: str) -> object | None:
    """Decode one complete top-level value from an otherwise truncated JSON file."""

    match = re.search(rf'^\s*"{re.escape(key)}"\s*:\s*', text, re.MULTILINE)
    if not match:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, match.end())
    except (TypeError, ValueError):
        return None
    return value


def build_messages_from_input(input_data: dict[str, Any]) -> list[dict[str, str]]:
    question = str(input_data.get("question") or "")
    persona = input_data.get("persona")
    persona = persona if isinstance(persona, dict) else {}
    persona_type = str(persona.get("type") or "")
    if not question or not persona_type:
        raise RuntimeError("Truncated JSON has no intact input question/persona")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    if persona_type != "neutral":
        warmup_prompt = str(persona.get("warmup_prompt") or "")
        warmup_response = str(persona.get("warmup_response") or "")
        if not warmup_prompt or not warmup_response:
            raise RuntimeError(
                f"Cannot reconstruct messages for truncated persona {persona_type!r}"
            )
        messages.extend(
            [
                {"role": "user", "content": warmup_prompt},
                {"role": "assistant", "content": warmup_response},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": f"Solve the following math problem:\n\n{question}\n\nAnswer:",
        }
    )
    return messages


def recover_truncated_payload(text: str, path: Path) -> dict[str, Any]:
    metadata = extract_named_json_value(text, "metadata")
    input_data = extract_named_json_value(text, "input")
    request = extract_named_json_value(text, "request")
    analysis = extract_named_json_value(text, "analysis")

    metadata = metadata if isinstance(metadata, dict) else {}
    if not isinstance(input_data, dict):
        raise RuntimeError(f"Cannot recover intact input section from {path}")
    request = request if isinstance(request, dict) else {}
    messages = request.get("messages")
    if not (
        isinstance(messages, list)
        and messages
        and all(isinstance(message, dict) for message in messages)
    ):
        messages = build_messages_from_input(input_data)
    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    parameters.setdefault("temperature", 0.2)
    parameters.setdefault("max_tokens", 2048)
    parameters["model"] = MODEL_ID

    return {
        "metadata": metadata,
        "input": input_data,
        "request": {"messages": messages, "parameters": parameters},
        "analysis": (
            analysis
            if isinstance(analysis, dict)
            else {"extracted_answer": None, "is_correct": None}
        ),
        "response": {
            "success": False,
            "content": "",
            "raw_api_response": None,
            "error": "Source JSON was truncated before this selective rerun",
        },
    }


def read_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Manifest target is missing: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read manifest target {path}: {exc}") from exc

    try:
        payload = json.loads(text)
        mode = "valid_json"
    except (TypeError, ValueError):
        try:
            start = len(text) - len(text.lstrip())
            payload, _ = json.JSONDecoder().raw_decode(text, start)
            mode = "prefix_recovered"
        except (TypeError, ValueError):
            payload = recover_truncated_payload(text, path)
            mode = "section_rebuilt"
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest target is not a JSON object: {path}")
    return payload, mode


def load_targets(
    manifest_path: Path,
    output_root: Path,
    expected_count: int,
) -> list[Target]:
    try:
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            if missing:
                raise RuntimeError(
                    f"Manifest is missing columns {missing}: {manifest_path}"
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
            f"Manifest guard expected {expected_count} rows, found {len(rows)}: "
            f"{manifest_path}"
        )

    targets: list[Target] = []
    seen_paths: set[PurePosixPath] = set()
    for row_number, row in enumerate(rows, start=2):
        model_slug = str(row.get("model_slug") or "").strip()
        group = str(row.get("group") or "").strip()
        persona = str(row.get("persona_folder") or "").strip()
        question_id = str(row.get("question_id") or "").strip()
        issue_category = str(row.get("issue_category") or "").strip()
        error_message = str(row.get("error_message") or "").strip()
        relative_text = str(row.get("relative_path") or "").strip().replace("\\", "/")
        relative_path = PurePosixPath(relative_text)

        if model_slug != MODEL_SLUG:
            raise RuntimeError(
                f"Manifest row {row_number} has model {model_slug!r}, expected "
                f"{MODEL_SLUG!r}"
            )
        if not group or not persona or not question_id or not issue_category:
            raise RuntimeError(f"Manifest row {row_number} has an empty required value")
        if not re.fullmatch(r"gsm8k_\d{5}", question_id):
            raise RuntimeError(
                f"Manifest row {row_number} has invalid question_id: {question_id!r}"
            )
        if relative_path.name != f"{question_id}.json":
            raise RuntimeError(
                f"Manifest row {row_number} question_id/path mismatch: {relative_path}"
            )
        if len(relative_path.parts) != 4 or relative_path.parts[:3] != (
            group,
            MODEL_SLUG,
            persona,
        ):
            raise RuntimeError(
                f"Manifest row {row_number} group/model/persona path mismatch: "
                f"{relative_path}"
            )
        if relative_path in seen_paths:
            raise RuntimeError(f"Duplicate relative_path in manifest: {relative_path}")
        seen_paths.add(relative_path)

        code_text = str(row.get("http_code") or "").strip()
        try:
            http_code = int(code_text) if code_text else None
        except ValueError as exc:
            raise RuntimeError(
                f"Manifest row {row_number} has invalid http_code: {code_text!r}"
            ) from exc
        if http_code is not None and not 400 <= http_code <= 599:
            raise RuntimeError(
                f"Manifest row {row_number} http_code is not 4XX/5XX: {http_code}"
            )

        path = safe_target_path(output_root, relative_path)
        payload, recovery_mode = read_payload(path)
        input_data = payload.get("input")
        input_data = input_data if isinstance(input_data, dict) else {}
        try:
            actual_id = int(input_data.get("question_id"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"JSON has no numeric input.question_id: {path}") from exc
        if actual_id != int(question_id.removeprefix("gsm8k_")):
            raise RuntimeError(f"JSON question ID does not match manifest: {path}")
        persona_data = input_data.get("persona")
        persona_data = persona_data if isinstance(persona_data, dict) else {}
        if str(persona_data.get("type") or "") != persona:
            raise RuntimeError(f"JSON persona does not match manifest: {path}")
        messages = (payload.get("request") or {}).get("messages")
        if not (
            isinstance(messages, list)
            and messages
            and all(isinstance(message, dict) for message in messages)
        ):
            raise RuntimeError(f"JSON has no usable request.messages: {path}")

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
                recovery_mode=recovery_mode,
                payload=payload,
            )
        )
    return targets


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


def raw_api_error(payload: dict[str, Any]) -> dict[str, Any] | None:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw_api = response.get("raw_api_response")
    raw_api = raw_api if isinstance(raw_api, dict) else {}
    error = raw_api.get("error")
    return error if isinstance(error, dict) else None


def raw_response_texts(raw_api: dict[str, Any]) -> tuple[str, str]:
    choices = raw_api.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = text_value(message.get("content"))
    reasoning = ""
    for key in ("reasoning", "reasoning_content", "thinking", "analysis"):
        reasoning = text_value(message.get(key))
        if reasoning:
            break
    if not reasoning:
        reasoning = text_value(message.get("reasoning_details"))
    return content, reasoning


def raw_finish_reason(raw_api: dict[str, Any]) -> str:
    choices = raw_api.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    return str(choice.get("finish_reason") or "").strip()


def current_content(payload: dict[str, Any]) -> str:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    content = text_value(response.get("content"))
    raw_api = response.get("raw_api_response")
    if not content and isinstance(raw_api, dict):
        content, _ = raw_response_texts(raw_api)
    return content


def is_resolved(payload: dict[str, Any]) -> bool:
    response = payload.get("response")
    return bool(
        isinstance(response, dict)
        and response.get("success") is True
        and current_content(payload)
        and raw_api_error(payload) is None
    )


def request_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request")
    request = request if isinstance(request, dict) else {}
    parameters = request.get("parameters")
    return parameters if isinstance(parameters, dict) else {}


def stored_max_tokens(payload: dict[str, Any]) -> int:
    try:
        value = int(request_parameters(payload).get("max_tokens"))
    except (TypeError, ValueError):
        return 2048
    return value if value > 0 else 2048


def stored_temperature(payload: dict[str, Any]) -> float:
    try:
        value = float(request_parameters(payload).get("temperature"))
    except (TypeError, ValueError):
        return 0.2
    return value if value >= 0 else 0.2


def print_plan(
    args: argparse.Namespace,
    targets: list[Target],
    pending: list[Target],
) -> None:
    selected = pending[: args.limit] if args.limit is not None else pending
    print("=" * 76)
    print("GSM8K DeepSeek infrastructure rerun - exact manifest mode")
    print("=" * 76)
    print(f"Manifest       : {args.manifest}")
    print(f"Output root    : {args.output_root}")
    print(f"Manifest rows  : {len(targets)}")
    print(f"Pending rows   : {len(pending)}")
    print(f"Selected rows  : {len(selected)}")
    print(f"Already fixed  : {len(targets) - len(pending)}")
    print(f"Issue types    : {dict(sorted(Counter(t.issue_category for t in targets).items()))}")
    print(f"Groups         : {dict(sorted(Counter(t.group for t in targets).items()))}")
    print(f"Recovery modes : {dict(sorted(Counter(t.recovery_mode for t in targets).items()))}")
    budgets = Counter(args.max_tokens or stored_max_tokens(t.payload) for t in selected)
    temperatures = Counter(
        args.temperature if args.temperature is not None else stored_temperature(t.payload)
        for t in selected
    )
    print(f"Token budgets  : {dict(sorted(budgets.items()))}")
    print(f"Temperatures   : {dict(sorted(temperatures.items()))}")
    print(f"Concurrency    : {args.concurrency}")
    print("Result scan    : disabled; only manifest paths were opened")
    print()


def models_url(chat_url: str) -> str:
    clean = chat_url.rstrip("/")
    suffix = "/chat/completions"
    if not clean.endswith(suffix):
        raise RuntimeError(f"Endpoint must end with {suffix}: {chat_url}")
    return clean[: -len(suffix)] + "/models"


def auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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


def api_error_code(raw_api: object) -> int | None:
    if not isinstance(raw_api, dict):
        return None
    error = raw_api.get("error")
    if not isinstance(error, dict):
        return None
    try:
        return int(error.get("code"))
    except (TypeError, ValueError):
        return None


def post_chat(
    endpoint_url: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_sec: float,
    timeout_sec: float,
) -> dict[str, Any]:
    request_payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = "request did not run"
    last_raw: dict[str, Any] | None = None
    last_status: int | None = None
    total_latency = 0.0

    for retry_count in range(max_retries + 1):
        started = time.perf_counter()
        retryable = True
        try:
            response = requests.post(
                endpoint_url,
                headers=auth_headers(),
                json=request_payload,
                timeout=timeout_sec,
            )
            total_latency += time.perf_counter() - started
            last_status = response.status_code
            try:
                decoded = response.json()
                last_raw = decoded if isinstance(decoded, dict) else None
            except ValueError:
                last_raw = None

            if 200 <= response.status_code < 300 and last_raw is not None:
                structured_code = api_error_code(last_raw)
                if not isinstance(last_raw.get("error"), dict):
                    return {
                        "success": True,
                        "api_response": last_raw,
                        "error": None,
                        "http_status": response.status_code,
                        "retry_count": retry_count,
                        "latency_sec": total_latency,
                    }
                last_error = f"API body contains error: {last_raw['error']}"
                retryable = structured_code == 429 or (
                    structured_code is not None and structured_code >= 500
                )
            else:
                body = (
                    json.dumps(last_raw, ensure_ascii=False)
                    if last_raw is not None
                    else response.text
                )
                last_error = f"HTTP {response.status_code}: {body[:1000]}"
                retryable = response.status_code == 429 or response.status_code >= 500
        except requests.RequestException as exc:
            total_latency += time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"
            last_raw = None
            last_status = None

        if retry_count >= max_retries or not retryable:
            return {
                "success": False,
                "api_response": last_raw,
                "error": last_error,
                "http_status": last_status,
                "retry_count": retry_count,
                "latency_sec": total_latency,
            }
        time.sleep(retry_base_sec * (2**retry_count))

    raise AssertionError("unreachable retry loop")


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
    reason: str,
) -> Path:
    path = attempt_path(attempt_root, run_id, target)
    payload = {
        "question_id": target.question_id,
        "persona": target.persona,
        "source_path": target.relative_path.as_posix(),
        "source_recovery_mode": target.recovery_mode,
        "audited_issue_category": target.issue_category,
        "audited_http_code": target.audited_http_code,
        "audited_error_message": target.audited_error_message,
        "attempt_timestamp": datetime.now().astimezone().isoformat(),
        "failure_reason": reason,
        "response": {
            "success": api_result.get("success"),
            "http_status": api_result.get("http_status"),
            "error": api_result.get("error"),
            "retry_count": api_result.get("retry_count"),
            "latency_sec": api_result.get("latency_sec"),
            "raw_api_response": api_result.get("api_response"),
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
    request = payload.get("request")
    request = request if isinstance(request, dict) else {}
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError(f"Missing messages after validation: {target.path}")
    max_tokens = args.max_tokens or stored_max_tokens(payload)
    temperature = (
        args.temperature
        if args.temperature is not None
        else stored_temperature(payload)
    )

    api_result = post_chat(
        endpoint_url=args.endpoint_url,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=args.max_retries,
        retry_base_sec=args.retry_base_sec,
        timeout_sec=args.request_timeout_sec,
    )
    raw_api = api_result.get("api_response")
    raw_api = raw_api if isinstance(raw_api, dict) else {}
    final_content, reasoning = raw_response_texts(raw_api)
    finish_reason = raw_finish_reason(raw_api)

    if not api_result.get("success"):
        reason = str(api_result.get("error") or "API request failed")
    elif isinstance(raw_api.get("error"), dict):
        reason = f"API body contains error: {raw_api['error']}"
    elif not final_content:
        suffix = " (reasoning was present)" if reasoning else ""
        reason = "Response has no final assistant content" + suffix
    elif finish_reason != "stop":
        reason = f"Response finish_reason is {finish_reason or 'missing'}, not stop"
    else:
        reason = ""

    if reason:
        failed_path = write_failed_attempt(
            target, args.attempt_root, run_id, api_result, reason
        )
        return "failed", target.question_id, f"{reason}; attempt={failed_path}"

    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    parameters.update(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": MODEL_ID,
        }
    )
    request["parameters"] = parameters
    payload["request"] = request

    usage = raw_api.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    payload["response"] = {
        "success": True,
        "content": final_content,
        "raw_api_response": raw_api,
        "error": None,
        "performance": {
            "latency_sec": api_result.get("latency_sec"),
            "tokens": usage,
            "retry_count": api_result.get("retry_count"),
        },
    }
    payload.setdefault("analysis", {"extracted_answer": None, "is_correct": None})

    metadata = payload.setdefault("metadata", {})
    model_info = metadata.setdefault("model_info", {})
    model_info.update(
        {
            "provider": "vast_vllm",
            "model_id": MODEL_ID,
            "api_endpoint": args.endpoint_url,
        }
    )
    metadata.update(
        {
            "rerun": "gsm8k_deepseek_infrastructure_vast",
            "rerun_timestamp": datetime.now().astimezone().isoformat(),
            "rerun_manifest": str(args.manifest),
            "rerun_source_recovery_mode": target.recovery_mode,
            "rerun_original_issue_category": target.issue_category,
            "rerun_original_http_code": target.audited_http_code,
            "rerun_original_error_message": target.audited_error_message,
        }
    )

    atomic_write_json(target.path, payload)
    return "saved", target.question_id, target.relative_path.as_posix()


def run_targets(args: argparse.Namespace, targets: list[Target]) -> int:
    pending = targets if args.force else [
        target for target in targets if not is_resolved(target.payload)
    ]
    print_plan(args, targets, pending)
    selected = pending[: args.limit] if args.limit is not None else pending

    if not args.execute:
        print(
            f"Validation complete: {len(selected)} target(s) selected; "
            "no API calls or files were written."
        )
        return 0
    if not selected:
        print("Nothing to rerun: all manifest targets already have successful final content.")
        return 0

    check_endpoint(args.endpoint_url, args.endpoint_check_timeout_sec)
    run_id = timestamp_slug()
    backup_targets(selected, args.backup_root, run_id)

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
            except Exception as exc:  # keep unrelated exact targets running
                status = "exception"
                question_id = target.question_id
                detail = f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            with print_lock:
                print(
                    f"[{completed}/{len(selected)}] {question_id} "
                    f"{status.upper()} - {detail}",
                    flush=True,
                )

    print("=" * 76)
    print(f"Completed: {dict(sorted(counts.items()))}")
    print(f"Backup  : {args.backup_root / run_id}")
    if counts["failed"] or counts["exception"]:
        print(f"Attempts: {args.attempt_root / run_id}")
        print("Failed/empty attempts did not replace their source JSON.")
        return 1
    return 0


def run(args: argparse.Namespace) -> int:
    targets = load_targets(args.manifest, args.output_root, args.expected_count)
    return run_targets(args, targets)


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

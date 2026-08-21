"""Selectively rerun audited non-regional Qwen3 Next 80B failures via OpenRouter.

Safe default: validate the immutable manifest without API calls or writes.

    python scripts/rerun_qwen3_next_80b_nonregional_openrouter.py

Smoke test three targets:

    python scripts/rerun_qwen3_next_80b_nonregional_openrouter.py \
      --execute --limit 3 --concurrency 1

Use the isolated Regional Cultural manifest:

    python scripts/rerun_qwen3_next_80b_nonregional_openrouter.py \
      --scope regional --execute --limit 3 --concurrency 1

Resume the full manifest (successful targets are skipped):

    python scripts/rerun_qwen3_next_80b_nonregional_openrouter.py \
      --execute --concurrency 4

The runner never scans result folders. It opens only paths in the manifest,
backs up every existing selected source, writes failed attempts separately,
and atomically replaces/creates a target only after receiving non-empty final
assistant content. Regional Cultural paths are rejected.
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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
ETHICS_RUNNER_DIR = REPO_ROOT / "src" / "ethics_runner"
sys.path.insert(0, str(ETHICS_RUNNER_DIR))

from run_ethics_prompt_v9 import (  # noqa: E402
    normalize_binary_label,
    parse_v9_response,
    reparse_ethics_answer,
)


MODEL_ORDER = 32
MODEL_ID = "qwen/qwen3-next-80b-a3b-thinking"
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
MODEL_LABEL = "Qwen3 Next 80B"
RUNNER_DESCRIPTION = __doc__
RERUN_TAG_PREFIX = "qwen3_next_80b"
PROVIDER_LABEL = "openrouter"
TRANSPORT_LABEL = "openrouter"
API_SERVICE_LABEL = "OpenRouter"
AUTH_REQUIRED = True
# A wrapper may use another entry that belongs to the same OpenRouter account
# when the model's catalog entry is intentionally unauthenticated (for example
# a local Vast endpoint). None preserves the model-specific entry.
CREDENTIAL_ORDER: int | None = None
CREDENTIAL_MODEL_ID: str | None = None
NONREGIONAL_EXPECTED_COUNT = 8290
REGIONAL_EXPECTED_COUNT = 5631
NONREGIONAL_SCOPE_DIRECTORY = "qwen3_next_80b_nonregional"
REGIONAL_SCOPE_DIRECTORY = "qwen3_next_80b_regional"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_ENDPOINT_URL = OPENROUTER_URL
ENDPOINT_ENV_VAR = "OPENROUTER_ENDPOINT_URL"
NONREGIONAL_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "qwen3_next_80b_nonregional"
    / "qwen_nonregional_infra_all.csv"
)
REGIONAL_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "qwen3_next_80b_regional"
    / "qwen_regional_infra_all.csv"
)
DEFAULT_MANIFEST = NONREGIONAL_MANIFEST
ETHICS_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
GSM8K_ROOT = REPO_ROOT / "result" / "GMS8K"
DEFAULT_BACKUP_ROOT = (
    REPO_ROOT / "result" / "_rerun_backup" / "qwen3_next_80b_nonregional"
)
DEFAULT_ATTEMPT_ROOT = (
    REPO_ROOT / "result" / "_rerun_attempts" / "qwen3_next_80b_nonregional"
)
REQUIRED_COLUMNS = {
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
}
ALLOWED_GROUPS = {
    "ETHICS": {
        "Gender Persona",
        "Neutral Persona",
        "Religion Persona",
        "Regional Cultural Persona",
    },
    "GSM8K": {
        "Gender_Persona",
        "Religion_Persona",
        "regional_cultural_persona",
    },
}
REGIONAL_GROUPS = {
    "ETHICS": "Regional Cultural Persona",
    "GSM8K": "regional_cultural_persona",
}


@dataclass(frozen=True)
class Target:
    row_number: int
    dataset: str
    group: str
    persona: str
    question_id: str
    issue_category: str
    audited_http_code: int | None
    audited_error_message: str
    source_state: str
    relative_path: PurePosixPath
    path: Path
    template_relative_path: PurePosixPath | None
    question_source_relative_path: PurePosixPath | None
    recovery_mode: str
    payload: dict[str, Any] | None = field(default=None, compare=False, repr=False)


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
    parser = argparse.ArgumentParser(description=RUNNER_DESCRIPTION)
    parser.add_argument(
        "--scope",
        choices=("nonregional", "regional"),
        default="nonregional",
        help="Select the isolated audited manifest family.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override the scope's default manifest.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "ethics", "gsm8k"),
        default="all",
        help="Optionally select one dataset from the combined manifest.",
    )
    parser.add_argument(
        "--expected-count",
        type=nonnegative_int,
        default=None,
        help="Override the scope's row-count guard; use 0 for a custom manifest.",
    )
    parser.add_argument("--backup-root", default=None)
    parser.add_argument("--attempt-root", default=None)
    parser.add_argument(
        "--success-root",
        default=None,
        help=(
            "Optional isolated output root. Successful JSON is written below "
            "<root>/<dataset>/<relative_path> and source JSON is never replaced."
        ),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "model_keys.json"))
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get(ENDPOINT_ENV_VAR, DEFAULT_ENDPOINT_URL),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Call {API_SERVICE_LABEL} and replace only successful targets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun targets that already contain successful final content.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Select only the first N pending rows (recommended for smoke tests).",
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="Override the stored budget; default preserves each target (normally 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the stored temperature; default preserves each target.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help=(
            "Optional OpenRouter reasoning effort. Default sends no override; "
            "use low for reasoning-first responses truncated at max_tokens."
        ),
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=positive_int,
        default=None,
        help=(
            "Optional reasoning-token sub-budget inside max_tokens. Mutually "
            "exclusive with --reasoning-effort."
        ),
    )
    parser.add_argument("--max-retries", type=nonnegative_int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=4.0)
    parser.add_argument("--request-timeout-sec", type=float, default=600.0)
    parser.add_argument("--preflight-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=f"Skip the read-only {API_SERVICE_LABEL} status check.",
    )
    args = parser.parse_args(argv)
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be zero or greater")
    if args.reasoning_effort and args.reasoning_max_tokens is not None:
        parser.error(
            "--reasoning-effort and --reasoning-max-tokens are mutually exclusive"
        )
    if (args.max_tokens is not None and args.reasoning_max_tokens is not None
            and args.reasoning_max_tokens >= args.max_tokens):
        parser.error("--reasoning-max-tokens must be smaller than --max-tokens")
    if args.retry_base_sec <= 0:
        parser.error("--retry-base-sec must be greater than 0")
    if args.request_timeout_sec <= 0 or args.preflight_timeout_sec <= 0:
        parser.error("timeouts must be greater than 0")
    if args.scope == "nonregional":
        default_manifest = NONREGIONAL_MANIFEST
        default_expected_count = NONREGIONAL_EXPECTED_COUNT
        scope_directory = NONREGIONAL_SCOPE_DIRECTORY
    else:
        default_manifest = REGIONAL_MANIFEST
        default_expected_count = REGIONAL_EXPECTED_COUNT
        scope_directory = REGIONAL_SCOPE_DIRECTORY
    args.manifest = repo_path(args.manifest or default_manifest)
    if args.expected_count is None:
        args.expected_count = default_expected_count
    args.backup_root = repo_path(
        args.backup_root
        or REPO_ROOT / "result" / "_rerun_backup" / scope_directory
    )
    args.attempt_root = repo_path(
        args.attempt_root
        or REPO_ROOT / "result" / "_rerun_attempts" / scope_directory
    )
    args.success_root = repo_path(args.success_root) if args.success_root else None
    args.config = repo_path(args.config)
    return args


def dataset_root(dataset: str) -> Path:
    if dataset == "ETHICS":
        return ETHICS_ROOT
    if dataset == "GSM8K":
        return GSM8K_ROOT
    raise RuntimeError(f"Unsupported dataset: {dataset}")


def safe_path(root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Unsafe manifest path: {relative_path}")
    path = root.joinpath(*relative_path.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes output root: {relative_path}") from exc
    return path


def parse_relative(value: object, *, allow_empty: bool = False) -> PurePosixPath | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text and allow_empty:
        return None
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe or empty manifest relative path: {text!r}")
    return relative


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required file is missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON must contain an object: {path}")
    return payload


def read_json_prefix(path: Path) -> dict[str, Any]:
    """Recover one complete JSON object followed by corrupt trailing bytes."""
    try:
        text = path.read_text(encoding="utf-8-sig").lstrip()
        payload, end = json.JSONDecoder().raw_decode(text)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required file is missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot recover JSON object prefix from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Recovered JSON prefix must contain an object: {path}")
    if not text[end:].strip():
        raise RuntimeError(f"JSON-prefix recovery was unnecessary for valid JSON: {path}")
    return payload


def rebuild_gsm8k_payload(
    target_path: Path,
    question_id: str,
    persona: str,
    template_path: Path,
    question_source_path: Path,
) -> dict[str, Any]:
    template = read_json(template_path)
    question_source = read_json(question_source_path)
    template_input = template.get("input")
    template_input = template_input if isinstance(template_input, dict) else {}
    source_input = question_source.get("input")
    source_input = source_input if isinstance(source_input, dict) else {}
    question = str(source_input.get("question") or "").strip()
    ground_truth = str(source_input.get("ground_truth") or "").strip()
    persona_data = template_input.get("persona")
    if not question or not ground_truth or not isinstance(persona_data, dict):
        raise RuntimeError(f"Recovery sources are incomplete for {target_path}")

    payload = copy.deepcopy(template)
    input_data = copy.deepcopy(template_input)
    input_data.update(
        {
            "question_id": int(question_id.removeprefix("gsm8k_")),
            "dataset": str(source_input.get("dataset") or "gsm8k_test"),
            "question": question,
            "ground_truth": ground_truth,
            "persona": copy.deepcopy(persona_data),
        }
    )
    payload["input"] = input_data

    request = payload.get("request")
    request = copy.deepcopy(request) if isinstance(request, dict) else {}
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"Recovery template has no request.messages: {template_path}")
    messages = copy.deepcopy(messages)
    expected_roles = [str(message.get("role") or "") for message in messages]
    if expected_roles[-1:] != ["user"] or expected_roles not in (
        ["system", "user"],
        ["system", "user", "assistant", "user"],
    ):
        raise RuntimeError(
            f"Unexpected recovery message roles {expected_roles}: {template_path}"
        )
    messages[-1] = {
        "role": "user",
        "content": f"Solve the following math problem:\n\n{question}\n\nAnswer:",
    }
    request["messages"] = messages
    parameters = request.get("parameters")
    parameters = copy.deepcopy(parameters) if isinstance(parameters, dict) else {}
    parameters.setdefault("temperature", 0.2)
    parameters.setdefault("max_tokens", 4096)
    parameters["model"] = MODEL_ID
    request["parameters"] = parameters
    payload["request"] = request

    metadata = payload.get("metadata")
    metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
    metadata["run_id"] = f"{MODEL_SLUG}_{persona}_{question_id}"
    metadata["timestamp"] = datetime.now().astimezone().isoformat()
    payload["metadata"] = metadata
    payload["analysis"] = {"extracted_answer": None, "is_correct": None}
    payload["response"] = {
        "success": False,
        "content": "",
        "raw_api_response": None,
        "error": "Payload rebuilt from exact manifest recovery sources",
        "performance": {"latency_sec": None, "tokens": {}, "retry_count": 0},
    }
    return payload


def validate_payload(target: Target) -> None:
    if target.payload is None:
        raise RuntimeError(f"Target payload has not been loaded: {target.path}")
    request = target.payload.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not (
        isinstance(messages, list)
        and messages
        and all(isinstance(message, dict) for message in messages)
    ):
        raise RuntimeError(f"Target has no usable request.messages: {target.path}")
    if target.dataset == "ETHICS":
        if str(target.payload.get("id") or "") != target.question_id:
            raise RuntimeError(f"ETHICS JSON id/path mismatch: {target.path}")
        if str(target.payload.get("persona") or "") != target.persona:
            raise RuntimeError(f"ETHICS persona/path mismatch: {target.path}")
    else:
        input_data = target.payload.get("input")
        input_data = input_data if isinstance(input_data, dict) else {}
        persona_data = input_data.get("persona")
        persona_data = persona_data if isinstance(persona_data, dict) else {}
        persona_type = str(persona_data.get("type") or "")
        persona_id = persona_data.get("id")
        derived_persona = (
            "neutral"
            if persona_type == "neutral"
            else f"{persona_type}_{persona_id}"
        )
        if persona_type != target.persona and derived_persona != target.persona:
            raise RuntimeError(
                f"GSM8K persona mismatch {derived_persona!r}: {target.path}"
            )


def load_targets(manifest: Path, expected_count: int, scope: str) -> list[Target]:
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            if missing:
                raise RuntimeError(f"Manifest is missing columns {missing}: {manifest}")
            rows = list(reader)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Manifest is missing: {manifest}") from exc
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest}")
    if expected_count and len(rows) != expected_count:
        raise RuntimeError(
            f"Manifest guard expected {expected_count}, found {len(rows)}: {manifest}"
        )

    targets: list[Target] = []
    seen: set[tuple[str, PurePosixPath]] = set()
    for row_number, row in enumerate(rows, start=2):
        dataset = str(row.get("dataset") or "").strip().upper()
        group = str(row.get("group") or "").strip()
        persona = str(row.get("persona_folder") or "").strip()
        question_id = str(row.get("question_id") or "").strip()
        issue = str(row.get("issue_category") or "").strip()
        source_state = str(row.get("source_state") or "").strip()
        model_slug = str(row.get("model_slug") or "").strip()
        relative = parse_relative(row.get("relative_path"))
        assert relative is not None
        if dataset not in ALLOWED_GROUPS or group not in ALLOWED_GROUPS[dataset]:
            raise RuntimeError(f"Manifest row {row_number} has forbidden group: {dataset}/{group}")
        is_regional = group == REGIONAL_GROUPS[dataset]
        if scope == "nonregional" and is_regional:
            raise RuntimeError(f"Regional path is forbidden at row {row_number}")
        if scope == "regional" and not is_regional:
            raise RuntimeError(f"Non-regional path is forbidden at row {row_number}")
        if model_slug != MODEL_SLUG:
            raise RuntimeError(f"Manifest row {row_number} model mismatch: {model_slug}")
        if not persona or not question_id or not issue or not source_state:
            raise RuntimeError(f"Manifest row {row_number} has an empty required value")
        if relative.parts[:3] != (group, MODEL_SLUG, persona):
            raise RuntimeError(f"Manifest row {row_number} path mismatch: {relative}")
        if relative.name != f"{question_id}.json":
            raise RuntimeError(f"Manifest row {row_number} ID/path mismatch: {relative}")
        key = (dataset, relative)
        if key in seen:
            raise RuntimeError(f"Duplicate manifest target: {dataset}/{relative}")
        seen.add(key)
        code_text = str(row.get("http_code") or "").strip()
        try:
            http_code = int(code_text) if code_text else None
        except ValueError as exc:
            raise RuntimeError(f"Invalid HTTP code at row {row_number}") from exc

        root = dataset_root(dataset)
        path = safe_path(root, relative)
        template_relative = parse_relative(
            row.get("template_relative_path"), allow_empty=True
        )
        question_relative = parse_relative(
            row.get("question_source_relative_path"), allow_empty=True
        )
        if source_state == "valid":
            recovery_mode = "valid_json"
        elif (
            source_state == "truncated"
            and template_relative is None
            and question_relative is None
        ):
            recovery_mode = "valid_json_prefix"
        elif dataset == "GSM8K" and source_state in {
            "empty",
            "truncated",
            "missing",
            "unreadable",
            "invalid_object",
        }:
            if template_relative is None or question_relative is None:
                raise RuntimeError(f"Recovery sources missing at manifest row {row_number}")
            recovery_mode = "rebuilt_from_manifest_sources"
        else:
            raise RuntimeError(
                f"Unsupported source state {source_state!r} at row {row_number}"
            )

        target = Target(
            row_number=row_number,
            dataset=dataset,
            group=group,
            persona=persona,
            question_id=question_id,
            issue_category=issue,
            audited_http_code=http_code,
            audited_error_message=str(row.get("error_message") or "").strip(),
            source_state=source_state,
            relative_path=relative,
            path=path,
            template_relative_path=template_relative,
            question_source_relative_path=question_relative,
            recovery_mode=recovery_mode,
        )
        targets.append(target)
    return targets


def hydrate_target(target: Target) -> Target:
    """Load one target only when a worker actually needs it."""
    if target.payload is not None:
        return target
    hydrated_source_state = target.source_state
    if target.source_state == "valid":
        payload = read_json(target.path)
    else:
        # A prior successful rerun may already have created/replaced a target
        # whose immutable manifest row still says missing/truncated. Prefer the
        # current valid result so resume can skip it correctly.
        payload = None
        if target.path.is_file():
            try:
                payload = read_json(target.path)
                hydrated_source_state = "valid"
            except RuntimeError:
                payload = None
        if payload is None and target.recovery_mode == "valid_json_prefix":
            payload = read_json_prefix(target.path)
        elif payload is None:
            if (
                target.template_relative_path is None
                or target.question_source_relative_path is None
            ):
                raise RuntimeError(f"Recovery sources missing for {target.path}")
            root = dataset_root(target.dataset)
            payload = rebuild_gsm8k_payload(
                target.path,
                target.question_id,
                target.persona,
                safe_path(root, target.template_relative_path),
                safe_path(root, target.question_source_relative_path),
            )
    hydrated = replace(
        target,
        payload=payload,
        source_state=hydrated_source_state,
    )
    validate_payload(hydrated)
    return hydrated


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


def response_texts(raw_api: dict[str, Any]) -> tuple[str, str, str]:
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
    return content, reasoning, str(choice.get("finish_reason") or "")


def raw_api_error(payload: dict[str, Any]) -> object | None:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    return api_response_error(raw)


def current_content(payload: dict[str, Any]) -> str:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    content = text_value(response.get("content"))
    raw = response.get("raw_api_response")
    if not content and isinstance(raw, dict):
        content, _, _ = response_texts(raw)
    if not content:
        prediction = payload.get("prediction")
        if isinstance(prediction, dict):
            content = text_value(prediction.get("raw"))
    return content


def is_resolved(target: Target) -> bool:
    if target.payload is None:
        raise RuntimeError(f"Cannot inspect an unloaded target: {target.path}")
    if target.source_state != "valid":
        return False
    response = target.payload.get("response")
    raw = response.get("raw_api_response") if isinstance(response, dict) else None
    _, _, finish_reason = response_texts(raw) if isinstance(raw, dict) else ("", "", "")
    return bool(
        isinstance(response, dict)
        and response.get("success") is True
        and current_content(target.payload)
        and raw_api_error(target.payload) is None
        and finish_reason not in {"error", "content_filter", "length"}
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
        return 4096
    return value if value > 0 else 4096


def stored_temperature(payload: dict[str, Any]) -> float:
    try:
        value = float(request_parameters(payload).get("temperature"))
    except (TypeError, ValueError):
        return 0.2
    return value if value >= 0 else 0.2


def warmup_content(payload: dict[str, Any]) -> str:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    content = text_value(response.get("content"))
    raw = response.get("raw_api_response")
    if not content and isinstance(raw, dict):
        content, _, _ = response_texts(raw)
    return content


def prepare_messages(target: Target) -> tuple[list[dict[str, Any]], str | None]:
    if target.payload is None:
        raise RuntimeError(f"Cannot prepare an unloaded target: {target.path}")
    request = target.payload.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"Target has no request messages: {target.path}")
    prepared = copy.deepcopy(messages)
    if target.dataset != "ETHICS" or "_implicit" not in target.persona:
        return prepared, None
    if any(message.get("role") == "assistant" for message in prepared):
        return prepared, None
    if len(prepared) != 3 or [message.get("role") for message in prepared] != [
        "system",
        "user",
        "user",
    ]:
        raise RuntimeError(f"Cannot safely repair implicit messages: {target.path}")
    warmup_relative = PurePosixPath(
        target.group,
        MODEL_SLUG,
        target.persona,
        "warmup",
        "persona_warmup.json",
    )
    warmup_path = safe_path(ETHICS_ROOT, warmup_relative)
    warmup = read_json(warmup_path)
    assistant_content = warmup_content(warmup)
    if not assistant_content:
        raise RuntimeError(f"Warm-up cache has no final content: {warmup_path}")
    warmup_request = warmup.get("request")
    warmup_messages = (
        warmup_request.get("messages") if isinstance(warmup_request, dict) else None
    )
    if not isinstance(warmup_messages, list) or warmup_messages != prepared[:2]:
        raise RuntimeError(f"Warm-up prompt does not match target: {warmup_path}")
    restored = [
        copy.deepcopy(prepared[0]),
        copy.deepcopy(prepared[1]),
        {"role": "assistant", "content": assistant_content},
        copy.deepcopy(prepared[2]),
    ]
    return restored, warmup_relative.as_posix()


def load_api_key(config_path: Path) -> tuple[str, str]:
    environment_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if environment_key:
        return environment_key, "OPENROUTER_API_KEY"
    try:
        entries = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "OPENROUTER_API_KEY is unset and config/model_keys.json is missing"
        ) from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"Model config must contain a list: {config_path}")
    credential_order = MODEL_ORDER if CREDENTIAL_ORDER is None else CREDENTIAL_ORDER
    credential_model_id = (
        MODEL_ID if CREDENTIAL_MODEL_ID is None else CREDENTIAL_MODEL_ID
    )
    for entry in entries:
        if not isinstance(entry, dict) or int(entry.get("order", -1)) != credential_order:
            continue
        if str(entry.get("model_id") or "") != credential_model_id:
            raise RuntimeError(
                f"Credential order {credential_order} has an unexpected model ID"
            )
        key = str(entry.get("api_key_alias") or "").strip()
        if not key:
            raise RuntimeError(
                f"Credential order {credential_order} has no configured key"
            )
        return key, str(config_path)
    raise RuntimeError(
        f"Credential order {credential_order} is missing from {config_path}"
    )


def openrouter_auth_url(chat_url: str) -> str:
    clean = chat_url.rstrip("/")
    suffix = "/chat/completions"
    if not clean.endswith(suffix):
        raise RuntimeError(f"Endpoint must end with {suffix}: {chat_url}")
    return clean[: -len(suffix)] + "/auth/key"


def auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers.update({
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://localhost",
            "X-Title": f"TA {MODEL_LABEL} selective rerun",
        })
    return headers


def check_openrouter_key(
    endpoint_url: str,
    api_key: str,
    timeout_sec: float,
) -> None:
    url = openrouter_auth_url(endpoint_url)
    try:
        response = requests.get(url, headers=auth_headers(api_key), timeout=timeout_sec)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"OpenRouter key preflight failed: {type(exc).__name__}: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else {}
    data = data if isinstance(data, dict) else {}
    remaining = data.get("limit_remaining")
    tier = "free" if data.get("is_free_tier") else "paid"
    print(f"[PREFLIGHT] OpenRouter authenticated; tier={tier}; limit_remaining={remaining}")


def sanitize_error(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+", "[URL omitted]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key|bearer)\s*[:=]\s*\S+",
        r"\1=[omitted]",
        text,
    )
    return text[:1000]


def api_response_error(raw_api: object) -> object | None:
    if not isinstance(raw_api, dict):
        return None
    error = raw_api.get("error")
    if error is None:
        choices = raw_api.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            error = choices[0].get("error")
    return error


def api_error_code(raw_api: object) -> int | None:
    error = api_response_error(raw_api)
    if not isinstance(error, dict):
        return None
    try:
        value = int(error.get("code"))
    except (TypeError, ValueError):
        return None
    return value if 400 <= value <= 599 else None


def post_chat(
    endpoint_url: str,
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_sec: float,
    timeout_sec: float,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
) -> dict[str, Any]:
    if reasoning_effort and reasoning_max_tokens is not None:
        raise ValueError("reasoning effort and max_tokens cannot both be set")
    request_payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        request_payload["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_max_tokens is not None:
        request_payload["reasoning"] = {"max_tokens": reasoning_max_tokens}
    total_latency = 0.0
    last_error = "request did not run"
    last_raw: dict[str, Any] | None = None
    last_status: int | None = None
    for retry_count in range(max_retries + 1):
        started = time.perf_counter()
        retryable = True
        try:
            response = requests.post(
                endpoint_url,
                headers=auth_headers(api_key),
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
            body_error = api_response_error(last_raw)
            structured_code = api_error_code(last_raw)
            has_error = body_error is not None
            _, _, finish_reason = (
                response_texts(last_raw) if isinstance(last_raw, dict)
                else ("", "", "")
            )
            abnormal_finish = finish_reason in {"error", "content_filter", "length"}
            if (200 <= response.status_code < 300 and last_raw is not None
                    and not has_error and not abnormal_finish):
                return {
                    "success": True,
                    "api_response": last_raw,
                    "http_status": response.status_code,
                    "error": None,
                    "retry_count": retry_count,
                    "latency_sec": total_latency,
                }
            if has_error:
                last_error = f"API body contains error: {sanitize_error(body_error)}"
                retryable = structured_code == 429 or (
                    structured_code is not None and structured_code >= 500
                )
            elif abnormal_finish:
                last_error = f"API body has abnormal finish_reason={finish_reason}"
                retryable = finish_reason in {"error", "length"}
            else:
                last_error = f"HTTP {response.status_code}: {sanitize_error(response.text)}"
                retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
        except requests.RequestException as exc:
            total_latency += time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {sanitize_error(exc)}"
            last_raw = None
            last_status = None
        if retry_count >= max_retries or not retryable:
            return {
                "success": False,
                "api_response": last_raw,
                "http_status": last_status,
                "error": last_error,
                "retry_count": retry_count,
                "latency_sec": total_latency,
            }
        time.sleep(retry_base_sec * (2**retry_count))
    raise AssertionError("unreachable retry loop")


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")


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


def backup_target(target: Target, root: Path, run_id: str) -> Path | None:
    """Back up exactly one pending source immediately before its API attempt."""
    if not target.path.exists():
        return None
    backup_path = (root / run_id / target.dataset.lower()).joinpath(
        *target.relative_path.parts
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target.path, backup_path)
    return backup_path


def attempt_path(root: Path, run_id: str, target: Target) -> Path:
    return (root / run_id / target.dataset.lower()).joinpath(
        *target.relative_path.parts
    )


def success_path(args: argparse.Namespace, target: Target) -> Path:
    """Return the selected success destination without changing source paths."""
    if args.success_root is None:
        return target.path
    dataset_output_root = args.success_root / target.dataset.lower()
    return safe_path(dataset_output_root, target.relative_path)


def isolated_success_is_resolved(
    args: argparse.Namespace,
    target: Target,
) -> bool:
    if args.success_root is None:
        return False
    destination = success_path(args, target)
    if not destination.is_file():
        return False
    payload = read_json(destination)
    isolated = replace(
        target,
        path=destination,
        payload=payload,
        source_state="valid",
        recovery_mode="valid_json",
    )
    validate_payload(isolated)
    return is_resolved(isolated)


def write_failed_attempt(
    target: Target,
    args: argparse.Namespace,
    run_id: str,
    api_result: dict[str, Any],
    reason: str,
) -> Path:
    path = attempt_path(args.attempt_root, run_id, target)
    payload = {
        "dataset": target.dataset,
        "question_id": target.question_id,
        "persona": target.persona,
        "source_path": target.relative_path.as_posix(),
        "source_state": target.source_state,
        "source_recovery_mode": target.recovery_mode,
        "audited_issue_category": target.issue_category,
        "audited_http_code": target.audited_http_code,
        "audited_error_message": target.audited_error_message,
        "attempt_timestamp": datetime.now().astimezone().isoformat(),
        "failure_reason": reason,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_max_tokens": args.reasoning_max_tokens,
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


def update_ethics_payload(
    payload: dict[str, Any],
    target: Target,
    request: dict[str, Any],
    raw_api: dict[str, Any],
    final_content: str,
    api_result: dict[str, Any],
    max_tokens: int,
    temperature: float,
    args: argparse.Namespace,
    warmup_source: str | None,
) -> None:
    explanation, parsed_answer, parsing_failed = parse_v9_response(final_content)
    reparsed_answer, reparse_source = reparse_ethics_answer(final_content)
    effective = parsed_answer if parsed_answer in ("0", "1") else reparsed_answer
    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    parameters.update(
        {"temperature": temperature, "max_tokens": max_tokens, "model": MODEL_ID}
    )
    if args.reasoning_effort:
        parameters["reasoning"] = {"effort": args.reasoning_effort}
    elif args.reasoning_max_tokens is not None:
        parameters["reasoning"] = {"max_tokens": args.reasoning_max_tokens}
    request["parameters"] = parameters
    payload["request"] = request
    payload["prediction"] = {
        "raw": final_content,
        "explanation": explanation,
        "parsed_answer": parsed_answer,
        "parsing_failed": parsing_failed,
        "parse_source": "prompt_v9",
    }
    payload["response"] = {
        "success": True,
        "raw_api_response": raw_api,
        "error": None,
        "retry_count": api_result.get("retry_count"),
        "latency_sec": api_result.get("latency_sec"),
    }
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "rerun": f"{RERUN_TAG_PREFIX}_{args.scope}_{TRANSPORT_LABEL}",
            "rerun_timestamp": datetime.now().astimezone().isoformat(),
            "rerun_manifest": str(args.manifest),
            "rerun_endpoint": args.endpoint_url,
            "rerun_original_issue_category": target.issue_category,
            "rerun_original_http_code": target.audited_http_code,
            "rerun_reparse_answer": reparsed_answer,
            "rerun_reparse_source": reparse_source,
        }
    )
    if warmup_source:
        metadata["rerun_warmup_recovered_from"] = warmup_source
    analysis = payload.setdefault("analysis", {})
    ground_truth = normalize_binary_label(
        analysis.get("ground_truth_label", analysis.get("GT"))
    )
    prediction = normalize_binary_label(effective)
    analysis["prediction_label"] = prediction
    analysis["is_correct"] = (
        prediction == ground_truth
        if prediction is not None and ground_truth is not None
        else None
    )


def update_gsm8k_payload(
    payload: dict[str, Any],
    target: Target,
    request: dict[str, Any],
    raw_api: dict[str, Any],
    final_content: str,
    api_result: dict[str, Any],
    max_tokens: int,
    temperature: float,
    args: argparse.Namespace,
) -> None:
    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    parameters.update(
        {"temperature": temperature, "max_tokens": max_tokens, "model": MODEL_ID}
    )
    if args.reasoning_effort:
        parameters["reasoning"] = {"effort": args.reasoning_effort}
    elif args.reasoning_max_tokens is not None:
        parameters["reasoning"] = {"max_tokens": args.reasoning_max_tokens}
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
    payload["analysis"] = {"extracted_answer": None, "is_correct": None}
    metadata = payload.setdefault("metadata", {})
    model_info = metadata.setdefault("model_info", {})
    model_info.update(
        {
            "provider": PROVIDER_LABEL,
            "model_id": MODEL_ID,
            "api_endpoint": args.endpoint_url,
        }
    )
    metadata.update(
        {
            "rerun": f"{RERUN_TAG_PREFIX}_{args.scope}_{TRANSPORT_LABEL}",
            "rerun_timestamp": datetime.now().astimezone().isoformat(),
            "rerun_manifest": str(args.manifest),
            "rerun_source_recovery_mode": target.recovery_mode,
            "rerun_original_issue_category": target.issue_category,
            "rerun_original_http_code": target.audited_http_code,
        }
    )


def process_target(
    target: Target,
    args: argparse.Namespace,
    api_key: str,
    run_id: str,
) -> tuple[str, str, str]:
    target = hydrate_target(target)
    if not args.force and is_resolved(target):
        return "skipped", target.question_id, "already contains a successful response"
    if not args.force and isolated_success_is_resolved(args, target):
        return "skipped", target.question_id, "isolated success already exists"
    destination = success_path(args, target)
    if args.success_root is None:
        backup_target(target, args.backup_root, run_id)
    assert target.payload is not None
    payload = copy.deepcopy(target.payload)
    request = payload.get("request")
    request = request if isinstance(request, dict) else {}
    messages, warmup_source = prepare_messages(target)
    request["messages"] = messages
    max_tokens = args.max_tokens or stored_max_tokens(payload)
    temperature = (
        args.temperature
        if args.temperature is not None
        else stored_temperature(payload)
    )
    api_result = post_chat(
        endpoint_url=args.endpoint_url,
        api_key=api_key,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=args.max_retries,
        retry_base_sec=args.retry_base_sec,
        timeout_sec=args.request_timeout_sec,
        reasoning_effort=args.reasoning_effort,
        reasoning_max_tokens=args.reasoning_max_tokens,
    )
    raw_api = api_result.get("api_response")
    raw_api = raw_api if isinstance(raw_api, dict) else {}
    final_content, reasoning, finish_reason = response_texts(raw_api)
    if not api_result.get("success"):
        reason = sanitize_error(api_result.get("error") or "API request failed")
    elif api_response_error(raw_api) is not None:
        reason = f"API body contains error: {sanitize_error(api_response_error(raw_api))}"
    elif finish_reason in {"error", "content_filter", "length"}:
        reason = f"Abnormal finish_reason={finish_reason}"
    elif not final_content:
        detail = "reasoning present" if reasoning else "no reasoning"
        reason = f"No final assistant content ({detail}; finish_reason={finish_reason})"
    else:
        reason = ""
    if reason:
        retry_count = int(api_result.get("retry_count") or 0)
        reason = f"{reason}; attempts={retry_count + 1} (retries={retry_count})"
        failed_path = write_failed_attempt(target, args, run_id, api_result, reason)
        return "failed", target.question_id, f"{reason}; attempt={failed_path}"

    if target.dataset == "ETHICS":
        update_ethics_payload(
            payload,
            target,
            request,
            raw_api,
            final_content,
            api_result,
            max_tokens,
            temperature,
            args,
            warmup_source,
        )
    else:
        update_gsm8k_payload(
            payload,
            target,
            request,
            raw_api,
            final_content,
            api_result,
            max_tokens,
            temperature,
            args,
        )
    atomic_write_json(destination, payload)
    return "saved", target.question_id, str(destination)


def print_plan(
    args: argparse.Namespace,
    targets: list[Target],
    selected: list[Target],
    already_fixed_checked: int,
) -> None:
    print("=" * 80)
    print(f"{MODEL_LABEL} {args.scope} selective rerun via {API_SERVICE_LABEL}")
    print("=" * 80)
    print(f"Model          : {MODEL_ID}")
    print(f"Endpoint       : {args.endpoint_url}")
    print(f"Manifest       : {args.manifest}")
    print(f"Manifest rows  : {len(targets)}")
    print(f"Dataset filter : {args.dataset}")
    print(f"Scope          : {args.scope}")
    print(f"Selected rows  : {len(selected)}")
    if args.limit is None and not args.force:
        print("Resume check   : lazy; successful files are skipped by workers")
    else:
        print(f"Already fixed  : {already_fixed_checked} while selecting this batch")
    print(f"Datasets       : {dict(sorted(Counter(t.dataset for t in targets).items()))}")
    print(f"Groups         : {dict(sorted(Counter(t.group for t in targets).items()))}")
    print(f"Issues         : {dict(sorted(Counter(t.issue_category for t in targets).items()))}")
    print(f"Source states  : {dict(sorted(Counter(t.source_state for t in targets).items()))}")
    print(f"Recovery modes : {dict(sorted(Counter(t.recovery_mode for t in targets).items()))}")
    token_setting = args.max_tokens if args.max_tokens is not None else "preserve per JSON (normally 4096)"
    temperature_setting = (
        args.temperature
        if args.temperature is not None
        else "preserve per JSON"
    )
    print(f"Token budget   : {token_setting}")
    print(f"Temperature    : {temperature_setting}")
    if args.reasoning_effort:
        reasoning_setting = f"effort={args.reasoning_effort}"
    elif args.reasoning_max_tokens is not None:
        reasoning_setting = f"max_tokens={args.reasoning_max_tokens}"
    else:
        reasoning_setting = "provider default"
    print(f"Reasoning      : {reasoning_setting}")
    print(f"Concurrency    : {args.concurrency}")
    print(f"Opposite scope : forbidden by validation ({args.scope})")
    print("Result scan    : disabled; only manifest/recovery paths are opened")
    if args.success_root is None:
        print("Success writes : replace only selected source JSON after backup")
    else:
        print(f"Success writes : isolated under {args.success_root}")
        print("Source writes  : disabled")
    print()


def run(args: argparse.Namespace) -> int:
    targets = load_targets(args.manifest, args.expected_count, args.scope)
    if args.dataset != "all":
        wanted = args.dataset.upper()
        targets = [target for target in targets if target.dataset == wanted]
    already_fixed_checked = 0
    if args.limit is not None and not args.force:
        selected = []
        for target in targets:
            hydrated = hydrate_target(target)
            if is_resolved(hydrated) or isolated_success_is_resolved(args, hydrated):
                already_fixed_checked += 1
                continue
            prepare_messages(hydrated)
            selected.append(hydrated)
            if len(selected) >= args.limit:
                break
    else:
        selected = targets[: args.limit] if args.limit is not None else targets
    print_plan(args, targets, selected, already_fixed_checked)
    if not args.execute:
        print(
            f"Validation complete: {len(selected)} target(s) selected; "
            "no API calls or files were written."
        )
        return 0
    if not selected:
        print("Nothing to rerun: every selected manifest target is resolved.")
        return 0

    if AUTH_REQUIRED:
        api_key, key_source = load_api_key(args.config)
        print(
            f"[AUTH] {API_SERVICE_LABEL} key loaded from {key_source}; "
            "key value is not logged"
        )
        if not args.skip_preflight:
            check_openrouter_key(
                args.endpoint_url, api_key, args.preflight_timeout_sec
            )
    else:
        api_key = ""
        print(f"[AUTH] {API_SERVICE_LABEL} is configured without authentication")
    run_id = timestamp_slug()
    counts: Counter[str] = Counter()
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_target, target, args, api_key, run_id): target
            for target in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                status, question_id, detail = future.result()
            except Exception as exc:  # continue exact remaining targets
                status = "exception"
                question_id = target.question_id
                detail = f"{type(exc).__name__}: {sanitize_error(exc)}"
            counts[status] += 1
            with print_lock:
                print(
                    f"[{completed}/{len(selected)}] {target.dataset} {question_id} "
                    f"{status.upper()} - {detail}",
                    flush=True,
                )

    print("=" * 80)
    print(f"Completed: {dict(sorted(counts.items()))}")
    if args.success_root is None:
        print(f"Backup   : {args.backup_root / run_id}")
    else:
        print(f"Success  : {args.success_root}")
        print("Sources  : unchanged (isolated-output mode)")
    if counts["failed"] or counts["exception"]:
        print(f"Attempts : {args.attempt_root / run_id}")
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
        print(f"[ERROR] {sanitize_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

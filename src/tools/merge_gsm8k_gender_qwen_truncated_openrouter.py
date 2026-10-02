"""Safely merge the 293 successful isolated Gender Qwen reruns.

The default is a read-only dry-run. ``--execute`` backs up and atomically
replaces only successful paths from the immutable 317-row rerun manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_gender_qwen_truncated_4096"
    / "gsm8k_gender_qwen_truncated_317.csv"
)
SUCCESS_ROOT = (
    REPO_ROOT
    / "result"
    / "_rerun_success"
    / "gsm8k_gender_qwen_truncated_4096"
    / "gsm8k"
)
ATTEMPT_ROOT = (
    REPO_ROOT
    / "result"
    / "_rerun_attempts"
    / "gsm8k_gender_qwen_truncated_4096"
)
SOURCE_ROOT = REPO_ROOT / "result" / "GMS8K"
BACKUP_ROOT = (
    REPO_ROOT
    / "result"
    / "_merge_backup"
    / "gsm8k_gender_qwen_truncated_4096"
)
AUDIT_ROOT = (
    REPO_ROOT
    / "result"
    / "_merge_audit"
    / "gsm8k_gender_qwen_truncated_4096"
)
EXPECTED_MANIFEST = 317
EXPECTED_SUCCESS = 293
EXPECTED_REMAINING = 24


@dataclass(frozen=True)
class MergeItem:
    relative_path: PurePosixPath
    source: Path
    rerun: Path
    original_sha256: str
    rerun_sha256: str
    status: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def safe_relative(value: object) -> PurePosixPath:
    text = str(value or "").strip().replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("Gender_Persona", MODEL_SLUG)
        or len(relative.parts) != 4
        or relative.suffix != ".json"
    ):
        raise RuntimeError(f"Unsafe or out-of-scope path: {text!r}")
    return relative


def rooted(root: Path, relative: PurePosixPath) -> Path:
    path = root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes root: {relative}") from exc
    return path


def load_manifest() -> dict[str, dict[str, str]]:
    try:
        with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(f"Cannot read manifest {MANIFEST}: {exc}") from exc
    if len(rows) != EXPECTED_MANIFEST:
        raise RuntimeError(
            f"Manifest guard expected {EXPECTED_MANIFEST}, found {len(rows)}"
        )
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        relative = safe_relative(row.get("relative_path"))
        key = relative.as_posix()
        if key in indexed:
            raise RuntimeError(f"Duplicate manifest path: {key}")
        if row.get("rerun_temperature") != "0" or row.get("rerun_max_tokens") != "4096":
            raise RuntimeError(f"Unexpected rerun parameters in manifest: {key}")
        indexed[key] = row
    return indexed


def response_fields(payload: dict[str, Any]) -> tuple[str, str, object | None]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    raw = raw if isinstance(raw, dict) else {}
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    content = content.strip() if isinstance(content, str) else ""
    return content, str(choice.get("finish_reason") or ""), raw.get("error")


def validate_pair(source: Path, rerun: Path, row: dict[str, str]) -> tuple[str, str, str]:
    source_payload = read_json(source)
    rerun_payload = read_json(rerun)
    source_sha = sha256(source)
    rerun_sha = sha256(rerun)
    expected_source_sha = str(row.get("source_sha256") or "")
    if source_sha == rerun_sha:
        return source_sha, rerun_sha, "already_merged"
    if source_sha != expected_source_sha:
        raise RuntimeError(
            f"Source changed after manifest creation: {source} "
            f"expected={expected_source_sha} current={source_sha}"
        )
    if source_payload.get("input") != rerun_payload.get("input"):
        raise RuntimeError(f"Rerun changed immutable input: {rerun}")
    source_request = source_payload.get("request")
    rerun_request = rerun_payload.get("request")
    source_request = source_request if isinstance(source_request, dict) else {}
    rerun_request = rerun_request if isinstance(rerun_request, dict) else {}
    if source_request.get("messages") != rerun_request.get("messages"):
        raise RuntimeError(f"Rerun changed request messages: {rerun}")
    source_parameters = source_request.get("parameters")
    rerun_parameters = rerun_request.get("parameters")
    source_parameters = source_parameters if isinstance(source_parameters, dict) else {}
    rerun_parameters = rerun_parameters if isinstance(rerun_parameters, dict) else {}
    source_other = {
        key: value
        for key, value in source_parameters.items()
        if key not in {"temperature", "max_tokens"}
    }
    rerun_other = {
        key: value
        for key, value in rerun_parameters.items()
        if key not in {"temperature", "max_tokens"}
    }
    if source_other != rerun_other:
        raise RuntimeError(f"Rerun changed non-sampling request parameters: {rerun}")
    if rerun_parameters.get("temperature") not in (0, 0.0):
        raise RuntimeError(f"Rerun temperature is not zero: {rerun}")
    if rerun_parameters.get("max_tokens") != 4096:
        raise RuntimeError(f"Rerun max_tokens is not 4096: {rerun}")
    response = rerun_payload.get("response")
    if not isinstance(response, dict) or response.get("success") is not True:
        raise RuntimeError(f"Rerun response is not successful: {rerun}")
    content, finish_reason, api_error = response_fields(rerun_payload)
    if not content or finish_reason != "stop" or api_error is not None:
        raise RuntimeError(
            f"Rerun has unusable final response: {rerun} "
            f"content={bool(content)} finish={finish_reason!r} error={api_error!r}"
        )
    return source_sha, rerun_sha, "pending"


def build_plan() -> tuple[list[MergeItem], set[str]]:
    manifest = load_manifest()
    rerun_files = sorted(SUCCESS_ROOT.rglob("*.json")) if SUCCESS_ROOT.is_dir() else []
    if len(rerun_files) != EXPECTED_SUCCESS:
        raise RuntimeError(
            f"Success guard expected {EXPECTED_SUCCESS}, found {len(rerun_files)}"
        )
    items: list[MergeItem] = []
    seen: set[str] = set()
    for rerun in rerun_files:
        relative = PurePosixPath(rerun.relative_to(SUCCESS_ROOT).as_posix())
        relative = safe_relative(relative.as_posix())
        key = relative.as_posix()
        if key not in manifest:
            raise RuntimeError(f"Rerun output is outside manifest: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate rerun output: {key}")
        seen.add(key)
        source = rooted(SOURCE_ROOT, relative)
        if not source.is_file():
            raise RuntimeError(f"Canonical source is missing: {source}")
        source_sha, rerun_sha, status = validate_pair(source, rerun, manifest[key])
        items.append(
            MergeItem(relative, source, rerun, source_sha, rerun_sha, status)
        )
    remaining = set(manifest) - seen
    if len(remaining) != EXPECTED_REMAINING:
        raise RuntimeError(
            f"Remaining guard expected {EXPECTED_REMAINING}, found {len(remaining)}"
        )
    return items, remaining


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    fields = [
        "relative_path",
        "status",
        "source_before_sha256",
        "rerun_sha256",
        "target_after_sha256",
        "backup_path",
    ]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items, remaining = build_plan()
    pending = [item for item in items if item.status == "pending"]
    already = [item for item in items if item.status == "already_merged"]
    print(f"manifest_rows={EXPECTED_MANIFEST}")
    print(f"successful_outputs={len(items)}")
    print(f"pending_merge={len(pending)}")
    print(f"already_merged={len(already)}")
    print(f"unresolved_not_merged={len(remaining)}")
    print("scope=GSM8K/Gender_Persona/Qwen3-Next-80B only")
    if not args.execute:
        print("dry_run=true source_files_modified=0")
        return 0

    merge_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")
    backup_run = BACKUP_ROOT / merge_id
    audit_rows: list[dict[str, object]] = []
    for item in items:
        if item.status == "already_merged":
            audit_rows.append(
                {
                    "relative_path": item.relative_path.as_posix(),
                    "status": "already_merged",
                    "source_before_sha256": item.original_sha256,
                    "rerun_sha256": item.rerun_sha256,
                    "target_after_sha256": item.rerun_sha256,
                    "backup_path": "",
                }
            )
            continue
        backup = rooted(backup_run, item.relative_path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, backup)
        atomic_copy(item.rerun, item.source)
        after_sha = sha256(item.source)
        if after_sha != item.rerun_sha256:
            raise RuntimeError(f"Post-merge hash mismatch: {item.source}")
        audit_rows.append(
            {
                "relative_path": item.relative_path.as_posix(),
                "status": "merged",
                "source_before_sha256": item.original_sha256,
                "rerun_sha256": item.rerun_sha256,
                "target_after_sha256": after_sha,
                "backup_path": str(backup),
            }
        )
    audit_path = AUDIT_ROOT / f"merge_{merge_id}.csv"
    write_audit(audit_path, audit_rows)
    print(f"merged={len(pending)}")
    print(f"backup={backup_run}")
    print(f"audit={audit_path}")
    print(f"unresolved_untouched={len(remaining)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

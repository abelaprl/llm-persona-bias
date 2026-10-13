"""Build exact non-regional Qwen3 Next 80B infrastructure rerun manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "qwen/qwen3-next-80b-a3b-thinking"
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
ETHICS_ROOT = REPO_ROOT / "result" / "benchmarks" / "ethics" / "runs" / "main" / "v9"
GSM8K_ROOT = REPO_ROOT / "result" / "benchmarks" / "gsm8k" / "runs" / "main"
OUTPUT_DIR = (
    REPO_ROOT / "result" / "rerun_manifests" / "qwen3_next_80b_nonregional"
)

ETHICS_SPECS = (
    ("Gender Persona", ETHICS_ROOT / "Gender Persona" / MODEL_SLUG, 1514),
    ("Neutral Persona", ETHICS_ROOT / "Neutral Persona" / MODEL_SLUG, 1514),
    ("Religion Persona", ETHICS_ROOT / "Religion Persona" / MODEL_SLUG, 1514),
)
GSM8K_SPECS = (
    ("Gender_Persona", GSM8K_ROOT / "Gender_Persona" / MODEL_SLUG, 1319),
    ("Religion_Persona", GSM8K_ROOT / "Religion_Persona" / MODEL_SLUG, 1319),
)
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
)


@dataclass(frozen=True)
class FileMeta:
    dataset: str
    group: str
    persona: str
    path: Path
    relative_path: PurePosixPath


@dataclass(frozen=True)
class Inspection:
    meta: FileMeta
    payload: dict[str, Any] | None
    source_state: str
    issue_category: str | None
    http_code: int | None
    error_message: str


def sanitize_error(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+", "[URL omitted]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key|bearer)\s*[:=]\s*\S+",
        r"\1=[omitted]",
        text,
    )
    return text[:500]


def response_error_parts(payload: dict[str, Any]) -> tuple[object, str, str]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw_api = response.get("raw_api_response")
    raw_api = raw_api if isinstance(raw_api, dict) else {}
    raw_error = raw_api.get("error")
    response_error = str(response.get("error") or "").strip()
    raw_message = ""
    if isinstance(raw_error, dict):
        raw_message = str(raw_error.get("message") or "").strip()
        metadata = raw_error.get("metadata")
        if not raw_message and isinstance(metadata, dict):
            raw_message = str(metadata.get("raw") or "").strip()
    elif isinstance(raw_error, str):
        raw_message = raw_error.strip()
    return raw_error, response_error, raw_message


def classify_payload(payload: dict[str, Any]) -> tuple[str, int | None, str] | None:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw_error, response_error, raw_message = response_error_parts(payload)
    if raw_error is None and response.get("success") is not False and not response_error:
        return None

    combined = " | ".join(item for item in (response_error, raw_message) if item)
    code: int | None = None
    if isinstance(raw_error, dict):
        try:
            candidate = int(raw_error.get("code"))
            code = candidate if 400 <= candidate <= 599 else None
        except (TypeError, ValueError):
            pass
    if code is None:
        match = re.search(r"HTTP\s+([45]\d\d)", combined, re.IGNORECASE)
        code = int(match.group(1)) if match else None

    if code is not None:
        category = f"http_{code}"
    elif re.search(
        r"NameResolutionError|Failed to resolve|getaddrinfo failed",
        combined,
        re.IGNORECASE,
    ):
        category = "network_dns"
    elif re.search(
        r"SSLError|UNEXPECTED_EOF|EOF occurred in violation",
        combined,
        re.IGNORECASE,
    ):
        category = "network_tls"
    elif re.search(
        r"ConnectionResetError|RemoteDisconnected|forcibly closed by the remote",
        combined,
        re.IGNORECASE,
    ):
        category = "network_remote_reset"
    elif re.search(
        r"ConnectionAbortedError|connection was aborted by the software",
        combined,
        re.IGNORECASE,
    ):
        category = "network_host_aborted"
    elif re.search(r"timed out|timeout", combined, re.IGNORECASE):
        category = "network_timeout"
    elif re.search(r"connection|max retries", combined, re.IGNORECASE):
        category = "network_connection"
    else:
        category = "non_http_infra"

    if category == "http_403" and "key limit exceeded" in combined.lower():
        message = "OpenRouter key limit exceeded (total limit)"
    elif category == "network_dns":
        message = "DNS resolution failure for openrouter.ai"
    elif category == "network_tls":
        message = "TLS/SSL connection interrupted"
    elif category == "network_remote_reset":
        message = "Remote endpoint reset or closed the connection"
    elif category == "network_host_aborted":
        message = "Connection aborted by local host (WinError 10053)"
    else:
        message = sanitize_error(combined) or category
    return category, code, message


def inspect_file(meta: FileMeta) -> Inspection:
    try:
        text = meta.path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return Inspection(
            meta,
            None,
            "unreadable",
            "invalid_json_unreadable",
            None,
            sanitize_error(exc),
        )
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        state = "empty" if not text.strip() else "truncated"
        return Inspection(
            meta,
            None,
            state,
            f"invalid_json_{state}",
            None,
            "JSON file is empty" if state == "empty" else "JSON is truncated",
        )
    if not isinstance(payload, dict):
        return Inspection(
            meta,
            None,
            "invalid_object",
            "invalid_json_object",
            None,
            "Top-level JSON value is not an object",
        )
    issue = classify_payload(payload)
    if issue is None:
        return Inspection(meta, payload, "valid", None, None, "")
    category, code, message = issue
    return Inspection(meta, payload, "valid", category, code, message)


def collect_files() -> tuple[list[FileMeta], list[dict[str, object]]]:
    files: list[FileMeta] = []
    inventory: list[dict[str, object]] = []
    for dataset, root, specs in (
        ("ETHICS", ETHICS_ROOT, ETHICS_SPECS),
        ("GSM8K", GSM8K_ROOT, GSM8K_SPECS),
    ):
        for group, model_dir, expected in specs:
            if not model_dir.is_dir():
                raise RuntimeError(f"Missing model directory: {model_dir}")
            for persona_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                main_paths = sorted(persona_dir.glob("*.json"))
                inventory.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "persona": persona_dir.name,
                        "count": len(main_paths),
                        "expected": expected,
                    }
                )
                for path in main_paths:
                    relative = PurePosixPath(path.relative_to(root).as_posix())
                    files.append(
                        FileMeta(dataset, group, persona_dir.name, path, relative)
                    )
    return files, inventory


def qid_number(question_id: str) -> int:
    match = re.fullmatch(r"gsm8k_(\d{5})", question_id)
    if not match:
        raise RuntimeError(f"Invalid GSM8K question ID: {question_id}")
    return int(match.group(1))


def choose_recovery_sources(
    target: FileMeta,
    inspections: list[Inspection],
) -> tuple[str, str]:
    question_id = target.path.stem
    target_number = qid_number(question_id)
    valid = [
        item
        for item in inspections
        if item.meta.dataset == "GSM8K" and item.payload is not None
    ]
    same_persona = [
        item
        for item in valid
        if item.meta.group == target.group and item.meta.persona == target.persona
    ]
    if not same_persona:
        raise RuntimeError(f"No recovery template for {target.relative_path}")
    template = min(
        same_persona,
        key=lambda item: abs(qid_number(item.meta.path.stem) - target_number),
    )
    question_sources = [
        item
        for item in valid
        if item.meta.path.stem == question_id
        and isinstance((item.payload or {}).get("input"), dict)
        and str((item.payload or {}).get("input", {}).get("question") or "").strip()
    ]
    if not question_sources:
        raise RuntimeError(f"No intact question source for {target.relative_path}")
    question_source = sorted(
        question_sources,
        key=lambda item: (
            item.meta.group != "Gender_Persona",
            item.meta.persona != "man_explicit_1",
            item.meta.relative_path.as_posix(),
        ),
    )[0]
    return (
        template.meta.relative_path.as_posix(),
        question_source.meta.relative_path.as_posix(),
    )


def build_rows(
    inspections: list[Inspection],
    inventory: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for item in inspections:
        if item.issue_category is None:
            continue
        template = ""
        question_source = ""
        if item.meta.dataset == "GSM8K" and item.source_state != "valid":
            template, question_source = choose_recovery_sources(item.meta, inspections)
        rows.append(
            {
                "dataset": item.meta.dataset,
                "model_slug": MODEL_SLUG,
                "group": item.meta.group,
                "persona_folder": item.meta.persona,
                "question_id": item.meta.path.stem,
                "issue_category": item.issue_category,
                "http_code": item.http_code or "",
                "error_message": item.error_message,
                "source_state": item.source_state,
                "template_relative_path": template,
                "question_source_relative_path": question_source,
                "relative_path": item.meta.relative_path.as_posix(),
            }
        )

    missing_rows: list[dict[str, object]] = []
    for entry in inventory:
        if entry["dataset"] != "GSM8K":
            continue
        group = str(entry["group"])
        persona = str(entry["persona"])
        expected = int(entry["expected"])
        model_dir = GSM8K_ROOT / group / MODEL_SLUG / persona
        for number in range(1, expected + 1):
            question_id = f"gsm8k_{number:05d}"
            path = model_dir / f"{question_id}.json"
            if path.exists():
                continue
            relative = PurePosixPath(path.relative_to(GSM8K_ROOT).as_posix())
            meta = FileMeta("GSM8K", group, persona, path, relative)
            template, question_source = choose_recovery_sources(meta, inspections)
            row = {
                "dataset": "GSM8K",
                "model_slug": MODEL_SLUG,
                "group": group,
                "persona_folder": persona,
                "question_id": question_id,
                "issue_category": "missing_json",
                "http_code": "",
                "error_message": "Expected output JSON does not exist",
                "source_state": "missing",
                "template_relative_path": template,
                "question_source_relative_path": question_source,
                "relative_path": relative.as_posix(),
            }
            rows.append(row)
            missing_rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["group"]),
            str(row["persona_folder"]),
            str(row["question_id"]),
        )
    )
    return rows, missing_rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--expected-ethics", type=int, default=2949)
    parser.add_argument("--expected-gsm8k", type=int, default=5341)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    files, inventory = collect_files()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspections = list(executor.map(inspect_file, files))
    rows, missing_rows = build_rows(inspections, inventory)
    ethics_rows = [row for row in rows if row["dataset"] == "ETHICS"]
    gsm8k_rows = [row for row in rows if row["dataset"] == "GSM8K"]

    if args.expected_ethics and len(ethics_rows) != args.expected_ethics:
        raise RuntimeError(
            f"ETHICS guard expected {args.expected_ethics}, found {len(ethics_rows)}"
        )
    if args.expected_gsm8k and len(gsm8k_rows) != args.expected_gsm8k:
        raise RuntimeError(
            f"GSM8K guard expected {args.expected_gsm8k}, found {len(gsm8k_rows)}"
        )

    write_manifest(OUTPUT_DIR / "qwen_ethics_infra.csv", ethics_rows)
    write_manifest(OUTPUT_DIR / "qwen_gsm8k_infra.csv", gsm8k_rows)
    write_manifest(OUTPUT_DIR / "qwen_nonregional_infra_all.csv", rows)

    print(f"Scanned main JSON : {len(files)}")
    print(f"ETHICS manifest   : {len(ethics_rows)}")
    print(f"GSM8K manifest    : {len(gsm8k_rows)}")
    print(f"Missing JSON rows : {len(missing_rows)}")
    print(f"Combined manifest : {len(rows)}")
    print(
        "Issue categories  : "
        f"{dict(sorted(Counter(str(row['issue_category']) for row in rows).items()))}"
    )
    print(f"Output directory  : {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

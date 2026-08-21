"""Build an immutable OpenRouter continuation manifest after a partial rerun.

Rows are retained only when their canonical raw JSON is still an API/provider
failure.  Failed-attempt archives supplied with ``--attempt-root`` are inspected
and content-filtered targets are excluded so a resume cannot pay for them again.

Safe default: validate and print counts.  Use ``--write`` to create the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import build_ethics_current_api_failure_manifests as audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, action="append", default=[])
    parser.add_argument("--exclude-all-attempts", action="store_true")
    parser.add_argument("--exclude-question-id", action="append", default=[])
    parser.add_argument("--exclude-source-path", action="append", default=[])
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def safe_raw_path(relative_path: str) -> Path:
    root = audit.RAW_ROOT.resolve()
    candidate = (root / Path(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"Manifest path escapes the raw root: {relative_path}")
    if not candidate.is_file():
        raise RuntimeError(f"Raw target does not exist: {candidate}")
    return candidate


def excluded_attempts(roots: list[Path], exclude_all: bool) -> set[str]:
    excluded: set[str] = set()
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Attempt root does not exist: {root}")
        for path in root.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                continue
            response = payload.get("response")
            response = response if isinstance(response, dict) else {}
            parts = audit.response_parts({"response": response}, "")
            category = audit.classify_missing(parts)
            if not exclude_all and category != "content_filter":
                continue
            source_path = str(payload.get("source_path") or "").strip()
            if not source_path:
                raise RuntimeError(f"Content-filter attempt lacks source_path: {path}")
            excluded.add(Path(source_path).as_posix())
    return excluded


def csv_bytes(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


def write_immutable(path: Path, content: bytes) -> str:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to replace non-identical output: {path}")
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
    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise RuntimeError(f"Manifest does not exist: {manifest}")

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise RuntimeError(f"Invalid manifest columns: {manifest}")
        fieldnames = list(reader.fieldnames)
        source_rows = list(reader)

    excluded = excluded_attempts(
        [path.resolve() for path in args.attempt_root], args.exclude_all_attempts
    )
    excluded_question_ids = set(args.exclude_question_id)
    excluded_source_paths = {
        Path(value).as_posix() for value in args.exclude_source_path
    }
    counts: Counter[str] = Counter()
    retained: list[dict[str, str]] = []
    for row in source_rows:
        relative_path = Path(str(row.get("relative_path") or "")).as_posix()
        raw_path = safe_raw_path(relative_path)
        payload = json.loads(raw_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Raw JSON is not an object: {raw_path}")
        category = audit.classify_missing(audit.response_parts(payload, ""))
        if relative_path in excluded:
            attempt_label = (
                "excluded_attempt" if args.exclude_all_attempts
                else "excluded_content_filter_attempt"
            )
            counts[attempt_label] += 1
        elif relative_path in excluded_source_paths:
            counts["excluded_source_path"] += 1
        elif str(row.get("question_id") or "") in excluded_question_ids:
            counts["excluded_question_id"] += 1
        elif category == "api_or_provider_error":
            retained.append(row)
            counts["retained_api_failure"] += 1
        else:
            counts[f"excluded_current_{category}"] += 1

    if len(retained) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} continuation rows, found {len(retained)}; "
            f"counts={dict(sorted(counts.items()))}"
        )

    content = csv_bytes(fieldnames, retained)
    status = write_immutable(args.output.resolve(), content) if args.write else "planned"
    print(
        f"{status} {args.output.resolve()} rows={len(retained)} "
        f"sha256={hashlib.sha256(content).hexdigest()}"
    )
    print(json.dumps(dict(sorted(counts.items())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2)

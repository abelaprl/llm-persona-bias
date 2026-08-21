"""Split the immutable DeepSeek API-failure manifest by stored token budget.

The current Vast template serves an 8192-token total context. Targets that
request 8192 output tokens therefore need a separate 16384-context server.
Safe default: validate and print counts. Use ``--write`` to create new files;
existing non-identical outputs are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "ethics_v9"
    / "api_failures_current_2026-08-21"
    / "deepseek_vllm_104.csv"
)


def safe_raw(relative_path: str) -> Path:
    root = RAW_ROOT.resolve()
    path = (root / Path(relative_path)).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Manifest path escapes raw root: {relative_path}")
    if not path.is_file():
        raise RuntimeError(f"Missing raw target: {path}")
    return path


def token_budget(payload: dict[str, object]) -> int:
    request = payload.get("request")
    request = request if isinstance(request, dict) else {}
    parameters = request.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    try:
        value = int(parameters.get("max_tokens", 4096))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invalid request.parameters.max_tokens") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid non-positive max_tokens: {value}")
    return value


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_MANIFEST.parent)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise RuntimeError(f"Invalid manifest columns: {args.manifest}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    buckets: dict[int, list[dict[str, str]]] = {4096: [], 8192: []}
    observed: Counter[int] = Counter()
    for row in rows:
        path = safe_raw(str(row.get("relative_path") or ""))
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Raw JSON is not an object: {path}")
        budget = token_budget(payload)
        observed[budget] += 1
        if budget not in buckets:
            raise RuntimeError(f"Unexpected token budget {budget}: {path}")
        buckets[budget].append(row)

    expected = {4096: 93, 8192: 11}
    if dict(observed) != expected:
        raise RuntimeError(f"Expected {expected}, observed {dict(observed)}")

    for budget, expected_count in expected.items():
        output = args.output_root / f"deepseek_vllm_max_tokens_{budget}_{expected_count}.csv"
        content = csv_bytes(fieldnames, buckets[budget])
        status = write_immutable(output, content) if args.write else "planned"
        print(f"{status} {output.resolve()} rows={len(buckets[budget])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2)

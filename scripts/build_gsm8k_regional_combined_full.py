"""Build an audited, full-text combined CSV for GSM8K regional personas.

This keeps every column from each model-level ``results.csv``.  The existing
``_combined/all_models_results.csv`` is intentionally left untouched because
it is a compact, 17-column accuracy-analysis table without full response text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Regional_Cultural_Persona"
)
OUTPUT_DIR = REPORT_ROOT / "_combined"
OUTPUT_CSV = OUTPUT_DIR / "all_models_results_full.csv"
SOURCE_MANIFEST = OUTPUT_DIR / "all_models_results_full_source_manifest.csv"
AUDIT_MANIFEST = OUTPUT_DIR / "all_models_results_full_manifest.json"

MODELS = (
    "deepseek_deepseek-r1-distill-qwen-32b",
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)
REGIONS = (
    "north_american",
    "east_asian",
    "south_asian",
    "sub_saharan_african",
)
VARIANTS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
PERSONAS = tuple(f"{region}_{variant}" for region in REGIONS for variant in VARIANTS)
EXPECTED_PER_CONDITION = 1_319
EXPECTED_ROWS = len(MODELS) * len(PERSONAS) * EXPECTED_PER_CONDITION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def temporary_path_for(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def source_files() -> list[tuple[str, Path]]:
    paths = [(model, REPORT_ROOT / model / "results.csv") for model in MODELS]
    missing = [path for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model result files: {missing}")
    return paths


def common_header(paths: list[tuple[str, Path]]) -> list[str]:
    headers = []
    for _, path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            headers.append(next(csv.reader(handle)))
    if not headers or any(header != headers[0] for header in headers[1:]):
        raise RuntimeError("The six model-level results.csv headers are not identical")
    if "response_content_full" not in headers[0]:
        raise RuntimeError("response_content_full is missing from the source schema")
    return headers[0]


def write_source_manifest(paths: list[tuple[str, Path]], row_counts: Counter) -> None:
    fields = ("model", "source_file", "rows", "bytes", "modified_local", "sha256")
    temporary = temporary_path_for(SOURCE_MANIFEST)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model, path in paths:
                writer.writerow(
                    {
                        "model": model,
                        "source_file": relative(path),
                        "rows": row_counts[model],
                        "bytes": path.stat().st_size,
                        "modified_local": datetime.fromtimestamp(path.stat().st_mtime)
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "sha256": sha256_file(path),
                    }
                )
        os.replace(temporary, SOURCE_MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)


def build() -> dict:
    csv.field_size_limit(sys.maxsize)
    paths = source_files()
    fields = common_header(paths)
    temporary = temporary_path_for(OUTPUT_CSV)
    condition_counts: Counter = Counter()
    model_counts: Counter = Counter()
    keys: set[tuple[str, str, str]] = set()
    visible_text = 0
    unavailable_text = 0

    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=fields,
                extrasaction="raise",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for expected_model, path in paths:
                with path.open(encoding="utf-8-sig", newline="") as source_handle:
                    reader = csv.DictReader(source_handle)
                    for row in reader:
                        model = (row.get("model") or "").strip()
                        persona = (row.get("persona") or "").strip()
                        question_id = (row.get("question_id") or "").strip()
                        if model != expected_model:
                            raise RuntimeError(
                                f"Model mismatch in {path}: {model!r} != {expected_model!r}"
                            )
                        if persona not in PERSONAS:
                            raise RuntimeError(f"Unexpected regional persona: {persona!r}")
                        if not question_id:
                            raise RuntimeError(f"Blank question_id in {path}")
                        key = (model, persona, question_id)
                        if key in keys:
                            raise RuntimeError(f"Duplicate response key: {key}")
                        keys.add(key)
                        model_counts[model] += 1
                        condition_counts[(model, persona)] += 1
                        if (row.get("response_content_full") or "").strip():
                            visible_text += 1
                        else:
                            unavailable_text += 1
                        writer.writerow(row)

        if len(keys) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Combined key count {len(keys):,} does not match {EXPECTED_ROWS:,}"
            )
        bad_conditions = {
            f"{model}/{persona}": condition_counts[(model, persona)]
            for model in MODELS
            for persona in PERSONAS
            if condition_counts[(model, persona)] != EXPECTED_PER_CONDITION
        }
        if bad_conditions:
            raise RuntimeError(f"Invalid condition coverage: {bad_conditions}")
        os.replace(temporary, OUTPUT_CSV)
    finally:
        temporary.unlink(missing_ok=True)

    write_source_manifest(paths, model_counts)
    payload = {
        "version": "regional-full-combined-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_file": relative(OUTPUT_CSV),
        "output_sha256": sha256_file(OUTPUT_CSV),
        "source_manifest": relative(SOURCE_MANIFEST),
        "source_policy": "six model-level results.csv files; no answer reparsing",
        "schema_columns": len(fields),
        "models": len(MODELS),
        "personas": len(PERSONAS),
        "expected_per_model_persona": EXPECTED_PER_CONDITION,
        "row_count": len(keys),
        "unique_response_keys": len(keys),
        "response_content_full_present": visible_text,
        "response_content_full_unavailable": unavailable_text,
        "model_row_counts": {model: model_counts[model] for model in MODELS},
        "validation": {
            "headers_identical": True,
            "all_condition_counts_equal_1319": True,
            "duplicate_response_keys": 0,
        },
    }
    temporary_manifest = temporary_path_for(AUDIT_MANIFEST)
    try:
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, AUDIT_MANIFEST)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Required guard before writing outputs"
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    payload = build()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

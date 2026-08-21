"""Build an audited full-text ETHICS CSV for Gender plus Neutral personas.

This is the ETHICS counterpart of
``scripts/build_gsm8k_regional_combined_full.py``.  It preserves every column
from the canonical model-level report CSVs and only concatenates rows; answers
are never reparsed or rescored here.

Run from the repository root::

    python src/MAIN_PARSING/ETHICS/build_gender_combined_full.py --execute

Outputs::

    result/ETHICS/reports/all_models_results_full.csv
    result/ETHICS/reports/all_models_results_full_source_manifest.csv
    result/ETHICS/reports/all_models_results_full_manifest.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS, REPO_ROOT, REPORT_ROOT  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "result" / "ETHICS" / "reports"
OUTPUT_CSV = OUTPUT_DIR / "all_models_results_full.csv"
SOURCE_MANIFEST = OUTPUT_DIR / "all_models_results_full_source_manifest.csv"
AUDIT_MANIFEST = OUTPUT_DIR / "all_models_results_full_manifest.json"

GENDER_PERSONAS = (
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
)
PERSONAS = (*GENDER_PERSONAS, "neutral")
EXPECTED_PER_CONDITION = 1_514
EXPECTED_ROWS = len(MODELS) * len(PERSONAS) * EXPECTED_PER_CONDITION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPORT_ROOT.parents[2].resolve()).as_posix()


def temporary_path_for(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def source_files() -> list[tuple[str, str, Path]]:
    """Return canonical sources in stable model/group order."""
    paths: list[tuple[str, str, Path]] = []
    for model in MODELS:
        paths.append(
            (
                model,
                "gender",
                REPORT_ROOT
                / "gender"
                / model
                / f"results_gender_{model}.csv",
            )
        )
        paths.append(
            (
                model,
                "neutral",
                REPORT_ROOT
                / "neutral"
                / model
                / f"results_neutral_{model}.csv",
            )
        )
    missing = [path for _, _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical report files: {missing}")
    return paths


def common_header(paths: list[tuple[str, str, Path]]) -> list[str]:
    headers: list[list[str]] = []
    for _, _, path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            headers.append(next(csv.reader(handle)))
    if not headers or any(header != headers[0] for header in headers[1:]):
        raise RuntimeError("The twelve source CSV headers are not identical")
    if "response_content_full" not in headers[0]:
        raise RuntimeError("response_content_full is missing from the source schema")
    return headers[0]


def write_source_manifest(
    paths: list[tuple[str, str, Path]], row_counts: Counter
) -> None:
    fields = (
        "model",
        "source_group",
        "source_file",
        "rows",
        "bytes",
        "modified_local",
        "sha256",
    )
    temporary = temporary_path_for(SOURCE_MANIFEST)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model, group, path in paths:
                writer.writerow(
                    {
                        "model": model,
                        "source_group": group,
                        "source_file": relative(path),
                        "rows": row_counts[(model, group)],
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

    source_counts: Counter = Counter()
    model_counts: Counter = Counter()
    persona_counts: Counter = Counter()
    subset_counts: Counter = Counter()
    status_counts: Counter = Counter()
    condition_counts: Counter = Counter()
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

            for expected_model, source_group, path in paths:
                expected_personas = (
                    frozenset(GENDER_PERSONAS)
                    if source_group == "gender"
                    else frozenset({"neutral"})
                )
                with path.open(encoding="utf-8-sig", newline="") as source_handle:
                    for row in csv.DictReader(source_handle):
                        model = (row.get("model") or "").strip()
                        persona = (row.get("persona_folder") or "").strip()
                        persona_group = (row.get("persona_group") or "").strip()
                        question_id = (row.get("question_id") or "").strip()

                        if model != expected_model:
                            raise RuntimeError(
                                f"Model mismatch in {path}: {model!r} != {expected_model!r}"
                            )
                        if persona not in expected_personas:
                            raise RuntimeError(
                                f"Unexpected persona in {path}: {persona!r}"
                            )
                        if persona_group != source_group:
                            raise RuntimeError(
                                f"persona_group mismatch in {path}: "
                                f"{persona_group!r} != {source_group!r}"
                            )
                        if not question_id:
                            raise RuntimeError(f"Blank question_id in {path}")

                        key = (model, persona, question_id)
                        if key in keys:
                            raise RuntimeError(f"Duplicate response key: {key}")
                        keys.add(key)

                        source_counts[(model, source_group)] += 1
                        model_counts[model] += 1
                        persona_counts[persona] += 1
                        subset_counts[(row.get("subset") or "").strip()] += 1
                        status_counts[(row.get("answer_status") or "").strip()] += 1
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

    write_source_manifest(paths, source_counts)
    payload = {
        "version": "ethics-gender-neutral-full-combined-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_file": relative(OUTPUT_CSV),
        "output_sha256": sha256_file(OUTPUT_CSV),
        "source_manifest": relative(SOURCE_MANIFEST),
        "source_policy": (
            "six canonical gender results CSVs plus six canonical neutral "
            "results CSVs; no answer reparsing or rescoring"
        ),
        "schema_columns": len(fields),
        "models": len(MODELS),
        "gender_personas": len(GENDER_PERSONAS),
        "neutral_personas": 1,
        "personas_total": len(PERSONAS),
        "expected_per_model_persona": EXPECTED_PER_CONDITION,
        "row_count": len(keys),
        "unique_response_keys": len(keys),
        "response_content_full_present": visible_text,
        "response_content_full_unavailable": unavailable_text,
        "model_row_counts": {model: model_counts[model] for model in MODELS},
        "persona_row_counts": {persona: persona_counts[persona] for persona in PERSONAS},
        "subset_row_counts": dict(sorted(subset_counts.items())),
        "answer_status_counts": dict(sorted(status_counts.items())),
        "validation": {
            "headers_identical": True,
            "all_condition_counts_equal_1514": True,
            "duplicate_response_keys": 0,
            "model_identity_matches_source": True,
            "persona_group_matches_source": True,
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

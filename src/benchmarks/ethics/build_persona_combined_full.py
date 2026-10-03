"""Build audited ETHICS full-text CSVs with the Neutral control included.

The output layout mirrors the GSM8K ``reports/<Persona_Group>/_combined``
layout requested for the thesis.  Every source column is preserved and rows
are concatenated without reparsing or rescoring model answers.

Run from the repository root::

    python src/benchmarks/ethics/build_persona_combined_full.py --group gender --execute
    python src/benchmarks/ethics/build_persona_combined_full.py --group religion --execute
    python src/benchmarks/ethics/build_persona_combined_full.py --group regional --execute

Outputs are written below ``result/benchmarks/ethics/reports``.  Each combined folder
contains the full CSV, a source-file manifest, and an audit manifest.
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


EXPECTED_PER_CONDITION = 1_514
DESTINATION_ROOT = REPO_ROOT / "result" / "ETHICS" / "reports"

GROUP_CONFIGS = {
    "gender": {
        "source_slug": "gender",
        "output_folder": "Gender_Persona",
        "personas": (
            "man_explicit_1",
            "man_implicit_1",
            "man_implicit_2",
            "man_implicit_3",
            "woman_explicit_1",
            "woman_implicit_1",
            "woman_implicit_2",
            "woman_implicit_3",
        ),
    },
    "religion": {
        "source_slug": "religion",
        "output_folder": "Religion_Persona",
        "personas": (
            "atheist_explicit",
            "atheist_implicit_1",
            "atheist_implicit_2",
            "atheist_implicit_3",
            "christian_explicit",
            "christian_implicit_1",
            "christian_implicit_2",
            "christian_implicit_3",
            "islam_explicit",
            "islam_implicit_1",
            "islam_implicit_2",
            "islam_implicit_3",
            "jewish_explicit",
            "jewish_implicit_1",
            "jewish_implicit_2",
            "jewish_implicit_3",
        ),
    },
    "regional": {
        "source_slug": "regional",
        "output_folder": "Regional_Cultural_Persona",
        "personas": (
            "east_asian_explicit",
            "east_asian_implicit_1",
            "east_asian_implicit_2",
            "east_asian_implicit_3",
            "north_american_explicit",
            "north_american_implicit_1",
            "north_american_implicit_2",
            "north_american_implicit_3",
            "south_asian_explicit",
            "south_asian_implicit_1",
            "south_asian_implicit_2",
            "south_asian_implicit_3",
            "sub_saharan_african_explicit",
            "sub_saharan_african_implicit_1",
            "sub_saharan_african_implicit_2",
            "sub_saharan_african_implicit_3",
        ),
    },
}


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


def output_paths(group_slug: str) -> tuple[Path, Path, Path]:
    config = GROUP_CONFIGS[group_slug]
    output_dir = DESTINATION_ROOT / config["output_folder"] / "_combined"
    return (
        output_dir / "all_models_results_full.csv",
        output_dir / "all_models_results_full_source_manifest.csv",
        output_dir / "all_models_results_full_manifest.json",
    )


def source_files(group_slug: str) -> list[tuple[str, str, Path]]:
    """Return group and neutral sources in stable model/group order."""
    config = GROUP_CONFIGS[group_slug]
    source_slug = config["source_slug"]
    paths: list[tuple[str, str, Path]] = []
    for model in MODELS:
        paths.append(
            (
                model,
                source_slug,
                REPORT_ROOT
                / source_slug
                / model
                / f"results_{source_slug}_{model}.csv",
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
    destination: Path,
    paths: list[tuple[str, str, Path]],
    row_counts: Counter,
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
    temporary = temporary_path_for(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model, source_group, path in paths:
                writer.writerow(
                    {
                        "model": model,
                        "source_group": source_group,
                        "source_file": relative(path),
                        "rows": row_counts[(model, source_group)],
                        "bytes": path.stat().st_size,
                        "modified_local": datetime.fromtimestamp(path.stat().st_mtime)
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "sha256": sha256_file(path),
                    }
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build(group_slug: str) -> dict:
    config = GROUP_CONFIGS[group_slug]
    group_personas = tuple(config["personas"])
    personas = (*group_personas, "neutral")
    expected_rows = len(MODELS) * len(personas) * EXPECTED_PER_CONDITION
    output_csv, source_manifest, audit_manifest = output_paths(group_slug)

    csv.field_size_limit(sys.maxsize)
    paths = source_files(group_slug)
    fields = common_header(paths)
    temporary = temporary_path_for(output_csv)

    source_counts: Counter = Counter()
    model_counts: Counter = Counter()
    persona_counts: Counter = Counter()
    subset_counts: Counter = Counter()
    status_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    condition_counts: Counter = Counter()
    condition_questions: dict[tuple[str, str], set[str]] = {}
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
                    frozenset(group_personas)
                    if source_group == config["source_slug"]
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

                        condition = (model, persona)
                        source_counts[(model, source_group)] += 1
                        model_counts[model] += 1
                        persona_counts[persona] += 1
                        subset_counts[(row.get("subset") or "").strip()] += 1
                        status_counts[(row.get("answer_status") or "").strip()] += 1
                        confidence_counts[
                            (row.get("extraction_confidence") or "").strip()
                        ] += 1
                        condition_counts[condition] += 1
                        condition_questions.setdefault(condition, set()).add(question_id)
                        if (row.get("response_content_full") or "").strip():
                            visible_text += 1
                        else:
                            unavailable_text += 1
                        writer.writerow(row)

        if len(keys) != expected_rows:
            raise RuntimeError(
                f"Combined key count {len(keys):,} does not match {expected_rows:,}"
            )
        bad_conditions = {
            f"{model}/{persona}": condition_counts[(model, persona)]
            for model in MODELS
            for persona in personas
            if condition_counts[(model, persona)] != EXPECTED_PER_CONDITION
        }
        if bad_conditions:
            raise RuntimeError(f"Invalid condition coverage: {bad_conditions}")

        question_sets = list(condition_questions.values())
        if not question_sets or any(qids != question_sets[0] for qids in question_sets[1:]):
            raise RuntimeError("The model/persona conditions do not share identical question IDs")
        os.replace(temporary, output_csv)
    finally:
        temporary.unlink(missing_ok=True)

    write_source_manifest(source_manifest, paths, source_counts)
    payload = {
        "version": f"ethics-{group_slug}-neutral-full-combined-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_file": relative(output_csv),
        "output_sha256": sha256_file(output_csv),
        "source_manifest": relative(source_manifest),
        "source_policy": (
            f"six canonical {group_slug} results CSVs plus six canonical neutral "
            "results CSVs; no answer reparsing or rescoring"
        ),
        "schema_columns": len(fields),
        "models": len(MODELS),
        "group_personas": len(group_personas),
        "neutral_personas": 1,
        "personas_total": len(personas),
        "expected_per_model_persona": EXPECTED_PER_CONDITION,
        "row_count": len(keys),
        "unique_response_keys": len(keys),
        "response_content_full_present": visible_text,
        "response_content_full_unavailable": unavailable_text,
        "model_row_counts": {model: model_counts[model] for model in MODELS},
        "persona_row_counts": {persona: persona_counts[persona] for persona in personas},
        "subset_row_counts": dict(sorted(subset_counts.items())),
        "answer_status_counts": dict(sorted(status_counts.items())),
        "extraction_confidence_counts": dict(sorted(confidence_counts.items())),
        "validation": {
            "headers_identical": True,
            "all_condition_counts_equal_1514": True,
            "all_conditions_share_identical_question_ids": True,
            "duplicate_response_keys": 0,
            "model_identity_matches_source": True,
            "persona_group_matches_source": True,
        },
    }

    temporary_manifest = temporary_path_for(audit_manifest)
    try:
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, audit_manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(GROUP_CONFIGS))
    parser.add_argument(
        "--execute", action="store_true", help="Required guard before writing outputs"
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required")
    payload = build(args.group)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

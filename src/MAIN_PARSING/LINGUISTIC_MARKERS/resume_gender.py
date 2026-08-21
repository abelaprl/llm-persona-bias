"""Safely finish a gender marker run after extraction already completed.

This recovery path is intentionally narrow: the target must have no manifest,
must contain the four extraction-stage CSVs, and every response row must match
the currently installed pipeline version. It never reparses answers or changes
the response-level marker values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import scipy

from .audit_sampler import build_audit_sample, build_recall_audit_sample
from .config import PIPELINE_VERSION, RANDOM_SEED
from .run_gender import _registry_digest, _write_csv
from .statistics import (
    build_descriptives,
    build_omnibus,
    build_paired_binary,
    build_paired_numeric,
    build_paired_openings,
    build_quality_control,
)


REQUIRED_EXTRACTION_FILES = (
    "response_records.csv",
    "marker_registry.csv",
    "input_inventory.csv",
    "quality_control.csv",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finish audit and statistics for a validated incomplete extraction bundle."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--audit-per-marker", type=int, default=50)
    parser.add_argument("--audit-recall-per-condition", type=int, default=2)
    return parser


def _validate_bundle(output_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Recovery directory does not exist: {output_dir}")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Refusing recovery: manifest.json already marks this run complete.")
    missing = [name for name in REQUIRED_EXTRACTION_FILES if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Recovery bundle is missing extraction files: {missing}")
    required_columns = {
        "pipeline_version",
        "benchmark",
        "model_id",
        "persona",
        "question_id",
        "visible_text",
        "text_available",
    }
    absent = required_columns.difference(frame.columns)
    if absent:
        raise ValueError(f"response_records.csv is missing columns: {sorted(absent)}")
    versions = set(frame["pipeline_version"].dropna().astype(str).unique())
    if versions != {PIPELINE_VERSION}:
        raise ValueError(
            f"Extraction version {sorted(versions)} does not match installed {PIPELINE_VERSION}."
        )
    duplicate = frame.duplicated(
        ["benchmark", "model_id", "persona", "question_id"], keep=False
    )
    if duplicate.any():
        example = frame.loc[
            duplicate, ["benchmark", "model_id", "persona", "question_id"]
        ].iloc[0]
        raise ValueError(f"Duplicate response key in recovery bundle: {example.to_dict()}")
    return frame


def resume(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    for name in REQUIRED_EXTRACTION_FILES:
        if not (output_dir / name).is_file():
            raise FileNotFoundError(f"Missing recovery input: {output_dir / name}")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Refusing recovery: manifest.json already exists.")

    print(f"[RECOVER] Reading {output_dir / 'response_records.csv'} ...", flush=True)
    frame = pd.read_csv(output_dir / "response_records.csv", low_memory=False)
    _validate_bundle(output_dir, frame)
    inventory = pd.read_csv(output_dir / "input_inventory.csv", keep_default_na=False)

    print("[RECOVER] Rebuilding QC and occurrence-level precision audit ...", flush=True)
    _write_csv(build_quality_control(frame), output_dir / "quality_control.csv")
    _write_csv(
        build_audit_sample(
            frame,
            sample_per_marker=args.audit_per_marker,
            seed=RANDOM_SEED,
        ),
        output_dir / "audit_marker_hits.csv",
    )
    _write_csv(
        build_recall_audit_sample(
            frame,
            sample_per_condition=args.audit_recall_per_condition,
            seed=RANDOM_SEED,
        ),
        output_dir / "audit_recall_responses.csv",
    )

    print("[RECOVER] Building descriptives ...", flush=True)
    _write_csv(build_descriptives(frame), output_dir / "condition_descriptives.csv")
    print("[RECOVER] Running paired numeric contrasts ...", flush=True)
    _write_csv(
        build_paired_numeric(frame, args.bootstrap_resamples),
        output_dir / "paired_numeric.csv",
    )
    print("[RECOVER] Running paired marker-presence contrasts ...", flush=True)
    _write_csv(
        build_paired_binary(frame, args.bootstrap_resamples),
        output_dir / "paired_binary.csv",
    )
    print("[RECOVER] Running paired opening-category contrasts ...", flush=True)
    _write_csv(
        build_paired_openings(frame, args.bootstrap_resamples),
        output_dir / "paired_openings.csv",
    )
    print("[RECOVER] Running five-condition omnibus checks ...", flush=True)
    _write_csv(build_omnibus(frame), output_dir / "omnibus.csv")

    gaps = int((inventory.get("input_status", "") != "available_complete").sum())
    generated = list(REQUIRED_EXTRACTION_FILES) + [
        "audit_marker_hits.csv",
        "audit_recall_responses.csv",
        "condition_descriptives.csv",
        "paired_numeric.csv",
        "paired_binary.csv",
        "paired_openings.csv",
        "omnibus.csv",
        "manifest.json",
    ]
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "registry_sha256": _registry_digest(),
        "benchmarks": frame["benchmark"].dropna().astype(str).drop_duplicates().tolist(),
        "models": frame["model_id"].dropna().astype(str).drop_duplicates().tolist(),
        "input_source": sorted(frame.get("input_source", pd.Series(dtype=str)).dropna().astype(str).unique()),
        "dataset_coverage": (
            "full_available_dataset_with_condition_gaps" if gaps else "full_available_dataset"
        ),
        "input_conditions_with_gaps": gaps,
        "max_records_per_persona": None,
        "response_records": len(frame),
        "bootstrap_resamples": args.bootstrap_resamples,
        "audit_per_marker": args.audit_per_marker,
        "audit_recall_per_condition": args.audit_recall_per_condition,
        "random_seed": RANDOM_SEED,
        "extract_only": False,
        "recovered_from_completed_extraction": True,
        "generated_files": generated,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ground_truth_used_for_extraction": False,
        "correctness_filter_applied": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Recovered run completed at {output_dir}", flush=True)
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    resume(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

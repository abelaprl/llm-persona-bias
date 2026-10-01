"""Command-line entry point for the gender linguistic-marker pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import scipy

from .audit_sampler import build_audit_sample, build_recall_audit_sample
from .config import DEFAULT_OUTPUT_ROOT, MAIN_MODELS, PIPELINE_VERSION, RANDOM_SEED
from .extractor import extract_markers
from .csv_loaders import load_csv_records
from .loaders import discover_raw_inputs, load_raw_records
from .registry import MARKERS, registry_rows
from .statistics import (
    build_descriptives,
    build_omnibus,
    build_paired_binary,
    build_paired_numeric,
    build_paired_openings,
    build_quality_control,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and compare gender linguistic markers on matched GSM8K/ETHICS questions."
        )
    )
    parser.add_argument(
        "--input-source",
        choices=("csv", "raw"),
        default="csv",
        help="Use parsed CSVs by default; raw JSON remains an explicit audit fallback.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("gsm8k", "ethics", "both"),
        default="both",
        help="Benchmarks to process; both remain separate in every output table.",
    )
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--audit-per-marker", type=int, default=50)
    parser.add_argument(
        "--audit-recall-per-condition",
        type=int,
        default=2,
        help="Blind full-response samples per model/persona for false-negative review.",
    )
    parser.add_argument(
        "--max-records-per-persona",
        "--max-files-per-persona",
        dest="max_records_per_persona",
        type=int,
        default=None,
        help=(
            "Diagnostic record limit per persona; --max-files-per-persona is a "
            "backward-compatible alias. "
            "Omit for the full datasets required by the thesis."
        ),
    )
    parser.add_argument(
        "--extract-only", action="store_true", help="Write extraction/QC outputs without tests."
    )
    return parser


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def _registry_digest() -> str:
    payload = json.dumps(registry_rows(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _benchmarks(choice: str) -> tuple[str, ...]:
    return ("gsm8k", "ethics") if choice == "both" else (choice,)


def _print_coverage(frame: pd.DataFrame) -> None:
    grouped = frame.groupby(["benchmark", "model_id", "persona"], sort=True)
    for (benchmark, model, persona), group in grouped:
        available = int(group["text_available"].fillna(False).sum())
        print(
            f"[{benchmark.upper()}] {model} | {persona:<22} "
            f"records={len(group):>5,} text={available:>5,} "
            f"missing={len(group) - available:>4,}"
        )


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; old results are never overwritten: {output_dir}"
        )
    subset_mode = args.max_records_per_persona is not None
    if subset_mode:
        print(
            "[WARN] Diagnostic subset mode is active; this output is not a full-dataset thesis result."
        )

    raw_records: list[dict[str, object]] = []
    input_inventory: list[dict[str, object]] = []
    for benchmark in _benchmarks(args.benchmark):
        if args.input_source == "csv":
            print(f"[{benchmark.upper()}] Loading visible responses from parsed CSV ...")
            records, inventory = load_csv_records(
                benchmark,
                models=args.models,
                max_records_per_persona=args.max_records_per_persona,
            )
        else:
            print(f"[{benchmark.upper()}] Discovering raw JSON ...")
            tasks, inventory = discover_raw_inputs(
                benchmark,
                models=args.models,
                max_files_per_persona=args.max_records_per_persona,
            )
            print(f"[{benchmark.upper()}] Loading {len(tasks):,} raw JSON records ...")
            records = load_raw_records(
                benchmark,
                models=args.models,
                workers=args.workers,
                max_files_per_persona=args.max_records_per_persona,
                tasks=tasks,
            )
        input_inventory.extend(inventory)
        gap_count = sum(
            row["input_status"] != "available_complete" for row in inventory
        )
        if gap_count:
            print(
                f"[{benchmark.upper()}] WARN: {gap_count} model/persona input condition(s) "
                "are incomplete, missing, or empty; see input_inventory.csv."
            )
        print(f"[{benchmark.upper()}] Loaded {len(records):,} records")
        raw_records.extend(records)

    if not raw_records:
        raise ValueError("No raw response files were found for the requested inputs.")

    print(f"[EXTRACT] Applying registry {PIPELINE_VERSION} to {len(raw_records):,} responses ...")
    extracted = [extract_markers(record) for record in raw_records]
    frame = pd.DataFrame(extracted)
    _print_coverage(frame)

    # Do not leave an output directory behind when discovery/loading/extraction
    # fails.  From this point onward the tables form one versioned run bundle.
    output_dir.mkdir(parents=True, exist_ok=False)

    # Raw content remains traceable through source_path. Avoid tripling a very large
    # text column in the output because visible_text/explanation_text are the actual
    # analysis surfaces.
    response_output = frame.drop(columns=["response_content"], errors="ignore")
    _write_csv(response_output, output_dir / "response_records.csv")
    _write_csv(pd.DataFrame(registry_rows()), output_dir / "marker_registry.csv")
    inventory_frame = pd.DataFrame(input_inventory)
    _write_csv(inventory_frame, output_dir / "input_inventory.csv")
    quality_control = build_quality_control(frame)
    _write_csv(quality_control, output_dir / "quality_control.csv")

    audit = build_audit_sample(
        frame,
        sample_per_marker=args.audit_per_marker,
        seed=RANDOM_SEED,
    )
    _write_csv(audit, output_dir / "audit_marker_hits.csv")
    recall_audit = build_recall_audit_sample(
        frame,
        sample_per_condition=args.audit_recall_per_condition,
        seed=RANDOM_SEED,
    )
    _write_csv(recall_audit, output_dir / "audit_recall_responses.csv")

    generated = [
        "response_records.csv",
        "marker_registry.csv",
        "input_inventory.csv",
        "quality_control.csv",
        "audit_marker_hits.csv",
        "audit_recall_responses.csv",
    ]
    if not args.extract_only:
        print("[STATS] Building condition descriptives ...")
        _write_csv(build_descriptives(frame), output_dir / "condition_descriptives.csv")
        print("[STATS] Running paired numeric contrasts ...")
        _write_csv(
            build_paired_numeric(frame, args.bootstrap_resamples),
            output_dir / "paired_numeric.csv",
        )
        print("[STATS] Running paired binary-marker contrasts ...")
        _write_csv(
            build_paired_binary(frame, args.bootstrap_resamples),
            output_dir / "paired_binary.csv",
        )
        print("[STATS] Running paired opening-category contrasts ...")
        _write_csv(
            build_paired_openings(frame, args.bootstrap_resamples),
            output_dir / "paired_openings.csv",
        )
        print("[STATS] Running five-condition omnibus checks ...")
        _write_csv(build_omnibus(frame), output_dir / "omnibus.csv")
        generated.extend(
            (
                "condition_descriptives.csv",
                "paired_numeric.csv",
                "paired_binary.csv",
                "paired_openings.csv",
                "omnibus.csv",
            )
        )

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "registry_sha256": _registry_digest(),
        "benchmarks": list(_benchmarks(args.benchmark)),
        "models": list(args.models),
        "input_source": args.input_source,
        "dataset_coverage": (
            "diagnostic_subset"
            if subset_mode
            else "full_available_dataset_with_condition_gaps"
            if any(
                row["input_status"] != "available_complete"
                for row in input_inventory
            )
            else "full_available_dataset"
        ),
        "input_conditions_with_gaps": sum(
            row["input_status"] != "available_complete" for row in input_inventory
        ),
        "max_records_per_persona": args.max_records_per_persona,
        "response_records": len(frame),
        "bootstrap_resamples": args.bootstrap_resamples,
        "audit_per_marker": args.audit_per_marker,
        "audit_recall_per_condition": args.audit_recall_per_condition,
        "random_seed": RANDOM_SEED,
        "extract_only": args.extract_only,
        "generated_files": generated + ["manifest.json"],
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
    print(f"[DONE] New outputs written to {output_dir}")
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

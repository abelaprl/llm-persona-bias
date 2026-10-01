"""Command-line pipeline for source-aligned GSM8K religion linguistic markers."""

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
from .extractor import extract_markers
from .religion_config import (
    DEFAULT_OUTPUT_ROOT,
    GSM8K_NEUTRAL_PARSED_CSV,
    GSM8K_RELIGION_PARSED_CSV,
    MAIN_MODELS,
    PIPELINE_VERSION,
    RANDOM_SEED,
)
from .religion_csv_loaders import load_religion_gsm8k_records
from .religion_registry import ALL_RELIGION_MARKERS, religion_registry_rows
from .religion_statistics import (
    build_religion_descriptives,
    build_religion_direct_availability,
    build_religion_direct_binary,
    build_religion_direct_numeric,
    build_religion_direct_numeric_correctness_sensitivity,
    build_religion_direct_opening_multinomial,
    build_religion_opening_multinomial,
    build_religion_paired_binary,
    build_religion_paired_numeric,
    build_religion_paired_numeric_correctness_sensitivity,
)
from .statistics import build_quality_control


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract paired linguistic markers for GSM8K religion personas."
    )
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--audit-per-marker", type=int, default=50)
    parser.add_argument("--audit-recall-per-condition", type=int, default=2)
    parser.add_argument("--max-records-per-persona", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def _registry_digest() -> str:
    payload = json.dumps(
        religion_registry_rows(), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _print_coverage(frame: pd.DataFrame) -> None:
    grouped = frame.groupby(["model_id", "persona"], sort=True)
    for (model, persona), group in grouped:
        available = int(group["text_available"].fillna(False).sum())
        print(
            f"[GSM8K] {model} | {persona:<22} records={len(group):>5,} "
            f"text={available:>5,} missing={len(group) - available:>4,}"
        )


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; results are write-once: {output_dir}"
        )
    subset_mode = args.max_records_per_persona is not None
    if subset_mode:
        print("[WARN] Diagnostic subset mode; this is not a thesis result.")

    print("[LOAD] Reading source-aligned religion responses and shared Neutral ...")
    records, inventory = load_religion_gsm8k_records(
        models=args.models,
        max_records_per_persona=args.max_records_per_persona,
    )
    if not records:
        raise ValueError("No religion GSM8K source records were loaded.")
    print(f"[LOAD] {len(records):,} actual source records")

    print(
        f"[EXTRACT] Applying {len(ALL_RELIGION_MARKERS)} registered markers "
        f"({PIPELINE_VERSION}) ..."
    )
    frame = pd.DataFrame(
        [
            extract_markers(
                record,
                ALL_RELIGION_MARKERS,
                pipeline_version=PIPELINE_VERSION,
            )
            for record in records
        ]
    )
    duplicate = frame.duplicated(
        ["model_id", "persona", "question_id"], keep=False
    )
    if duplicate.any():
        raise ValueError("Duplicate response keys remained after extraction.")
    _print_coverage(frame)

    output_dir.mkdir(parents=True, exist_ok=False)
    # response_records is the normalized combined Religion + Neutral analysis
    # CSV. Raw response_content is dropped because visible/explanation surfaces
    # retain the auditable text actually used by every marker.
    response_output = frame.drop(columns=["response_content"], errors="ignore")
    _write_csv(response_output, output_dir / "response_records.csv")
    _write_csv(pd.DataFrame(religion_registry_rows()), output_dir / "marker_registry.csv")
    inventory_frame = pd.DataFrame(inventory)
    _write_csv(inventory_frame, output_dir / "input_inventory.csv")
    _write_csv(build_quality_control(frame), output_dir / "quality_control.csv")
    _write_csv(
        build_audit_sample(
            frame,
            sample_per_marker=args.audit_per_marker,
            seed=RANDOM_SEED,
            markers=ALL_RELIGION_MARKERS,
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

    generated = [
        "response_records.csv",
        "marker_registry.csv",
        "input_inventory.csv",
        "quality_control.csv",
        "audit_marker_hits.csv",
        "audit_recall_responses.csv",
    ]
    if not args.extract_only:
        tables = (
            ("condition_descriptives.csv", build_religion_descriptives(frame)),
            (
                "paired_numeric.csv",
                build_religion_paired_numeric(frame, args.bootstrap_resamples),
            ),
            (
                "correctness_sensitivity_paired_numeric.csv",
                build_religion_paired_numeric_correctness_sensitivity(
                    frame, args.bootstrap_resamples
                ),
            ),
            (
                "paired_binary.csv",
                build_religion_paired_binary(frame, args.bootstrap_resamples),
            ),
            ("paired_opening_multinomial.csv", build_religion_opening_multinomial(frame)),
            (
                "direct_identity_numeric.csv",
                build_religion_direct_numeric(frame, args.bootstrap_resamples),
            ),
            (
                "direct_identity_correctness_sensitivity.csv",
                build_religion_direct_numeric_correctness_sensitivity(
                    frame, args.bootstrap_resamples
                ),
            ),
            (
                "direct_identity_binary.csv",
                build_religion_direct_binary(frame, args.bootstrap_resamples),
            ),
            (
                "direct_identity_availability.csv",
                build_religion_direct_availability(frame, args.bootstrap_resamples),
            ),
            (
                "direct_identity_opening_multinomial.csv",
                build_religion_direct_opening_multinomial(frame),
            ),
        )
        for filename, table in tables:
            print(f"[STATS] Writing {filename} ({len(table):,} rows) ...")
            _write_csv(table, output_dir / filename)
            generated.append(filename)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "status": "candidate_pending_manual_precision_recall_audit",
        "registry_sha256": _registry_digest(),
        "benchmark": "gsm8k",
        "models": list(args.models),
        "input_source": "source_aligned_csv",
        "source_files": {
            "religion": str(GSM8K_RELIGION_PARSED_CSV),
            "neutral": str(GSM8K_NEUTRAL_PARSED_CSV),
        },
        "source_sha256": {
            "religion": _sha256(GSM8K_RELIGION_PARSED_CSV),
            "neutral": _sha256(GSM8K_NEUTRAL_PARSED_CSV),
        },
        "target_records": len(args.models) * 17 * 1319,
        "response_records": len(frame),
        "source_gap_records": int(inventory_frame["n_missing_records"].sum()),
        "dataset_coverage": (
            "diagnostic_subset" if subset_mode else "full_available_with_source_gaps"
        ),
        "max_records_per_persona": args.max_records_per_persona,
        "bootstrap_resamples": args.bootstrap_resamples,
        "holm_family_within_identity": "7 planned condition contrasts per model-identity-outcome",
        "holm_family_direct_identity": "24 planned comparisons (6 identity pairs x 4 conditions) per model-outcome",
        "implicit_proxies_aggregated_for_inference": False,
        "religion_identities_pooled": False,
        "correctness_filter_applied": False,
        "correctness_sensitivity_filter": "both paired responses correct after pairing",
        "hidden_reasoning_used": False,
        "text_field": "visible_text",
        "ground_truth_used_for_extraction": False,
        "extract_only": args.extract_only,
        "generated_files": generated + ["manifest.json"],
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] New output bundle: {output_dir}")
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

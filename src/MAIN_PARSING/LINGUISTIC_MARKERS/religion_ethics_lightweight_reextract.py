"""Lightweight multiprocess re-extraction for ETHICS religion markers.

This entry point intentionally avoids the summary/statistics modules (and
therefore SciPy) while worker processes are alive.  It validates every source
key and text-surface metric against a prior response artifact, replaces all 60
religion count/density/presence columns from raw canonical responses, and
writes a write-once intermediate CSV for the full audit finalizer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .ethics_religion_registry import (
    ETHICS_RELIGION_REGISTRY_VERSION,
    RELIGION_MARKERS,
)
from .extractor import extract_markers
from .religion_config import IDENTITY_CONDITIONS
from .religion_csv_loaders import religion_persona_metadata


REQUIRED_COLUMNS = (
    "model",
    "persona_folder",
    "file",
    "question_id",
    "subset",
    "response_content_full",
    "answer_status",
    "abstain_reason",
    "parse_source",
    "reasoning_tokens",
    "truncated",
    "construct_valid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def _as_number(value: object) -> int | float | None:
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _condition_bucket(persona: str) -> str:
    if persona == "neutral":
        return "neutral"
    if persona.endswith("_explicit"):
        return "explicit"
    if "_implicit_" in persona:
        return "implicit"
    raise ValueError(f"Unexpected religion persona: {persona}")


def _iter_batches(
    frame: pd.DataFrame, batch_size: int = 100
) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    batch: list[dict[str, object]] = []
    for record_index, row in enumerate(frame.itertuples(index=False), start=1):
        identity, condition = religion_persona_metadata(row.persona_folder)
        reasoning_tokens = _as_number(row.reasoning_tokens)
        response_content = str(row.response_content_full or "")
        batch.append(
            {
                "model_id": row.model,
                "benchmark": "ethics",
                "subset": row.subset,
                "identity": identity,
                "condition": condition,
                "condition_bucket": row.condition_bucket,
                "persona": row.persona_folder,
                "question_id": row.question_id,
                "raw_source_file": row.file,
                "source_path": str(row.file),
                "source_row": record_index + 1,
                "response_content": response_content,
                "response_success": bool(response_content.strip()),
                "reasoning_present": bool(reasoning_tokens and reasoning_tokens > 0),
                "response_truncated": _as_bool(row.truncated),
                "answer_status": row.answer_status or None,
                "abstain_reason": row.abstain_reason or None,
                "parse_source": row.parse_source or None,
                "construct_valid": _as_bool(row.construct_valid),
                "reasoning_tokens": reasoning_tokens,
            }
        )
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def _extract_batch(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        extracted = extract_markers(
            record,
            markers=RELIGION_MARKERS,
            pipeline_version=ETHICS_RELIGION_REGISTRY_VERSION,
        )
        extracted.pop("response_content", None)
        extracted.pop("explanation_text", None)
        rows.append(extracted)
    return rows


def _extract_chunk(
    frame: pd.DataFrame,
    executor: ProcessPoolExecutor,
    workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generated = executor.map(
        _extract_batch,
        _iter_batches(frame),
        chunksize=1,
        buffersize=workers * 3,
    )
    for batch in generated:
        rows.extend(batch)
    return pd.DataFrame.from_records(rows)


def _extract_chunk_locally(frame: pd.DataFrame) -> pd.DataFrame:
    """Single-process fallback for systems with insufficient paging capacity."""

    rows: list[dict[str, object]] = []
    for batch in _iter_batches(frame):
        rows.extend(_extract_batch(batch))
    return pd.DataFrame.from_records(rows)


def _validate_and_merge(base: pd.DataFrame, religion: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(religion):
        raise ValueError("Base/source chunk lengths differ.")
    for base_col, new_col in (
        ("model_id", "model_id"),
        ("persona", "persona"),
        ("question_id", "question_id"),
    ):
        if not np.array_equal(
            base[base_col].astype(str).to_numpy(),
            religion[new_col].astype(str).to_numpy(),
        ):
            raise ValueError(f"Base/source key order differs at {base_col}.")
    for column in ("subset", "answer_status", "text_status", "opening_category"):
        if not np.array_equal(
            base[column].fillna("").astype(str).to_numpy(),
            religion[column].fillna("").astype(str).to_numpy(),
        ):
            raise ValueError(f"Base/source surface metadata differs at {column}.")
    if not np.array_equal(
        base["text_available"].map(_as_bool).to_numpy(),
        religion["text_available"].map(_as_bool).to_numpy(),
    ):
        raise ValueError("Base/source text availability differs.")
    for column in ("word_count", "sentence_count", "avg_sentence_length", "ttr_300"):
        old = pd.to_numeric(base[column], errors="coerce").to_numpy(float)
        new = pd.to_numeric(religion[column], errors="coerce").to_numpy(float)
        if not np.allclose(old, new, rtol=1e-6, atol=1e-7, equal_nan=True):
            raise ValueError(f"Base/source surface metric differs at {column}.")

    result = base.copy()
    religion_columns = [
        f"{marker.marker_id}__{suffix}"
        for marker in RELIGION_MARKERS
        for suffix in ("count", "density", "present")
    ]
    missing = set(religion_columns).difference(religion.columns)
    if missing:
        raise ValueError(f"Re-extraction is missing columns: {sorted(missing)}")
    for column in religion_columns:
        result[column] = religion[column].to_numpy()
    result["visible_text"] = religion["visible_text"].fillna("").astype(str).to_numpy()
    return result


def run(
    *,
    input_csv: Path,
    base_output: Path,
    stage_path: Path,
    workers: int,
    chunk_size: int,
    resume_partial: bool = False,
    recycle_pool_every: int = 5,
    max_new_rows: int = 0,
) -> dict[str, object]:
    input_csv = input_csv.resolve()
    base_output = base_output.resolve()
    stage_path = stage_path.resolve()
    partial_path = stage_path.with_name(stage_path.name + ".partial")
    stage_manifest_path = stage_path.with_name(stage_path.name + ".manifest.json")
    progress_path = partial_path.with_name(partial_path.name + ".progress.json")
    for target in (stage_path, stage_manifest_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite re-extraction stage: {target}")
    if partial_path.exists() and not resume_partial:
        raise FileExistsError(
            f"Partial stage exists; pass --resume-partial after inspection: {partial_path}"
        )
    if resume_partial and not partial_path.is_file():
        raise FileNotFoundError(f"No partial stage to resume: {partial_path}")
    base_manifest_path = base_output / "manifest.json"
    base_response_path = base_output / "response_level_markers.csv.gz"
    if not base_manifest_path.is_file() or not base_response_path.is_file():
        raise FileNotFoundError("Base manifest/response artifact is incomplete.")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    input_sha = _sha256(input_csv)
    if base_manifest.get("input_sha256") != input_sha:
        raise ValueError("Base output and canonical input SHA-256 differ.")

    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = set(REQUIRED_COLUMNS).difference(header)
    if missing:
        raise ValueError(f"Canonical CSV is missing columns: {sorted(missing)}")
    seen_keys: set[tuple[str, str, str]] = set()
    partial_key_order: list[tuple[str, str, str]] = []
    processed = 0
    if resume_partial:
        partial_header = pd.read_csv(partial_path, nrows=0).columns
        required_partial = {"model_id", "persona", "question_id", "visible_text"}
        missing_partial = required_partial.difference(partial_header)
        if missing_partial:
            raise ValueError(f"Partial stage is missing columns: {sorted(missing_partial)}")
        for key_chunk in pd.read_csv(
            partial_path,
            usecols=["model_id", "persona", "question_id"],
            dtype=str,
            keep_default_na=False,
            chunksize=10_000,
        ):
            keys = list(
                zip(
                    key_chunk["model_id"],
                    key_chunk["persona"],
                    key_chunk["question_id"],
                )
            )
            duplicate = next((key for key in keys if key in seen_keys), None)
            if duplicate is not None or len(set(keys)) != len(keys):
                raise ValueError(f"Duplicate key in partial stage: {duplicate}")
            seen_keys.update(keys)
            partial_key_order.extend(keys)
            processed += len(keys)
        if processed % chunk_size:
            raise ValueError(
                f"Partial rows ({processed:,}) are not aligned to chunk size {chunk_size:,}."
            )
        print(f"[resume] validated {processed:,} partial response keys", flush=True)

    resume_rows = processed
    source_reader = pd.read_csv(
        input_csv,
        usecols=list(REQUIRED_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        chunksize=chunk_size,
    )
    base_reader = pd.read_csv(
        base_response_path,
        low_memory=False,
        chunksize=chunk_size,
    )
    if resume_rows:
        offset = 0
        for _ in range(resume_rows // chunk_size):
            source_skip = next(source_reader)
            base_skip = next(base_reader)
            source_keys = list(
                zip(
                    source_skip["model"].astype(str),
                    source_skip["persona_folder"].astype(str),
                    source_skip["question_id"].astype(str),
                )
            )
            base_keys = list(
                zip(
                    base_skip["model_id"].astype(str),
                    base_skip["persona"].astype(str),
                    base_skip["question_id"].astype(str),
                )
            )
            expected_keys = partial_key_order[offset : offset + len(source_keys)]
            if source_keys != base_keys or source_keys != expected_keys:
                raise ValueError("Partial/source/base key order differs during resume.")
            offset += len(source_keys)
        del partial_key_order
        gc.collect()
    expected_personas = {
        persona
        for conditions in IDENTITY_CONDITIONS.values()
        for persona in conditions.values()
    }
    first_write = not resume_partial
    executor: ProcessPoolExecutor | None = None
    chunks_with_pool = 0
    checkpoint_reached = False
    try:
        while True:
            try:
                source = next(source_reader)
            except StopIteration:
                source = None
            try:
                base = next(base_reader)
            except StopIteration:
                base = None
            if source is None or base is None:
                if not (source is None and base is None):
                    raise ValueError("Canonical source and base response lengths differ.")
                break
            unknown = set(source["persona_folder"]).difference(expected_personas)
            if unknown:
                raise ValueError(f"Unexpected persona folders: {sorted(unknown)}")
            source["condition_bucket"] = source["persona_folder"].map(_condition_bucket)
            if workers == 1:
                religion = _extract_chunk_locally(source)
            else:
                if executor is None:
                    executor = ProcessPoolExecutor(max_workers=workers)
                    chunks_with_pool = 0
                religion = _extract_chunk(source, executor, workers)
            if workers > 1 and executor is not None:
                chunks_with_pool += 1
            if workers > 1 and chunks_with_pool >= recycle_pool_every:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
                gc.collect()
            merged = _validate_and_merge(base, religion)
            keys = list(
                zip(
                    merged["model_id"].astype(str),
                    merged["persona"].astype(str),
                    merged["question_id"].astype(str),
                )
            )
            duplicate = next((key for key in keys if key in seen_keys), None)
            if duplicate is not None:
                raise ValueError(f"Duplicate response key in stage: {duplicate}")
            seen_keys.update(keys)
            merged.to_csv(
                partial_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8",
            )
            first_write = False
            processed += len(merged)
            print(f"[lightweight-reextract] {processed:,} validated responses", flush=True)
            del source, base, religion, merged, keys
            gc.collect()
            if max_new_rows and processed - resume_rows >= max_new_rows:
                checkpoint_reached = True
                print(
                    f"[checkpoint] paused safely at {processed:,} validated responses",
                    flush=True,
                )
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if checkpoint_reached:
        progress = {
            "stage_type": "lightweight_partial_checkpoint",
            "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
            "input_sha256": input_sha,
            "base_manifest_sha256": _sha256(base_manifest_path),
            "validated_rows": processed,
            "unique_response_keys": len(seen_keys),
            "partial_file": str(partial_path),
            "workers": workers,
            "chunk_size": chunk_size,
            "max_new_rows": max_new_rows,
        }
        progress_path.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return progress

    expected_rows = int(base_manifest.get("rows_input", 0))
    if processed != expected_rows or len(seen_keys) != expected_rows:
        raise ValueError(
            f"Incomplete stage: rows={processed:,}, keys={len(seen_keys):,}, "
            f"expected={expected_rows:,}"
        )
    os.replace(partial_path, stage_path)
    progress_path.unlink(missing_ok=True)
    manifest = {
        "stage_type": "lightweight_full_religion_reextract",
        "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
        "input_csv": str(input_csv),
        "input_sha256": input_sha,
        "base_output": str(base_output),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "rows": processed,
        "unique_response_keys": len(seen_keys),
        "religion_markers_reextracted": len(RELIGION_MARKERS),
        "religion_columns_replaced": len(RELIGION_MARKERS) * 3,
        "workers": workers,
        "chunk_size": chunk_size,
        "resumed_from_partial_rows": resume_rows,
        "worker_pool_recycled_every_chunks": recycle_pool_every,
        "stage_file": str(stage_path),
        "stage_sha256": _sha256(stage_path),
    }
    stage_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] lightweight stage: {stage_path}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=(
            repo_root
            / "result"
            / "ETHICS"
            / "reports"
            / "Religion_Persona"
            / "_combined"
            / "all_models_results_full.csv"
        ),
    )
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--stage-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=3_000)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--recycle-pool-every", type=int, default=5)
    parser.add_argument("--max-new-rows", type=int, default=0)
    args = parser.parse_args(argv)
    run(
        input_csv=args.input_csv,
        base_output=args.base_output,
        stage_path=args.stage_path,
        workers=max(1, args.workers),
        chunk_size=max(100, args.chunk_size),
        resume_partial=args.resume_partial,
        recycle_pool_every=max(1, args.recycle_pool_every),
        max_new_rows=max(0, args.max_new_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

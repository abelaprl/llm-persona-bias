"""CSV-first loaders for the gender linguistic-marker pipeline.

The parsed CSVs already contain the visible response text required by this
analysis.  Ground truth and correctness may be retained as QC metadata, but
they never determine whether a response enters marker extraction.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import (
    ETHICS_PARSED_ROOT,
    EXPECTED_QUESTIONS,
    GENDER_PERSONAS,
    GSM8K_PARSED_CSV,
    MAIN_MODELS,
    REPO_ROOT,
)
from .loaders import _persona_metadata


COMMON_COLUMNS = (
    "model",
    "persona_folder",
    "file",
    "question_id",
    "dataset",
    "subset",
    "response_content_full",
    "answer_status",
    "behavior_category",
    "parse_source",
    "is_correct",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "truncated",
    "truncated_by_metadata",
    "truncated_by_content",
    "extraction_issue",
    "error_category",
)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _bool_or_none(value: object) -> bool | None:
    text = str(value or "").strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _number_or_none(value: object) -> int | float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _stable_question_id(row: dict[str, object], benchmark: str) -> str:
    source_file = str(row.get("file") or "").strip()
    if source_file:
        filename = re.split(r"[\\/]+", source_file)[-1]
        return Path(filename).stem
    question_id = str(row.get("question_id") or "").strip()
    if benchmark == "gsm8k":
        raise ValueError(
            "GSM8K CSV row has no file column; its question_id column is known to be corrupt."
        )
    return question_id


def _read_filtered_csv(
    path: Path,
    models: set[str],
    personas: set[str],
    chunksize: int = 10_000,
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in COMMON_COLUMNS if column in header]
    required = {"model", "persona_folder", "response_content_full"}
    missing = required.difference(usecols)
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {sorted(missing)}")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        selected = chunk[
            chunk["model"].isin(models) & chunk["persona_folder"].isin(personas)
        ].copy()
        if selected.empty:
            continue
        selected["_source_row"] = selected.index.to_numpy() + 2
        selected["_source_csv"] = _relative(path)
        chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _csv_row_to_record(row: dict[str, object], benchmark: str) -> dict[str, object]:
    model = str(row.get("model") or "")
    persona = str(row.get("persona_folder") or "")
    identity, condition = _persona_metadata(persona)
    content = str(row.get("response_content_full") or "")
    finish_reason = str(row.get("finish_reason") or "") or None
    truncated = any(
        _bool_or_none(row.get(column)) is True
        for column in ("truncated", "truncated_by_metadata", "truncated_by_content")
    ) or finish_reason == "length"
    error_category = str(row.get("error_category") or "").strip()
    extraction_issue = str(row.get("extraction_issue") or "").strip()
    reasoning_tokens = _number_or_none(row.get("reasoning_tokens"))
    subset = str(row.get("subset") or "").strip()
    if not subset:
        dataset = str(row.get("dataset") or "").strip()
        subset = "all" if benchmark == "gsm8k" else dataset or "unknown"
    return {
        "model_id": model,
        "benchmark": benchmark,
        "subset": subset,
        "identity": identity,
        "condition": condition,
        "persona": persona,
        "question_id": _stable_question_id(row, benchmark),
        "input_source": "parsed_csv",
        "source_path": str(row.get("_source_csv") or ""),
        "source_row": _number_or_none(row.get("_source_row")),
        "raw_source_file": str(row.get("file") or "") or None,
        "response_content": content,
        "response_success": bool(content.strip()),
        "json_valid": None,
        "load_error": "",
        "response_error_present": bool(
            error_category and error_category.casefold() not in {"none", "no_error"}
        ),
        "reasoning_present": bool(reasoning_tokens and reasoning_tokens > 0),
        "finish_reason": finish_reason,
        "response_truncated": truncated,
        "answer_status": str(row.get("answer_status") or "") or None,
        "behavior_category": str(row.get("behavior_category") or "") or None,
        "parse_source": str(row.get("parse_source") or "") or None,
        "is_correct": _bool_or_none(row.get("is_correct")),
        "extraction_issue": extraction_issue or None,
        "error_category": error_category or None,
        "prompt_tokens": _number_or_none(row.get("prompt_tokens")),
        "completion_tokens": _number_or_none(row.get("completion_tokens")),
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": _number_or_none(row.get("total_tokens")),
    }


def _inventory_row(
    benchmark: str,
    model: str,
    persona: str,
    source_path: Path,
    source_exists: bool,
    available: int,
    selected: int,
) -> dict[str, object]:
    expected = EXPECTED_QUESTIONS[benchmark]
    if not source_exists:
        status = "missing_source"
    elif available == 0:
        status = "empty_source"
    elif available < expected:
        status = "available_incomplete"
    else:
        status = "available_complete"
    return {
        "benchmark": benchmark,
        "model_id": model,
        "persona": persona,
        "input_source": "parsed_csv",
        "source_path": _relative(source_path),
        "source_exists": source_exists,
        "n_expected_records": expected,
        "n_records_available": available,
        "n_records_selected": selected,
        "n_missing_records": max(0, expected - available),
        "input_status": status,
    }


def _select_and_inventory(
    frame: pd.DataFrame,
    benchmark: str,
    models: tuple[str, ...],
    personas: tuple[str, ...],
    source_for: dict[tuple[str, str], Path],
    max_records_per_persona: int | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not frame.empty:
        frame = frame.copy()
        frame["_stable_id"] = [
            _stable_question_id(row, benchmark) for row in frame.to_dict("records")
        ]
        duplicate = frame.duplicated(["model", "persona_folder", "_stable_id"], keep=False)
        if duplicate.any():
            example = frame.loc[duplicate, ["model", "persona_folder", "_stable_id"]].iloc[0]
            raise ValueError(f"Duplicate parsed-CSV response key: {example.to_dict()}")
        frame = frame.sort_values(["model", "persona_folder", "_stable_id"], kind="stable")

    chosen: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for model in models:
        for persona in personas:
            condition_rows = frame[
                (frame.get("model", pd.Series(dtype=str)) == model)
                & (frame.get("persona_folder", pd.Series(dtype=str)) == persona)
            ]
            available = len(condition_rows)
            selected_rows = (
                condition_rows.head(max_records_per_persona)
                if max_records_per_persona is not None
                else condition_rows
            )
            if not selected_rows.empty:
                chosen.append(selected_rows)
            source_path = source_for[(model, persona)]
            inventory.append(
                _inventory_row(
                    benchmark,
                    model,
                    persona,
                    source_path,
                    source_path.is_file(),
                    available,
                    len(selected_rows),
                )
            )
    selected = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    return selected, inventory


def load_csv_records(
    benchmark: str,
    models: Iterable[str] = MAIN_MODELS,
    max_records_per_persona: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load visible responses from current parsed CSVs plus an input inventory."""
    benchmark = benchmark.casefold()
    if benchmark not in {"gsm8k", "ethics"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    models = tuple(str(model) for model in models)
    personas = tuple(GENDER_PERSONAS)
    wanted_models = set(models)
    wanted_personas = set(personas)

    if benchmark == "gsm8k":
        source_for = {
            (model, persona): GSM8K_PARSED_CSV for model in models for persona in personas
        }
        frame = _read_filtered_csv(
            GSM8K_PARSED_CSV, wanted_models, wanted_personas
        )
    else:
        frames: list[pd.DataFrame] = []
        source_for: dict[tuple[str, str], Path] = {}
        for model in models:
            gender_path = (
                ETHICS_PARSED_ROOT / "gender" / model / f"results_gender_{model}.csv"
            )
            neutral_path = (
                ETHICS_PARSED_ROOT / "neutral" / model / f"results_neutral_{model}.csv"
            )
            for persona in personas:
                source_for[(model, persona)] = neutral_path if persona == "neutral" else gender_path
            for path in (gender_path, neutral_path):
                loaded = _read_filtered_csv(path, {model}, wanted_personas)
                if not loaded.empty:
                    frames.append(loaded)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    selected, inventory = _select_and_inventory(
        frame,
        benchmark,
        models,
        personas,
        source_for,
        max_records_per_persona,
    )
    records = [
        _csv_row_to_record(row, benchmark) for row in selected.to_dict("records")
    ]
    return records, inventory

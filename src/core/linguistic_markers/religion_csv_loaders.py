"""Load source-aligned GSM8K religion responses plus the shared Neutral baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from .religion_config import (
    EXPECTED_QUESTIONS,
    GSM8K_NEUTRAL_PARSED_CSV,
    GSM8K_RELIGION_PARSED_CSV,
    MAIN_MODELS,
    RELIGION_PERSONAS,
    REPO_ROOT,
)


SOURCE_COLUMNS = (
    "model",
    "persona_folder",
    "file",
    "question_id",
    "stable_question_id",
    "source_file",
    "source_row",
    "dataset",
    "response_content_full",
    "response_success",
    "response_error_present",
    "error_class",
    "error_message",
    "input_json_valid",
    "technical_failure",
    "content_empty",
    "reasoning_present",
    "think_block_stripped",
    "answer_status",
    "parse_source",
    "is_correct",
    "finish_reason",
    "truncated_by_metadata",
    "truncated_by_content",
    "extraction_issue",
    "error_category",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def religion_persona_metadata(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "neutral"
    for identity in ("islam", "christian", "jewish", "atheist"):
        prefix = f"{identity}_"
        if persona.startswith(prefix):
            suffix = persona[len(prefix) :]
            return identity, "explicit" if suffix == "explicit" else suffix
    raise ValueError(f"Unexpected religion persona: {persona}")


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


def _read_source(
    path: Path,
    *,
    models: set[str],
    personas: set[str],
    chunksize: int = 10_000,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in SOURCE_COLUMNS if column in header]
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
        selected["_source_csv"] = _relative(path)
        chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _stable_question_id(row: dict[str, object]) -> str:
    stable = str(row.get("stable_question_id") or "").strip()
    if stable:
        return stable
    raw_file = str(row.get("file") or "").replace("\\", "/").strip()
    if raw_file:
        return Path(raw_file).stem
    raise ValueError("GSM8K source row has neither stable_question_id nor file.")


def _row_to_record(row: dict[str, object]) -> dict[str, object]:
    persona = str(row.get("persona_folder") or "")
    identity, condition = religion_persona_metadata(persona)
    content = str(row.get("response_content_full") or "")
    finish_reason = str(row.get("finish_reason") or "").strip() or None
    raw_success = _bool_or_none(row.get("response_success"))
    technical_failure = _bool_or_none(row.get("technical_failure")) is True
    error_class = str(row.get("error_class") or row.get("error_category") or "").strip()
    reasoning_tokens = _number_or_none(row.get("reasoning_tokens"))
    reasoning_present = _bool_or_none(row.get("reasoning_present"))
    if reasoning_present is None:
        reasoning_present = bool(reasoning_tokens and reasoning_tokens > 0)
    truncated = any(
        _bool_or_none(row.get(column)) is True
        for column in ("truncated_by_metadata", "truncated_by_content")
    ) or finish_reason == "length"
    return {
        "model_id": str(row.get("model") or ""),
        "benchmark": "gsm8k",
        "subset": "all",
        "identity": identity,
        "condition": condition,
        "persona": persona,
        "question_id": _stable_question_id(row),
        "input_source": "source_aligned_csv",
        "source_path": str(row.get("_source_csv") or ""),
        "source_row": _number_or_none(row.get("source_row")),
        "raw_source_file": str(row.get("source_file") or row.get("file") or "") or None,
        "response_content": content,
        "response_success": raw_success if raw_success is not None else bool(content.strip()),
        "json_valid": _bool_or_none(row.get("input_json_valid")),
        "load_error": str(row.get("error_message") or ""),
        "response_error_present": (
            _bool_or_none(row.get("response_error_present")) is True
            or technical_failure
            or bool(error_class and error_class.casefold() not in {"none", "no_error"})
        ),
        "reasoning_present": reasoning_present,
        "finish_reason": finish_reason,
        "response_truncated": truncated,
        "answer_status": str(row.get("answer_status") or "") or None,
        "behavior_category": None,
        "parse_source": str(row.get("parse_source") or "") or None,
        "is_correct": _bool_or_none(row.get("is_correct")),
        "extraction_issue": str(row.get("extraction_issue") or "") or None,
        "error_category": error_class or None,
        "technical_failure": technical_failure,
        "content_empty": _bool_or_none(row.get("content_empty")),
        "prompt_tokens": _number_or_none(row.get("prompt_tokens")),
        "completion_tokens": _number_or_none(row.get("completion_tokens")),
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": _number_or_none(row.get("total_tokens")),
    }


def load_religion_gsm8k_records(
    models: Iterable[str] = MAIN_MODELS,
    max_records_per_persona: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return actual source rows; absent source keys remain explicit inventory gaps."""

    models = tuple(str(model) for model in models)
    wanted_models = set(models)
    religion_personas = set(RELIGION_PERSONAS) - {"neutral"}
    religion = _read_source(
        GSM8K_RELIGION_PARSED_CSV,
        models=wanted_models,
        personas=religion_personas,
    )
    neutral = _read_source(
        GSM8K_NEUTRAL_PARSED_CSV,
        models=wanted_models,
        personas={"neutral"},
    )
    frame = pd.concat([religion, neutral], ignore_index=True, sort=False)
    if not frame.empty:
        # Both publication source-aligned tables already carry a validated
        # stable_question_id. Avoid converting the 200+ MB text frame to a
        # list of Python dictionaries merely to read that column.
        stable = frame.get(
            "stable_question_id", pd.Series("", index=frame.index, dtype=str)
        ).fillna("").astype(str).str.strip()
        missing_stable = stable.eq("")
        if missing_stable.any():
            file_stem = (
                frame.get("file", pd.Series("", index=frame.index, dtype=str))
                .fillna("")
                .astype(str)
                .str.replace("\\", "/", regex=False)
                .str.rsplit("/", n=1)
                .str[-1]
                .str.replace(r"\.json$", "", regex=True)
            )
            stable = stable.mask(missing_stable, file_stem)
        if stable.eq("").any():
            raise ValueError("Source row has neither stable_question_id nor file.")
        frame["_stable_id"] = stable
        duplicate = frame.duplicated(["model", "persona_folder", "_stable_id"], keep=False)
        if duplicate.any():
            example = frame.loc[duplicate, ["model", "persona_folder", "_stable_id"]].iloc[0]
            raise ValueError(f"Duplicate religion response key: {example.to_dict()}")
        frame = frame.sort_values(["model", "persona_folder", "_stable_id"], kind="stable")

    selected_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for model in models:
        for persona in RELIGION_PERSONAS:
            group = frame[
                frame["model"].eq(model) & frame["persona_folder"].eq(persona)
            ]
            chosen = (
                group.head(max_records_per_persona)
                if max_records_per_persona is not None
                else group
            )
            if not chosen.empty:
                selected_frames.append(chosen)
            source_path = (
                GSM8K_NEUTRAL_PARSED_CSV
                if persona == "neutral"
                else GSM8K_RELIGION_PARSED_CSV
            )
            available = len(group)
            inventory.append(
                {
                    "benchmark": "gsm8k",
                    "model_id": model,
                    "persona": persona,
                    "input_source": "source_aligned_csv",
                    "source_path": _relative(source_path),
                    "source_exists": source_path.is_file(),
                    "n_expected_records": EXPECTED_QUESTIONS,
                    "n_records_available": available,
                    "n_records_selected": len(chosen),
                    "n_missing_records": max(0, EXPECTED_QUESTIONS - available),
                    "input_status": (
                        "available_complete"
                        if available >= EXPECTED_QUESTIONS
                        else "available_incomplete"
                        if available
                        else "missing_source"
                    ),
                }
            )

    selected = (
        pd.concat(selected_frames, ignore_index=True, sort=False)
        if selected_frames
        else pd.DataFrame()
    )
    records = [_row_to_record(row) for row in selected.to_dict("records")]
    return records, inventory

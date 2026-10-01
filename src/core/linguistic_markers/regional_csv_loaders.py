"""Load full GSM8K regional-cultural responses plus the shared Neutral baseline."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .regional_config import (
    EXPECTED_QUESTIONS,
    GSM8K_NEUTRAL_PARSED_CSV,
    GSM8K_REGIONAL_PARSED_CSV,
    MAIN_MODELS,
    REGIONAL_IDENTITIES,
    REGIONAL_PERSONAS,
    REPO_ROOT,
)


SOURCE_COLUMNS = (
    "model",
    "persona_folder",
    "persona",
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


def regional_persona_metadata(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "neutral"
    for identity in REGIONAL_IDENTITIES:
        prefix = f"{identity}_"
        if persona.startswith(prefix):
            suffix = persona[len(prefix) :]
            return identity, "explicit" if suffix == "explicit" else suffix
    raise ValueError(f"Unexpected regional persona: {persona}")


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


def _text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


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
    if "model" not in usecols or "response_content_full" not in usecols:
        raise ValueError(f"CSV {path} lacks model or response_content_full")
    persona_column = "persona_folder" if "persona_folder" in usecols else "persona"
    if persona_column not in usecols:
        raise ValueError(f"CSV {path} lacks a persona column")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        selected = chunk[
            chunk["model"].isin(models) & chunk[persona_column].isin(personas)
        ].copy()
        if selected.empty:
            continue
        selected["_persona_key"] = selected[persona_column]
        selected["_source_csv"] = _relative(path)
        chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=[*usecols, "_persona_key", "_source_csv"])
    return pd.concat(chunks, ignore_index=True, sort=False)


def _stable_question_id(row: dict[str, object]) -> str:
    stable = _text(row.get("stable_question_id"))
    if stable:
        return stable
    raw_file = _text(row.get("file")).replace("\\", "/")
    match = re.search(r"gsm8k_(\d+)\.json$", raw_file)
    if match:
        return f"gsm8k_{int(match.group(1)):05d}"
    raw_question = _text(row.get("question_id"))
    if raw_question.isdigit():
        return f"gsm8k_{int(raw_question):05d}"
    if raw_question:
        return raw_question
    raise ValueError("GSM8K source row has no stable question identifier")


def _row_to_record(row: dict[str, object]) -> dict[str, object]:
    persona = str(row.get("_persona_key") or "")
    identity, condition = regional_persona_metadata(persona)
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


def load_regional_gsm8k_records(
    models: Iterable[str] = MAIN_MODELS,
    max_records_per_persona: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return available regional and Neutral source rows plus explicit gaps."""

    models = tuple(str(model) for model in models)
    wanted_models = set(models)
    regional_personas = set(REGIONAL_PERSONAS) - {"neutral"}
    regional = _read_source(
        GSM8K_REGIONAL_PARSED_CSV,
        models=wanted_models,
        personas=regional_personas,
    )
    neutral = _read_source(
        GSM8K_NEUTRAL_PARSED_CSV,
        models=wanted_models,
        personas={"neutral"},
    )
    frame = pd.concat([regional, neutral], ignore_index=True, sort=False)
    if not frame.empty:
        stable = (
            frame.get("stable_question_id", pd.Series("", index=frame.index, dtype=str))
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "NaN": ""})
        )
        missing_stable = stable.eq("")
        if missing_stable.any():
            from_file = (
                frame.get("file", pd.Series("", index=frame.index, dtype=str))
                .fillna("")
                .astype(str)
                .str.replace("\\", "/", regex=False)
                .str.extract(r"(gsm8k_\d+)\.json$", expand=False)
                .fillna("")
            )
            stable = stable.mask(missing_stable, from_file)
        missing_stable = stable.eq("")
        if missing_stable.any():
            numeric_question = pd.to_numeric(
                frame.get("question_id", pd.Series("", index=frame.index)),
                errors="coerce",
            )
            from_question = numeric_question.map(
                lambda value: f"gsm8k_{int(value):05d}" if pd.notna(value) else ""
            )
            stable = stable.mask(missing_stable, from_question)
        if stable.eq("").any():
            raise ValueError("Regional source contains a row without a stable question ID")
        frame["_stable_id"] = stable
        duplicate = frame.duplicated(["model", "_persona_key", "_stable_id"], keep=False)
        if duplicate.any():
            example = frame.loc[duplicate, ["model", "_persona_key", "_stable_id"]].iloc[0]
            raise ValueError(f"Duplicate regional response key: {example.to_dict()}")
        frame = frame.sort_values(["model", "_persona_key", "_stable_id"], kind="stable")

    selected_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for model in models:
        for persona in REGIONAL_PERSONAS:
            group = frame[
                frame["model"].eq(model) & frame["_persona_key"].eq(persona)
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
                else GSM8K_REGIONAL_PARSED_CSV
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

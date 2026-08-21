"""Raw JSON loaders for GSM8K and ETHICS gender conditions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from typing import Iterable

from .config import (
    ETHICS_MAIN_ROOT,
    ETHICS_PARSED_ROOT,
    GENDER_PERSONAS,
    GSM8K_GENDER_ROOT,
    MAIN_MODELS,
    REPO_ROOT,
)


def _load_json_recoverable(path: Path) -> tuple[dict[str, object], bool, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return {}, False, f"{type(exc).__name__}: {exc}"
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}, True, ""
    except json.JSONDecodeError as exc:
        stripped = text.lstrip("\ufeff\x00\r\n\t ")
        try:
            payload, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            return {}, False, f"JSONDecodeError: {exc}"
        if not isinstance(payload, dict) or stripped[end:].strip():
            return {}, False, f"JSONDecodeError: {exc}"
        return payload, False, f"JSONDecodeError_recovered: {exc}"


def _choice(payload: dict[str, object]) -> dict[str, object]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    raw = raw if isinstance(raw, dict) else {}
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _content(payload: dict[str, object]) -> str:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    direct = response.get("content")
    if isinstance(direct, str):
        return direct
    choice = _choice(payload)
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    prediction = payload.get("prediction")
    prediction = prediction if isinstance(prediction, dict) else {}
    raw_prediction = prediction.get("raw")
    return raw_prediction if isinstance(raw_prediction, str) else ""


def _reasoning_present(payload: dict[str, object]) -> bool:
    choice = _choice(payload)
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return True
    details = message.get("reasoning_details")
    return isinstance(details, list) and bool(details)


def _usage(payload: dict[str, object]) -> dict[str, object]:
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    raw = response.get("raw_api_response")
    raw = raw if isinstance(raw, dict) else {}
    usage = raw.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _persona_metadata(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "neutral"
    identity = "woman" if persona.startswith("woman_") else "man"
    condition = "explicit" if "_explicit_" in persona else persona.split(f"{identity}_", 1)[1]
    return identity, condition


def _record_from_path(task: tuple[str, str, str, Path]) -> dict[str, object]:
    benchmark, model, persona, path = task
    payload, json_valid, load_error = _load_json_recoverable(path)
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    choice = _choice(payload)
    content = _content(payload)
    success_value = response.get("success")
    if isinstance(success_value, bool):
        response_success = success_value
    else:
        response_success = bool(content or choice) and not bool(response.get("error"))
    identity, condition = _persona_metadata(persona)
    subset = str(payload.get("subset") or ("all" if benchmark == "gsm8k" else "unknown"))
    usage = _usage(payload)
    try:
        source_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(path)
    return {
        "model_id": model,
        "benchmark": benchmark,
        "subset": subset,
        "identity": identity,
        "condition": condition,
        "persona": persona,
        "question_id": path.stem,
        "input_source": "raw_json",
        "source_path": source_path,
        "source_row": None,
        "raw_source_file": source_path,
        "response_content": content,
        "response_success": response_success,
        "json_valid": json_valid,
        "load_error": load_error,
        "response_error_present": bool(response.get("error")),
        "reasoning_present": _reasoning_present(payload),
        "finish_reason": choice.get("finish_reason"),
        "response_truncated": choice.get("finish_reason") == "length",
        "answer_status": None,
        "behavior_category": None,
        "parse_source": None,
        "is_correct": None,
        **usage,
    }


def _enrich_ethics_status(
    records: list[dict[str, object]], models: Iterable[str]
) -> None:
    """Attach current answer-parser status without using it to select text.

    These fields are quality-control metadata only.  Marker extraction never
    branches on answer status, parse source, correctness, or ground truth.
    """
    by_model: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    wanted = {str(model) for model in models}
    for model in wanted:
        status_map: dict[tuple[str, str], dict[str, str]] = {}
        paths = (
            ETHICS_PARSED_ROOT
            / "gender"
            / model
            / f"results_gender_{model}.csv",
            ETHICS_PARSED_ROOT
            / "neutral"
            / model
            / f"results_neutral_{model}.csv",
        )
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    persona = row.get("persona_folder") or ""
                    question_id = row.get("question_id") or Path(row.get("file") or "").stem
                    status_map[(persona, question_id)] = {
                        "answer_status": row.get("answer_status") or "",
                        "behavior_category": row.get("behavior_category") or "",
                        "parse_source": row.get("parse_source") or "",
                        "is_correct": row.get("is_correct") or "",
                    }
        by_model[model] = status_map
    for record in records:
        status = by_model.get(str(record["model_id"]), {}).get(
            (str(record["persona"]), str(record["question_id"]))
        )
        if status:
            record.update(status)


def discover_raw_files(
    benchmark: str,
    models: Iterable[str] = MAIN_MODELS,
    max_files_per_persona: int | None = None,
) -> list[tuple[str, str, str, Path]]:
    tasks, _ = discover_raw_inputs(benchmark, models, max_files_per_persona)
    return tasks


def discover_raw_inputs(
    benchmark: str,
    models: Iterable[str] = MAIN_MODELS,
    max_files_per_persona: int | None = None,
) -> tuple[list[tuple[str, str, str, Path]], list[dict[str, object]]]:
    """Discover available files and retain explicit condition-level gaps.

    A missing/empty condition is not silently converted into response rows.  It
    is recorded in the returned inventory so downstream paired tests naturally
    have zero complete pairs while the provenance table explains why.
    """
    benchmark = benchmark.casefold()
    if benchmark not in {"gsm8k", "ethics"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    tasks: list[tuple[str, str, str, Path]] = []
    inventory: list[dict[str, object]] = []
    for model in models:
        for persona in GENDER_PERSONAS:
            if benchmark == "gsm8k":
                folder = GSM8K_GENDER_ROOT / model / persona
                pattern = "gsm8k_*.json"
            else:
                group = "Neutral Persona" if persona == "neutral" else "Gender Persona"
                folder = ETHICS_MAIN_ROOT / group / model / persona
                pattern = "ethics_*.json"
            folder_exists = folder.is_dir()
            files = sorted(folder.glob(pattern)) if folder_exists else []
            available = len(files)
            if max_files_per_persona is not None:
                files = files[:max_files_per_persona]
            tasks.extend((benchmark, model, persona, path) for path in files)
            try:
                source_folder = str(folder.relative_to(REPO_ROOT))
            except ValueError:
                source_folder = str(folder)
            inventory.append(
                {
                    "benchmark": benchmark,
                    "model_id": model,
                    "persona": persona,
                    "input_source": "raw_json",
                    "source_path": source_folder,
                    "source_exists": folder_exists,
                    "n_expected_records": 1319 if benchmark == "gsm8k" else 1514,
                    "n_records_available": available,
                    "n_records_selected": len(files),
                    "n_missing_records": max(
                        0, (1319 if benchmark == "gsm8k" else 1514) - available
                    ),
                    "input_status": (
                        "missing_source"
                        if not folder_exists
                        else "empty_source"
                        if not available
                        else "available_incomplete"
                        if available < (1319 if benchmark == "gsm8k" else 1514)
                        else "available_complete"
                    ),
                }
            )
    return tasks, inventory


def load_raw_records(
    benchmark: str,
    models: Iterable[str] = MAIN_MODELS,
    workers: int = 8,
    max_files_per_persona: int | None = None,
    tasks: list[tuple[str, str, str, Path]] | None = None,
) -> list[dict[str, object]]:
    """Load raw response records; hidden reasoning is recorded but never returned."""
    models = tuple(models)
    if tasks is None:
        tasks = discover_raw_files(benchmark, models, max_files_per_persona)
    if workers <= 1:
        records = [_record_from_path(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_record_from_path, tasks, chunksize=64))
    keys: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (
            str(record["benchmark"]),
            str(record["model_id"]),
            str(record["persona"]),
            str(record["question_id"]),
        )
        if key in keys:
            raise ValueError(f"Duplicate response key: {key}")
        keys.add(key)
    if benchmark.casefold() == "ethics":
        _enrich_ethics_status(records, models)
    return records

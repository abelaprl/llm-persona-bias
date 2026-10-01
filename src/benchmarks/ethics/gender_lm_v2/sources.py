"""Allowlisted inputs and immutable source provenance."""
from pathlib import Path
import hashlib
import pandas as pd

KEYS = ["model_id", "persona", "subset", "question_id"]
SOURCE_COLUMNS = ["model", "persona_folder", "file", "question_id", "subset", "response_content_full"]


def reserve_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_keys(frame: pd.DataFrame) -> None:
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Source keys must be non-null and unique")


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"model": "model_id", "persona_folder": "persona"})
    validate_keys(frame)
    return frame

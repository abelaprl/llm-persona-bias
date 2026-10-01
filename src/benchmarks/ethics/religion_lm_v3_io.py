"""Final-only input and immutable run output helpers."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
KEYS = ['model_id', 'persona', 'subset', 'question_id']
SOURCE_COLUMNS = ['model','persona_folder','subset','question_id','response_content_full','answer_status']


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_final_rows(path):
    rows = pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype=str, keep_default_na=False)
    rows = rows.rename(columns={'model':'model_id','persona_folder':'persona','response_content_full':'final_output'})
    if rows[KEYS].eq('').any().any() or rows.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate response key')
    rows['source_row'] = range(len(rows))
    rows['response_key'] = rows[KEYS].apply(lambda r: json.dumps(r.tolist(), separators=(',',':')), axis=1)
    return rows


def create_run_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def build_source_manifest(paths):
    return {str(Path(p).resolve()): {'sha256':sha256(p), 'bytes':Path(p).stat().st_size} for p in paths}

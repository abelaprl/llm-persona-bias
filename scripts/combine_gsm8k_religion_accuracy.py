"""Combine one validated Religion Persona accuracy.csv per model.

The output preserves the source header and every source row verbatim at the
CSV field level. DeepSeek uses the validated Vast rerun report; the other five
models use their canonical report directories.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports" / "Religion_Persona"
VAST_ROOT = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports_vast_rerun_20260814"
    / "Religion_Persona"
)
OUTPUT = CANONICAL_ROOT / "accuracy_all_models.csv"

MODEL_SOURCES = [
    (
        "deepseek_deepseek-r1-distill-qwen-32b",
        VAST_ROOT / "deepseek_deepseek-r1-distill-qwen-32b" / "accuracy.csv",
    ),
    (
        "meta-llama_llama-3.1-8b-instruct",
        CANONICAL_ROOT / "meta-llama_llama-3.1-8b-instruct" / "accuracy.csv",
    ),
    (
        "meta-llama_llama-3.3-70b-instruct",
        CANONICAL_ROOT / "meta-llama_llama-3.3-70b-instruct" / "accuracy.csv",
    ),
    (
        "mistralai_ministral-8b-2512",
        CANONICAL_ROOT / "mistralai_ministral-8b-2512" / "accuracy.csv",
    ),
    (
        "openai_gpt-oss-20b",
        CANONICAL_ROOT / "openai_gpt-oss-20b" / "accuracy.csv",
    ),
    (
        "qwen_qwen3-next-80b-a3b-thinking",
        CANONICAL_ROOT / "qwen_qwen3-next-80b-a3b-thinking" / "accuracy.csv",
    ),
]


def main() -> int:
    header: list[str] | None = None
    combined: list[dict[str, str]] = []

    for expected_model, source in MODEL_SOURCES:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            source_header = list(reader.fieldnames or [])
            rows = list(reader)

        if header is None:
            header = source_header
        elif source_header != header:
            raise RuntimeError(f"column mismatch: {source}")
        if len(rows) != 17:
            raise RuntimeError(f"expected 17 rows, found {len(rows)}: {source}")
        if any(row.get("model") != expected_model for row in rows):
            raise RuntimeError(f"model value mismatch: {source}")
        personas = [row.get("persona") for row in rows]
        if len(set(personas)) != 17 or personas.count("OVERALL") != 1:
            raise RuntimeError(f"persona rows are incomplete or duplicated: {source}")
        combined.extend(rows)

    if header is None:
        raise RuntimeError("no source header was loaded")
    if len(combined) != 102:
        raise RuntimeError(f"expected 102 combined rows, found {len(combined)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=header,
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(combined)
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"models={len(MODEL_SOURCES)}")
    print(f"rows={len(combined)}")
    print(f"columns={len(header)}")
    print(f"output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

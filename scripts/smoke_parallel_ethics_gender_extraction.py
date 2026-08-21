"""Compare sequential and process-parallel extraction on a fixed small slice."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MAIN_PARSING.LINGUISTIC_MARKERS.gender_dataset_audit import (
    default_audit_config,
    extract_response_markers,
    load_refined_ethics,
)


def main() -> None:
    config = default_audit_config(REPO_ROOT)
    source = load_refined_ethics(config.input_csv).head(24)
    sequential = extract_response_markers(source, workers=1)
    parallel = extract_response_markers(source, workers=2)
    pd.testing.assert_frame_equal(sequential, parallel, check_dtype=True)
    print(f"parallel extraction smoke test passed: rows={len(parallel)} columns={len(parallel.columns)}")


if __name__ == "__main__":
    main()

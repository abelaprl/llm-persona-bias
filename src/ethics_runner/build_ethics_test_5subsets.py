"""
Build reproducible ETHICS test datasets from 4 Hendrycks ETHICS subsets.

Rules implemented:
- Use ONLY test split
- Sample exactly 200 rows per subset
- Sampling is random without replacement, fixed seed
- Save per-subset sampled JSON + one combined JSON (800 rows total)

Usage:
  python src/ethics_runner/build_ethics_test_5subsets.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "hendrycks/ethics"
SUBSETS = [
    "commonsense",
    "deontology",
    "justice",
    "virtue",
]
SPLIT = "test"
SAMPLE_PER_SUBSET = 200
SEED = 42

OUTPUT_DIR = Path("data")
OUT_COMBINED = OUTPUT_DIR / "ETHICS_test_4subsets_800.json"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def pick_excuse_text(row: dict[str, Any]) -> str | None:
    """Return excuse/rationale text when present in source schema."""
    for key in ["excuse", "rationale", "reason", "explanation"]:
        if key in row:
            text = normalize_text(row.get(key))
            if text:
                return text
    return None


def pick_scenario_text(row: dict[str, Any]) -> str:
    """Best-effort scenario extraction across ETHICS subset schemas."""
    for key in [
        "scenario",
        "input",
        "text",
        "statement",
        "action",
        "question",
    ]:
        if key in row:
            text = normalize_text(row.get(key))
            if text:
                return text

    # Fallback: compact JSON string from all primitive fields if no single text key.
    compact = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            compact[k] = v
    return normalize_text(json.dumps(compact, ensure_ascii=False))


def sample_subset(subset: str, rng_seed: int) -> list[dict[str, Any]]:
    # Test split is used to avoid train/validation leakage into evaluation analysis.
    ds = load_dataset(DATASET_NAME, subset, split=SPLIT)
    total = len(ds)

    if total < SAMPLE_PER_SUBSET:
        raise ValueError(
            f"Subset '{subset}' has only {total} rows in split '{SPLIT}', "
            f"cannot sample {SAMPLE_PER_SUBSET}."
        )

    rng = random.Random(rng_seed)
    selected_indices = sorted(rng.sample(range(total), SAMPLE_PER_SUBSET))

    sampled: list[dict[str, Any]] = []
    for i, source_idx in enumerate(selected_indices, start=1):
        row = dict(ds[int(source_idx)])
        scenario = pick_scenario_text(row)
        excuse = pick_excuse_text(row)
        sampled.append(
            {
                "id": f"ethics_{subset}_{i:04d}",
                "subset": subset,
                "sample_id": i,
                "source_index": int(source_idx),
                "scenario": scenario,
                "excuse": excuse,
                "label": row.get("label"),
                "split": SPLIT,
                "raw": row,
            }
        )

    return sampled


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    # Fixed per-subset seed derivation ensures deterministic and reproducible sampling.
    for subset in SUBSETS:
        subset_seed = SEED + SUBSETS.index(subset)
        sampled_rows = sample_subset(subset=subset, rng_seed=subset_seed)

        per_subset_path = OUTPUT_DIR / f"ETHICS_{subset}_test_200.json"
        save_json(per_subset_path, sampled_rows)
        print(f"Saved {len(sampled_rows)} rows -> {per_subset_path}")

        all_rows.extend(sampled_rows)

    # Combined dataset keeps all selected subsets in one unified file.
    save_json(OUT_COMBINED, all_rows)
    print(f"Saved {len(all_rows)} rows -> {OUT_COMBINED}")


if __name__ == "__main__":
    main()

"""
Build reproducible ETHICS test datasets from 4 Hendrycks ETHICS subsets.

Rules implemented:
- Use ONLY test split
- Start from 200-row base per subset
- Top up each subset to exactly 10% rows (floor, minimum 1)
- Sampling is random without replacement, fixed seed
- Skip source_index that already exists in base rows

Usage:
  python src/ethics_runner/build_ethics_test_5subsets.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET_NAME = "hendrycks/ethics"
SUBSETS = [
    "commonsense",
    "deontology",
    "justice",
    "virtue",
]
SPLIT = "test"
SAMPLE_FRACTION = 0.10
SEED = 42

OUTPUT_DIR = Path("data")
BASE_PER_SUBSET = 200
OUT_COMBINED = OUTPUT_DIR / "ETHICS_test_4subsets_10percent_from200.json"


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


def load_subset_test_rows(subset: str) -> list[dict[str, Any]]:
    file_path = hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=f"data/{subset}/{SPLIT}.csv",
    )
    df = pd.read_csv(file_path)
    # Keep plain JSON-serializable values.
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    return [dict(row) for row in records]


def sample_subset(
    subset: str,
    rng_seed: int,
    sample_size: int,
    excluded_source_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []

    # Test split is used to avoid train/validation leakage into evaluation analysis.
    rows = load_subset_test_rows(subset)
    total = len(rows)

    excluded = excluded_source_indices or set()
    candidate_indices = [idx for idx in range(total) if idx not in excluded]

    if not candidate_indices:
        print(f"Skipped subset '{subset}': no unseen source_index left.")
        return []

    sampled_count = min(sample_size, len(candidate_indices))
    if sampled_count < sample_size:
        print(
            f"Subset '{subset}': requested {sample_size} top-up rows, "
            f"but only {sampled_count} unseen rows are available."
        )

    rng = random.Random(rng_seed)
    selected_indices = sorted(rng.sample(candidate_indices, sampled_count))

    sampled: list[dict[str, Any]] = []
    for i, source_idx in enumerate(selected_indices, start=1):
        row = dict(rows[int(source_idx)])
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


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}.")
    return [row for row in data if isinstance(row, dict)]


def keep_unique_source_index_rows(
    rows: list[dict[str, Any]],
    subset: str,
) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen_source_index: set[int] = set()
    for row in rows:
        row_subset = row.get("subset")
        if row_subset is None:
            row["subset"] = subset
        elif row_subset != subset:
            continue

        source_index = row.get("source_index")
        if not isinstance(source_index, int):
            continue
        if source_index in seen_source_index:
            continue

        seen_source_index.add(source_index)
        unique_rows.append(row)
    return unique_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_final_rows: list[dict[str, Any]] = []
    all_new_rows: list[dict[str, Any]] = []

    # Fixed per-subset seed derivation ensures deterministic and reproducible sampling.
    for subset in SUBSETS:
        subset_rows = load_subset_test_rows(subset)
        target_sample_size = max(1, math.floor(len(subset_rows) * SAMPLE_FRACTION))

        base_path = OUTPUT_DIR / f"ETHICS_{subset}_test_{BASE_PER_SUBSET}.json"
        base_rows = keep_unique_source_index_rows(load_json_list(base_path), subset=subset)

        if len(base_rows) > target_sample_size:
            base_rows = base_rows[:target_sample_size]

        existing_ids = {
            str(row.get("id"))
            for row in base_rows
            if isinstance(row.get("id"), str) and row.get("id")
        }
        existing_source_indices = {
            int(row["source_index"])
            for row in base_rows
            if isinstance(row.get("source_index"), int)
        }

        needed_topup = max(0, target_sample_size - len(base_rows))
        subset_seed = SEED + SUBSETS.index(subset)
        sampled_rows = sample_subset(
            subset=subset,
            rng_seed=subset_seed,
            sample_size=needed_topup,
            excluded_source_indices=existing_source_indices,
        )

        next_sample_id = max(
            [
                int(row["sample_id"])
                for row in base_rows
                if isinstance(row.get("sample_id"), int)
            ]
            or [0]
        ) + 1
        for row in sampled_rows:
            row["sample_id"] = next_sample_id
            row["id"] = f"ethics_{subset}_{next_sample_id:04d}"
            while row["id"] in existing_ids:
                next_sample_id += 1
                row["sample_id"] = next_sample_id
                row["id"] = f"ethics_{subset}_{next_sample_id:04d}"
            existing_ids.add(row["id"])
            existing_source_indices.add(int(row["source_index"]))
            next_sample_id += 1

        final_subset_rows = base_rows + sampled_rows
        per_subset_path = OUTPUT_DIR / f"ETHICS_{subset}_test_10percent_from200.json"
        save_json(per_subset_path, final_subset_rows)
        print(
            f"Saved {len(final_subset_rows)} rows (base={len(base_rows)}, "
            f"added={len(sampled_rows)}, target={target_sample_size}) -> {per_subset_path}"
        )

        all_final_rows.extend(final_subset_rows)
        all_new_rows.extend(sampled_rows)

    save_json(OUT_COMBINED, all_final_rows)
    print(
        "Saved "
        f"{len(all_new_rows)} top-up rows (total={len(all_final_rows)})"
        f" -> {OUT_COMBINED}"
    )


if __name__ == "__main__":
    main()

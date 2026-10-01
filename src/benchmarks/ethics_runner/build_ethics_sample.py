"""
Build ETHICS commonsense balanced 20-sample set from validation split.

Rules:
- Remove scenarios with < 5 words
- Remove duplicate scenarios (case-insensitive)
- Split by label (0 -> wrong, 1 -> acceptable)
- Sample exactly 10 from each pool
- Save exactly 20 rows to data/ethics_sample_20.json

Usage:
  python src/benchmarks/ethics_runner/build_ethics_sample.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

INPUT_PATH = Path("data/ETHICS_commonsense_validation.json")
OUTPUT_PATH = Path("data/ethics_sample_20.json")
RNG_SEED = 42


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_records() -> list[dict]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for row in data:
        # Dataset uses "input" as text field; map it to scenario.
        scenario = normalize_spaces(str(row.get("scenario") or row.get("input") or ""))
        label = row.get("label")

        if scenario and label in (0, 1):
            records.append({"scenario": scenario, "label": int(label)})

    return records


def prefilter(records: list[dict]) -> list[dict]:
    seen = set()
    filtered: list[dict] = []

    for rec in records:
        scenario = rec["scenario"]

        if word_count(scenario) < 5:
            continue

        key = scenario.casefold()
        if key in seen:
            continue

        seen.add(key)
        filtered.append(rec)

    return filtered


def split_pools(records: list[dict]) -> tuple[list[dict], list[dict]]:
    wrong_pool = [r for r in records if r["label"] == 0]
    acceptable_pool = [r for r in records if r["label"] == 1]
    return wrong_pool, acceptable_pool


def sample_balanced(wrong_pool: list[dict], acceptable_pool: list[dict]) -> list[dict]:
    if len(wrong_pool) < 10:
        raise ValueError(f"Not enough wrong samples (label=0): {len(wrong_pool)} < 10")
    if len(acceptable_pool) < 10:
        raise ValueError(f"Not enough acceptable samples (label=1): {len(acceptable_pool)} < 10")

    random.shuffle(wrong_pool)
    random.shuffle(acceptable_pool)

    selected = wrong_pool[:10] + acceptable_pool[:10]
    random.shuffle(selected)
    return selected


def format_output(selected: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, rec in enumerate(selected, start=1):
        label = rec["label"]
        out.append(
            {
                "id": f"ethics_{i:04d}",
                "scenario": rec["scenario"],
                "label": label,
                "category": "wrong" if label == 0 else "acceptable",
            }
        )
    return out


def main() -> None:
    random.seed(RNG_SEED)

    records = load_records()
    filtered = prefilter(records)
    wrong_pool, acceptable_pool = split_pools(filtered)
    selected = sample_balanced(wrong_pool, acceptable_pool)
    output = format_output(selected)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    wrong_count = sum(1 for x in output if x["label"] == 0)
    acceptable_count = sum(1 for x in output if x["label"] == 1)

    print("ETHICS 20-sample file generated")
    print(f"Input records: {len(records)}")
    print(f"After prefilter: {len(filtered)}")
    print(f"Pool counts -> wrong(label=0): {len(wrong_pool)}, acceptable(label=1): {len(acceptable_pool)}")
    print(f"Output rows: {len(output)}")
    print(f"Output balance -> wrong: {wrong_count}, acceptable: {acceptable_count}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

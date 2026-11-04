"""
Build a 650-row ELI5 subset using streaming reservoir sampling.

Default outputs:
- E:/Tugas Akhir/TugasAkhir/data/eli5_subset_650.json
- E:/Tugas Akhir/TugasAkhir/result/ELI5/eli5_subset_650_meta.json

Usage:
  python src/eli5_runner/build_eli5_subset_650.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_OUTPUT_DATA = Path("E:/Tugas Akhir/TugasAkhir/data/eli5_subset_650.json")
DEFAULT_OUTPUT_META = Path("E:/Tugas Akhir/TugasAkhir/result/ELI5/eli5_subset_650_meta.json")
DEFAULT_DATASET_NAME = "sentence-transformers/eli5"
DEFAULT_DATASET_CONFIG = "pair"
DEFAULT_SPLIT = "train"
DEFAULT_K = 650
DEFAULT_SEED = 42
MIN_POSITIVE_LEN = 200
MAX_POSITIVE_LEN = 3000


@dataclass
class FilterStats:
    scanned_total: int = 0
    valid_total: int = 0
    invalid_missing_anchor: int = 0
    invalid_missing_positive: int = 0
    invalid_anchor_too_short: int = 0
    invalid_positive_len_out_of_range: int = 0
    invalid_low_alpha_ratio: int = 0


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    non_space = sum(not ch.isspace() for ch in text)
    if non_space == 0:
        return 0.0
    return letters / non_space


def load_streaming_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
):
    return load_dataset(dataset_name, dataset_config, split=split, streaming=True)


def is_valid_item(item: dict[str, Any], stats: FilterStats) -> tuple[bool, dict[str, str] | None]:
    anchor = normalize_text(item.get("anchor") or item.get("question"))
    positive = normalize_text(item.get("positive") or item.get("answer"))

    if not anchor:
        stats.invalid_missing_anchor += 1
        return False, None

    if not positive:
        stats.invalid_missing_positive += 1
        return False, None

    if len(anchor) < 12:
        stats.invalid_anchor_too_short += 1
        return False, None

    positive_len = len(positive)
    if positive_len < MIN_POSITIVE_LEN or positive_len > MAX_POSITIVE_LEN:
        stats.invalid_positive_len_out_of_range += 1
        return False, None

    # Keep a light quality gate so punctuation-only noise is filtered out.
    if alpha_ratio(anchor) < 0.55 or alpha_ratio(positive) < 0.55:
        stats.invalid_low_alpha_ratio += 1
        return False, None

    return True, {"anchor": anchor, "positive": positive}


def reservoir_sample(
    stream,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], FilterStats]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    stats = FilterStats()

    for stream_index, raw_item in enumerate(stream):
        stats.scanned_total += 1

        is_valid, cleaned = is_valid_item(raw_item, stats)
        if not is_valid or cleaned is None:
            continue

        stats.valid_total += 1
        candidate = {
            "id": f"eli5_{stats.valid_total:06d}",
            "anchor": cleaned["anchor"],
            "positive": cleaned["positive"],
            "source_stream_index": stream_index,
        }

        if len(reservoir) < k:
            reservoir.append(candidate)
            continue

        replacement_index = rng.randint(0, stats.valid_total - 1)
        if replacement_index < k:
            reservoir[replacement_index] = candidate

    # Stable deterministic order for downstream runs and diffs.
    reservoir.sort(key=lambda x: x["source_stream_index"])
    for idx, item in enumerate(reservoir, start=1):
        item["id"] = f"eli5_subset_{idx:04d}"

    return reservoir, stats


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_metadata(
    dataset_name: str,
    dataset_config: str,
    split: str,
    k: int,
    seed: int,
    sampled_count: int,
    stats: FilterStats,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "config": dataset_config,
            "split": split,
            "streaming": True,
        },
        "sampling": {
            "method": "reservoir_sampling",
            "target_k": k,
            "actual_k": sampled_count,
            "seed": seed,
        },
        "filters": {
            "positive_min_len": MIN_POSITIVE_LEN,
            "positive_max_len": MAX_POSITIVE_LEN,
            "anchor_min_len": 12,
            "alpha_ratio_min": 0.55,
        },
        "stats": {
            "scanned_total": stats.scanned_total,
            "valid_total": stats.valid_total,
            "invalid_missing_anchor": stats.invalid_missing_anchor,
            "invalid_missing_positive": stats.invalid_missing_positive,
            "invalid_anchor_too_short": stats.invalid_anchor_too_short,
            "invalid_positive_len_out_of_range": stats.invalid_positive_len_out_of_range,
            "invalid_low_alpha_ratio": stats.invalid_low_alpha_ratio,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 650 ELI5 subset via streaming reservoir sampling")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_data", type=Path, default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("--output_meta", type=Path, default=DEFAULT_OUTPUT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/4] Loading streaming dataset...")
    stream = load_streaming_dataset(args.dataset_name, args.dataset_config, args.split)

    print("[2/4] Running reservoir sampling...")
    sampled_items, stats = reservoir_sample(stream=stream, k=args.k, seed=args.seed)

    print("[3/4] Writing outputs...")
    save_json(args.output_data, sampled_items)

    metadata = build_metadata(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        k=args.k,
        seed=args.seed,
        sampled_count=len(sampled_items),
        stats=stats,
    )
    save_json(args.output_meta, metadata)

    print("[4/4] Done")
    print(f"  Sampled: {len(sampled_items)}")
    print(f"  Output data: {args.output_data}")
    print(f"  Output meta: {args.output_meta}")


if __name__ == "__main__":
    main()

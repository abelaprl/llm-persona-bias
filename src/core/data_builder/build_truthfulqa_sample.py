"""
Build a 20-question TruthfulQA sample focused on persona-bias categories.

Rules:
- Target categories: 10 categories x 2 questions each
- No duplicate questions
- Output: data/truthfulqa/truthfulqa_sample_20.json

Usage:
  python src/core/data_builder/build_truthfulqa_sample.py
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/truthfulqa/TruthfulQA.json")
OUTPUT_PATH = Path("data/truthfulqa/truthfulqa_sample_20.json")
RNG_SEED = 42

TARGET_CATEGORIES = [
    "Conspiracies",
    "Superstitions",
    "Religion",
    "Politics",
    "Law",
    "Psychology",
    "Health",
    "Economics",
    "Culture",
    "History",
]

# TruthfulQA.json does not contain a literal "Culture" category.
# We map it to Sociology as the closest cultural/social-bias category.
CATEGORY_ALIASES = {
    "Conspiracies": ["Conspiracies"],
    "Superstitions": ["Superstitions"],
    "Religion": ["Religion"],
    "Politics": ["Politics"],
    "Law": ["Law"],
    "Psychology": ["Psychology"],
    "Health": ["Health"],
    "Economics": ["Economics"],
    "Culture": ["Culture", "Sociology"],
    "History": ["History"],
}


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_truthfulqa() -> list[dict]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise RuntimeError("TruthfulQA dataset is empty or invalid")

    rows: list[dict] = []
    for item in data:
        question = safe_text(item.get("question"))
        answer = safe_text(item.get("answer"))
        category = safe_text(item.get("category"))
        q_type = safe_text(item.get("type"))
        source = safe_text(item.get("source"))

        if not question or not answer or not category:
            continue

        rows.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "type": q_type,
                "source": source,
            }
        )

    return rows


def build_category_pools(rows: list[dict]) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {cat: [] for cat in TARGET_CATEGORIES}

    for target_cat in TARGET_CATEGORIES:
        aliases = {x.casefold() for x in CATEGORY_ALIASES[target_cat]}
        filtered = [r for r in rows if (r.get("category") or "").casefold() in aliases]
        pools[target_cat] = filtered

    return pools


def sample_balanced_20(pools: dict[str, list[dict]]) -> list[dict]:
    rng = random.Random(RNG_SEED)
    selected: list[dict] = []
    used_questions: set[str] = set()

    for idx, target_cat in enumerate(TARGET_CATEGORIES):
        pool = list(pools[target_cat])
        rng.shuffle(pool)

        picks_for_cat: list[dict] = []
        for rec in pool:
            q_key = normalize_question(rec["question"])
            if q_key in used_questions:
                continue

            rec2 = dict(rec)
            rec2["target_category"] = target_cat
            picks_for_cat.append(rec2)
            used_questions.add(q_key)

            if len(picks_for_cat) == 2:
                break

        if len(picks_for_cat) < 2:
            raise RuntimeError(
                f"Not enough unique questions for category '{target_cat}'. "
                f"Need 2, got {len(picks_for_cat)}"
            )

        selected.extend(picks_for_cat)

    # Global shuffle so category order does not leak target sequence.
    rng.shuffle(selected)
    return selected


def build_output(selected: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, rec in enumerate(selected, start=1):
        out.append(
            {
                "id": f"truth_{i:04d}",
                "question": rec["question"],
                "ground_truth": rec["answer"],
                "category": rec["target_category"],
                "type": rec.get("type", ""),
                "source": rec.get("source", ""),
            }
        )
    return out


def main() -> None:
    rows = load_truthfulqa()
    pools = build_category_pools(rows)

    for cat in TARGET_CATEGORIES:
        if len(pools[cat]) < 2:
            raise RuntimeError(f"Category '{cat}' has fewer than 2 candidates ({len(pools[cat])})")

    selected = sample_balanced_20(pools)
    output = build_output(selected)

    if len(output) != 20:
        raise RuntimeError(f"Output size must be 20, got {len(output)}")

    unique_q = {normalize_question(x["question"]) for x in output}
    if len(unique_q) != 20:
        raise RuntimeError("Duplicate questions detected in output")

    cat_counts = Counter(x["category"] for x in output)
    for cat in TARGET_CATEGORIES:
        if cat_counts.get(cat, 0) != 2:
            raise RuntimeError(f"Category '{cat}' must have exactly 2 samples, got {cat_counts.get(cat, 0)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("TruthfulQA 20-sample dataset generated")
    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(output)}")
    print("Category distribution:")
    for cat in TARGET_CATEGORIES:
        print(f"  - {cat}: {cat_counts[cat]}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

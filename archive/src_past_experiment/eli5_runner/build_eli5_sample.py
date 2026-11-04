"""
Build ELI5 20-sample smart stratified set based on answer complexity.

Usage:
  python src/eli5_runner/build_eli5_sample.py

Output:
  data/eli5_sample_20.json
"""

from __future__ import annotations

import json
import random
import re
import string
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

OUTPUT_PATH = Path("data/eli5_sample_20.json")
RNG_SEED = 42
INPUT_PATH = Path("data/ELI5_pair_train_5k.json")
CATEGORY_RULES = {
    "short": (30, 80),
    "medium": (81, 150),
    "long": (151, 250),
    "very_long": (251, 10_000),
}

TARGET_COUNTS = {
    "short": 5,
    "medium": 5,
    "long": 5,
    "very_long": 5,
}

DOMAIN_KEYWORDS = {
    "science": {
        "science", "physics", "chemistry", "biology", "space", "planet",
        "atom", "evolution", "brain", "medicine", "health", "dna",
    },
    "social": {
        "society", "culture", "history", "politics", "religion", "law",
        "education", "family", "gender", "war", "country", "government",
    },
    "technology": {
        "computer", "internet", "software", "hardware", "ai", "algorithm",
        "programming", "code", "network", "phone", "data", "technology",
    },
    "economics": {
        "economy", "economic", "market", "money", "finance", "inflation",
        "trade", "tax", "bank", "price", "business", "salary",
    },
}

STOPWORDS_EN = {
    "the", "is", "are", "a", "an", "and", "or", "to", "of", "in", "for", "on",
    "with", "that", "this", "why", "how", "what", "when", "where", "who", "it",
    "be", "as", "at", "by", "from", "was", "were", "do", "does", "did", "can",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def contains_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.|_url_\d+_", text, flags=re.IGNORECASE))


def contains_code_block(text: str) -> bool:
    if "```" in text:
        return True
    # Common code-like patterns in plain text.
    return bool(re.search(r"\b(def|class|import|function|var|let|const)\b.*[:({]", text))


def looks_english(text: str) -> bool:
    """Heuristic English detector to avoid hard dependency on language-id models."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 6:
        return False

    stop_hits = sum(1 for w in words if w in STOPWORDS_EN)
    stop_ratio = stop_hits / max(len(words), 1)

    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    all_letters = sum(ch.isalpha() for ch in text)
    ascii_ratio = ascii_letters / max(all_letters, 1)

    # Reasonable thresholds for ELI5-style English text.
    return stop_ratio >= 0.06 and ascii_ratio >= 0.85


def category_from_answer_length(answer_length: int) -> str | None:
    for cat, (lo, hi) in CATEGORY_RULES.items():
        if lo <= answer_length <= hi:
            return cat
    return None


def tokenize_for_similarity(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS_EN}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def infer_domains(question: str, answer: str) -> set[str]:
    text = f"{question} {answer}".lower()
    matched = set()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in kws):
            matched.add(domain)
    return matched


def load_eli5_records() -> list[dict]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)

    records: list[dict] = []
    for row in ds:
        question = row.get("question") or ""
        answer = row.get("answer") or ""
        question = normalize_spaces(str(question))
        answer = normalize_spaces(str(answer))
        records.append({"question": question, "answer": answer})
    return records


def prefilter(records: Iterable[dict]) -> list[dict]:
    seen_questions = set()
    cleaned: list[dict] = []

    for rec in records:
        q = rec["question"]
        a = rec["answer"]

        q_words = count_words(q)
        a_words = count_words(a)

        if q_words < 5:
            continue
        if a_words < 30:
            continue
        if a_words > 400:
            continue

        q_key = q.lower()
        if q_key in seen_questions:
            continue

        if contains_url(q):
            continue
        if contains_code_block(a):
            continue
        if not looks_english(f"{q} {a}"):
            continue

        seen_questions.add(q_key)
        rec2 = dict(rec)
        rec2["answer_length"] = a_words
        rec2["question_tokens"] = tokenize_for_similarity(q)
        rec2["domains"] = infer_domains(q, a)
        cleaned.append(rec2)

    return cleaned


def build_pools(records: Iterable[dict]) -> dict[str, list[dict]]:
    pools = {"short": [], "medium": [], "long": [], "very_long": []}
    for rec in records:
        cat = category_from_answer_length(rec["answer_length"])
        if cat:
            rec2 = dict(rec)
            rec2["category"] = cat
            pools[cat].append(rec2)
    return pools


def build_pools_quantile(records: Iterable[dict]) -> tuple[dict[str, list[dict]], tuple[float, float, float]]:
    """Fallback complexity bins using quartiles of answer length."""
    rows = list(records)
    lengths = [r["answer_length"] for r in rows]
    q1, q2, q3 = statistics.quantiles(lengths, n=4, method="inclusive")

    pools = {"short": [], "medium": [], "long": [], "very_long": []}

    for rec in rows:
        l = rec["answer_length"]
        rec2 = dict(rec)
        if l <= q1:
            rec2["category"] = "short"
            pools["short"].append(rec2)
        elif l <= q2:
            rec2["category"] = "medium"
            pools["medium"].append(rec2)
        elif l <= q3:
            rec2["category"] = "long"
            pools["long"].append(rec2)
        else:
            rec2["category"] = "very_long"
            pools["very_long"].append(rec2)

    return pools, (q1, q2, q3)


def is_similar_to_selected(candidate: dict, selected: list[dict], threshold: float = 0.6) -> bool:
    c_tokens = candidate.get("question_tokens", set())
    for s in selected:
        sim = jaccard(c_tokens, s.get("question_tokens", set()))
        if sim >= threshold:
            return True
    return False


def pick_category_samples(pool: list[dict], n: int, already_selected: list[dict]) -> list[dict]:
    pool_copy = list(pool)
    random.shuffle(pool_copy)

    picks: list[dict] = []

    # Pass 1: prefer diversity against global selected.
    for rec in pool_copy:
        if len(picks) >= n:
            break
        if is_similar_to_selected(rec, already_selected + picks, threshold=0.6):
            continue
        picks.append(rec)

    # Pass 2: relax threshold slightly.
    if len(picks) < n:
        for rec in pool_copy:
            if len(picks) >= n:
                break
            if rec in picks:
                continue
            if is_similar_to_selected(rec, already_selected + picks, threshold=0.72):
                continue
            picks.append(rec)

    # Pass 3: fallback fill to ensure exact count.
    if len(picks) < n:
        for rec in pool_copy:
            if len(picks) >= n:
                break
            if rec in picks:
                continue
            picks.append(rec)

    return picks[:n]


def improve_cross_domain_mix(samples: list[dict]) -> list[dict]:
    """
    Light post-pass: preserve size/category but try to keep domain coverage broad.
    This function keeps existing sample order if coverage is already good.
    """
    covered = set()
    for s in samples:
        covered.update(s.get("domains", set()))

    # If we already cover >=3 target domains, keep as-is.
    if len(covered & set(DOMAIN_KEYWORDS.keys())) >= 3:
        return samples

    return samples


def validate_or_raise(samples: list[dict]) -> None:
    if len(samples) != 20:
        raise ValueError(f"Validation failed: total samples={len(samples)} (expected 20)")

    q_set = set()
    cat_counter = Counter()

    for s in samples:
        q = s.get("question", "").strip().lower()
        a = s.get("answer", "").strip()

        if not q or not a:
            raise ValueError("Validation failed: empty question/answer found")

        if q in q_set:
            raise ValueError("Validation failed: duplicate question found")
        q_set.add(q)

        cat = s.get("category")
        cat_counter[cat] += 1

    for cat, expected in TARGET_COUNTS.items():
        if cat_counter.get(cat, 0) != expected:
            raise ValueError(
                f"Validation failed: {cat}={cat_counter.get(cat, 0)} (expected {expected})"
            )


def format_output(samples: list[dict]) -> list[dict]:
    out = []
    for idx, s in enumerate(samples, start=1):
        out.append(
            {
                "id": f"eli5_{idx:04d}",
                "question": s["question"],
                "answer": s["answer"],
                "answer_length": s["answer_length"],
                "category": s["category"],
            }
        )
    return out


def main() -> None:
    random.seed(RNG_SEED)

    print("[1/6] Loading dataset...")
    records = load_eli5_records()
    print(f"  Source: {INPUT_PATH}")
    print(f"  Loaded records: {len(records)}")

    print("[2/6] Applying mandatory pre-filters...")
    filtered = prefilter(records)
    print(f"  After filters: {len(filtered)}")

    print("[3/6] Building category pools...")
    pools = build_pools(filtered)
    for cat in ["short", "medium", "long", "very_long"]:
        print(f"  {cat}: {len(pools[cat])}")

    # If strict fixed ranges cannot satisfy 5-per-category, fallback to quantile bins.
    shortages = [cat for cat, need in TARGET_COUNTS.items() if len(pools[cat]) < need]
    if shortages:
        print("  [WARN] Fixed thresholds cannot satisfy target counts for:", ", ".join(shortages))
        pools, (q1, q2, q3) = build_pools_quantile(filtered)
        print("  [INFO] Using quantile fallback thresholds:")
        print(f"    short <= {q1:.1f}")
        print(f"    medium <= {q2:.1f}")
        print(f"    long <= {q3:.1f}")
        print("    very_long > q3")
        for cat in ["short", "medium", "long", "very_long"]:
            print(f"  {cat}: {len(pools[cat])}")

    print("[4/6] Smart stratified sampling (5 each category)...")
    selected: list[dict] = []

    for cat in ["short", "medium", "long", "very_long"]:
        picks = pick_category_samples(pools[cat], TARGET_COUNTS[cat], selected)
        if len(picks) < TARGET_COUNTS[cat]:
            raise RuntimeError(f"Insufficient samples in {cat} after smart sampling")
        selected.extend(picks)

    selected = improve_cross_domain_mix(selected)

    # Shuffle final order but keep deterministic with seed.
    random.shuffle(selected)

    print("[5/6] Validating sampled set...")
    validate_or_raise(selected)

    output = format_output(selected)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("[6/6] Done.")
    print(f"  Saved: {OUTPUT_PATH}")

    cat_counter = Counter(x["category"] for x in output)
    print("  Category distribution:")
    for cat in ["short", "medium", "long", "very_long"]:
        print(f"    - {cat}: {cat_counter.get(cat, 0)}")


if __name__ == "__main__":
    main()

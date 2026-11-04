"""
Validate and fix 20 SQuAD samples for persona evaluation.

Usage:
  python src/squad_runner/fix_samples.py \
      --input data/SQuAD_20_samples.json \
      --output data/squad_20_cleaned.json \
      --overwrite-input
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_answer(answer: str) -> str:
    cleaned = normalize_spaces(answer or "")
    # Remove dangling unmatched opening parenthesis at the end.
    cleaned = re.sub(r"\(\s*$", "", cleaned).rstrip()
    # Remove dangling separator fragments at the end.
    cleaned = re.sub(r"[,;:\-]\s*$", "", cleaned).rstrip()
    return cleaned


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def is_list_type(answer: str) -> bool:
    text = normalize_spaces(answer)
    lowered = f" {text.lower()} "
    words = count_words(text)

    comma_segments = [segment.strip() for segment in text.split(",") if segment.strip()]
    has_multi_comma_segments = len(comma_segments) >= 2

    has_and = " and " in lowered and words >= 3

    has_enumeration = bool(re.search(r"\b[^,]+,\s*[^,]+\s+and\s+[^,]+\b", text, flags=re.IGNORECASE))
    has_x_and_y = bool(re.search(r"\b\w+(?:\s+\w+)*\s+and\s+\w+(?:\s+\w+)*\b", text, flags=re.IGNORECASE)) and words >= 3

    if has_multi_comma_segments:
        # Avoid false positive: "Entity, a/an/the ..." is appositive explanation, not list.
        if len(comma_segments) == 2:
            second = comma_segments[1].lower()
            second_clean = re.sub(r"[^a-z.]", "", second)
            org_suffixes = {
                "inc", "inc.", "ltd", "ltd.", "corp", "corp.", "co", "co.",
                "llc", "llp", "plc", "sa", "ag", "gmbh"
            }
            if second.startswith(("a ", "an ", "the ")):
                return False
            if second_clean in org_suffixes:
                return False
        return True

    if has_and or has_enumeration or has_x_and_y:
        return True

    return False


def question_type(question: str) -> str:
    q = normalize_spaces(question).lower()
    if q.startswith("what"):
        return "what"
    if q.startswith("who"):
        return "who"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"
    if q.startswith("which"):
        return "what"
    return "other"


def assign_category(length: int, list_flag: bool) -> str:
    if list_flag:
        return "list"
    if length <= 3:
        return "short"
    if length <= 10:
        return "medium"
    return "long"


def validate_category_distribution(category_counts: Counter) -> list[str]:
    issues = []
    for category in ["short", "medium", "long", "list"]:
        cnt = category_counts.get(category, 0)
        if cnt < 4 or cnt > 6:
            issues.append(f"{category}={cnt} outside tolerance [4,6]")
    return issues


def process_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    changes = []
    suspicious = []
    cleaned_samples = []

    for item in samples:
        original_category = item.get("category")
        original_list = item.get("is_list_type")

        answer = clean_answer(str(item.get("answer", "")))
        ans_len = count_words(answer)
        list_flag = is_list_type(answer)
        q_type = question_type(str(item.get("question", "")))
        category = assign_category(ans_len, list_flag)

        updated = dict(item)
        updated["answer"] = answer
        updated["answer_length"] = ans_len
        updated["is_list_type"] = list_flag
        updated["question_type"] = q_type
        updated["category"] = category
        cleaned_samples.append(updated)

        if original_category != category or bool(original_list) != list_flag:
            changes.append({
                "id": item.get("id"),
                "category": f"{original_category} -> {category}",
                "is_list_type": f"{original_list} -> {list_flag}",
                "question": item.get("question", ""),
                "answer": answer,
            })

        if ans_len < 2 or ans_len > 25:
            suspicious.append({
                "id": item.get("id"),
                "answer_length": ans_len,
                "answer": answer,
                "question": item.get("question", ""),
            })

    category_counts = Counter(s["category"] for s in cleaned_samples)
    qtype_counts = Counter(s["question_type"] for s in cleaned_samples)
    list_count = sum(1 for s in cleaned_samples if s["is_list_type"])

    summary = {
        "total": len(cleaned_samples),
        "category_distribution": dict(category_counts),
        "question_type_distribution": dict(qtype_counts),
        "list_count": list_count,
        "distribution_issues": validate_category_distribution(category_counts),
        "changed_samples": changes,
        "suspicious_answers": suspicious,
    }

    return cleaned_samples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and relabel 20 SQuAD samples")
    parser.add_argument("--input", default="data/SQuAD_20_samples.json", help="Input JSON file")
    parser.add_argument("--output", default="data/squad_20_cleaned.json", help="Output cleaned JSON file")
    parser.add_argument("--overwrite-input", action="store_true", help="Overwrite input file with cleaned content")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Input JSON must be a list of sample objects")

    cleaned_samples, summary = process_samples(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_samples, f, indent=2, ensure_ascii=False)

    if args.overwrite_input:
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_samples, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("SQUAD SAMPLE FIX SUMMARY")
    print("=" * 72)
    print(f"Total samples: {summary['total']}")
    print(f"Category distribution: {summary['category_distribution']}")
    print(f"Question type distribution: {summary['question_type_distribution']}")
    print(f"List count: {summary['list_count']}")

    if summary["distribution_issues"]:
        print("[WARN] Category distribution outside tolerance:")
        for issue in summary["distribution_issues"]:
            print(f"  - {issue}")
    else:
        print("[OK] Category distribution within tolerance (±1 around 5)")

    print(f"\nChanged samples: {len(summary['changed_samples'])}")
    for c in summary["changed_samples"]:
        print(f"  - {c['id']}: category {c['category']}, list {c['is_list_type']}")

    print(f"\nSuspicious answers (<2 or >25 words): {len(summary['suspicious_answers'])}")
    for s in summary["suspicious_answers"]:
        print(f"  - {s['id']}: len={s['answer_length']} | answer='{s['answer']}'")

    print(f"\nSaved cleaned file: {output_path}")
    if args.overwrite_input:
        print(f"Overwritten input file: {input_path}")


if __name__ == "__main__":
    main()

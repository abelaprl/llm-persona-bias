"""
Final polishing and validation for SQuAD 20 samples.

Usage:
  python src/squad_runner/final_cleaning.py \
      --input data/squad_20_cleaned.json \
      --output data/squad_20_final.json
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def fix_known_typos(answer: str) -> str:
    if answer == "tempted to enter the test site":
        return "attempted to enter the test site"
    return answer


def fix_truncated_answer(answer: str) -> str:
    text = normalize_spaces(answer)

    # If there is unmatched opening parenthesis, append closing parenthesis.
    if text.count("(") > text.count(")"):
        text = text + ")"

    # Strip trailing incomplete punctuation marks repeatedly.
    text = re.sub(r"[\(,;]+\s*$", "", text).rstrip()

    return text


def assign_question_type(question: str) -> str:
    q = normalize_spaces(question).lower()

    if "who" in q:
        return "who"
    if q.startswith("what") or q.startswith("which"):
        return "what"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"
    return "other"


def assign_category(answer_length: int, is_list_type: bool) -> str:
    if is_list_type:
        return "list"
    if answer_length <= 3:
        return "short"
    if answer_length <= 10:
        return "medium"
    return "long"


def normalize_for_em_f1(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_distribution(category_counts: Counter) -> list[str]:
    issues = []
    for cat in ["short", "medium", "long", "list"]:
        val = category_counts.get(cat, 0)
        if val < 4 or val > 6:
            issues.append(f"{cat}={val} outside tolerance [4,6]")
    return issues


def process(samples: list[dict]) -> tuple[list[dict], dict]:
    final_samples = []
    quality_flags = {
        "answer_length_lt_2": [],
        "answer_length_gt_25": [],
        "context_gt_400": [],
        "answer_not_in_context": [],
    }
    changed = []

    for item in samples:
        updated = dict(item)

        original = {
            "answer": item.get("answer", ""),
            "answer_length": item.get("answer_length"),
            "question_type": item.get("question_type"),
            "category": item.get("category"),
        }

        answer = str(item.get("answer", ""))
        answer = fix_known_typos(answer)
        answer = fix_truncated_answer(answer)
        answer = normalize_spaces(answer)

        answer_length = count_words(answer)
        q_type = assign_question_type(str(item.get("question", "")))
        is_list_type = bool(item.get("is_list_type", False))
        category = assign_category(answer_length, is_list_type)
        normalized_answer = normalize_for_em_f1(answer)

        updated["answer"] = answer
        updated["answer_length"] = answer_length
        updated["question_type"] = q_type
        updated["category"] = category
        updated["normalized_answer"] = normalized_answer

        # Quality flags
        context = str(item.get("context", ""))
        if answer_length < 2:
            quality_flags["answer_length_lt_2"].append(item.get("id"))
        if answer_length > 25:
            quality_flags["answer_length_gt_25"].append(item.get("id"))
        if count_words(context) > 400:
            quality_flags["context_gt_400"].append(item.get("id"))
        if answer.lower() not in context.lower():
            quality_flags["answer_not_in_context"].append(item.get("id"))

        if (
            original["answer"] != updated["answer"]
            or original["answer_length"] != updated["answer_length"]
            or original["question_type"] != updated["question_type"]
            or original["category"] != updated["category"]
        ):
            changed.append({
                "id": item.get("id"),
                "answer": f"{original['answer']} -> {updated['answer']}",
                "answer_length": f"{original['answer_length']} -> {updated['answer_length']}",
                "question_type": f"{original['question_type']} -> {updated['question_type']}",
                "category": f"{original['category']} -> {updated['category']}",
            })

        final_samples.append(updated)

    category_counts = Counter(x.get("category", "") for x in final_samples)
    qtype_counts = Counter(x.get("question_type", "") for x in final_samples)

    summary = {
        "total": len(final_samples),
        "category_distribution": dict(category_counts),
        "question_type_distribution": dict(qtype_counts),
        "distribution_issues": validate_distribution(category_counts),
        "quality_flags": quality_flags,
        "changed": changed,
    }
    return final_samples, summary


def print_summary(summary: dict) -> None:
    print("=" * 72)
    print("FINAL SQUAD CLEANING SUMMARY")
    print("=" * 72)
    print(f"Total samples: {summary['total']}")

    cat = summary["category_distribution"]
    print("Category Distribution:")
    print(f"short: {cat.get('short', 0)}")
    print(f"medium: {cat.get('medium', 0)}")
    print(f"long: {cat.get('long', 0)}")
    print(f"list: {cat.get('list', 0)}")

    if summary["distribution_issues"]:
        print("[WARNING] Category distribution out of tolerance:")
        for issue in summary["distribution_issues"]:
            print(f"  - {issue}")
    else:
        print("[OK] Category distribution within tolerance (5 ± 1)")

    print("\nQuestion Type Distribution:")
    for k, v in sorted(summary["question_type_distribution"].items()):
        print(f"{k}: {v}")

    qf = summary["quality_flags"]
    print("\nQuality Flags:")
    print(f"answer_length < 2: {len(qf['answer_length_lt_2'])}")
    print(f"answer_length > 25: {len(qf['answer_length_gt_25'])}")
    print(f"context > 400 words: {len(qf['context_gt_400'])}")
    print(f"answer not found in context: {len(qf['answer_not_in_context'])}")

    if qf["answer_length_lt_2"]:
        print(f"  IDs answer_length < 2: {qf['answer_length_lt_2']}")
    if qf["answer_length_gt_25"]:
        print(f"  IDs answer_length > 25: {qf['answer_length_gt_25']}")
    if qf["context_gt_400"]:
        print(f"  IDs context > 400 words: {qf['context_gt_400']}")
    if qf["answer_not_in_context"]:
        print(f"  IDs answer not in context: {qf['answer_not_in_context']}")

    print(f"\nChanged samples: {len(summary['changed'])}")
    for ch in summary["changed"]:
        print(
            f"  - {ch['id']}: answer({ch['answer']}), answer_length({ch['answer_length']}), "
            f"question_type({ch['question_type']}), category({ch['category']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final polishing for SQuAD 20 samples")
    parser.add_argument("--input", default="data/squad_20_cleaned.json", help="Input JSON path")
    parser.add_argument("--output", default="data/squad_20_final.json", help="Output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Input JSON must be a list")
    if len(samples) != 20:
        print(f"[WARNING] Expected 20 samples but got {len(samples)}")

    final_samples, summary = process(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_samples, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"\nSaved final dataset: {output_path}")


if __name__ == "__main__":
    main()

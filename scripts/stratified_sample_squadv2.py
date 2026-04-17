import argparse
import json
import os
import re
from typing import Any

import pandas as pd


DEFAULT_INPUT = r"D:\Abel\TugasAkhir\data\squad_v2_validation.json"
DEFAULT_OUTPUT = r"D:\Abel\TugasAkhir\data\squad_v2_sampled_10percent.csv"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def first_answer_text(answers: Any) -> str:
    if answers is None:
        return ""

    if isinstance(answers, dict):
        text_values = answers.get("text")
        if isinstance(text_values, list):
            for text in text_values:
                if isinstance(text, str) and text.strip():
                    return text.strip()
        if isinstance(text_values, str):
            return text_values.strip()

    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            elif isinstance(item, str) and item.strip():
                return item.strip()

    if isinstance(answers, str):
        return answers.strip()

    return ""


def load_squad_json(input_path: str) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return pd.DataFrame(raw)

    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        rows = []
        for article in raw["data"]:
            title = article.get("title") if isinstance(article, dict) else None
            paragraphs = article.get("paragraphs", []) if isinstance(article, dict) else []
            for paragraph in paragraphs:
                context = paragraph.get("context") if isinstance(paragraph, dict) else None
                qas = paragraph.get("qas", []) if isinstance(paragraph, dict) else []
                for qa in qas:
                    if not isinstance(qa, dict):
                        continue
                    rows.append(
                        {
                            "id": qa.get("id"),
                            "title": title,
                            "context": context,
                            "question": qa.get("question"),
                            "is_impossible": qa.get("is_impossible", False),
                            "answers": qa.get("answers"),
                            "plausible_answers": qa.get("plausible_answers"),
                        }
                    )
        return pd.DataFrame(rows)

    raise ValueError("Unsupported JSON format. Expected list of rows or SQuAD-style object with 'data'.")


def detect_input_path(preferred_path: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path

    fallback_candidates = [
        r"E:\Tugas Akhir\TugasAkhir\data\SQUADv2_validation.json",
        r"E:\Tugas Akhir\TugasAkhir\data\SQuAD_validation.json",
    ]
    for candidate in fallback_candidates:
        if os.path.exists(candidate):
            return candidate

    return preferred_path


def detect_output_path(preferred_output: str, resolved_input: str) -> str:
    if os.path.exists(os.path.dirname(preferred_output)):
        return preferred_output

    input_dir = os.path.dirname(resolved_input)
    if input_dir:
        return os.path.join(input_dir, "squad_v2_sampled_10percent.csv")

    return preferred_output


def get_question_type(question: Any) -> str:
    if not isinstance(question, str):
        return "other"

    match = re.search(r"[A-Za-z]+", question.strip().lower())
    first_word = match.group(0) if match else "other"
    allowed = {"what", "who", "when", "where", "why", "how"}
    return first_word if first_word in allowed else "other"


def get_answer_length(row: pd.Series) -> str:
    if normalize_bool(row.get("is_impossible", False)):
        return "none"

    answer_text = first_answer_text(row.get("answers"))
    if not answer_text:
        answer_text = first_answer_text(row.get("plausible_answers"))

    word_count = len([w for w in str(answer_text).strip().split() if w])

    if word_count <= 3:
        return "short"
    if word_count <= 10:
        return "medium"
    return "long"


def distribution_table(original_df: pd.DataFrame, sampled_df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, float]:
    orig = original_df[column].value_counts(normalize=True).mul(100)
    samp = sampled_df[column].value_counts(normalize=True).mul(100)

    idx = sorted(set(orig.index).union(set(samp.index)))
    comp = pd.DataFrame(
        {
            "original_pct": orig.reindex(idx, fill_value=0.0),
            "sampled_pct": samp.reindex(idx, fill_value=0.0),
        }
    )
    comp["abs_pct_diff"] = (comp["sampled_pct"] - comp["original_pct"]).abs()
    mapd = float(comp["abs_pct_diff"].mean())
    return comp, mapd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified 10% sample from SQuAD v2 style dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output CSV file")
    parser.add_argument("--frac", type=float, default=0.10, help="Sampling fraction per stratum")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = detect_input_path(args.input)
    output_path = detect_output_path(args.output, input_path)

    df = load_squad_json(input_path)

    # Step 2: shuffle rows with fixed seed.
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Step 3: create stratification features.
    df["is_impossible"] = df["is_impossible"].apply(normalize_bool)
    df["question_type"] = df["question"].apply(get_question_type)
    df["answer_length"] = df.apply(get_answer_length, axis=1)

    # Step 4: combine into strata key.
    df["strata"] = (
        df["is_impossible"].astype(str)
        + "|"
        + df["question_type"].astype(str)
        + "|"
        + df["answer_length"].astype(str)
    )

    # Step 5: stratified sampling with safe minimum 1 per group.
    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n = max(1, int(round(len(group) * args.frac)))
        n = min(n, len(group))
        return group.sample(n=n, random_state=args.seed)

    sampled_df = (
        df.groupby("strata", group_keys=False)
        .apply(sample_group)
        .reset_index(drop=True)
    )

    # Step 6: combine and save.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled_df.to_csv(output_path, index=False)

    # Validation: print distribution comparisons and mean absolute pct difference.
    print(f"Input path:  {input_path}")
    print(f"Output path: {output_path}")
    print(f"Original rows: {len(df)}")
    print(f"Sampled rows:  {len(sampled_df)}")

    for feature in ["is_impossible", "question_type", "answer_length"]:
        comp, mapd = distribution_table(df, sampled_df, feature)
        print("\n" + "=" * 72)
        print(f"Distribution comparison: {feature}")
        print(comp.to_string(float_format=lambda x: f"{x:0.2f}"))
        print(f"Mean absolute percentage difference ({feature}): {mapd:0.2f}")


if __name__ == "__main__":
    main()

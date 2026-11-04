from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


QUESTION_TYPES = {"what", "who", "when", "where", "why", "how"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a representative 10%% SQuAD v2 sample with stratified sampling."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\Abel\TugasAkhir\data\squad_v2_validation.json",
        help="Path to SQuAD v2 validation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\Abel\TugasAkhir\data\squad_v2_sampled_10percent.csv",
        help="Path to output sampled CSV file.",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.10,
        help="Sampling fraction per stratum (default: 0.10).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed used for shuffling and sampling.",
    )
    return parser.parse_args()


def load_squad_to_dataframe(input_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict[str, Any]] = []
    for article in payload.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers") or []
                answer_text = ""
                if answers:
                    answer_text = str(answers[0].get("text") or "").strip()

                rows.append(
                    {
                        "title": title,
                        "context": context,
                        "id": qa.get("id", ""),
                        "question": str(qa.get("question") or "").strip(),
                        "is_impossible": bool(qa.get("is_impossible", False)),
                        "answer_text": answer_text,
                    }
                )

    return pd.DataFrame(rows)


def normalize_question_type(question: str) -> str:
    words = str(question).strip().split()
    if not words:
        return "other"
    first_word = words[0].lower().strip("'\"()[]{}.,!?;:")
    return first_word if first_word in QUESTION_TYPES else "other"


def categorize_answer_length(is_impossible: bool, answer_text: str) -> str:
    if is_impossible:
        return "none"

    word_count = len([w for w in str(answer_text).split() if w])
    if 1 <= word_count <= 3:
        return "short"
    if 4 <= word_count <= 10:
        return "medium"
    return "long"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["question_type"] = out["question"].apply(normalize_question_type)
    out["answer_length"] = out.apply(
        lambda r: categorize_answer_length(bool(r["is_impossible"]), str(r["answer_text"])),
        axis=1,
    )
    out["strata"] = (
        out["is_impossible"].astype(str)
        + "|"
        + out["question_type"].astype(str)
        + "|"
        + out["answer_length"].astype(str)
    )
    return out


def stratified_sample(df: pd.DataFrame, sample_frac: float, random_state: int) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("strata", dropna=False):
        n = max(1, int(round(len(group) * sample_frac)))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df


def distribution_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).sort_index()
    props = (counts / len(df)).rename("proportion")
    return pd.DataFrame({"count": counts, "proportion": props})


def compare_distributions(original_df: pd.DataFrame, sampled_df: pd.DataFrame, column: str) -> float:
    orig = distribution_table(original_df, column)
    samp = distribution_table(sampled_df, column)

    all_idx = orig.index.union(samp.index)
    orig_prop = orig.reindex(all_idx).fillna(0.0)["proportion"]
    samp_prop = samp.reindex(all_idx).fillna(0.0)["proportion"]
    diff = (orig_prop - samp_prop).abs()

    print(f"\n[{column}] distribution comparison")
    report = pd.DataFrame(
        {
            "original_count": orig.reindex(all_idx).fillna(0.0)["count"].astype(int),
            "sampled_count": samp.reindex(all_idx).fillna(0.0)["count"].astype(int),
            "original_prop": orig_prop,
            "sampled_prop": samp_prop,
            "abs_diff": diff,
        }
    )
    print(report.to_string())

    return float(diff.mean() * 100.0)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1) Load dataset using pandas-compatible workflow.
    df = load_squad_to_dataframe(input_path)
    if df.empty:
        raise RuntimeError("Loaded dataset is empty.")

    # 2) Shuffle dataset.
    df = df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    # 3-4) Create features and strata.
    df = add_features(df)

    # 5-6) Stratified 10% sampling with safe minimum per group.
    sampled_df = stratified_sample(df, sample_frac=args.sample_frac, random_state=args.random_state)

    # Validation.
    print(f"Original rows: {len(df)}")
    print(f"Sampled rows:  {len(sampled_df)}")
    print(f"Effective fraction: {len(sampled_df) / len(df):.4f}")

    m1 = compare_distributions(df, sampled_df, "is_impossible")
    m2 = compare_distributions(df, sampled_df, "question_type")
    m3 = compare_distributions(df, sampled_df, "answer_length")

    print("\nMean absolute percentage difference (percentage points)")
    print(f"- is_impossible: {m1:.4f}")
    print(f"- question_type: {m2:.4f}")
    print(f"- answer_length: {m3:.4f}")
    print(f"- overall_mean: {(m1 + m2 + m3) / 3.0:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved sampled CSV: {output_path}")


if __name__ == "__main__":
    main()

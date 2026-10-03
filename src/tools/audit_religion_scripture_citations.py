"""Measure named scripture citations missed by the current authority marker."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "result/benchmarks/gsm8k/reports/Religion_Persona/linguistic_markers"
    / "religion_gsm8k_full_source_aligned_v1_3_20260818"
)

BIBLE_BOOKS = (
    "Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|Samuel|"
    "Kings|Chronicles|Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs?|Ecclesiastes|"
    "Song of Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|"
    "Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|"
    "Malachi|Matthew|Mark|Luke|John|Acts|Romans|Corinthians|Galatians|"
    "Ephesians|Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|"
    "Hebrews|James|Peter|Jude|Revelation"
)
CITATION_PATTERN = re.compile(
    rf"\b(?:[1-3]\s+)?(?:{BIBLE_BOOKS})\s+\d{{1,3}}:\d{{1,3}}"
    rf"(?:[-–]\d{{1,3}})?\b",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sample-model")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    columns = [
        "model_id",
        "persona",
        "question_id",
        "visible_text",
        "religion_scripture_authority__present",
    ]
    found: list[dict[str, object]] = []
    for chunk in pd.read_csv(
        args.run_dir / "response_records.csv",
        usecols=columns,
        low_memory=False,
        chunksize=10_000,
    ):
        for row in chunk.itertuples(index=False):
            text = "" if pd.isna(row.visible_text) else str(row.visible_text)
            matches = [match.group(0) for match in CITATION_PATTERN.finditer(text)]
            if matches:
                found.append(
                    {
                        "model_id": row.model_id,
                        "persona": row.persona,
                        "question_id": row.question_id,
                        "matches": "; ".join(dict.fromkeys(matches)),
                        "already_registered": float(
                            row.religion_scripture_authority__present
                        )
                        > 0,
                        "visible_text": text,
                    }
                )
    frame = pd.DataFrame(found)
    print(f"Citation-positive responses: {len(frame)}")
    print(f"Already captured by v1.2 scripture marker: {int(frame['already_registered'].sum())}")
    missed = frame[~frame["already_registered"]]
    print(f"Missed by v1.2 scripture marker: {len(missed)}")
    print(
        frame.groupby(["model_id", "persona", "already_registered"], sort=True)
        .size()
        .rename("n_responses")
        .reset_index()
        .to_string(index=False)
    )
    display_rows = missed
    if args.sample_model:
        display_rows = missed[missed["model_id"].eq(args.sample_model)]
        display_rows = display_rows.sample(
            n=min(args.sample_size, len(display_rows)), random_state=args.seed
        ).sort_values(["persona", "question_id"])
        print(
            f"\nFull-text sample for {args.sample_model}: "
            f"{len(display_rows)}/{len(missed[missed['model_id'].eq(args.sample_model)])}"
        )
    for index, row in enumerate(display_rows.itertuples(index=False), start=1):
        print("\n" + "=" * 100)
        print(
            f"[{index}] {row.model_id} | {row.persona} | {row.question_id} | "
            f"{row.matches}"
        )
        print("-" * 100)
        print(row.visible_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

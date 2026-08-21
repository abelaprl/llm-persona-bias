"""Scan and sample corpus-grounded religious salutation/formula candidates.

This utility is intentionally independent of the marker registry.  It is used
for recall auditing: candidate expressions are located in the visible response
surface, counted by model/persona, and printed in full for manual review.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE_CSV = (
    REPO_ROOT
    / "result/GMS8K/reports/Religion_Persona/linguistic_markers"
    / "religion_gsm8k_full_source_aligned_v1_3_20260818/response_records.csv"
)

CANDIDATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "assalamu_alaikum": re.compile(
        r"\b(?:as[- ]?sala?mu|assala?mu|sala?mu)\s+ala[yi]kum\b", re.IGNORECASE
    ),
    "wa_alaikum_salam": re.compile(
        r"\bwa\s+ala[yi]kum\s+(?:as[- ]?)?sala?m\b", re.IGNORECASE
    ),
    "shalom": re.compile(r"\bshalom\b", re.IGNORECASE),
    "lshana": re.compile(
        r"\bl['’]?shana\s+haba['’]?ah(?:\s+b['’]?yerushalayim)?\b",
        re.IGNORECASE,
    ),
    "inshallah": re.compile(r"\b(?:in\s*sha|insha)['’ -]?allah\b", re.IGNORECASE),
    "jazakallah": re.compile(
        r"\b(?:jaza?k(?:um|allahu)?\s+allah(?:u)?(?:\s+khayran)?|"
        r"jaza?kallahu?\s+khayran)\b",
        re.IGNORECASE,
    ),
    "amen": re.compile(r"\bame+n\b", re.IGNORECASE),
    "bismillah": re.compile(r"\bbismillah\b", re.IGNORECASE),
    "alhamdulillah": re.compile(r"\balhamdulillah\b", re.IGNORECASE),
    "mashallah": re.compile(r"\bmasha['’ ]?allah\b", re.IGNORECASE),
    "subhanallah": re.compile(r"\bsubhanallah\b", re.IGNORECASE),
    "baruch_hashem": re.compile(r"\bbaruch\s+hashem\b", re.IGNORECASE),
    "mazel_tov": re.compile(r"\bmazel\s+tov\b", re.IGNORECASE),
    "chag_sameach": re.compile(r"\bchag\s+sameach\b", re.IGNORECASE),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-csv", type=Path, default=DEFAULT_RESPONSE_CSV)
    parser.add_argument("--sample-model")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser


def _candidate_names(text: object) -> list[str]:
    value = "" if pd.isna(text) else str(text)
    return [name for name, pattern in CANDIDATE_PATTERNS.items() if pattern.search(value)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    usecols = ["model_id", "persona", "question_id", "visible_text", "text_available"]
    if args.sample_model:
        selected_chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            args.response_csv, usecols=usecols, low_memory=False, chunksize=10_000
        ):
            selected = chunk[chunk["model_id"].eq(args.sample_model)]
            if not selected.empty:
                selected_chunks.append(selected)
        frame = pd.concat(selected_chunks, ignore_index=True)
    else:
        frame = pd.read_csv(args.response_csv, usecols=usecols, low_memory=False)
    frame = frame[frame["text_available"].fillna(False).astype(bool)].copy()
    frame["candidate_names"] = frame["visible_text"].map(_candidate_names)
    hits = frame[frame["candidate_names"].map(bool)].explode("candidate_names")

    summary = (
        hits.groupby(["candidate_names", "model_id", "persona"], sort=True)
        .agg(n_responses=("question_id", "nunique"))
        .reset_index()
    )
    print(summary.to_string(index=False))
    unique_response_count = hits[["model_id", "persona", "question_id"]].drop_duplicates().shape[0]
    print(f"\nUnique candidate responses: {unique_response_count:,}")
    print(f"Candidate response-pattern rows: {len(hits):,}")

    if args.sample_model:
        candidates = frame[
            frame["model_id"].eq(args.sample_model) & frame["candidate_names"].map(bool)
        ]
        n = min(args.sample_size, len(candidates))
        sample = candidates.sample(n=n, random_state=args.seed).sort_values(
            ["persona", "question_id"]
        )
        print(f"\nFULL-TEXT SAMPLE: {args.sample_model} ({n}/{len(candidates)})")
        for index, row in enumerate(sample.itertuples(index=False), start=1):
            print("\n" + "=" * 96)
            print(
                f"[{index}] {row.model_id} | {row.persona} | {row.question_id} | "
                f"{','.join(row.candidate_names)}"
            )
            print("-" * 96)
            print(row.visible_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

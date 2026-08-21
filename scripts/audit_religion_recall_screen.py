"""Broad, registry-independent screen for manual religion-marker recall audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MAIN_PARSING.LINGUISTIC_MARKERS.extractor import extract_markers  # noqa: E402
from src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_registry import (  # noqa: E402
    RELIGION_MARKERS,
)


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "result/GMS8K/reports/Religion_Persona/linguistic_markers"
    / "religion_gsm8k_full_source_aligned_v1_3_20260818"
)

# This intentionally casts a wider net than the registered lexicon.  Terms
# with ambiguous secular uses are retained so the reviewer, not the screen,
# decides whether a response is genuinely religion-positive.
BROAD_RELIGION_SCREEN = re.compile(
    r"\b(?:"
    r"muslims?|islam(?:ic)?|christians?|christianity|jews?|jewish|judaism|judaic|"
    r"atheists?|atheism|atheistic|agnostics?|agnosticism|secularism|non-?religious|"
    r"religions?|religious(?:ly)?|faith|faithful|spiritual(?:ity)?|beliefs?|"
    r"allah|god|gods|lord|deity|divine|jesus|christ|heaven|hell|soul|grace|"
    r"qur['’]?an|koran|hadith|bible|biblical|scriptures?|torah|talmud|gospels?|"
    r"pray(?:er|ers|ing)?|worship(?:ping|ing)?|ramadan|eid|jumu['’]?ah|"
    r"salah|salat|shalat|shabbat|sabbath|yom\s+kippur|kaddish|shema|lent|lenten|"
    r"communion|halal|haram|kosher|zakat|sharia|mitzvah|"
    r"mosque|masjid|synagogue|church|congregation|temple|rabbi|imam|priest|pastor|"
    r"pilgrimage|hajj|rituals?|hanukkah|passover|christmas|easter|bless(?:es|ed|ing)?|"
    r"assala?mu|sala?mu|shalom|insha['’ -]?allah|jaza?kallah|ame+n|"
    r"chag\s+sameach|mazel\s+tov|l['’]?shana"
    r")\b",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--full-text", action="store_true")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    recall = pd.read_csv(args.run_dir / "audit_recall_responses.csv", low_memory=False)
    rows: list[dict[str, object]] = []
    for row in recall.itertuples(index=False):
        text = "" if pd.isna(row.visible_text) else str(row.visible_text)
        broad_matches = list(dict.fromkeys(match.group(0) for match in BROAD_RELIGION_SCREEN.finditer(text)))
        extracted = extract_markers(
            {"benchmark": "gsm8k", "response_content": text}, RELIGION_MARKERS
        )
        marker_ids = [
            marker.marker_id
            for marker in RELIGION_MARKERS
            if extracted[f"{marker.marker_id}__present"] == 1
        ]
        if broad_matches or marker_ids:
            rows.append(
                {
                    "model_id": row.model_id,
                    "persona": row.persona,
                    "question_id": row.question_id,
                    "broad_matches": "; ".join(broad_matches),
                    "registered_marker_ids": ";".join(marker_ids),
                    "visible_text": text,
                }
            )

    print(f"Blind recall rows: {len(recall)}")
    print(f"Rows selected by broad screen or registry: {len(rows)}")
    for index, row in enumerate(rows, start=1):
        print("\n" + "=" * 100)
        print(
            f"[{index}] {row['model_id']} | {row['persona']} | {row['question_id']}\n"
            f"BROAD={row['broad_matches']}\nREGISTERED={row['registered_marker_ids']}"
        )
        if args.full_text:
            print("-" * 100)
            print(row["visible_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

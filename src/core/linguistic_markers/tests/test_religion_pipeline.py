from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.linguistic_markers.extractor import extract_markers
from src.core.linguistic_markers.religion_config import RELIGION_PERSONAS
from src.core.linguistic_markers.religion_csv_loaders import (
    load_religion_gsm8k_records,
    religion_persona_metadata,
)
from src.core.linguistic_markers.religion_registry import (
    ALL_RELIGION_MARKERS,
)
from src.core.linguistic_markers.religion_statistics import (
    build_religion_direct_availability,
    build_religion_direct_numeric,
    build_religion_direct_opening_multinomial,
    build_religion_paired_numeric,
)


class ReligionLoaderTests(unittest.TestCase):
    def test_persona_metadata(self):
        self.assertEqual(religion_persona_metadata("neutral"), ("neutral", "neutral"))
        self.assertEqual(
            religion_persona_metadata("jewish_implicit_2"),
            ("jewish", "implicit_2"),
        )

    def test_loader_combines_religion_with_shared_neutral_and_keeps_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            religion_path = root / "religion.csv"
            neutral_path = root / "neutral.csv"
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona_folder": "islam_explicit",
                        "file": "islam_explicit/gsm8k_00001.json",
                        "stable_question_id": "gsm8k_00001",
                        "response_content_full": "May Allah bless you.",
                        "response_success": True,
                        "is_correct": True,
                    }
                ]
            ).to_csv(religion_path, index=False)
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona_folder": "neutral",
                        "file": "neutral/gsm8k_00001.json",
                        "stable_question_id": "gsm8k_00001",
                        "response_content_full": "One plus one is two.",
                        "is_correct": True,
                    }
                ]
            ).to_csv(neutral_path, index=False)
            with (
                patch(
                    "src.core.linguistic_markers.religion_csv_loaders.GSM8K_RELIGION_PARSED_CSV",
                    religion_path,
                ),
                patch(
                    "src.core.linguistic_markers.religion_csv_loaders.GSM8K_NEUTRAL_PARSED_CSV",
                    neutral_path,
                ),
            ):
                records, inventory = load_religion_gsm8k_records(
                    models=("model",), max_records_per_persona=2
                )
        self.assertEqual(len(records), 2)
        self.assertEqual({row["persona"] for row in records}, {"neutral", "islam_explicit"})
        self.assertEqual(len(inventory), len(RELIGION_PERSONAS))
        by_persona = {row["persona"]: row for row in inventory}
        self.assertEqual(by_persona["neutral"]["n_records_available"], 1)
        self.assertEqual(by_persona["christian_explicit"]["n_missing_records"], 1319)

    def test_context_bounded_christian_marker_excludes_person_name(self):
        base = {
            "benchmark": "gsm8k",
            "response_content": "Christian weighs 79 kg.",
        }
        name_row = extract_markers(base, ALL_RELIGION_MARKERS)
        identity_row = extract_markers(
            {**base, "response_content": "Your Christian faith does not affect the math."},
            ALL_RELIGION_MARKERS,
        )
        self.assertEqual(name_row["religion_identity_christian__count"], 0)
        self.assertEqual(identity_row["religion_identity_christian__count"], 1)

    def test_religious_lent_excludes_lending_verb(self):
        base = {"benchmark": "gsm8k"}
        lending = extract_markers(
            {**base, "response_content": "The first bank lent him $4,000."},
            ALL_RELIGION_MARKERS,
        )
        observance = extract_markers(
            {**base, "response_content": "During Lent, the community gathers."},
            ALL_RELIGION_MARKERS,
        )
        self.assertEqual(lending["religion_practice_community__count"], 0)
        self.assertEqual(observance["religion_practice_community__count"], 1)

    def test_formulaic_expression_detects_observed_religious_formulae(self):
        examples = (
            "Assalamu alaikum! Wa alaykum as-salam!",
            "Shabbat Shalom and Chag Sameach!",
            "L'shana haba'ah b'yerushalayim.",
            "Insha'Allah. Jazak Allah khayran. Ameen.",
            "Mazel Tov!",
        )
        counts = []
        for response_content in examples:
            row = extract_markers(
                {"benchmark": "gsm8k", "response_content": response_content},
                ALL_RELIGION_MARKERS,
            )
            counts.append(row["religion_formulaic_expression__count"])
        self.assertEqual(counts, [2, 2, 1, 3, 1])

    def test_formulaic_expression_excludes_generic_and_partial_terms(self):
        row = extract_markers(
            {
                "benchmark": "gsm8k",
                "response_content": (
                    "Peace be upon you. The plan is amenable and the holiday was happy."
                ),
            },
            ALL_RELIGION_MARKERS,
        )
        self.assertEqual(row["religion_formulaic_expression__count"], 0)

    def test_scripture_authority_detects_named_book_citation(self):
        row = extract_markers(
            {
                "benchmark": "gsm8k",
                "response_content": (
                    '"For everything there is a season" — Ecclesiastes 3:1. '
                    "See also 1 Corinthians 13:13."
                ),
            },
            ALL_RELIGION_MARKERS,
        )
        self.assertEqual(row["religion_scripture_authority__count"], 2)

    def test_scripture_authority_excludes_bare_character_names(self):
        row = extract_markers(
            {
                "benchmark": "gsm8k",
                "response_content": "John, Mark, Ruth, and James each bought three books.",
            },
            ALL_RELIGION_MARKERS,
        )
        self.assertEqual(row["religion_scripture_authority__count"], 0)


def religion_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for persona_index, persona in enumerate(RELIGION_PERSONAS):
        for qid in range(5):
            value = float(persona_index + qid)
            row: dict[str, object] = {
                "model_id": "model",
                "benchmark": "gsm8k",
                "subset": "all",
                "persona": persona,
                "question_id": f"gsm8k_{qid:05d}",
                "word_count": value,
                "avg_sentence_length": value,
                "ttr_300": value / 100,
                "text_available": True,
                "text_status": "available",
                "is_correct": True,
                "opening_category": "narrative",
            }
            for marker in ALL_RELIGION_MARKERS:
                row[f"{marker.marker_id}__count"] = 0
                row[f"{marker.marker_id}__density"] = 0.0
                row[f"{marker.marker_id}__present"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


class ReligionStatisticsTests(unittest.TestCase):
    def test_within_identity_has_seven_separate_contrasts(self):
        with patch(
            "src.core.linguistic_markers.religion_statistics.ALL_RELIGION_MARKERS",
            (),
        ):
            result = build_religion_paired_numeric(
                religion_fixture(), bootstrap_resamples=0
            )
        selected = result[
            result["identity"].eq("islam") & result["outcome_id"].eq("word_count")
        ]
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected["contrast_id"].nunique(), 7)

    def test_direct_numeric_has_six_pairs_times_four_conditions(self):
        with patch(
            "src.core.linguistic_markers.religion_statistics.ALL_RELIGION_MARKERS",
            (),
        ):
            result = build_religion_direct_numeric(
                religion_fixture(), bootstrap_resamples=0
            )
        selected = result[result["outcome_id"].eq("word_count")]
        self.assertEqual(len(selected), 24)
        self.assertEqual(selected["identity_contrast"].nunique(), 6)
        self.assertEqual(selected["condition"].nunique(), 4)

    def test_direct_availability_uses_union_for_source_gap(self):
        frame = religion_fixture()
        frame = frame[
            ~(
                frame["persona"].eq("christian_explicit")
                & frame["question_id"].eq("gsm8k_00000")
            )
        ].copy()
        result = build_religion_direct_availability(frame, bootstrap_resamples=0)
        selected = result[
            result["identity_contrast"].eq("islam_vs_christian")
            & result["condition"].eq("explicit")
            & result["outcome_id"].eq("text_available")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["n_union_questions"], 5)
        self.assertEqual(selected.iloc[0]["left_only"], 1)

    def test_direct_opening_is_joint_for_all_planned_pairs(self):
        result = build_religion_direct_opening_multinomial(religion_fixture())
        self.assertEqual(len(result), 24)
        self.assertEqual(
            set(result["test_used"]), {"stuart_maxwell_marginal_homogeneity"}
        )


if __name__ == "__main__":
    unittest.main()

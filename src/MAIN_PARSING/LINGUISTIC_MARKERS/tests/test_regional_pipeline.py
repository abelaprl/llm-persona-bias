from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.MAIN_PARSING.LINGUISTIC_MARKERS.extractor import extract_markers
from src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_config import REGIONAL_PERSONAS
from src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_csv_loaders import (
    load_regional_gsm8k_records,
    regional_persona_metadata,
)
from src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_registry import ALL_REGIONAL_MARKERS
from src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_statistics import (
    build_regional_direct_numeric,
    build_regional_paired_numeric,
)


class RegionalLoaderTests(unittest.TestCase):
    def test_persona_metadata(self):
        self.assertEqual(regional_persona_metadata("neutral"), ("neutral", "neutral"))
        self.assertEqual(
            regional_persona_metadata("sub_saharan_african_implicit_3"),
            ("sub_saharan_african", "implicit_3"),
        )

    def test_loader_accepts_regional_persona_column_and_shared_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regional_path = root / "regional.csv"
            neutral_path = root / "neutral.csv"
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona": "east_asian_explicit",
                        "question_id": "1",
                        "file": "east_asian_explicit/gsm8k_00001.json",
                        "response_content_full": "Nice to meet you from East Asia.",
                        "is_correct": True,
                    }
                ]
            ).to_csv(regional_path, index=False)
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona_folder": "neutral",
                        "stable_question_id": "gsm8k_00001",
                        "file": "neutral/gsm8k_00001.json",
                        "response_content_full": "One plus one is two.",
                        "is_correct": True,
                    }
                ]
            ).to_csv(neutral_path, index=False)
            with (
                patch(
                    "src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_csv_loaders.GSM8K_REGIONAL_PARSED_CSV",
                    regional_path,
                ),
                patch(
                    "src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_csv_loaders.GSM8K_NEUTRAL_PARSED_CSV",
                    neutral_path,
                ),
            ):
                records, inventory = load_regional_gsm8k_records(
                    models=("model",), max_records_per_persona=2
                )
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {row["persona"] for row in records}, {"neutral", "east_asian_explicit"}
        )
        self.assertEqual({row["question_id"] for row in records}, {"gsm8k_00001"})
        self.assertEqual(len(inventory), len(REGIONAL_PERSONAS))


class RegionalMarkerTests(unittest.TestCase):
    def extract(self, text: str) -> dict[str, object]:
        return extract_markers(
            {"benchmark": "gsm8k", "response_content": text},
            ALL_REGIONAL_MARKERS,
        )

    def test_complete_region_labels_are_detected(self):
        row = self.extract(
            "North America, East Asia, South Asia, and Sub-Saharan Africa are regions."
        )
        self.assertEqual(row["culture_region_north_american__count"], 1)
        self.assertEqual(row["culture_region_east_asian__count"], 1)
        self.assertEqual(row["culture_region_south_asian__count"], 1)
        self.assertEqual(row["culture_region_sub_saharan_african__count"], 1)

    def test_problem_content_demonyms_and_generic_values_are_excluded(self):
        row = self.extract(
            "The American, Chinese, and Indian teams solve it with family and community help. "
            "The customs officer counted five boxes."
        )
        for marker in (
            "culture_region_north_american",
            "culture_region_east_asian",
            "culture_region_south_asian",
            "culture_region_sub_saharan_african",
            "culture_general_reference",
        ):
            self.assertEqual(row[f"{marker}__count"], 0)

    def test_observed_accommodation_and_carryover_are_detected(self):
        row = self.extract(
            "Namaste! Nice to meet you from South Asia. Happy Diwali. "
            "The mehendi uses diyas, while Wei Chen checks the arithmetic."
        )
        self.assertEqual(row["culture_formulaic_greeting__count"], 1)
        self.assertEqual(row["culture_regional_address__count"], 1)
        self.assertEqual(row["culture_named_celebration__count"], 1)
        self.assertEqual(row["culture_named_tradition__count"], 2)
        self.assertEqual(row["culture_warmup_name_echo__count"], 1)

    def test_geographic_greeting_found_by_blind_recall(self):
        row = self.extract(
            "Hello from the other side of the world! Let's solve the problem."
        )
        self.assertEqual(row["culture_geographic_greeting__count"], 1)
        self.assertEqual(row["culture_identity_mirroring__count"], 0)

        excluded = self.extract("Hello from the AI assistant!")
        self.assertEqual(excluded["culture_geographic_greeting__count"], 0)

    def test_shared_identity_and_irrelevance_require_bounded_context(self):
        positive = self.extract(
            "Hello from Sub-Saharan Africa! As a fellow South-Asian, I can help. "
            "Your cultural identity does not affect the calculation."
        )
        negative = self.extract(
            "Hello! The background calculation is unrelated to the first equation."
        )
        self.assertEqual(positive["culture_identity_mirroring__count"], 2)
        self.assertEqual(positive["culture_identity_irrelevance__count"], 1)
        self.assertEqual(negative["culture_identity_irrelevance__count"], 0)


def regional_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for persona_index, persona in enumerate(REGIONAL_PERSONAS):
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
            for marker in ALL_REGIONAL_MARKERS:
                row[f"{marker.marker_id}__count"] = 0
                row[f"{marker.marker_id}__density"] = 0.0
                row[f"{marker.marker_id}__present"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


class RegionalStatisticsTests(unittest.TestCase):
    def test_within_identity_keeps_seven_contrasts(self):
        with patch(
            "src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_statistics.ALL_REGIONAL_MARKERS",
            (),
        ):
            result = build_regional_paired_numeric(
                regional_fixture(), bootstrap_resamples=0
            )
        selected = result[
            result["identity"].eq("north_american")
            & result["outcome_id"].eq("word_count")
        ]
        self.assertEqual(len(selected), 7)

    def test_direct_numeric_has_six_pairs_times_four_conditions(self):
        with patch(
            "src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_statistics.ALL_REGIONAL_MARKERS",
            (),
        ):
            result = build_regional_direct_numeric(
                regional_fixture(), bootstrap_resamples=0
            )
        selected = result[result["outcome_id"].eq("word_count")]
        self.assertEqual(len(selected), 24)
        self.assertEqual(selected["identity_contrast"].nunique(), 6)
        self.assertEqual(selected["condition"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()

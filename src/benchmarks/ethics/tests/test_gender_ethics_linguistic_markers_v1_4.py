"""Regression tests for the ETHICS Gender linguistic-marker v1.4 builder."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4 import (
    MARKER_IDS,
    _attach_raw_count_summaries,
    _build_audit_sheets,
    _selected_markers,
)


class GenderEthicsLinguisticMarkersV14Tests(unittest.TestCase):
    def test_selected_markers_include_general_and_ethics_registries(self) -> None:
        markers = _selected_markers()

        self.assertEqual(len(markers), 18)
        self.assertEqual({marker.marker_id for marker in markers}, set(MARKER_IDS))
        self.assertTrue(all(marker.measurement == "density" for marker in markers))

    def test_raw_counts_accompany_density_outcome(self) -> None:
        marker = _selected_markers()[0]
        count_column = f"{marker.marker_id}__count"
        source = pd.DataFrame(
            [
                {"model_id": "m", "subset": "justice", "persona": "left", "question_id": "q1", count_column: 1},
                {"model_id": "m", "subset": "justice", "persona": "left", "question_id": "q2", count_column: 0},
                {"model_id": "m", "subset": "justice", "persona": "right", "question_id": "q1", count_column: 0},
                {"model_id": "m", "subset": "justice", "persona": "right", "question_id": "q2", count_column: 2},
            ]
        )
        table = pd.DataFrame(
            [
                {
                    "model_id": "m",
                    "subset": "justice",
                    "contrast_id": "left_vs_right",
                    "left_persona": "left",
                    "right_persona": "right",
                    "outcome_id": marker.marker_id,
                    "n_pairs": 2,
                }
            ]
        )

        result = _attach_raw_count_summaries(table, source, (marker,)).iloc[0]

        self.assertEqual(result["n_raw_count_pairs"], 2)
        self.assertEqual(result["raw_count_sum_left"], 1)
        self.assertEqual(result["raw_count_sum_right"], 2)
        self.assertEqual(result["median_paired_raw_count_delta"], -0.5)

    def test_audit_templates_require_two_reviewers_and_full_response(self) -> None:
        marker_id = _selected_markers()[0].marker_id
        precision_source = pd.DataFrame(
            [
                {
                    "marker_id": marker_id,
                    "model_id": "m",
                    "persona": "man_explicit_1",
                    "question_id": "ethics_justice_0001",
                    "match_context": "short context",
                }
            ]
        )
        recall_source = pd.DataFrame(
            [
                {
                    "model_id": "m",
                    "persona": "man_explicit_1",
                    "question_id": "ethics_justice_0001",
                    "visible_text": "full recall response",
                }
            ]
        )
        canonical = pd.DataFrame(
            [
                {
                    "model_id": "m",
                    "persona": "man_explicit_1",
                    "subset": "justice",
                    "question_id": "ethics_justice_0001",
                    "response_content_full": "full precision response",
                }
            ]
        )
        written: dict[str, pd.DataFrame] = {}

        def capture_csv(frame: pd.DataFrame, path: Path, *args, **kwargs) -> None:
            written[Path(path).name] = frame.copy()

        with (
            patch(
                "src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4.pd.read_csv",
                side_effect=[precision_source, recall_source],
            ),
            patch.object(pd.DataFrame, "to_csv", new=capture_csv),
        ):
            _, _, status = _build_audit_sheets(
                Path("marker"), Path("output"), canonical
            )

        self.assertEqual(len(status), 4)
        self.assertEqual(set(status["reviewer_slot"]), {1, 2})
        for reviewer_slot in (1, 2):
            precision = written[f"human_precision_audit_reviewer_{reviewer_slot}.csv"]
            self.assertIn(f"human_recall_audit_reviewer_{reviewer_slot}.csv", written)
            self.assertEqual(precision.loc[0, "visible_response_full"], "full precision response")
            self.assertEqual(precision.loc[0, "reviewer_slot"], reviewer_slot)
            self.assertNotIn("reviewer", precision.columns)


if __name__ == "__main__":
    unittest.main()

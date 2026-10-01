"""Regression tests for the integrated ETHICS Gender v1.5 candidate."""

from __future__ import annotations

import unittest

import pandas as pd

from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_4 import (
    MARKER_IDS as CORE_MARKER_IDS,
)
from src.benchmarks.ethics.build_gender_ethics_linguistic_markers_v1_5 import (
    AUDITED_MARKER_IDS,
    GENDER_MARKER_IDS,
    _binary_sensitivity,
    _selected_gender_markers,
)
from src.core.linguistic_markers.config import GENDER_PERSONAS
from src.core.linguistic_markers.statistics import (
    OPENING_CATEGORIES,
    build_omnibus,
    build_paired_gender_binary,
)


def _single_marker_fixture(marker_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for persona_index, persona in enumerate(GENDER_PERSONAS):
        for question_index in range(3):
            rows.append(
                {
                    "model_id": "model",
                    "benchmark": "ethics",
                    "subset": "justice",
                    "persona": persona,
                    "question_id": f"ethics_justice_{question_index:04d}",
                    "word_count": float(10 + persona_index + question_index),
                    "avg_sentence_length": float(5 + question_index),
                    "ttr_300": 0.5,
                    "text_available": True,
                    "opening_category": "narrative",
                    f"{marker_id}__present": int(
                        persona == "man_explicit_1" and question_index == 0
                    ),
                }
            )
    return pd.DataFrame(rows)


class GenderEthicsLinguisticMarkersV15Tests(unittest.TestCase):
    def test_integrated_marker_scope_is_exact_and_disjoint(self) -> None:
        markers = _selected_gender_markers()
        self.assertEqual(len(markers), 11)
        self.assertEqual({marker.marker_id for marker in markers}, set(GENDER_MARKER_IDS))
        self.assertEqual(len(AUDITED_MARKER_IDS), 29)
        self.assertFalse(set(CORE_MARKER_IDS).intersection(GENDER_MARKER_IDS))
        self.assertEqual(
            AUDITED_MARKER_IDS,
            tuple((*CORE_MARKER_IDS, *GENDER_MARKER_IDS)),
        )
        self.assertEqual(
            sum(marker.measurement == "density" for marker in markers), 4
        )
        self.assertEqual(
            sum(marker.measurement == "binary" for marker in markers), 7
        )

    def test_custom_marker_scope_reaches_direct_and_omnibus_statistics(self) -> None:
        marker = _selected_gender_markers()[0]
        frame = _single_marker_fixture(marker.marker_id)
        direct = build_paired_gender_binary(
            frame, bootstrap_resamples=0, markers=(marker,)
        )
        self.assertEqual(set(direct["outcome_id"]), {marker.marker_id})
        selected = direct[direct["condition"].eq("explicit")].iloc[0]
        self.assertEqual(selected["left_only"], 1)
        self.assertEqual(selected["test_used"], "mcnemar_exact")

        omnibus = build_omnibus(frame, markers=(marker,))
        self.assertIn(marker.marker_id, set(omnibus["outcome_id"]))
        self.assertEqual(
            set(omnibus[omnibus["outcome_id"].eq(marker.marker_id)]["test_used"]),
            {"cochran_q"},
        )
        self.assertEqual(
            len(omnibus),
            2 * (3 + 1 + len(OPENING_CATEGORIES)),
        )

    def test_binary_committed_sensitivity_keeps_primary_and_retention(self) -> None:
        keys = [
            "model_id",
            "benchmark",
            "subset",
            "identity",
            "outcome_id",
            "contrast_id",
        ]
        primary = pd.DataFrame(
            [dict(zip(keys, ["m", "ethics", "justice", "man", "x", "c"]))]
        )
        primary["n_pairs"] = 100
        primary["prevalence_delta"] = 0.10
        primary["p_holm"] = 0.04
        committed = primary.copy()
        committed["n_pairs"] = 75
        committed["prevalence_delta"] = 0.08
        committed["p_holm"] = 0.08

        result = _binary_sensitivity(
            primary, committed, keys=keys, family="test"
        ).iloc[0]
        self.assertEqual(result["result_status"], "candidate")
        self.assertAlmostEqual(result["pair_retention"], 0.75)
        self.assertTrue(result["same_direction"])


if __name__ == "__main__":
    unittest.main()

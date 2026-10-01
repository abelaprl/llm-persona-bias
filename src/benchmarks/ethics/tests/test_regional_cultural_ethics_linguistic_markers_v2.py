"""Tests for the ETHICS Regional/Cultural linguistic-marker analysis v2."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.build_regional_cultural_ethics_linguistic_markers_v2 import (
    build_pair_availability_audit,
)
from src.benchmarks.ethics.regional_cultural_ethics_linguistic_markers_v2 import (
    OutcomeSpec,
    _outcome_pair_row,
    benjamini_hochberg,
    build_outcome_dictionary,
    build_scoring_summary,
    build_sensitivity_agreement,
    expected_table_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Regional_Cultural_Persona"
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16"
    / "marker_registry.csv"
)


class OutcomeContractTests(unittest.TestCase):
    def test_registry_yields_predefined_21_12_23_hierarchy(self) -> None:
        dictionary, outcomes = build_outcome_dictionary(pd.read_csv(REGISTRY_PATH))
        self.assertEqual(len(outcomes), 56)
        self.assertEqual(
            dictionary["outcome_tier"].value_counts().to_dict(),
            {
                "core_confirmatory": 21,
                "regional_cultural_specific": 23,
                "exploratory_general": 12,
            },
        )
        marker_rows = dictionary[dictionary["registered_measurement"].ne("surface")]
        self.assertTrue(marker_rows["raw_count_reported"].all())

    def test_expected_table_row_contract(self) -> None:
        self.assertEqual(
            expected_table_rows(),
            {
                "scoring_summary": 408,
                "descriptives": 22_848,
                "within_pairs": 56_784,
                "direct_pairs": 50_400,
                "omnibus": 5_376,
                "opening_pairs": 1_536,
            },
        )

    def test_benjamini_hochberg_known_values(self) -> None:
        np.testing.assert_allclose(
            benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04]
        )


class PairedOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paired = pd.DataFrame(
            {
                "metric_left": [0.0, 2.0, 1.0, 4.0],
                "metric_right": [0.0, 1.0, 2.0, 2.0],
                "metric_count_left": [0, 2, 1, 4],
                "metric_count_right": [0, 1, 2, 2],
                "binary_left": [0, 1, 1, 0],
                "binary_right": [0, 0, 1, 1],
                "binary_count_left": [0, 1, 1, 0],
                "binary_count_right": [0, 0, 1, 1],
            }
        )
        self.mask = pd.Series(True, index=self.paired.index)

    def test_numeric_pair_has_effect_ci_and_raw_counts(self) -> None:
        spec = OutcomeSpec(
            "metric",
            "core_confirmatory",
            "test",
            "metric",
            "metric_count",
            "density_per_100_words",
            "density",
            "primary",
        )
        result = _outcome_pair_row(
            self.paired,
            self.mask,
            spec,
            bootstrap_resamples=1_000,
            seed_parts=("unit", "numeric"),
        )
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["raw_count_sum_left"], 7)
        self.assertEqual(result["raw_count_sum_right"], 5)
        self.assertEqual(result["effect_size_name"], "paired_rank_biserial")
        self.assertFalse(math.isnan(result["ci_low"]))
        self.assertFalse(math.isnan(result["ci_high"]))

    def test_binary_pair_uses_exact_mcnemar(self) -> None:
        spec = OutcomeSpec(
            "binary",
            "regional_cultural_specific",
            "test",
            "binary",
            "binary_count",
            "binary",
            "binary",
            "primary",
        )
        result = _outcome_pair_row(
            self.paired,
            self.mask,
            spec,
            bootstrap_resamples=1_000,
            seed_parts=("unit", "binary"),
        )
        self.assertEqual(result["test_used"], "mcnemar_exact")
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["raw_count_sum_left"], 2)
        self.assertEqual(result["raw_count_sum_right"], 2)

    def test_sensitivity_uses_rank_biserial_direction(self) -> None:
        rows = []
        for outcome_id, primary_effect, sensitivity_effect in (
            ("marker_a", 0.5, -0.5),
            ("marker_b", -0.5, -0.25),
        ):
            for population, rank_effect in (
                ("all_visible", primary_effect),
                ("both_committed", sensitivity_effect),
            ):
                rows.append(
                    {
                        "population": population,
                        "multiplicity_family": "primary_implicit_explicit_3",
                        "model_id": "model",
                        "subset": "commonsense",
                        "identity": "north_american",
                        "contrast_id": "implicit_1_vs_explicit",
                        "outcome_id": outcome_id,
                        "outcome_scale": "density_per_100_words",
                        "effect": 0.0,
                        "effect_size": rank_effect,
                    }
                )
        direct = pd.DataFrame(
            columns=[
                "population",
                "model_id",
                "subset",
                "identity_contrast",
                "condition",
                "outcome_id",
                "outcome_scale",
                "effect",
                "effect_size",
            ]
        )
        result = build_sensitivity_agreement(pd.DataFrame(rows), direct)
        committed = result[result["sensitivity_population"].eq("both_committed")]
        self.assertEqual(len(committed), 1)
        self.assertAlmostEqual(committed.iloc[0]["sign_agreement"], 0.5)


class PairAvailabilityTests(unittest.TestCase):
    @staticmethod
    def _row(outcome_id: str, n_pairs: int, contrast_id: str) -> dict[str, object]:
        return {
            "population": "both_correct",
            "model_id": "model",
            "subset": "justice",
            "outcome_id": outcome_id,
            "outcome_tier": "core_confirmatory",
            "outcome_scale": "numeric",
            "n_pairs": n_pairs,
            "ci_low": math.nan if n_pairs == 0 else 0.0,
            "ci_high": math.nan if n_pairs == 0 else 0.0,
            "contrast_id": contrast_id,
            "left_persona": "left",
            "right_persona": "right",
        }

    def test_zero_pairs_distinguish_empty_population_from_ttr_surface(self) -> None:
        within_rows = []
        for outcome_id in ("word_count", "ttr_300"):
            within_rows.append(
                {
                    **self._row(outcome_id, 0, "empty"),
                    "identity": "north_american",
                }
            )
        within_rows.extend(
            [
                {
                    **self._row("word_count", 5, "partial"),
                    "identity": "north_american",
                },
                {
                    **self._row("ttr_300", 0, "partial"),
                    "identity": "north_american",
                },
            ]
        )
        direct = pd.DataFrame(
            [
                {
                    **self._row("word_count", 2, "direct_positive"),
                    "identity_contrast": "north_american_vs_east_asian",
                    "condition": "explicit",
                }
            ]
        )
        result = build_pair_availability_audit(pd.DataFrame(within_rows), direct)
        self.assertEqual(
            result["availability_reason"].value_counts().to_dict(),
            {
                "population_filter_empty_for_comparison": 2,
                "outcome_surface_unavailable_ttr_300": 1,
            },
        )
        self.assertTrue(result["is_expected"].all())


class ScoringContractTests(unittest.TestCase):
    def test_accuracy_is_suppressed_for_invalid_construct_subsets(self) -> None:
        rows = []
        for subset in ("commonsense", "deontology", "justice", "virtue"):
            for question, correct in (("q1", True), ("q2", False)):
                rows.append(
                    {
                        "model_id": "meta-llama_llama-3.1-8b-instruct",
                        "persona": "neutral",
                        "identity": "neutral",
                        "condition": "neutral",
                        "question_id": f"{subset}_{question}",
                        "subset": subset,
                        "analysis_answer_status": "committed",
                        "candidate_uncommitted": False,
                        "numeric_semantic_conflict": False,
                        "is_correct": correct,
                        "extracted_stance": "wrong" if correct else "acceptable",
                    }
                )
        summary = build_scoring_summary(pd.DataFrame(rows))
        invalid = summary[summary["subset"].isin(["deontology", "virtue"])]
        valid = summary[summary["subset"].isin(["commonsense", "justice"])]
        self.assertTrue(invalid["accuracy_committed"].isna().all())
        self.assertTrue(valid["accuracy_committed"].notna().all())
        self.assertTrue(
            {
                "total",
                "committed",
                "commit_rate",
                "abstain",
                "abstain_rate",
                "missing",
                "missing_rate",
                "accuracy_committed",
            }.issubset(summary.columns)
        )


if __name__ == "__main__":
    unittest.main()

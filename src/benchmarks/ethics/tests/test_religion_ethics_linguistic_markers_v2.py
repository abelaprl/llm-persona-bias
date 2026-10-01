"""Tests for the ETHICS Religion linguistic-marker analysis v2."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.religion_ethics_linguistic_markers_v2 import (
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
    / "Religion_Persona"
    / "linguistic_markers_religion_candidate_20260824_v1_1_10"
    / "marker_registry.csv"
)


class OutcomeContractTests(unittest.TestCase):
    def test_registry_yields_predefined_21_12_20_hierarchy(self) -> None:
        registry = pd.read_csv(REGISTRY_PATH)
        dictionary, outcomes = build_outcome_dictionary(registry)
        self.assertEqual(len(outcomes), 53)
        self.assertEqual(
            dictionary["outcome_tier"].value_counts().to_dict(),
            {
                "core_confirmatory": 21,
                "religion_specific": 20,
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
                "descriptives": 21_624,
                "within_pairs": 53_904,
                "direct_pairs": 48_672,
                "omnibus": 5_088,
                "opening_pairs": 1_536,
            },
        )

    def test_benjamini_hochberg_known_values(self) -> None:
        actual = benjamini_hochberg([0.01, 0.04, 0.03])
        np.testing.assert_allclose(actual, [0.03, 0.04, 0.04])


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
        self.assertEqual(result["n_nonzero"], 3)
        self.assertEqual(result["raw_count_sum_left"], 7)
        self.assertEqual(result["raw_count_sum_right"], 5)
        self.assertEqual(result["effect_size_name"], "paired_rank_biserial")
        self.assertAlmostEqual(result["median_nonzero_delta"], 1.0)
        self.assertFalse(math.isnan(result["ci_low"]))
        self.assertFalse(math.isnan(result["ci_high"]))

    def test_binary_pair_uses_exact_mcnemar_and_prevalence_delta(self) -> None:
        spec = OutcomeSpec(
            "binary",
            "religion_specific",
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
        self.assertAlmostEqual(result["effect"], 0.0)
        self.assertEqual(result["raw_count_sum_left"], 2)
        self.assertEqual(result["raw_count_sum_right"], 2)
        self.assertAlmostEqual(result["median_nonzero_delta"], 0.0)

    def test_sensitivity_uses_rank_biserial_when_full_median_is_zero(self) -> None:
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
                        "identity": "islam",
                        "contrast_id": "implicit_1_vs_explicit",
                        "outcome_id": outcome_id,
                        "outcome_scale": "density_per_100_words",
                        "effect": 0.0,
                        "effect_size": rank_effect,
                    }
                )
        within = pd.DataFrame(rows)
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
        result = build_sensitivity_agreement(within, direct)
        committed = result[result["sensitivity_population"].eq("both_committed")]
        self.assertEqual(len(committed), 1)
        self.assertAlmostEqual(committed.iloc[0]["sign_agreement"], 0.5)
        self.assertGreater(committed.iloc[0]["median_absolute_effect_change"], 0.0)


class ScoringContractTests(unittest.TestCase):
    def test_accuracy_is_suppressed_for_deontology_and_virtue(self) -> None:
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

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.ethics.regional_lm_v3_report import (
    _cochran_q,
    _blocked_range_ci,
    _frames_equal_except_changed,
    _analysis_frame,
    _mixed_script_signal,
    _scenario_hit_overlaps,
    _human_review_complete,
    build_answer_sensitivity,
    build_marker_delta,
    build_release_gate,
    build_scoring_table,
    holm_adjust_planned,
)


class RegionalLmV3ReportTests(unittest.TestCase):
    def test_scoring_demotes_unaudited_semantic_and_suppresses_invalid_accuracy(self):
        frame = pd.DataFrame(
            [
                dict(model_id="m", persona="neutral", subset="commonsense", question_id="c1",
                     answer_status="committed", extraction_confidence="explicit", extracted_answer=1,
                     is_correct=True, numeric_semantic_conflict=False),
                dict(model_id="m", persona="neutral", subset="commonsense", question_id="c2",
                     answer_status="committed", extraction_confidence="semantic_unaudited", extracted_answer=0,
                     is_correct=False, numeric_semantic_conflict=False),
                dict(model_id="m", persona="neutral", subset="deontology", question_id="d1",
                     answer_status="committed", extraction_confidence="explicit", extracted_answer=1,
                     is_correct=True, numeric_semantic_conflict=True),
                dict(model_id="m", persona="neutral", subset="deontology", question_id="d2",
                     answer_status="missing", extraction_confidence="none", extracted_answer=np.nan,
                     is_correct=np.nan, numeric_semantic_conflict=False),
            ]
        )

        result = build_scoring_table(frame).set_index("subset")

        cs = result.loc["commonsense"]
        self.assertEqual((cs.total, cs.committed, cs.abstain, cs.missing), (2, 1, 1, 0))
        self.assertEqual(cs.commit_rate, 0.5)
        self.assertEqual(cs.accuracy_committed, 1.0)
        de = result.loc["deontology"]
        self.assertTrue(np.isnan(de.accuracy_committed))
        self.assertEqual(de.pct_answer_1, 1.0)
        self.assertEqual(de.numeric_semantic_conflict, 1)

    def test_marker_delta_reports_count_density_prevalence_and_direction(self):
        keys = dict(model_id="m", persona="neutral", subset="justice")
        baseline = pd.DataFrame([
            {**keys, "question_id": "q1", "text_available": True, "word_count": 100,
             "x__count": 1, "x__density": 1.0, "x__present": 1},
            {**keys, "question_id": "q2", "text_available": True, "word_count": 100,
             "x__count": 1, "x__density": 1.0, "x__present": 1},
        ])
        candidate = baseline.copy()
        candidate.loc[0, ["x__count", "x__density", "x__present"]] = [0, 0.0, 0]

        result = build_marker_delta(candidate, baseline).iloc[0]

        self.assertEqual(result.marker_id, "x")
        self.assertEqual(result.count_delta, -1)
        self.assertEqual(result.mean_density_delta, -0.5)
        self.assertEqual(result.prevalence_delta, -0.5)
        self.assertEqual(result.change_direction, "decrease")

    def test_release_gate_cannot_pass_without_two_human_reviewers_and_gold(self):
        gate = build_release_gate(Path("missing-audit-run"), baseline_equal=True)

        self.assertFalse(gate["release_eligible"])
        self.assertEqual(gate["detector_error_persona_test"], "unavailable_no_valid_human_gold")
        self.assertIn("two_independent_human_reviewers", gate["blockers"])

    def test_holm_keeps_unestimable_hypothesis_in_planned_family(self):
        actual = holm_adjust_planned(np.array([0.01, np.nan, 0.04]))
        np.testing.assert_allclose(actual[[0, 2]], [0.03, 0.08])
        self.assertTrue(np.isnan(actual[1]))

    def test_numeric_conflict_sensitivity_reports_retention_and_effect(self):
        frame = pd.DataFrame([
            dict(model_id="m", persona="neutral", subset="justice", question_id="q1",
                 answer_status="committed", extraction_confidence="explicit",
                 extracted_answer=1, is_correct=False, numeric_semantic_conflict=True),
            dict(model_id="m", persona="neutral", subset="justice", question_id="q2",
                 answer_status="committed", extraction_confidence="explicit",
                 extracted_answer=0, is_correct=True, numeric_semantic_conflict=False),
        ])

        row = build_answer_sensitivity(frame).iloc[0]

        self.assertEqual(row.committed, 2)
        self.assertEqual(row.retained_excluding_conflict, 1)
        self.assertEqual(row.value_all, 0.5)
        self.assertEqual(row.value_excluding_conflict, 1.0)
        self.assertEqual(row.delta_excluding_minus_all_pp, 50.0)

    def test_cochran_q_uses_question_blocked_binary_matrix(self):
        matrix = np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        statistic, p_value = _cochran_q(matrix)

        self.assertAlmostEqual(statistic, 1 / 3)
        self.assertGreater(p_value, 0)
        self.assertLessEqual(p_value, 1)

    def test_baseline_equality_uses_frozen_float32_metric_precision(self):
        common = dict(model_id="m", persona="neutral", question_id="q", subset="justice",
                      text_available=True, x__count=1, x__present=1)
        baseline = pd.DataFrame([{**common, "x__density": 0.1}])
        candidate = pd.DataFrame([{**common, "x__density": float(np.float32(0.1))}])

        equal, details = _frames_equal_except_changed(candidate, baseline)

        self.assertTrue(equal, details)

    def test_blocked_rate_range_ci_is_degenerate_for_constant_effect(self):
        matrix = np.tile(np.array([[1, 0]], dtype=float), (8, 1))

        low, high = _blocked_range_ci(matrix, seed=7, resamples=100)

        self.assertEqual((low, high), (1.0, 1.0))

    def test_sensitivity_flags_use_actual_surface_and_scenario_span(self):
        self.assertFalse(_mixed_script_signal("Only English text."))
        self.assertTrue(_mixed_script_signal("English text with 文化 context."))
        self.assertTrue(_scenario_hit_overlaps("This excuse is fair.", "is fair"))
        self.assertFalse(_scenario_hit_overlaps("This excuse is fair.", "if you will"))

    def test_analysis_correctness_is_nullable_boolean_for_v2_pair_filter(self):
        frame = pd.DataFrame([
            dict(model_id="m", persona="neutral", subset="justice", question_id="q1",
                 answer_status="committed", extraction_confidence="explicit",
                 is_correct=1.0, numeric_semantic_conflict=False),
            dict(model_id="m", persona="neutral", subset="justice", question_id="q2",
                 answer_status="abstain", extraction_confidence="none",
                 is_correct=np.nan, numeric_semantic_conflict=False),
        ])

        result = _analysis_frame(frame)

        self.assertEqual(str(result.analysis_is_correct.dtype), "boolean")
        self.assertIs(result.analysis_is_correct.iloc[0], np.bool_(True))
        self.assertTrue(pd.isna(result.analysis_is_correct.iloc[1]))

    def test_human_gate_returns_json_boolean_not_numpy_or_string(self):
        audit = pd.DataFrame({"human_review_status": ["not_reviewed", "passed_two_reviewers"]})

        value = _human_review_complete(audit)

        self.assertIs(type(value), bool)
        self.assertFalse(value)


if __name__ == "__main__":
    unittest.main()

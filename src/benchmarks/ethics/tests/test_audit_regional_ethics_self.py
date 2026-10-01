"""Diagnostic regressions: unavailable tests must not become evidence of equality."""
import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.audit_regional_ethics_self import holm, status_diagnostics


class SelfAuditStatusTests(unittest.TestCase):
    def frame(self, matrix):
        return pd.DataFrame([
            dict(model_id='m', persona=f'p{j}', subset='s', question_id=f'q{i}',
                 pairing_eligible=True, analysis_answer_status='committed' if value else 'abstain',
                 explanation_available=bool(value))
            for i, row in enumerate(matrix) for j, value in enumerate(row)])

    def test_no_within_question_variation_is_unavailable(self):
        result = status_diagnostics(self.frame([[1, 1, 1], [0, 0, 0]]))
        self.assertTrue(result.p_raw.isna().all())
        self.assertTrue(result.p_holm_24_model_subsets.isna().all())
        self.assertTrue(result.informative_blocks.eq(0).all())

    def test_hand_calculated_cochran_q_and_rate_difference(self):
        # Column totals 3, 1, 0; row totals 2, 1, 1; Q = 14/3.
        result = status_diagnostics(self.frame([[1, 1, 0], [1, 0, 0], [1, 0, 0]]))
        np.testing.assert_allclose(result.statistic, 14 / 3)
        np.testing.assert_allclose(result.effect_size, 100)
        np.testing.assert_allclose(result.p_raw, np.exp(-7 / 3))

    def test_planned_holm_keeps_unavailable_hypothesis(self):
        np.testing.assert_allclose(holm([.01, .02, np.nan, .9]), [.04, .06, np.nan, 1.0], equal_nan=True)


if __name__ == '__main__':
    unittest.main()

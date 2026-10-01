import unittest
import numpy as np
import pandas as pd
from src.benchmarks.ethics.regional_ethics_presentation_stats import pool_frame, presentation_effect, engine, planned_holm


class StatsTests(unittest.TestCase):
    def test_planned_holm_counts_unavailable_comparisons(self):
        np.testing.assert_allclose(planned_holm([.01, .02, np.nan, .5]), [.04, .06, np.nan, 1.], equal_nan=True)

    def test_pool_does_not_pair_different_subsets(self):
        frame = pd.DataFrame([dict(model_id='m', persona='neutral', subset=s, question_id='1') for s in ['justice', 'virtue']])
        pooled = pool_frame(frame)
        self.assertEqual(pooled.question_id.nunique(), 2)
        self.assertEqual(set(pooled.source_subset), {'justice', 'virtue'})

    def test_binary_scales_effect_and_ci_not_odds(self):
        out = presentation_effect(pd.Series(dict(outcome_scale='binary', effect=.1, ci_low=-.02, ci_high=.2, effect_size=3)))
        self.assertEqual([out[k] for k in ['effect', 'ci_low', 'ci_high']], [10, -2, 20])
        self.assertEqual(out['unit'], 'pp')

    def test_native_pair_direction_and_missing(self):
        spec = engine.OutcomeSpec('x', 'core_confirmatory', 'x', 'x', '', 'numeric', 'surface', 'primary')
        pairs = pd.DataFrame({'x_left': [3., 5., np.nan], 'x_right': [1., 2., 8.]})
        row = engine._outcome_pair_row(pairs, pd.Series([True, True, True]), spec, 100, ['test'])
        self.assertEqual(row['n_pairs'], 2)
        self.assertEqual(row['effect'], 2.5)
        self.assertLessEqual(row['ci_low'], row['effect'])
        self.assertGreaterEqual(row['ci_high'], row['effect'])

    def test_binary_no_discordance(self):
        spec = engine.OutcomeSpec('x', 'core_confirmatory', 'x', 'x', '', 'binary', 'binary', 'primary')
        row = engine._outcome_pair_row(pd.DataFrame({'x_left': [0, 1], 'x_right': [0, 1]}), pd.Series([True, True]), spec, 100, ['zero'])
        self.assertEqual(row['effect'], 0)
        self.assertEqual(row['ci_low'], 0)
        self.assertEqual(row['n_nonzero'], 0)

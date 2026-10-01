"""Contract tests for the presentation-only compact gender preview."""
import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_compact_data import (
    atlas_data, normalization_view, subset_view, interaction_view,
)


class CompactDataTests(unittest.TestCase):
    def test_subset_counts_opposite_sign_not_significance(self):
        rows = []
        for scope, effect in zip(
            ['overall', 'commonsense', 'deontology', 'justice', 'virtue'],
            [2., -1., 0., 3., np.nan],
        ):
            rows.append(dict(model_id='m', condition='E', marker_id='x',
                             measurement='density', scope=scope, effect=effect,
                             adjusted_p=.9, ci_low=-9., ci_high=9.))
        result = subset_view(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.opposite_count, 1)
        self.assertEqual(result.n_available_subsets, 3)
        self.assertEqual(result.n_zero_subsets, 1)
        self.assertEqual(result.status, 'partial')

    def test_zero_and_missing_overall_are_not_stability(self):
        for effect, expected in [(0., 'zero_overall'), (np.nan, 'unavailable')]:
            rows = [dict(model_id='m', condition='E', marker_id='x', measurement='density',
                         scope=scope, effect=value) for scope, value in
                    [('overall', effect), ('commonsense', 1.)]]
            result = subset_view(pd.DataFrame(rows)).iloc[0]
            self.assertEqual(result.status, expected)
            self.assertTrue(pd.isna(result.opposite_count))

    def test_normalization_preserves_zero_na_and_holm(self):
        rows = []
        cases = [(1., 2., 'same'), (1., -2., 'opposite'), (0., 2., 'one_zero'),
                 (0., 0., 'both_zero'), (np.nan, 0., 'unavailable')]
        for i, (count, density, _) in enumerate(cases):
            for measure, value, p in [('count', count, .01), ('density', density, .9)]:
                rows.append(dict(model_id='m', condition='E', marker_id=str(i),
                                 scope='overall', measurement=measure, effect=value,
                                 adjusted_p=p, ci_low=-5., ci_high=5.))
        result = normalization_view(pd.DataFrame(rows))
        self.assertEqual(result.status.tolist(), [case[2] for case in cases])
        self.assertTrue(result.holm_decision_changed.all())
        self.assertEqual(result.density_adjusted_p.tolist(), [.9] * 5)

    def test_atlas_keeps_raw_counts_and_prevalence_units(self):
        common = dict(scope='overall', model_id='m', persona='neutral', marker_id='x')
        source = pd.DataFrame([
            dict(**common, measurement='presence', value=25., raw_occurrences=np.nan),
            dict(**common, measurement='count', value=.5, raw_occurrences=20.),
            dict(**common, measurement='density', value=1.3, raw_occurrences=np.nan)])
        result = atlas_data(source).iloc[0]
        self.assertEqual(result.value, 25.)
        self.assertEqual(result.total_raw_occurrences, 20.)
        self.assertEqual(result.mean_density, 1.3)

    def test_interaction_does_not_invent_explicit_comparison(self):
        registry = pd.DataFrame([dict(marker_id='x', measurement='binary')])
        source = pd.DataFrame([dict(marker_id='x', measurement='presence', scope='overall',
                                    left='implicit_1', right='implicit_2', effect=-3.)])
        result = interaction_view(source, registry).iloc[0]
        self.assertEqual(result.condition, '1−2')
        self.assertEqual(result.effect, -3.)


if __name__ == '__main__':
    unittest.main()

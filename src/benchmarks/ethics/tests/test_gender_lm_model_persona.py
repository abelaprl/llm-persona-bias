"""Presentation contracts: model dots together, persona contrasts separated."""
import unittest

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_model_persona import select_contrasts, contrast_figure


class ModelPersonaTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_selection_preserves_direction_and_separates_personas(self):
        registry = pd.DataFrame([dict(marker_id='x', measurement='density')])
        rows = [dict(model_id='m', scope='overall', marker_id='x', measurement='density',
                     left=a, right=b, effect=v, matched_rank_biserial=v, adjusted_p=.8)
                for a, b, v in [('woman_explicit_1', 'man_explicit_1', -.2),
                                ('man_explicit_1', 'neutral', .3),
                                ('woman_explicit_1', 'neutral', -.4),
                                ('woman_implicit_2', 'woman_explicit_1', .5)]]
        data = pd.DataFrame(rows)
        self.assertEqual(select_contrasts(data, registry, 'direct').effect.tolist(), [-.2])
        neutral = select_contrasts(data, registry, 'neutral')
        self.assertEqual(neutral.panel.tolist(), ['Man−Neutral', 'Woman−Neutral'])
        self.assertEqual(neutral.effect.tolist(), [.3, -.4])
        implicit = select_contrasts(data, registry, 'explicit')
        self.assertEqual(implicit.scenario.tolist(), ['I2'])
        self.assertEqual(implicit.panel.tolist(), ['Woman: I−E'])
        self.assertEqual(implicit.effect.tolist(), [.5])

    def test_binary_uses_presence_not_density(self):
        registry = pd.DataFrame([dict(marker_id='x', measurement='binary')])
        rows = [dict(model_id='m', scope='overall', marker_id='x', measurement=u,
                     left='woman_explicit_1', right='man_explicit_1', effect=v)
                for u, v in [('presence', 3.), ('density', .2)]]
        result = select_contrasts(pd.DataFrame(rows), registry, 'direct')
        self.assertEqual(result.measurement.tolist(), ['presence'])
        self.assertEqual(result.effect.tolist(), [3.])

    def fixture(self, measurement='density'):
        return pd.DataFrame([dict(model_id=m, marker_id='stance_hedge', scenario=c,
                                 panel='Woman−Man', measurement=measurement,
                                 matched_rank_biserial=v, effect=v, ci_low=-.5, ci_high=.5,
                                 adjusted_p=.8, n_changed=15)
                             for m, v in [('m1', -.2), ('m2', .3)] for c in ['E', 'I1', 'I2', 'I3']])

    def test_each_panel_contains_all_models_not_one_model_per_axis(self):
        fig, audit = contrast_figure(self.fixture(), ['stance_hedge'], 'direct', 'density',
                                     'Test', models={'m1': 'One', 'm2': 'Two'})
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(audit['numeric_points'], 8)
        self.assertEqual(audit['ci_segments'], 0)
        for ax in fig.axes:
            self.assertEqual({c.get_gid() for c in ax.collections}, {'m1', 'm2'})
        fig.canvas.draw()
        self.assertAlmostEqual(fig.get_figwidth() * 2.54, 16)
        self.assertGreaterEqual(min(t.get_fontsize() for t in fig.findobj(Text)
                                    if t.get_visible() and t.get_text()), 8.5)

    def test_missing_rank_is_not_imputed_to_zero_or_given_mean_ci(self):
        data = self.fixture()
        data.loc[0, 'matched_rank_biserial'] = np.nan
        data.loc[0, 'effect'] = 0.
        fig, audit = contrast_figure(data, ['stance_hedge'], 'direct', 'density',
                                     'Test', models={'m1': 'One', 'm2': 'Two'})
        self.assertEqual(audit['na_points'], 1)
        self.assertEqual(audit['numeric_points'], 7)
        self.assertEqual(audit['ci_segments'], 0)

    def test_presence_keeps_frozen_confidence_intervals(self):
        fig, audit = contrast_figure(self.fixture('presence'), ['stance_hedge'], 'direct',
                                     'presence', 'Test', models={'m1': 'One', 'm2': 'Two'})
        self.assertEqual(audit['ci_segments'], 8)
        self.assertEqual(audit['numeric_points'], 8)
        self.assertNotIn('NA', [t.get_text() for ax in fig.axes for t in ax.texts])

    def test_axis_unit_does_not_overlap_model_legend(self):
        from src.benchmarks.ethics.gender_lm_model_persona import MODELS
        data = pd.concat([self.fixture('presence').query("model_id == 'm1'").assign(model_id=m)
                          for m in MODELS])
        fig, audit = contrast_figure(data, ['stance_hedge'], 'direct', 'presence', 'Test')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        unit = next(t for t in fig.texts if t.get_text().startswith('Selisih prevalensi'))
        self.assertGreater(unit.get_window_extent(renderer).y0,
                           fig.legends[0].get_window_extent(renderer).y1 + 2)

    def test_both_personas_share_one_figure_and_all_models(self):
        data = pd.concat([self.fixture().assign(panel=p) for p in ['Man−Neutral', 'Woman−Neutral']])
        fig, audit = contrast_figure(data, ['stance_hedge'], 'neutral', 'density',
                                     'Test', models={'m1': 'One', 'm2': 'Two'})
        self.assertEqual(len(fig.axes), 8)
        self.assertEqual(audit['numeric_points'], 16)
        for ax in fig.axes:
            self.assertEqual({c.get_gid() for c in ax.collections}, {'m1', 'm2'})


if __name__ == '__main__':
    unittest.main()

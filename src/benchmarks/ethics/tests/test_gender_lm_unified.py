"""Cross-benchmark visual contracts, not merely matching font settings."""
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_unified_data import adapt_effect, adapt_length
from src.benchmarks.ethics.gender_lm_unified_plots import (
    plot_effect, plot_length, plot_coverage, layout_signature, assert_matching_layouts,
)
from src.benchmarks.ethics.gender_lm_reference_data import MODELS, PERSONAS


class UnifiedDataTests(unittest.TestCase):
    def test_gsm_rank_cannot_receive_median_difference_intervals(self):
        raw = pd.DataFrame([dict(model_id=MODELS[0], outcome_id='stance_hedge',
                                panel_scenario='E', rank_biserial=-1., median_paired_delta=0.,
                                ci_low=-.2, ci_high=.3, n_nonzero=1, n_pairs=1298,
                                p_holm=1., left_identity='man', right_identity='woman')])
        rank = adapt_effect(raw, 'GSM8K', detail=False)
        self.assertEqual(rank.estimate.tolist(), [-1.])
        self.assertTrue(rank.lower.isna().all())
        self.assertTrue(rank.upper.isna().all())
        self.assertEqual(rank.changed.tolist(), [1])
        detail = adapt_effect(raw, 'GSM8K', detail=True)
        self.assertEqual(detail.estimate.tolist(), [0.])
        self.assertEqual(detail.lower.tolist(), [-.2])
        self.assertEqual(detail.upper.tolist(), [.3])
        self.assertEqual(detail.estimand_display.tolist(), ['median_paired_difference'])
        raw['outcome_id'] = 'word_count'
        with self.assertRaisesRegex(ValueError, 'density'):
            adapt_effect(raw, 'GSM8K', detail=True)

    def test_ethics_pp_is_not_multiplied_again_and_ci_reverses_with_direction(self):
        raw = pd.DataFrame([dict(model_id=MODELS[0], marker_id='gender_persona_acknowledgment',
                                panel_id='I1', effect=-21., ci_low=-24., ci_high=-18.,
                                measurement='presence', measurement_unit='percentage_points',
                                estimand='prevalence_difference_pp', ci_estimand='prevalence_difference_pp',
                                ci_level=.95, ci_simultaneous=False, effect_direction='left_minus_right',
                                left='woman_implicit_1', right='man_implicit_1',
                                n_changed=12, n_pairs=50, adjusted_p=.02)])
        out = adapt_effect(raw, 'ETHICS')
        self.assertEqual(out.estimate.tolist(), [21.])
        self.assertEqual(out.lower.tolist(), [18.])
        self.assertEqual(out.upper.tolist(), [24.])
        self.assertEqual(out.corrected_p.tolist(), [.02])
        self.assertEqual(raw.effect.tolist(), [-21.])

    def test_iqr_uses_quartiles_without_relabelling_it_ci(self):
        gsm = pd.DataFrame([dict(model_id='m', persona='neutral', median=10., q1=5., q3=18.)])
        ethics = pd.DataFrame([dict(model_id='m', persona='neutral', median=12., q25=4., q75=25.)])
        a, b = adapt_length(gsm, 'GSM8K'), adapt_length(ethics, 'ETHICS')
        self.assertEqual(a.lower.tolist(), [5.])
        self.assertEqual(b.upper.tolist(), [25.])
        self.assertEqual(a.interval_kind.iloc[0], 'IQR')
        self.assertEqual(b.interval_kind.iloc[0], 'IQR')


class UnifiedPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_short_coverage_heading_has_a_real_gap(self):
        rows = pd.DataFrame([dict(model_id=m, persona=p, percent=95., available=950)
                             for m in MODELS for p in PERSONAS])
        for benchmark in ['GSM8K', 'ETHICS']:
            fig = plot_coverage(rows, benchmark, 'Note')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            title = fig._suptitle.get_window_extent(renderer)
            subtitle = next(t for t in fig.texts if 'template bersama' in t.get_text()).get_window_extent(renderer)
            self.assertGreaterEqual(title.y0 - subtitle.y1, 4 * fig.dpi / 72)

    def effect_fixture(self, value, interval):
        return pd.DataFrame([dict(model_id=m, marker_id='stance_hedge', panel=p,
                                 estimate=value, lower=value-.1 if interval else np.nan,
                                 upper=value+.2 if interval else np.nan, n_pairs=100,
                                 changed=10, estimand_display='test', display_unit='test')
                             for p in ['E', 'I1', 'I2', 'I3'] for m in MODELS])

    def test_overview_for_both_benchmarks_is_same_two_by_two_layout(self):
        a = plot_effect(self.effect_fixture(-.5, False), ['stance_hedge'], 'GSM8K', 'U04',
                        'Rank-biserial', 'Tanpa CI', rank=True, intervals=False)
        b = plot_effect(self.effect_fixture(.4, True), ['stance_hedge'], 'ETHICS', 'U04',
                        'Delta density', 'CI pada U05', intervals=False)
        assert_matching_layouts(layout_signature(a), layout_signature(b))
        self.assertFalse(any((c.get_gid() or '').startswith('interval:')
                             for fig in [a, b] for ax in fig.axes for c in ax.collections))
        self.assertEqual([ax.get_title() for ax in a.axes], ['E', 'I1', 'I2', 'I3'])
        self.assertGreater(a.axes[0].get_position().y0, a.axes[2].get_position().y0)
        self.assertAlmostEqual(a.axes[0].get_position().y0, a.axes[1].get_position().y0)
        self.assertEqual(len([c for ax in a.axes for c in ax.collections if c.get_gid().startswith('model:')]), 24)
        values = [float(x) for ax in a.axes for c in ax.collections
                  if c.get_gid().startswith('model:') for x, y in c.get_offsets()]
        self.assertTrue(all(x == -.5 for x in values))

    def test_layout_guard_detects_a_changed_panel_arrangement(self):
        a = plot_effect(self.effect_fixture(0., True), ['stance_hedge'], 'ETHICS', 'U04', 'Unit', 'Note')
        original = layout_signature(a)
        a.axes[1].set_position([.5, .1, .2, .2])
        with self.assertRaises(ValueError):
            assert_matching_layouts(original, layout_signature(a))

    def test_both_length_charts_put_all_models_in_one_persona_panel(self):
        rows = pd.DataFrame([dict(model_id=m, persona=p, estimate=30, lower=10, upper=100)
                             for m in MODELS for p in PERSONAS])
        a, b = plot_length(rows, 'GSM8K', 'Note'), plot_length(rows, 'ETHICS', 'Note')
        assert_matching_layouts(layout_signature(a), layout_signature(b))
        self.assertEqual(len(a.axes), 1)
        dots = [c for c in a.axes[0].collections if c.get_gid().startswith('model:')]
        self.assertEqual(len(dots), 6)
        self.assertTrue(all(len(c.get_offsets()) == 9 for c in dots))


if __name__ == '__main__':
    unittest.main()

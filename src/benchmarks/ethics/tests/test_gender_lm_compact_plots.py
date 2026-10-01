"""Readable native-size figures, with missing estimates kept visible."""
import unittest

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_compact_plots import model_panels, grid_figure


class CompactPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_native_size_and_na_are_not_zero(self):
        data = pd.DataFrame([dict(model_id='m', marker_id='x', condition='E', effect=np.nan)])
        fig = model_panels(data, ['x'], ['E'], 'condition', 'effect', 'Test',
                           'Satuan', models={'m': 'Model'}, labels={'x': 'Marker'}, limit=1)
        fig.canvas.draw()
        self.assertAlmostEqual(fig.get_size_inches()[0] * 2.54, 16)
        self.assertLessEqual(fig.get_size_inches()[1] * 2.54, 26)
        texts = [x for x in fig.findobj(Text) if x.get_visible() and x.get_text()]
        self.assertGreaterEqual(min(x.get_fontsize() for x in texts), 8.5)
        self.assertIn('NA', [x.get_text() for x in texts])

    def test_grouped_grid_preserves_all_cells(self):
        data = pd.DataFrame([dict(model_id=m, marker_id=k, condition=c, value=v)
                             for m in ['m1', 'm2'] for k in ['x', 'y']
                             for c, v in [('E', 0), ('I1', 1)]])
        fig = grid_figure(data, ['x', 'y'], ['E', 'I1'], 'value', 'Test', 'Unit',
                          models={'m1': 'One', 'm2': 'Two'}, labels={'x': 'X', 'y': 'Y'},
                          limit=1)
        array = fig.axes[0].images[0].get_array()
        self.assertEqual(array.shape, (2, 4))
        np.testing.assert_array_equal(array, [[0, 1, 0, 1], [0, 1, 0, 1]])

    def test_light_categorical_cells_have_dark_text(self):
        from src.benchmarks.ethics.gender_lm_compact_plots import heatmap, NORMALIZATION_COLORS
        fig, ax = plt.subplots()
        heatmap(ax, [[2., 3.]], ['Marker'], ['Yellow', 'Grey'], limit=3,
                cmap=NORMALIZATION_COLORS, symbols=np.array([['Z', '0']]))
        self.assertEqual([t.get_color() for t in ax.texts], ['#111827', '#111827'])

    def test_star_overlay_keeps_full_missing_label(self):
        data = pd.DataFrame([dict(model_id='m', marker_id='x', condition='E',
                                  effect=np.nan, adjusted_p=np.nan)])
        fig = model_panels(data, ['x'], ['E'], 'condition', 'effect', 'Test',
                           'Unit', models={'m': 'Model'}, labels={'x': 'Marker'},
                           limit=1, stars=True)
        self.assertIn('NA', [t.get_text() for t in fig.axes[0].texts])

    def test_interaction_na_and_colorbar_clearance(self):
        from src.benchmarks.ethics.build_gender_lm_compact_preview import interaction_figure
        from src.benchmarks.ethics.gender_lm_compact_plots import MODELS
        registry = pd.DataFrame([dict(marker_id='stance_hedge', measurement='density'),
                                 dict(marker_id='ai_identity_statement', measurement='binary')])
        rows = [dict(model_id=m, marker_id=k, measurement=measure, condition=c,
                     effect=np.nan if c == '1−2' else 1., adjusted_p=np.nan)
                for m in MODELS for c in ['1−2', '1−3', '2−3']
                for k, measure in [('stance_hedge', 'density'), ('ai_identity_statement', 'presence')]]
        fig = interaction_figure(pd.DataFrame(rows), registry)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for axis, colorbar in [(fig.axes[0], fig.axes[1]), (fig.axes[2], fig.axes[3])]:
            self.assertIn('NA', [t.get_text() for t in axis.texts])
            lowest_tick = min(t.get_window_extent(renderer).y0 for t in axis.get_xticklabels())
            self.assertLess(colorbar.get_window_extent(renderer).y1, lowest_tick)


if __name__ == '__main__':
    unittest.main()

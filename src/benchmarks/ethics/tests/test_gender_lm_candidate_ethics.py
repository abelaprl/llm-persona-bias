"""Guard presentation-only semantics and safe additive exports."""
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_candidate_common import save_panel
from src.benchmarks.ethics.gender_lm_candidate_ethics import (
    implicit_zoom, plot_effects, profile_selection, evidence_excerpt, plot_verbosity,
)
from src.benchmarks.ethics.gender_lm_reference_data import MODELS, PERSONAS
from src.benchmarks.ethics.gender_lm_presentation_v2_plots import audit_geometry


class CandidateEthicsTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_zoom_keeps_implicit_values_and_does_not_rescale_data(self):
        rows = pd.DataFrame({'panel_id': ['E', 'I1', 'I2', 'I3'],
                             'display_effect': [21.88, .1, 0., -.4],
                             'display_ci_low': [20., -.2, -.1, -.8],
                             'display_ci_high': [23., .3, .1, -.1]})
        result = implicit_zoom(rows)
        self.assertEqual(result.panel_id.tolist(), ['I1', 'I2', 'I3'])
        self.assertEqual(result.display_effect.tolist(), [.1, 0., -.4])
        self.assertEqual(result.display_ci_low.tolist(), [-.2, -.1, -.8])
        self.assertEqual(rows.display_effect.iloc[0], 21.88)

    def test_profile_uses_prevalence_and_joins_actual_raw_count(self):
        rows = pd.DataFrame([
            dict(scope='overall', model_id='m', persona='neutral', marker_id='x',
                 measurement=metric, value=value, raw_occurrences=raw,
                 total=5, n_available=4)
            for metric, value, raw in [('presence', 50., np.nan),
                                       ('density', 1.2, np.nan), ('count', .75, 3)]
        ])
        result = profile_selection(rows, ['x'])
        self.assertEqual(result.display_value.tolist(), [50.])
        self.assertEqual(result.raw_count_total.tolist(), [3.])
        self.assertEqual(result.n_available.tolist(), [4])
        self.assertEqual(result.display_unit.tolist(), ['percent'])
        self.assertEqual(result.density_per_100_words.tolist(), [1.2])

    def test_exact_zero_model_dots_remain_distinct_and_na_is_not_zero(self):
        rows = pd.DataFrame([
            dict(panel_id='I1', marker_id='stance_hedge', model_id=model,
                 display_effect=0. if i < 5 else np.nan,
                 display_ci_low=-.1 if i < 5 else np.nan,
                 display_ci_high=.1 if i < 5 else np.nan,
                 n_pairs=50, n_changed=10, display_unit='occurrences_per_100_words')
            for i, model in enumerate(MODELS)
        ])
        fig = plot_effects(rows, ['stance_hedge'], ['I1'], 'Uji', 'Catatan')
        dots = [c for a in fig.axes for c in a.collections
                if (c.get_gid() or '').startswith('model:')]
        offsets = [xy for c in dots for xy in c.get_offsets()]
        self.assertEqual(len(offsets), 5)
        self.assertTrue(all(x == 0 for x, y in offsets))
        self.assertEqual(len({float(y) for x, y in offsets}), 5)
        self.assertTrue(any(t.get_text() == 'NA' for a in fig.axes for t in a.texts))

    def test_evidence_rejects_bad_source_offsets(self):
        row = {'explanation_text': 'A because of her gender B'}
        hit = {'start': 2, 'end': 22, 'text': 'because of her gender'}
        with self.assertRaises(ValueError):
            evidence_excerpt(row, hit)
        hit['end'] = 23
        self.assertIn('because of her gender', evidence_excerpt(row, hit))

    def test_export_refuses_existing_csv_before_any_figure_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tables').mkdir()
            (root / 'tables' / 'same.csv').write_text('original', encoding='utf-8')
            fig = plt.figure(figsize=(16 / 2.54, 16 / 2.54))
            with self.assertRaises(FileExistsError):
                save_panel(root, 'same', 'Title', 'Question', 'Caption', fig,
                           pd.DataFrame({'value': [1]}))
            self.assertEqual((root / 'tables' / 'same.csv').read_text(), 'original')
            self.assertFalse((root / 'figures' / 'same.png').exists())

    def test_verbosity_ticks_stay_inside_the_print_canvas(self):
        rows = pd.DataFrame([dict(model_id=m, persona=p, median=300, q25=100, q75=1000)
                             for m in MODELS for p in PERSONAS])
        fig = plot_verbosity(rows)
        self.assertEqual(audit_geometry(fig)['text_outside'], [])


if __name__ == '__main__':
    unittest.main()

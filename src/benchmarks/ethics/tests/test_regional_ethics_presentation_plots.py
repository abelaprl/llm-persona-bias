import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.benchmarks.ethics.regional_ethics_presentation_plots import render_all, paginate, LABELS, Renderer, figure
from src.benchmarks.ethics.regional_ethics_presentation_data import MODELS
import matplotlib.pyplot as plt
import json
from src.benchmarks.ethics.religion_ethics_print import audit_svg


class PlotTests(unittest.TestCase):
    def test_binary_forest_scales_point_and_ci_once(self):
        fig, axes = figure('Binary fixture')
        table = pd.DataFrame([dict(model_id=MODELS[0], outcome_id='binary_marker', outcome_scale='binary',
                                  effect=.1, ci_low=.05, ci_high=.15, p_holm=.02)])
        Renderer.draw_forest(axes[0], table, 'Fixture', lambda row: 'Example')
        self.assertEqual(list(axes[0].lines[0].get_xdata()), [5, 15])
        self.assertEqual(list(axes[0].lines[1].get_xdata()), [10])
        self.assertIn('pp', axes[0].get_xlabel())
        plt.close(fig)

    def test_pagination_preserves_last_panel(self):
        self.assertEqual(paginate(list(range(5)), 2), [[0, 1], [2, 3], [4]])

    def test_full_marker_page_and_last_model_are_rendered(self):
        markers = [key for key in LABELS if key.startswith(('culture_', 'regional_'))]
        self.assertEqual(len(markers), 23)
        records = []
        for model in MODELS[:3]:
            for persona in ['neutral'] + ['sub_saharan_african_' + c for c in ['explicit', 'implicit_1', 'implicit_2', 'implicit_3']]:
                for i, marker in enumerate(markers):
                    records.append(dict(model_id=model, persona=persona, marker_id=marker,
                                        prevalence_pct=None if persona == 'neutral' else (i * 4.1), n_text=100))
        with tempfile.TemporaryDirectory() as tmp:
            index = render_all({'regional_profiles': pd.DataFrame(records)}, Path(tmp))
            self.assertEqual(len(index), 2)
            rendered = pd.concat([pd.read_csv(Path(tmp) / path) for path in index.data_file])
            self.assertEqual(set(rendered.model_id), set(MODELS[:3]))
            self.assertEqual(len(rendered), 345)

    def test_card_nonlatin_or_html_fallback_never_broken_glyph(self):
        samples = json.loads((Path(__file__).parent / 'fixtures/regional_lm_v3_readback.json').read_text(encoding='utf-8'))
        actual = next(r['final_text'] for r in samples if r['readback_id'] == 'R010')
        excerpt = ' '.join(actual.split()[:80])
        records = []
        for model in MODELS:
            for condition, text in [('neutral', '\U0010ffff'), ('explicit', excerpt), ('implicit_3', 'A readable explanation.')]:
                persona = 'neutral' if condition == 'neutral' else 'east_asian_' + condition
                records.append(dict(model_id=model, identity='east_asian', question_id='fixture', persona=persona,
                                    response_key=model + '|' + persona, analysis_surface_status='linguistic_text',
                                    word_count=len(text.split()), lexical=text, excerpt=text, excerpt_start=0,
                                    excerpt_end=len(text), hits=[], raw_final_sha256='fixture'))
        with tempfile.TemporaryDirectory() as tmp:
            render = Renderer(Path(tmp))
            render.cards(records)
            self.assertEqual(len(render.index), 6)
            first = pd.read_csv(Path(tmp) / render.index[0]['data_file'])
            self.assertEqual(first.iloc[0].glyph_fallback, 'HTML')
            self.assertNotEqual(first.iloc[2].glyph_fallback, 'HTML')

    def test_print_size_na_and_long_labels(self):
        records = []
        markers = ['culture_region_sub_saharan_african', 'culture_identity_irrelevance']
        for model in ['m1', 'm2']:
            for persona in ['neutral', 'sub_saharan_african_explicit', 'sub_saharan_african_implicit_1', 'sub_saharan_african_implicit_2', 'sub_saharan_african_implicit_3']:
                for marker in markers:
                    records.append(dict(model_id=model, persona=persona, marker_id=marker, prevalence_pct=None if persona == 'neutral' else 0, n_text=0 if persona == 'neutral' else 5))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            index = render_all({'regional_profiles': pd.DataFrame(records)}, output)
            self.assertEqual(len(index), 1)
            for path in index.svg:
                audit = audit_svg(output / path)
                self.assertAlmostEqual(audit['width_cm'], 16, places=4)
                self.assertLessEqual(audit['height_cm'], 26)
                self.assertGreaterEqual(audit['min_font_pt'], 8.5)

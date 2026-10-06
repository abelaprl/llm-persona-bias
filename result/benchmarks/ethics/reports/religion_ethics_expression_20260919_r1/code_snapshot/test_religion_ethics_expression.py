"""Protect frozen measurements, paired examples and exact highlight spans."""
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import nbformat


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        name = 'src.MAIN_PARSING.ETHICS.religion_ethics_expression'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Expression presentation adapter is missing')
        self.mod = importlib.import_module(name)

    def test_heatmap_keeps_zero_na_and_original_density(self):
        # Replacing NA with zero or pooling response densities must fail.
        table = pd.DataFrame([
            dict(model_id='m', persona=p, marker_id='religion_figure_reference',
                 prevalence_text=v, mean_response_density=d, raw_count=n,
                 linguistic_available=t, total=4, word_total=w, abstain_text=a)
            for p, v, d, n, t, w, a in [('neutral', 0., 0., 0, 2, 40, 1),
                                       ('islam_explicit', 50., 5.5, 2, 2, 110, 1),
                                       ('islam_implicit_1', np.nan, np.nan, 0, 0, 0, 0)]
        ])
        actual = self.mod.heatmap_table(table, ['religion_figure_reference'], ['m'],
                                        ['neutral', 'islam_explicit', 'islam_implicit_1'])
        self.assertEqual(actual.prevalence_pct.iloc[0], 0.)
        self.assertEqual(actual.prevalence_pct.iloc[1], 50.)
        self.assertTrue(pd.isna(actual.prevalence_pct.iloc[2]))
        self.assertEqual(actual.mean_response_density.iloc[1], 5.5)
        self.assertEqual(actual.raw_count.sum(), 2)
        self.assertEqual(actual.abstain_text.iloc[1], 1)
        with self.assertRaises(ValueError):
            self.mod.heatmap_table(pd.concat([table, table.iloc[:1]]),
                                  ['religion_figure_reference'], ['m'], ['neutral'])

    def test_missing_heatmap_cell_is_not_silently_created(self):
        table = pd.DataFrame([dict(model_id='m', persona='neutral', marker_id='religion_x',
                                   prevalence_text=0., linguistic_available=1)])
        with self.assertRaises(ValueError):
            self.mod.heatmap_table(table, ['religion_x'], ['m'], ['neutral', 'islam_explicit'])

    def test_selection_keeps_same_model_subset_question_and_allows_label_only(self):
        # Joining only question_id or filtering N to text would break this fixture.
        records = []
        for subset, score in [('justice', 2), ('virtue', 1)]:
            for persona in ['neutral', 'islam_explicit', 'islam_implicit_1', 'islam_implicit_2', 'islam_implicit_3']:
                records.append(dict(model_id='m', persona=persona, subset=subset, question_id='same',
                                    response_key=subset + persona, word_count=20,
                                    surface_status='label_only' if persona == 'neutral' else 'linguistic_text',
                                    religion_a__count=score if persona != 'neutral' else 0,
                                    religion_b__count=int(persona == 'islam_implicit_2')))
        rows = pd.DataFrame(records)
        selected, ranking = self.mod.select_card_rows(rows, ['religion_a', 'religion_b'], {'m': 'islam'})
        self.assertEqual(selected.persona.tolist(), ['neutral', 'islam_explicit', 'islam_implicit_2'])
        self.assertEqual(selected.subset.tolist(), ['justice'] * 3)
        self.assertEqual(selected.question_id.tolist(), ['same'] * 3)
        self.assertEqual(selected.surface_status.iloc[0], 'label_only')
        self.assertEqual(len(ranking), 2)
        self.assertEqual(int(ranking.selected.sum()), 1)

    def test_missing_final_counts_stay_na_and_do_not_crash_example_ranking(self):
        records = [dict(model_id='m', persona=p, subset='justice', question_id='q',
                        response_key=p, word_count=0 if missing else 10,
                        surface_status='empty_final' if missing else 'linguistic_text',
                        religion_a__count=np.nan if missing else 1)
                   for p, missing in [('neutral', True), ('islam_explicit', False),
                                      ('islam_implicit_1', True), ('islam_implicit_2', False),
                                      ('islam_implicit_3', False)]]
        selected, _ = self.mod.select_card_rows(pd.DataFrame(records), ['religion_a'], {'m': 'islam'})
        self.assertEqual(selected.persona.tolist(), ['neutral', 'islam_explicit', 'islam_implicit_2'])
        self.assertTrue(pd.isna(selected.religion_a__count.iloc[0]))
        self.assertEqual(selected.surface_status.iloc[0], 'empty_final')

    def test_highlights_preserve_overlap_raw_text_and_reject_bad_offsets(self):
        text = 'A & Torah teaching.'
        hits = [dict(raw_start=4, raw_end=9, raw_matched_text='Torah',
                     marker_id='religion_scripture_reference'),
                dict(raw_start=4, raw_end=18, raw_matched_text='Torah teaching',
                     marker_id='religion_moral_authority_framing')]
        segments = self.mod.highlight_segments(text, hits)
        self.assertEqual(''.join(s['text'] for s in segments), text)
        overlap = next(s for s in segments if s['text'] == 'Torah')
        self.assertEqual(set(overlap['markers']),
                         {'religion_scripture_reference', 'religion_moral_authority_framing'})
        self.assertEqual(overlap['group'], 'overlap')
        html = self.mod.highlight_html(text, hits)
        self.assertIn('A &amp; ', html)
        self.assertIn('religion_scripture_reference', html)
        with self.assertRaises(ValueError):
            self.mod.highlight_segments(text, [dict(hits[0], raw_matched_text='wrong')])

    def test_excerpt_is_contiguous_and_keeps_whole_highlight(self):
        text = ('word ' * 60) + 'Torah' + (' more' * 60)
        hit = dict(raw_start=300, raw_end=305, raw_matched_text='Torah', marker_id='religion_scripture_reference')
        start, end = self.mod.excerpt_window(text, [hit], max_words=45)
        self.assertGreater(start, 0)
        self.assertLess(end, len(text))
        self.assertLessEqual(start, 300)
        self.assertGreaterEqual(end, 305)
        self.assertLessEqual(len(text[start:end].split()), 45)
        self.assertEqual(self.mod.excerpt_window('0', [], max_words=45), (0, 1))

    def test_visuals_export_full_cells_and_paired_card_provenance(self):
        # Dropping a marker or stripping unhighlighted text must break exported evidence.
        name = 'src.MAIN_PARSING.ETHICS.religion_ethics_expression_plots'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Expression renderer is missing')
        plots = importlib.import_module(name)
        from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import MODELS, CONDITIONS
        records = [dict(model_id=m, persona=p, marker_id=marker, prevalence_pct=0.,
                        linguistic_available=2, mean_response_density=0., raw_count=0)
                   for m in MODELS for p in ['neutral'] + ['islam_' + c for c in CONDITIONS]
                   for marker in self.mod.RELIGION_MARKERS]
        cards = [dict(card_id='card_01', model_id=MODELS[0], card_identity='islam', subset='justice',
                      question_id='fixture', persona=p, final_output='A & Torah teaching.', word_count=4,
                      answer_status='abstain', surface_status='linguistic_text', response_key=p,
                      source_file='fixture.csv', source_row=i, raw_final_sha256='fixture',
                      hits=[dict(raw_start=4, raw_end=9, raw_matched_text='Torah',
                                 marker_id='religion_scripture_reference', rule_id='fixture')])
                 for i, p in enumerate(['neutral', 'islam_explicit', 'islam_implicit_1'])]
        with tempfile.TemporaryDirectory(prefix='religion_expression_') as tmp:
            result = plots.render_expression(Path(tmp), pd.DataFrame(records), cards, identities=['islam'])
            self.assertEqual(len(result), 2)
            exported = pd.read_csv(Path(tmp) / result.iloc[0].table)
            self.assertEqual(len(exported), 600)
            self.assertEqual(exported.marker_id.nunique(), 20)
            self.assertTrue((Path(tmp) / result.iloc[1].png).is_file())
            evidence = pd.read_csv(Path(tmp) / result.iloc[1].table)
            self.assertEqual(evidence.excerpt.tolist(), ['A & Torah teaching.'] * 3)
            self.assertEqual(evidence.excerpt_start.tolist(), [0] * 3)
            full_html = (Path(tmp) / 'response_cards.html').read_text(encoding='utf-8')
            self.assertIn('A &amp;', full_html)
            self.assertIn('religion_scripture_reference', full_html)
            self.assertIn('fixture.csv', full_html)

    def test_notebook_extension_keeps_existing_cells_and_uses_own_bundle(self):
        name = 'src.MAIN_PARSING.ETHICS.build_religion_ethics_expression'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Supplement builder is missing')
        builder = importlib.import_module(name)
        original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Existing'),
                                                  nbformat.v4.new_code_cell('x = 42')])
        actual = builder.extend_notebook(original, 'test_run')
        self.assertEqual(actual.cells[:2], original.cells)
        self.assertEqual(len(original.cells), 2)
        code = '\n'.join(c.source for c in actual.cells if c.cell_type == 'code')
        self.assertIn('religion_ethics_expression_test_run', code)
        self.assertNotIn('load_final_rows', code)
        self.assertNotIn('extract', code)


if __name__ == '__main__':
    unittest.main()

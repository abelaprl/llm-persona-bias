"""Protect print readability and dictionary-category diversity semantics."""
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat


class PrintTests(unittest.TestCase):
    def module(self):
        name = 'src.MAIN_PARSING.ETHICS.religion_ethics_print'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Print adapter is missing')
        return importlib.import_module(name)

    def fixture(self):
        return pd.DataFrame([
            dict(response_key='a', model_id='m', persona='neutral', subset='justice',
                 question_id='a', surface_status='linguistic_text', answer_status='abstain',
                 word_count=100, x__count=2, y__count=1),
            dict(response_key='b', model_id='m', persona='neutral', subset='justice',
                 question_id='b', surface_status='linguistic_text', answer_status='committed',
                 word_count=10, x__count=0, y__count=0),
            dict(response_key='c', model_id='m', persona='neutral', subset='justice',
                 question_id='c', surface_status='label_only', answer_status='committed',
                 word_count=0, x__count=0, y__count=0),
            dict(response_key='d', model_id='m', persona='neutral', subset='justice',
                 question_id='d', surface_status='empty_final', answer_status='missing',
                 word_count=0, x__count=np.nan, y__count=np.nan),
        ])

    def test_diversity_preserves_no_hit_abstain_and_unavailable(self):
        mod = self.module()
        rows = mod.diversity_rows(self.fixture(), ['x', 'y'])
        self.assertEqual(rows.distinct_markers.iloc[:2].tolist(), [2, 0])
        self.assertTrue(rows.distinct_markers.iloc[2:].isna().all())
        self.assertEqual(rows.raw_marker_count.iloc[0], 3)
        self.assertEqual(rows.marker_density.iloc[0], 3.)
        summary, distribution, lengths = mod.diversity_tables(rows, 2)
        record = summary.iloc[0]
        self.assertEqual((record.total, record.linguistic_available, record.abstain_text), (4, 2, 1))
        self.assertEqual(record.mean_distinct_markers, 1.)
        self.assertEqual(record.mean_marker_density, 1.5)
        self.assertEqual(distribution.responses.tolist(), [1, 0, 1])
        self.assertEqual(distribution.cumulative_pct.tolist(), [50., 50., 100.])
        self.assertEqual(lengths.responses.sum(), 2)

    def test_invalid_counts_or_duplicate_keys_are_rejected(self):
        mod = self.module()
        for value in [np.nan, -1, .5]:
            rows = self.fixture()
            rows.loc[0, 'x__count'] = value
            with self.assertRaises(ValueError):
                mod.diversity_rows(rows, ['x', 'y'])
        with self.assertRaises(ValueError):
            mod.diversity_rows(pd.concat([self.fixture(), self.fixture().iloc[:1]]), ['x', 'y'])

    def test_zero_denominator_does_not_become_zero_diversity(self):
        mod = self.module()
        rows = mod.diversity_rows(self.fixture().iloc[2:].copy(), ['x', 'y'])
        summary, distribution, _ = mod.diversity_tables(rows, 2)
        self.assertTrue(pd.isna(summary.mean_distinct_markers.iloc[0]))
        self.assertTrue(distribution.cumulative_pct.isna().all())

    def test_print_pagination_keeps_values_legends_and_readable_fonts(self):
        mod = self.module()
        with tempfile.TemporaryDirectory(prefix='religion_print_') as directory:
            renderer = mod.PrintRenderer(Path(directory))
            fig, axes = plt.subplots(1, 3)
            for i, ax in enumerate(axes):
                ax.plot([0, 1], [i, i + 1], label='data')
                ax.set_title('Panel ' + str(i))
                ax.set_yticks([0, 1], ['Long marker label', 'Other marker'])
                ax.set_xlabel('Delta prevalence (poin persentase)')
            table = pd.DataFrame({'effect': [0., 1., 2.], 'p_holm': [.01, .1, np.nan]})
            renderer.finish(fig, 'fixture', 'Judul yang tetap terbaca', table, 'Caption', 'test')
            self.assertEqual(len(renderer.index), 2)
            self.assertEqual(sum(r['panels'] for r in renderer.index), 3)
            restored = pd.read_csv(Path(directory) / 'figure_tables/fixture.csv')
            np.testing.assert_allclose(restored.effect, [0, 1, 2])
            self.assertTrue(pd.isna(restored.p_holm.iloc[2]))
            texts = []
            for record in renderer.index:
                audit = mod.audit_svg(Path(directory) / record['svg'])
                self.assertAlmostEqual(audit['width_cm'], 16, places=5)
                self.assertLessEqual(audit['height_cm'], 26)
                self.assertGreaterEqual(audit['min_font_pt'], 8.5)
                self.assertEqual(audit['font_families'], ['DejaVu Sans'])
                tree = ET.parse(Path(directory) / record['svg'])
                texts.extend(''.join(e.itertext()) for e in tree.iter() if e.tag.endswith('}text'))
            for label in ['Panel 0', 'Panel 1', 'Panel 2', 'Llama 3.1 8B']:
                self.assertIn(label, texts)

    def test_expression_pages_keep_all_cells_and_exact_excerpts(self):
        mod = self.module()
        self.assertTrue(hasattr(mod, 'render_expression_print'), 'Page-sized expression views are missing')
        from src.MAIN_PARSING.ETHICS.religion_ethics_expression import RELIGION_MARKERS
        from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import MODELS, CONDITIONS
        records = [dict(model_id=m, persona=p, marker_id=k, prevalence_pct=50.,
                        linguistic_available=2, raw_count=1, mean_response_density=2.)
                   for m in MODELS for p in ['neutral'] + ['islam_' + c for c in CONDITIONS]
                   for k in RELIGION_MARKERS]
        cards = [dict(card_id='card_01', model_id=MODELS[0], card_identity='islam', subset='justice',
                      question_id='fixture', persona=p, final_output='A & Torah teaching.', word_count=4,
                      answer_status='abstain', surface_status='linguistic_text', response_key=p,
                      raw_final_sha256='fixture', source_file='fixture.csv', source_row=i,
                      hits=[dict(raw_start=4, raw_end=9, raw_matched_text='Torah',
                                 marker_id='religion_scripture_reference', rule_id='fixture')])
                 for i,p in enumerate(['neutral', 'islam_explicit', 'islam_implicit_1'])]
        with tempfile.TemporaryDirectory(prefix='religion_print_expression_') as directory:
            root = Path(directory)
            (root / 'figures').mkdir()
            (root / 'figure_tables').mkdir()
            index = mod.render_expression_print(root, pd.DataFrame(records), cards, identities=['islam'])
            self.assertEqual(len(index), 4)
            profiles = pd.concat([pd.read_csv(root / r['table']) for r in index[:3]])
            self.assertEqual(len(profiles), 600)
            self.assertEqual(len(profiles.drop_duplicates(['model_id','persona','marker_id'])), 600)
            self.assertEqual(profiles.prevalence_pct.unique().tolist(), [50.])
            examples = pd.read_csv(root / index[-1]['table'])
            self.assertEqual(examples.excerpt.tolist(), ['A & Torah teaching.'] * 3)
            for record in index:
                audit = mod.audit_svg(root / record['svg'])
                self.assertAlmostEqual(audit['width_cm'], 16., places=5)
                self.assertGreaterEqual(audit['min_font_pt'], 8.5)

    def test_ecdf_exports_zero_bin_and_does_not_rescale_percent(self):
        mod = self.module()
        self.assertTrue(hasattr(mod, 'render_diversity'), 'Diversity distribution view is missing')
        from src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data import MODELS
        rows = self.fixture()
        rows['model_id'] = MODELS[0]
        summary, distribution, lengths = mod.diversity_tables(mod.diversity_rows(rows, ['x','y']), 2)
        with tempfile.TemporaryDirectory(prefix='religion_print_ecdf_') as directory:
            root = Path(directory)
            (root / 'figures').mkdir()
            (root / 'figure_tables').mkdir()
            index = mod.render_diversity(root, distribution, marker_count=2, identities=['islam'])
            table = pd.read_csv(root / index[0]['table'])
            self.assertEqual(table.cumulative_pct.tolist(), [50.,50.,100.])
            self.assertEqual(table.responses.sum(), 2)

    def test_notebook_revision_preserves_existing_analysis_and_uses_flat_figures(self):
        name = 'src.MAIN_PARSING.ETHICS.build_religion_ethics_print'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Print notebook builder is missing')
        builder = importlib.import_module(name)
        original = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell('User analysis remains'),
            nbformat.v4.new_code_cell('def show_section(section):\n    pass\n'),
            nbformat.v4.new_code_cell("SUPPLEMENT = ROOT / 'original'\ndef show_expression(section):\n    pass\n\nshow_expression('expression_heatmaps')"),
            nbformat.v4.new_markdown_cell('## 6. Existing section'),
            nbformat.v4.new_code_cell('user_calculation = 42'),
        ])
        result = builder.prepare_print_notebook(original, 'result/print_fixture')
        self.assertEqual(result.cells[0].source, 'User analysis remains')
        self.assertEqual(result.cells[-1].source, 'user_calculation = 42')
        self.assertEqual(len(original.cells), 5)
        namespace = {}
        # Execute the preserved user computation, not an assertion on its spelling.
        exec(result.cells[-1].source, namespace)
        self.assertEqual(namespace['user_calculation'], 42)
        self.assertEqual(result.metadata['print_layout']['width_cm'], 16)
        self.assertTrue(any('## 5D.' in c.source for c in result.cells))

    def test_stale_local_picture_falls_back_to_verified_bundle(self):
        from src.MAIN_PARSING.ETHICS import build_religion_ethics_print as builder
        self.assertTrue(hasattr(builder, 'choose_figure'), 'Hash-aware fallback is missing')
        with tempfile.TemporaryDirectory(prefix='religion_print_fallback_') as directory:
            root = Path(directory)
            local, bundle = root / 'local.png', root / 'bundle.png'
            local.write_bytes(b'old visual')
            bundle.write_bytes(b'verified visual')
            digest = hashlib.sha256(b'verified visual').hexdigest()
            self.assertEqual(builder.choose_figure(local, bundle, digest), bundle)
            local.write_bytes(b'verified visual')
            self.assertEqual(builder.choose_figure(local, bundle, digest), local)
            local.write_bytes(b'bad')
            bundle.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                builder.choose_figure(local, bundle, digest)

    def test_real_card_excerpts_fit_without_shortening_or_font_shrink(self):
        mod = self.module()
        root = Path(__file__).resolve().parents[4]
        source = root / 'result/ETHICS/reports/religion_ethics_expression_20260919_r1/response_cards.json'
        if not source.exists():
            self.skipTest('Frozen research fixtures are not installed')
        cards = json.loads(source.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(prefix='religion_print_real_cards_') as directory:
            target = Path(directory)
            (target / 'figures').mkdir()
            (target / 'figure_tables').mkdir()
            index = mod.render_expression_print(target, pd.DataFrame(), cards, identities=[])
            self.assertEqual(len(index), 6)
            for record in index:
                expected = pd.read_csv(source.parent / 'figure_tables' / (record['parent_id'] + '.csv'))
                actual = pd.read_csv(target / record['table'])
                self.assertEqual(actual.excerpt.tolist(), expected.excerpt.tolist())
                self.assertGreaterEqual(mod.audit_svg(target / record['svg'])['min_font_pt'], 8.5)
                if 'card_05' in record['figure_id']:
                    self.assertLess(mod.audit_svg(target / record['svg'])['height_cm'], 20)
                    self.assertIn('label_only', (target / record['svg']).read_text(encoding='utf-8'))

    def test_cache_rejects_missing_pages_even_if_parent_csv_is_complete(self):
        from src.MAIN_PARSING.ETHICS import build_religion_ethics_print as builder
        self.assertTrue(hasattr(builder, 'validate_main_cache'), 'Cache page completeness check is missing')
        source = pd.DataFrame([dict(figure_id='overview', section='verbosity')])
        cache = pd.DataFrame([dict(figure_id=f'overview_p{i:02d}', parent_id='overview',
                                   section='verbosity', page=i, panels=2) for i in [1,2]])
        builder.validate_main_cache(cache, source)
        with self.assertRaises(ValueError):
            builder.validate_main_cache(cache.iloc[:1], source)
        broken = cache.copy()
        broken.loc[1, 'panels'] = 1
        with self.assertRaises(ValueError):
            builder.validate_main_cache(broken, source)


if __name__ == '__main__':
    unittest.main()

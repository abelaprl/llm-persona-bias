"""Guard denominators and source inference while changing presentation."""
import importlib
import unittest
from pathlib import Path
import re

import numpy as np
import pandas as pd
import nbformat


class PresentationDataTests(unittest.TestCase):
    def setUp(self):
        module = 'src.MAIN_PARSING.ETHICS.religion_ethics_gsm8k_style_data'
        self.assertIsNotNone(importlib.util.find_spec(module), 'Presentation adapter is missing')
        self.adapter = importlib.import_module(module)

    def test_coverage_keeps_abstain_text_and_excludes_label_only_from_words(self):
        rows = pd.DataFrame([
            ['m', 'neutral', 'linguistic_text', 'committed', 10],
            ['m', 'neutral', 'linguistic_text', 'abstain', 30],
            ['m', 'neutral', 'label_only', 'committed', 1],
            ['m', 'neutral', 'empty_final', 'missing', 0],
        ], columns=['model_id', 'persona', 'surface_status', 'answer_status', 'word_count'])
        result = self.adapter.coverage_table(rows).iloc[0]
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['linguistic_available'], 2)
        self.assertEqual(result['abstain_text'], 1)
        self.assertEqual(result['word_total'], 40)
        self.assertEqual(result['median_words'], 20)
        self.assertEqual(result['coverage_pct'], 50)

    def test_contrast_selection_preserves_all_conditions_direction_and_na(self):
        rows = []
        for condition in ['explicit', 'implicit_1', 'implicit_2', 'implicit_3']:
            rows.append(dict(model_id='m', left_persona='islam_' + condition,
                             right_persona='neutral', marker_id='stance_hedge',
                             contrast_kind='vs_neutral', population='text_all', metric='density',
                             effect=-2.5, ci_low=np.nan, ci_high=np.nan, p_holm=0.03, n_pairs=7))
        rows.append(dict(rows[0], population='both_committed_text', effect=99))
        result = self.adapter.select_contrasts(pd.DataFrame(rows), 'vs_neutral', ['stance_hedge'], 'density')
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result.condition), {'explicit', 'implicit_1', 'implicit_2', 'implicit_3'})
        self.assertEqual(result.effect.tolist(), [-2.5] * 4)
        self.assertTrue(result.ci_low.isna().all())
        self.assertEqual(result.n_pairs.tolist(), [7] * 4)

    def test_sensitivity_joins_identity_and_condition_without_cross_pairs(self):
        base = dict(model_id='m', marker_id='x', contrast_kind='vs_neutral',
                    right_persona='neutral', metric='density', n_pairs=10)
        rows = pd.DataFrame([
            dict(base, left_persona='islam_explicit', population='text_all', effect=2.),
            dict(base, left_persona='jewish_explicit', population='text_all', effect=-3.),
            dict(base, left_persona='islam_explicit', population='both_committed_text', effect=-1.),
            dict(base, left_persona='jewish_explicit', population='both_committed_text', effect=np.nan),
        ])
        result = self.adapter.sensitivity_table(rows)
        self.assertEqual(len(result), 2)
        islam = result[result.left_persona.eq('islam_explicit')].iloc[0]
        self.assertEqual(islam.effect_main, 2.)
        self.assertEqual(islam.effect_sensitivity, -1.)
        self.assertTrue(islam.direction_reversed)
        jewish = result[result.left_persona.eq('jewish_explicit')].iloc[0]
        self.assertTrue(pd.isna(jewish.effect_sensitivity))
        self.assertFalse(jewish.evaluable)

    def test_term_density_uses_response_mean_and_deduplicates_same_span(self):
        hits = pd.DataFrame([
            ['a', 'm', 'islam_explicit', 0, 3, 'God', 'religion_deity_invocation'],
            ['a', 'm', 'islam_explicit', 0, 3, 'God', 'religion_generic_reference'],
            ['b', 'm', 'islam_explicit', 0, 3, 'God', 'religion_deity_invocation'],
        ], columns=['response_key', 'model_id', 'persona', 'raw_start', 'raw_end', 'matched_text', 'marker_id'])
        rows = pd.DataFrame([
            ['a', 'm', 'islam_explicit', 'linguistic_text', 10],
            ['b', 'm', 'islam_explicit', 'linguistic_text', 100],
            ['c', 'm', 'islam_explicit', 'linguistic_text', 100],
        ], columns=['response_key', 'model_id', 'persona', 'surface_status', 'word_count'])
        result = self.adapter.term_table(hits, rows).iloc[0]
        self.assertEqual(result.raw_count, 2)
        self.assertEqual(result.n_text, 3)
        self.assertAlmostEqual(result.mean_response_density, 11 / 3)
        self.assertEqual(result.term, 'god')

    def test_examples_can_cover_model_missing_from_targeted_package(self):
        packages = {'targeted.json': [dict(model_id='m', final_output='First')],
                    'open_round_1.json': [dict(model_id='n', final_output='Second')]}
        result = self.adapter.select_examples(packages, models=['m', 'n'])
        self.assertEqual([r['final_output'] for r in result], ['First', 'Second'])
        self.assertEqual(result[1]['example_source_file'], 'open_round_1.json')

    def test_archiving_preserves_html_link_targets_and_rerun_document_location(self):
        from src.MAIN_PARSING.ETHICS import religion_ethics_gsm8k_style_notebook as presentation
        self.assertTrue(hasattr(presentation, 'rebase_notebook'), 'Notebook relocation is missing')
        root = Path('D:/TugasAkhir')
        old = root / 'notebooks/religion/linguistic marker/ETHICS'
        new = old / 'archive/example'
        cell = nbformat.v4.new_code_cell("DOCUMENT_DIR = ROOT / 'notebooks/religion/linguistic marker/ETHICS'")
        cell.outputs = [nbformat.v4.new_output('display_data', data={'text/html': '<a href="figures/a.png">PNG</a>'})]
        notebook = nbformat.v4.new_notebook(cells=[cell])
        moved = presentation.rebase_notebook(notebook, old, new, root)
        link = re.search('href="([^"]+)"', moved.cells[0].outputs[0].data['text/html']).group(1)
        self.assertEqual((new / link).resolve(), (old / 'figures/a.png').resolve())
        namespace = {'ROOT': root}
        exec(moved.cells[0].source, namespace)
        self.assertEqual(namespace['DOCUMENT_DIR'], new)


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
from src.benchmarks.ethics.regional_ethics_presentation_view import classify_surface, engine_frame, coverage_table, apply_pairing_exclusions, paired_engine_frame


class ViewTests(unittest.TestCase):
    def test_exclusion_only_affects_pairs_not_original_or_descriptive(self):
        source = pd.DataFrame({'response_key': ['m|neutral|q', 'm|explicit|q'],
            'text_available': [True, True], 'explanation_available': [True, True],
            'analysis_answer_status': ['abstain', 'committed'], 'word_count': [100, 50],
            'a__count': [2, 1], 'a__density': [2., 2.]})
        view = apply_pairing_exclusions(source, {'m|neutral|q'})
        self.assertEqual(len(view), 2)
        self.assertEqual(view.explanation_available.tolist(), [True, True])
        self.assertEqual(view.a__count.tolist(), [2, 1])
        descriptive = engine_frame(view, pd.DataFrame())
        paired = paired_engine_frame(view, pd.DataFrame())
        self.assertEqual(descriptive.text_available.tolist(), [True, True])
        self.assertEqual(paired.text_available.tolist(), [False, True])
        self.assertNotIn('pairing_eligible', source)

    def test_unknown_exclusion_key_rejected(self):
        with self.assertRaises(ValueError):
            apply_pairing_exclusions(pd.DataFrame({'response_key': ['known']}), {'unknown'})

    def test_binary_not_explanation(self):
        for text in ['0', '1', '**1**', 'Answer: 0', 'Final answer: 1.']:
            self.assertEqual(classify_surface(text, text, 'available'), 'label_only')
        self.assertEqual(classify_surface('', '', 'missing'), 'missing')
        self.assertEqual(classify_surface('0.5', '0.5', 'available'), 'technical_only')
        self.assertEqual(classify_surface('I am not sure.', 'I am not sure.', 'available'), 'linguistic_text')
        self.assertEqual(classify_surface('Because it hurts.', 'Because it hurts.', 'technical_only'), 'technical_only')

    def test_mask_copy_keeps_abstain(self):
        view = pd.DataFrame({'text_available': [True, True], 'explanation_available': [False, True],
                             'analysis_answer_status': ['committed', 'abstain'], 'word_count': [1, 10],
                             'a__count': [0, 0], 'a__density': [0., 0.], 'a__present': [False, False]})
        result = engine_frame(view, pd.DataFrame())
        self.assertTrue(pd.isna(result.loc[0, 'a__count']))
        self.assertEqual(result.loc[1, 'a__count'], 0)
        self.assertTrue(result.loc[1, 'text_available'])
        self.assertEqual(view.loc[0, 'word_count'], 1)

import unittest
import pandas as pd
from src.benchmarks.ethics.regional_ethics_expression_evidence import validate_hit, context_flags, excerpt_window, highlighted_html, select_cards, scenario_for_response


class EvidenceTests(unittest.TestCase):
    def test_actual_scenario_override_is_response_specific(self):
        scenarios = {'q': 'Dataset scenario'}
        overrides = {'m|neutral|q': 'Actual different scenario'}
        self.assertEqual(scenario_for_response('m|neutral|q', 'q', scenarios, overrides), 'Actual different scenario')
        self.assertEqual(scenario_for_response('m|explicit|q', 'q', scenarios, overrides), 'Dataset scenario')

    def test_cards_do_not_pair_mismatched_scenarios(self):
        records = []
        marker = 'culture_region_east_asian'
        for q, count in [('q1', 10), ('q2', 1)]:
            for persona in ['neutral', 'east_asian_explicit', 'east_asian_implicit_1']:
                records.append(dict(model_id='m', persona=persona, subset='justice', question_id=q,
                                    response_key=f'm|{persona}|{q}', text_available=True,
                                    pairing_eligible=not (q == 'q1' and persona == 'neutral'),
                                    analysis_surface_status='linguistic_text', word_count=100, **{marker + '__count': count}))
        cards = select_cards(pd.DataFrame(records), pd.DataFrame(), [marker])
        self.assertEqual(set(cards.question_id), {'q2'})

    def test_surface_offsets(self):
        hit = dict(start=0, end=10, matched_text='East Asian')
        validate_hit('East Asian', hit)
        with self.assertRaises(ValueError):
            validate_hit('**East Asian**', hit)

    def test_overlap_independent_unknown_not_false(self):
        self.assertEqual(context_flags('East Asian', 'An East Asian person', 'East Asian food'),
                         dict(persona_literal_overlap=True, scenario_literal_overlap=True))
        self.assertIsNone(context_flags('x', None, '')['persona_literal_overlap'])

    def test_excerpt_is_contiguous(self):
        text = ' '.join(str(i) for i in range(200))
        start, end = excerpt_window(text, [dict(start=390, end=393)], 100)
        self.assertLessEqual(len(text[start:end].split()), 100)
        self.assertIn('100', text[start:end])

    def test_html_escapes_and_overlapping_markers(self):
        text = '<x> East Asian'
        hit = dict(start=4, end=14, matched_text='East Asian', marker_id='a')
        out = highlighted_html(text, [hit, dict(hit, marker_id='b')])
        self.assertIn('&lt;x&gt;', out)
        self.assertIn('a, b', out)
        self.assertEqual(out.count('<mark'), 1)

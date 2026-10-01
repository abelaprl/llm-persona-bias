import unittest
import pandas as pd
from src.benchmarks.ethics.regional_ethics_presentation_data import validate_rows, validate_gate, join_catalog, persona_parts


class DataTests(unittest.TestCase):
    def test_duplicate_response_is_rejected(self):
        row = dict(model_id='m', persona='neutral', subset='justice', question_id='q1', response_key='m|neutral|q1')
        with self.assertRaises(ValueError):
            validate_rows(pd.DataFrame([row, row]))

    def test_incorrect_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(pd.DataFrame([dict(model_id='m', persona='neutral', subset='justice', question_id='q1', response_key='wrong')]))

    def test_gate_not_truthy_string(self):
        with self.assertRaises(ValueError):
            validate_gate({'release_eligible': 'False'})
        self.assertFalse(validate_gate({'release_eligible': False, 'human_review_complete': 'False'})['human_review_complete'])

    def test_persona_underscore_identity(self):
        self.assertEqual(persona_parts('sub_saharan_african_implicit_3'), ('sub_saharan_african', 'implicit_3'))
        with self.assertRaises(ValueError):
            persona_parts('islam_explicit')

    def test_catalog_incomplete_rejected(self):
        with self.assertRaises(ValueError):
            join_catalog(pd.DataFrame([{'marker_id': 'x'}]), pd.DataFrame())

import inspect
import unittest

from src.MAIN_PARSING.GSM8K.Religion_Persona import (
    rescue_uncommitted,
    rules_llama31_8b,
)


class ReligionRescueTests(unittest.TestCase):
    def row(self, **updates):
        base = {
            'model': 'model',
            'persona': 'persona',
            'file': 'persona/example.json',
            'answer_status': 'abstain',
            'abstain_reason': 'candidate_only',
            'candidate_answer': '42',
            'response_content_full': 'An intermediate candidate is 42.',
            'technical_failure': 'False',
            'ground_truth_answer': '42',
            'finish_reason': 'stop',
            'content_empty': 'False',
        }
        base.update(updates)
        return base

    def test_promotes_existing_candidate_without_ground_truth_selection(self):
        row, manifest = rescue_uncommitted.apply_rescue(
            self.row(), rules_llama31_8b
        )
        self.assertEqual(row['answer_status'], 'committed')
        self.assertEqual(row['extracted_answer'], 42)
        self.assertEqual(row['priority_hit'], 'existing_candidate')
        self.assertTrue(row['is_correct'])
        self.assertIsNotNone(manifest)
        self.assertNotIn('ground', inspect.signature(
            rescue_uncommitted.choose_rescue
        ).parameters)

    def test_no_final_answer_uses_recorded_last_numeric_token(self):
        row, manifest = rescue_uncommitted.apply_rescue(
            self.row(
                abstain_reason='no_final_answer',
                candidate_answer='',
                response_content_full='The first amount is 10. It ends at 17.',
                ground_truth_answer='17',
            ),
            rules_llama31_8b,
        )
        self.assertEqual(row['extracted_answer'], 17)
        self.assertIsNotNone(manifest)

    def test_empty_truncation_remains_unresolved(self):
        original = self.row(
            abstain_reason='truncated', candidate_answer='',
            response_content_full='', content_empty='True', finish_reason='length',
        )
        row, manifest = rescue_uncommitted.apply_rescue(
            original, rules_llama31_8b
        )
        self.assertEqual(row, original)
        self.assertIsNone(manifest)

    def test_technical_failure_is_never_promoted(self):
        original = self.row(
            technical_failure='True',
            response_content_full='Answer: 42',
        )
        row, manifest = rescue_uncommitted.apply_rescue(
            original, rules_llama31_8b
        )
        self.assertEqual(row, original)
        self.assertIsNone(manifest)

    def test_existing_committed_row_is_never_changed(self):
        original = self.row(
            answer_status='committed', abstain_reason='',
            extracted_answer='11', candidate_answer='',
        )
        row, manifest = rescue_uncommitted.apply_rescue(
            original, rules_llama31_8b
        )
        self.assertEqual(row, original)
        self.assertIsNone(manifest)


if __name__ == '__main__':
    unittest.main()

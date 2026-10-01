import unittest
import pandas as pd
from src.benchmarks.ethics.run_religion_lm_v3 import extract_batch, verify_audit_bundle


class PipelineTests(unittest.TestCase):
    def test_metadata_does_not_affect_extraction(self):
        a = dict(model_id='a',persona='neutral',subset='justice',question_id='q',final_output='As an AI, I can discuss religion.',answer_status='abstain',response_key='a')
        b = dict(a,model_id='b',persona='jewish_explicit',response_key='b',ground_truth='SECRET')
        (ra,ha),(rb,hb) = extract_batch([a,b])
        cols = [c for c in ra if '__' in c]
        self.assertEqual({c:ra[c] for c in cols},{c:rb[c] for c in cols})
        self.assertNotIn('ground_truth',rb)
        self.assertEqual(len(ha),len(hb))

    def test_missing_human_audit_rejects_validated_run(self):
        with self.assertRaises(ValueError):
            verify_audit_bundle(None)


if __name__=='__main__':
    unittest.main()

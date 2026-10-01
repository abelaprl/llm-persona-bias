import unittest
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_discovery import build_discovery_queue, scan_observed_forms


class DiscoveryTests(unittest.TestCase):
    def test_queue_development_only_and_no_hit_required(self):
        rows = pd.DataFrame([dict(response_key=str(i), model_id='m', persona='neutral', subset='justice', question_id=str(i), final_output='No registered marker here.') for i in range(10)])
        split = rows[['subset','question_id']].copy()
        split['partition'] = ['development']*5+['held_out']*5
        queue = build_discovery_queue(rows,split,{'0'}, {'per_model':20}, 18222008)
        self.assertEqual(set(queue.response_key), {'1','2','3','4'})

    def test_probe_response_and_occurrence_counts_differ(self):
        rows = pd.DataFrame([dict(model_id='m',persona='neutral',subset='justice',final_output='niyyah niyyah')])
        result = scan_observed_forms(rows, {'p':{'pattern':r'\bniyyah\b','sample_id':'R069'}})
        self.assertEqual(result.response_hits.iloc[0],1)
        self.assertEqual(result.occurrence_hits.iloc[0],2)


if __name__ == '__main__':
    unittest.main()

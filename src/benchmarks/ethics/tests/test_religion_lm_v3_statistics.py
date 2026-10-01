import unittest
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_statistics import build_pairs, bootstrap_effects, holm_adjust


class StatisticsTests(unittest.TestCase):
    def test_pairing_preserves_subset(self):
        rows = pd.DataFrame([dict(model_id='m',persona=p,subset=s,question_id='same',v=v) for p,s,v in [('a','justice',3),('b','justice',1),('a','virtue',9),('b','virtue',5)]])
        pairs = build_pairs(rows,'a','b')
        self.assertEqual(len(pairs),2)
        self.assertEqual(list(pairs.v_left-pairs.v_right),[2,4])

    def test_effect_direction_and_equal_subset(self):
        d = np.array([[1.],[1.],[1.],[9.],[9.]])
        a = bootstrap_effects(d, np.array(['a','a','a','b','b']), 123, 100, True)
        b = bootstrap_effects(-d,np.array(['a','a','a','b','b']),123,100,True)
        self.assertAlmostEqual(a['effect'][0],5)
        self.assertAlmostEqual(a['effect'][0],-b['effect'][0])
        self.assertAlmostEqual(a['ci_low'][0],-b['ci_high'][0])

    def test_single_question_subset_has_no_inferential_interval(self):
        result = bootstrap_effects(np.array([[1.],[9.]]),np.array(['a','b']),123,100,True)
        self.assertTrue(np.isnan(result['ci_low'][0]))

    def test_holm_with_missing(self):
        result = holm_adjust(np.array([.01,.04,np.nan]))
        np.testing.assert_allclose(result[:2],[.02,.04])
        self.assertTrue(np.isnan(result[2]))


if __name__ == '__main__':
    unittest.main()

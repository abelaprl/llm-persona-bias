import unittest
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_gold import evaluate_against_gold, reviewer_agreement

class GoldTests(unittest.TestCase):
    def test_false_negative_and_empty_positive_not_perfect_recall(self):
        gold = pd.DataFrame({'audit_id':['a','b','c'],'marker_id':['x']*3,'present':[1,1,0]})
        pred = gold.copy(); pred['present'] = [1,0,1]
        design = pd.DataFrame({'audit_id':['a','b','c'],'question_id':['a','b','c'],'subset':['s']*3,'inclusion_probability':[.5]*3})
        result = evaluate_against_gold(pred,gold,design,resamples=100)['markers'][0]
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,1))
        self.assertEqual(result['recall'],.5)
        gold['present'] = 0
        self.assertIsNone(evaluate_against_gold(pred,gold,design,resamples=100)['markers'][0]['recall'])

    def test_incomplete_annotation_is_rejected(self):
        gold = pd.DataFrame({'audit_id':['a'],'marker_id':['x'],'present':[None]})
        with self.assertRaises(ValueError):
            evaluate_against_gold(gold,gold,pd.DataFrame())

    def test_independent_reviewers_agreement(self):
        a = pd.DataFrame({'audit_id':['a','b'],'marker_id':['x','x'],'present':[1,0],'reviewer_id':['one']*2})
        b = a.copy(); b['reviewer_id']='two'
        self.assertEqual(reviewer_agreement(a,b)['cohen_kappa'],1)
        with self.assertRaises(ValueError):
            reviewer_agreement(a,a)

if __name__=='__main__': unittest.main()

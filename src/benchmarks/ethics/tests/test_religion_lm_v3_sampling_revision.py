import unittest
import pandas as pd
from src.benchmarks.ethics.religion_v3_sampling import sample_recall

class SamplingRevisionTests(unittest.TestCase):
    def test_independent_cell_streams_and_every_subset(self):
        rows = pd.DataFrame([dict(model_id=m,persona=p,subset=s,question_id=s+str(i),response_key=f'{m}|{p}|{s}|{i}',final_output='0')
            for m in ('a','b','c') for p in ('neutral','islam_explicit') for s in ('commonsense','deontology','justice','virtue') for i in range(30)])
        split = rows[['subset','question_id']].drop_duplicates();split['partition']='held_out'
        result = sample_recall(rows,split)
        self.assertEqual(len(result),60)
        self.assertGreater(result.question_id.nunique(),10)
        self.assertTrue(result.groupby(['model_id','persona']).subset.nunique().eq(4).all())
        self.assertTrue(result.inclusion_probability.gt(0).all())
        self.assertEqual(result.to_dict('records'),sample_recall(rows,split).to_dict('records'))

    def test_development_is_ineligible(self):
        rows = pd.DataFrame([dict(model_id='a',persona='neutral',subset=s,question_id=s+str(i),response_key=f'{s}|{i}',final_output='0')
            for s in ('commonsense','deontology','justice','virtue') for i in range(10)])
        split = rows[['subset','question_id']].copy();split['partition']='held_out';split.loc[split.question_id.str.endswith('0'),'partition']='development'
        result = sample_recall(rows,split)
        self.assertFalse(result.question_id.str.endswith('0').any())

if __name__=='__main__': unittest.main()

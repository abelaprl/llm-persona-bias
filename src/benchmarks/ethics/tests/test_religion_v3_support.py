import unittest
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_v3_support import paired_marker_support


class PairSupportTests(unittest.TestCase):
    def test_density_is_response_mean_and_equal_subset_is_explicit(self):
        pair = pd.DataFrame({'subset':['justice','justice','virtue'],
            'word_count_left':[10,100,10], 'word_count_right':[20,20,20],
            'x__count_left':[1,1,3], 'x__count_right':[1,0,0]})
        plain = paired_marker_support(pair,['x'],False)
        equal = paired_marker_support(pair,['x'],True)
        self.assertAlmostEqual(plain.loc[0,'density_left'],(10+1+30)/3)
        self.assertAlmostEqual(equal.loc[0,'density_left'],(5.5+30)/2)
        self.assertEqual(plain.loc[0,'raw_count_left'],5)
        self.assertEqual(plain.loc[0,'word_total_left'],120)

    def test_label_only_has_zero_presence_and_no_density(self):
        pair = pd.DataFrame({'subset':['justice'], 'word_count_left':[0],
            'word_count_right':[10], 'x__count_left':[0], 'x__count_right':[1]})
        row = paired_marker_support(pair,['x'],False).iloc[0]
        self.assertEqual(row.prevalence_left,0)
        self.assertTrue(np.isnan(row.density_left))

    def test_equal_subset_degeneracy_is_checked_within_each_stratum(self):
        pair=pd.DataFrame({'subset':['justice','justice','virtue','virtue'],
            'word_count_left':[10]*4,'word_count_right':[10]*4,
            'x__count_left':[1,1,0,0],'x__count_right':[0,0,1,1]})
        pooled=paired_marker_support(pair,['x'],False).iloc[0]
        equal=paired_marker_support(pair,['x'],True).iloc[0]
        self.assertFalse(pooled.constant_density_difference)
        self.assertTrue(equal.constant_density_difference)
        self.assertTrue(equal.constant_prevalence_difference)

    def test_support_reconciles_all_contrasts_and_populations(self):
        from functools import lru_cache
        from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
        from src.benchmarks.ethics.religion_lm_v3_registry import get_registry
        from src.benchmarks.ethics.religion_lm_v3_statistics import IDENTITIES,CONDITIONS,estimate_contrasts
        from src.benchmarks.ethics.religion_v3_support import augment_contrasts
        @lru_cache(None)
        def extract(text):return extract_final(text)[0]
        personas=['neutral']+[i+'_'+c for i in IDENTITIES for c in CONDITIONS]
        texts=['','1','Compassion and fairness matter.','As a helpful AI, I can discuss harm and suffering.']
        rows=[]
        for i,persona in enumerate(personas):
            for subset in ('commonsense','deontology','justice','virtue'):
                for q in range(4):
                    row=dict(extract(texts[(i+q)%4]),model_id='m',persona=persona,subset=subset,
                        question_id=str(q),answer_status='abstain' if q%2 else 'committed')
                    rows.append(row)
        frame=pd.DataFrame(rows)
        contrasts=estimate_contrasts(frame,resamples=20)
        contrasts['inference_scope_status']='available'
        result=augment_contrasts(frame,contrasts,[m.marker_id for m in get_registry()])
        self.assertEqual(len(result),52*9*59)
        finite=result.effect.notna()
        np.testing.assert_allclose(result.loc[finite,'effect'],(result.estimate_left-result.estimate_right).loc[finite],atol=1e-12)


if __name__=='__main__':unittest.main()

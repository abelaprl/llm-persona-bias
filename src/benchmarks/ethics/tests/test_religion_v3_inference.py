import unittest
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_v3_inference import repair_inference,holm_planned,exact_mcnemar


class InferenceTests(unittest.TestCase):
    def row(self,**changes):
        result=dict(metric='prevalence_text',population='both_committed_text',n_pairs=2,effect=100.,
            ci_low=100.,ci_high=100.,p_raw=1/10001,p_holm=.0108,inference_scope_status='available',
            discordant_left_only=2,discordant_right_only=0,constant_density_difference=True,
            constant_prevalence_difference=True)
        result.update(informative_density_pairs=2,informative_prevalence_pairs=2,
            density_sign_flip_p=.5,density_sign_flip_method='exact_paired_sign_flip',
            prevalence_sign_flip_p=.5,prevalence_sign_flip_method='exact_paired_sign_flip')
        return dict(result,**changes)

    def test_two_discordant_pairs_are_not_strong_evidence(self):
        result=repair_inference(pd.DataFrame([self.row()])).iloc[0]
        self.assertEqual(result.p_raw,.5)
        self.assertTrue(np.isnan(result.ci_low))
        self.assertEqual(result.p_method,'exact_paired_McNemar')

    def test_constant_density_does_not_get_artificial_precision(self):
        result=repair_inference(pd.DataFrame([self.row(metric='density',effect=1.,ci_low=1.,ci_high=1.)])).iloc[0]
        self.assertEqual(result.p_raw,.5)
        self.assertTrue(np.isnan(result.ci_high))

    def test_zero_discordance_has_no_evidence_but_no_zero_width_interval(self):
        result=repair_inference(pd.DataFrame([self.row(n_pairs=100,effect=0.,ci_low=0.,ci_high=0.,
            discordant_left_only=0,discordant_right_only=0)])).iloc[0]
        self.assertEqual(result.p_raw,1.)
        self.assertTrue(np.isnan(result.ci_low))

    def test_planned_holm_keeps_unestimable_hypothesis_in_family_size(self):
        result=holm_planned(np.array([.01,np.nan]))
        self.assertAlmostEqual(result[0],.02)
        self.assertTrue(np.isnan(result[1]))

    def test_extreme_exact_probability_is_flagged_instead_of_zero(self):
        value,floored=exact_mcnemar(1514,0)
        self.assertGreater(value,0)
        self.assertTrue(floored)

    def test_nonconstant_sparse_interval_is_suppressed(self):
        result=repair_inference(pd.DataFrame([self.row(metric='density',effect=1.5,ci_low=1.,ci_high=2.,
            constant_density_difference=False)])).iloc[0]
        self.assertEqual(result.p_raw,.5)
        self.assertTrue(np.isnan(result.ci_low))
        self.assertEqual(result.ci_status,'fewer_than_six_informative_pairs')

    def test_sign_flip_exact_matches_paired_permutation(self):
        from scipy.stats import permutation_test
        from src.benchmarks.ethics.religion_v3_randomization import paired_sign_flip
        d=np.array([[1.,1.],[2.,-2.],[0.,3.]])
        weights=np.array([.25,.25,.5])
        p,methods,n=paired_sign_flip(d,weights,seed=42)
        for i in range(2):
            expected=permutation_test((d[:,i]*weights,),lambda x:np.abs(x.sum()),
                permutation_type='samples',alternative='greater',n_resamples=np.inf).pvalue
            self.assertEqual(p[i],expected)
        self.assertEqual(p[0],.5)
        self.assertEqual(n.tolist(),[2,3])
        self.assertTrue(all(m=='exact_paired_sign_flip' for m in methods))

    def test_sign_flip_monte_carlo_is_reproducible_and_nonzero(self):
        from src.benchmarks.ethics.religion_v3_randomization import paired_sign_flip
        d=np.ones((20,2));d[:,1]=0
        first=paired_sign_flip(d,np.ones(20)/20,seed=42,resamples=200)
        second=paired_sign_flip(d,np.ones(20)/20,seed=42,resamples=200)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertGreater(first[0][0],0)
        self.assertEqual(first[0][1],1)
        self.assertEqual(first[1][0],'monte_carlo_paired_sign_flip')


if __name__=='__main__':unittest.main()

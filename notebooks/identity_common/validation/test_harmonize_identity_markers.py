import importlib.util
from pathlib import Path
import unittest
import json
import numpy as np
from scipy.stats import binomtest

path = Path(__file__).resolve().parents[3] / 'src/benchmarks/ethics/harmonize_identity_markers.py'
spec = importlib.util.spec_from_file_location('common', path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class IdentityMeasurementTests(unittest.TestCase):
    def test_hidden_reasoning_is_not_evidence(self):
        for domain in ('gender', 'religion'):
            values, spans = common.measure_one((domain, '<think>You, as a woman, should pray. May God bless you.</think>42'))
            self.assertFalse(spans)
            self.assertTrue(values['visible_available'])
            self.assertFalse(values['explanation_available'])

    def test_missing_is_not_zero(self):
        for domain in ('gender', 'religion'):
            values, _ = common.measure_one((domain, ''))
            self.assertFalse(values['visible_available'])
            for key, value in values.items():
                if key.endswith('__present'):
                    self.assertTrue(np.isnan(value), key)

    def test_scenario_pronoun_is_not_user_identity(self):
        values, _ = common.measure_one(('gender', 'She has five apples. Her brother takes two.'))
        self.assertEqual(values['feminine_pronouns__present'], 1)
        self.assertEqual(values['gender_persona_direct_address__present'], 0)
        self.assertEqual(values['gender_second_person_identity_reference__present'], 0)

    def test_identity_and_pronouns_are_separate(self):
        values, _ = common.measure_one(('gender', 'You, as a woman, can check the calculation.'))
        self.assertEqual(values['gender_persona_direct_address__present'], 1)
        self.assertEqual(values['feminine_pronouns__present'], 0)

    def test_religious_terms_are_not_automatically_perspective(self):
        values, _ = common.measure_one(('religion', 'A church has 20 chairs. Add 10 chairs to get 30.'))
        self.assertEqual(values['religion_perspective_adoption__present'], 0)
        self.assertEqual(values['religion_moral_authority_framing__present'], 0)

    def test_mcnemar_uses_discordant_pairs(self):
        a = np.array([1]*8 + [0]*12, dtype=float)
        b = np.array([0]*8 + [1]*2 + [0]*10, dtype=float)
        v = common.estimate(a,b,True,np.random.default_rng(12))
        self.assertEqual(v['n_pairs'], 20)
        self.assertEqual(v['discordant_left_only'], 8)
        self.assertEqual(v['discordant_right_only'], 2)
        self.assertAlmostEqual(v['p_raw'], binomtest(8,10,.5).pvalue)
        self.assertAlmostEqual(v['delta'], 30)

    def test_holm_retains_unavailable_hypotheses(self):
        np.testing.assert_allclose(common.holm([.01,.04,.2,np.nan]), [.04,.12,.4,np.nan])

    def test_contrast_coverage(self):
        for identities,n in [(['man','woman'],18),(['islam','christian','jewish','atheist'],52)]:
            ps = ['neutral']+[i+'_'+c for i in identities for c in ['explicit_1','implicit_1','implicit_2','implicit_3']]
            self.assertEqual(len(common.contrasts(ps)), n)

    def test_reviewed_irrelevance_responses(self):
        audit = Path(__file__).with_name('repair_validation_50.json')
        if not audit.exists():
            self.skipTest('Full-response audit fixture is shipped with the result bundle.')
        rows = json.loads(audit.read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 50)
        for row in rows:
            values, _ = common.measure_one(('religion',row['full_response']))
            self.assertEqual(values['religion_perspective_adoption__present'],0,row['source_row'])
            self.assertEqual(values['religion_identity_irrelevance__present'],1,row['source_row'])


if __name__ == '__main__':
    unittest.main()

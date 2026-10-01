import unittest
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface

class Extensions(unittest.TestCase):
    def test_observed_variants_and_provenance(self):
        row,hits = extract_final('As a neutral AI assistant, I can discuss intention (niyyah). ### **Conclusion:** Context matters.')
        for marker in ('ai_identity_statement','religion_named_concept_reference','organization_section_header'):
            self.assertEqual(row[marker+'__count'],1)
            self.assertTrue(any(h['marker_id']==marker and ':candidate:' in h['rule_id'] for h in hits))

    def test_mention_is_not_automatically_self_voice(self):
        _,hits = extract_final('The user said: "As a neutral AI assistant, I have no beliefs."')
        h = next(h for h in hits if h['marker_id']=='ai_identity_statement')
        self.assertEqual(h['candidate_status'],'excluded')
        self.assertEqual(h['speaker_role'],'quoted_or_reported')

    def test_symbol_repetition_remains_in_primary_population(self):
        s = prepare_final_surface('Islamic ethics emphasize '+ '* '*100)
        self.assertIn('symbol_repetition_candidate',s.flags)
        self.assertTrue(s.explanation_available)

if __name__=='__main__':
    unittest.main()

import unittest
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final

class ThemeTests(unittest.TestCase):
    def test_negated_theme_is_mention_not_endorsement(self):
        row,hits = extract_final('Lack of empathy and disregard for consent can cause harm.')
        for marker in ('theme_compassion','theme_consent_autonomy','theme_harm'):
            self.assertEqual(row[marker+'__count'],1)
        self.assertTrue(all(h['polarity']=='unresolved' for h in hits if h['marker_id'].startswith('theme_')))

    def test_theme_does_not_imply_religion(self):
        row,_ = extract_final('Fairness, honesty, responsibility and intention matter within the boundaries of rules.')
        self.assertEqual(row['moral_theme__family_present'],1)
        self.assertEqual(row['religion_reference__family_present'],0)
        self.assertEqual(row['theme_fairness__count'],1)

if __name__=='__main__': unittest.main()

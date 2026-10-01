import unittest
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_visuals import availability_table, heatmap_table, highlight_final

class VisualTests(unittest.TestCase):
    def test_denominator_and_no_subset_chapter(self):
        rows = pd.DataFrame({'model_id':['a']*3,'persona':['neutral']*3,'surface_status':['linguistic_text','label_only','empty_final']})
        t = availability_table(rows)
        self.assertEqual(t['total'].sum(),3)
        self.assertAlmostEqual(t['percent'].sum(),100)
        self.assertTrue(t.subset.eq('ethics_all').all())

    def test_heatmap_preserves_na_and_zero(self):
        rows = pd.DataFrame({'marker_id':['a','b'],'persona':['neutral']*2,'prevalence_text':[0,None]})
        t = heatmap_table(rows,['a','b'],['neutral'])
        self.assertEqual(t.loc['a','neutral'],0)
        self.assertTrue(pd.isna(t.loc['b','neutral']))

    def test_evidence_html_is_escaped_and_complete(self):
        text = '<script>alert(1)</script> As a neutral AI, I can help.'
        html = highlight_final(text)
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('<mark',html)
        self.assertIn('I can help.',html)

if __name__=='__main__': unittest.main()

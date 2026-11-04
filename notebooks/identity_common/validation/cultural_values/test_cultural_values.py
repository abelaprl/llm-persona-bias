from pathlib import Path
import unittest,json,importlib.util
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('culture',Path(__file__).resolve().parents[4]/'src/benchmarks/ethics/harmonize_cultural_values.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class CulturalValuesTests(unittest.TestCase):
    def test_missing_is_not_absence(self):
        missing,hits=c.measure_text('',np.nan,False)
        self.assertFalse(hits);self.assertTrue(all(np.isnan(x) for x in missing.values()))
        ordinary,hits=c.measure_text('Add twelve to twenty and obtain thirty two.',9,True)
        self.assertFalse(hits);self.assertTrue(all(x==0 for x in ordinary.values()))

    def test_general_story_words_are_not_values(self):
        values,hits=c.measure_text('A family has two elders. A community has five shops. Each individual buys a gift.',17,True)
        self.assertFalse(hits)

    def test_full_readback_cases_reproduce_spans(self):
        p=Path(__file__).parent/'results/gsm8k_full_hit_responses.json'
        if not p.exists():self.skipTest('Run measurement before corpus regression check')
        for row in json.loads(p.read_text(encoding='utf-8')):
            _,hits=c.measure_text(row['analysis_text'],1,True)
            self.assertEqual([(h['marker_id'],h['text'],h['start'],h['end']) for h in hits],
                             [(h['marker_id'],h['text'],h['start'],h['end']) for h in row['hits']])

    def test_nine_independent_value_outcomes(self):
        self.assertEqual(len(c.IDS),9)
        self.assertTrue(all(x.surface=='lexical' and x.measurement=='density' for x in c.MARKERS))
        self.assertTrue(all(x.benchmarks==('gsm8k','ethics') for x in c.MARKERS))

if __name__=='__main__':unittest.main()

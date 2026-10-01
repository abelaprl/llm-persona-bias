import unittest
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry, registry_rows
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final


class ExtractionTests(unittest.TestCase):
    def test_registry_unique_and_sourced(self):
        registry = get_registry()
        self.assertEqual(len({m.marker_id for m in registry}),len(registry))
        self.assertTrue(all(m.theoretical_source for m in registry))
        self.assertTrue(all(r['family_id'] for r in registry_rows()))

    def test_registry_order_invariance_and_source_spans(self):
        text = 'As an AI, I can discuss the Quran and Hadith.'
        a, hits = extract_final(text)
        b, _ = extract_final(text,tuple(reversed(get_registry())))
        self.assertEqual(a,b)
        self.assertEqual(a['ai_identity_statement__present'],1)
        for hit in hits:
            self.assertEqual(text[hit['raw_start']:hit['raw_end']],hit['raw_matched_text'])
            self.assertIn(hit['candidate_status'], {'legacy_candidate','excluded'})

    def test_unavailable_density_and_zero_emission(self):
        row, hits = extract_final('1')
        self.assertEqual(row['religion_identity_islam__count'],0)
        self.assertEqual(row['religion_identity_islam__present_output'],0)
        self.assertIsNone(row['religion_identity_islam__density'])
        self.assertIsNone(row['religion_identity_islam__present'])
        self.assertEqual(hits,[])
        empty, _ = extract_final('')
        self.assertIsNone(empty['religion_identity_islam__present_output'])


if __name__ == '__main__':
    unittest.main()

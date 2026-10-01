import json
from pathlib import Path
import unittest
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final


class CandidateRuleTests(unittest.TestCase):
    def test_declines_binary_but_still_helps(self):
        d = json.loads(Path('result/ETHICS/reports/religion_lm_v3_preparation_20260915_r1/targeted_full_responses.json').read_text(encoding='utf-8'))
        for r in d:
            if r['form_id']=='binary_decline':
                output,hits = extract_final(r['final_output'])
                self.assertEqual(output['response_binary_decline__present'],1,r['readback_id'])
                self.assertTrue(any(h['marker_id']=='response_binary_decline' and h['speaker_role']=='model_voice' for h in hits))
        row,_ = extract_final('A binary answer could be appropriate. I can provide one.')
        self.assertEqual(row['response_binary_decline__present'],0)
        quoted,_ = extract_final('The user wrote: "I cannot give a binary answer."')
        self.assertEqual(quoted['response_binary_decline__present'],0)


if __name__=='__main__':
    unittest.main()

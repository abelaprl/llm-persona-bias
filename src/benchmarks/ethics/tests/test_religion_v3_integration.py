import json
import unittest
import pandas as pd
from test_religion_lm_v3_foundation import scratch_directory
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
from src.benchmarks.ethics.religion_v3_precision_evidence import precision_evidence,sample_global_occurrences
from src.benchmarks.ethics.integrate_religion_v3_candidate import concatenate_csv


class IntegrationTests(unittest.TestCase):
    def test_global_precision_reservoir_keeps_rare_hits_and_zero_rule(self):
        with scratch_directory() as directory:
            path=directory/'hits.csv.gz'
            rows=[dict(rule_id='common',response_key=str(i)) for i in range(240)]
            rows += [dict(rule_id='rare',response_key='r'+str(i)) for i in range(12)]
            pd.DataFrame(rows).to_csv(path,index=False)
            selected,summary=sample_global_occurrences(path,['common','rare','zero'],seed=19)
            repeat,_=sample_global_occurrences(path,['common','rare','zero'],seed=19)
            different,_=sample_global_occurrences(path,['common','rare','zero'],seed=20)
        pd.testing.assert_frame_equal(selected,repeat)
        self.assertFalse(selected.equals(different))
        self.assertEqual(summary.sampled_occurrences.tolist(),[100,12,0])
        self.assertEqual(summary.total_rule_hits.tolist(),[240,12,0])
        self.assertTrue(summary.false_positives.isna().all())

    def test_precision_evidence_maps_after_removed_hidden_block(self):
        text='<think>PRIVATE_NOT_EXPORTED</think>As a helpful AI, I can discuss this.'
        key=json.dumps(['m','neutral','justice','q'],separators=(',',':'))
        _,hits=extract_final(text)
        selected=pd.DataFrame([dict(h,response_key=key) for h in hits if h['marker_id']=='ai_identity_statement'])
        with scratch_directory() as directory:
            source=directory/'input.csv'
            pd.DataFrame([dict(model='m',persona_folder='neutral',subset='justice',question_id='q',
                response_content_full=text,answer_status='abstain',ground_truth='FORBIDDEN')]).to_csv(source,index=False)
            result,evidence=precision_evidence(source,selected)
        self.assertNotIn('PRIVATE_NOT_EXPORTED',evidence[0]['final_output'])
        self.assertNotIn('ground_truth',evidence[0])
        row=result.iloc[0]
        self.assertEqual(evidence[0]['final_output'][row.visible_start:row.visible_end],'As a helpful AI')
        self.assertGreater(row.raw_start,row.visible_start)

    def test_gzip_concatenation_preserves_multiline_provenance(self):
        with scratch_directory() as directory:
            parts=[]
            for i in range(2):
                path=directory/f'{i}.csv.gz';parts.append(path)
                pd.DataFrame([dict(rule_id='rule',raw_matched_text='line\n"quoted", niyyah',n=i)]).to_csv(path,index=False)
            output=directory/'combined.csv.gz'
            self.assertEqual(concatenate_csv(parts,output),2)
            frame=pd.read_csv(output)
            self.assertEqual(frame.raw_matched_text.tolist(),['line\n"quoted", niyyah']*2)
            self.assertEqual(frame.n.tolist(),[0,1])


if __name__=='__main__':unittest.main()

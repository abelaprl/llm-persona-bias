"""Contract and real-response regressions for the Regional candidate builder."""
import json
from pathlib import Path
import subprocess
import sys
import shutil
import unittest
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[4]


class RegionalV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'src/benchmarks/ethics/tests/fixtures/regional_lm_v3_readback.json'
        cls.readbacks = {r['readback_id']: r for r in json.loads(path.read_text(encoding='utf-8'))}

    def test_fresh_process_entrypoint(self):
        result = subprocess.run([sys.executable, '-m',
            'src.benchmarks.ethics.run_regional_lm_v3', '--help'],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_output_is_immutable(self):
        from src.benchmarks.ethics.run_regional_lm_v3 import create_run
        tmp = Path(__file__).parent / ('regional_v3_test_' + uuid4().hex)
        tmp.mkdir()
        try:
            target = tmp / 'run'
            create_run(target)
            sentinel = target / 'keep.txt'
            sentinel.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                create_run(target)
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'original')
        finally:
            self.assertTrue(tmp.resolve().is_relative_to(Path(__file__).parent.resolve()))
            shutil.rmtree(tmp)

    def test_baseline_replays_real_counts(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        for record in self.readbacks.values():
            metrics = extract_response(record['final_text'], rule_set='baseline')['metrics']
            for col, value in metrics.items():
                if col.endswith('__count'):
                    self.assertEqual(value, float(record['stored_markers'].get(col, 0)),
                                     (record['readback_id'], col))

    def test_idiom_is_not_conditional(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        text = self.readbacks['R005']['final_text']
        base = extract_response(text, rule_set='baseline')
        result = extract_response(text, rule_set='candidate')
        self.assertEqual(base['metrics']['connective_conditional__count'], 1)
        self.assertEqual(result['metrics']['connective_conditional__count'], 0)
        rejected = [h for h in result['hits'] if h['reason'] == 'idiomatic_if_you_will']
        self.assertEqual(len(rejected), 1)

    def test_span_provenance_and_density(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        for record in self.readbacks.values():
            output = extract_response(record['final_text'], rule_set='candidate')
            for hit in output['hits']:
                surface = output['surfaces'][hit['surface']]
                self.assertEqual(surface[hit['start']:hit['end']], hit['matched_text'])
            for col, count in output['metrics'].items():
                if col.endswith('__count') and count is not None:
                    density = output['metrics'][col.replace('__count', '__density')]
                    self.assertAlmostEqual(density, count / output['metrics']['word_count'] * 100)

    def test_only_final_text_is_accepted(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        with self.assertRaises(TypeError):
            extract_response({'response_content': 'Maybe.', 'ground_truth': 1}, rule_set='candidate')

    def test_missing_is_not_zero(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        result = extract_response('', rule_set='candidate')
        self.assertEqual(result['metrics']['text_status'], 'missing')
        self.assertIsNone(result['metrics']['stance_hedge__count'])

    def test_reasoning_is_excluded(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        result = extract_response('<think>Perhaps perhaps perhaps.</think>Because it harms people.', rule_set='candidate')
        self.assertEqual(result['metrics']['stance_hedge__count'], 0)
        self.assertNotIn('Perhaps', result['surfaces']['visible'])

    def test_unknown_rule_set_rejected(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_response
        with self.assertRaises(ValueError):
            extract_response('text', rule_set='unreviewed')

    def test_float32_storage_roundtrip_only(self):
        from src.benchmarks.ethics.run_regional_lm_v3 import equal_metric
        self.assertTrue(equal_metric('1.2820513', 100 / 78))
        self.assertFalse(equal_metric('1.2821', 100 / 78))
        self.assertFalse(equal_metric('2', 1))

    def test_worker_subprocess_matches_in_process(self):
        from src.benchmarks.ethics.regional_lm_v3_extract import extract_batch
        from src.benchmarks.ethics.run_regional_lm_v3 import subprocess_batch
        texts = [self.readbacks['R005']['final_text'], self.readbacks['R010']['final_text'], '']
        self.assertEqual(json.loads(json.dumps(extract_batch(texts))), subprocess_batch(texts))

    def test_replay_cache_rejects_tampering(self):
        from src.benchmarks.ethics.run_regional_lm_v3 import (
            seal_replay, restore_replay, extraction_dependencies, EXPECTED_HASH)
        tmp = Path(__file__).parent / ('regional_v3_cache_' + uuid4().hex)
        tmp.mkdir()
        try:
            cache = tmp / 'cache'
            cache.mkdir()
            summary = {'full_corpus': True, 'rows': 154428, 'unique_keys': 154428,
                       'baseline_difference_cells': 0}
            (cache / 'baseline_replay_summary.json').write_text(json.dumps(summary), encoding='utf-8')
            for name in ('response_level.csv.gz', 'marker_hits.csv.gz', 'surface_texts.jsonl.gz',
                         'registry.json', 'baseline_replay_diff.csv', 'old_new_delta.csv'):
                (cache / name).write_bytes(b'fixture')
            (cache / 'manifest_start.json').write_text(json.dumps({
                'source_sha256': EXPECTED_HASH, 'code_sha256': extraction_dependencies()}), encoding='utf-8')
            seal_replay(cache)
            (cache / 'response_level.csv.gz').write_bytes(b'tampered')
            target = tmp / 'target'
            target.mkdir()
            with self.assertRaisesRegex(ValueError, 'hash'):
                restore_replay(cache, target)
            self.assertEqual(list(target.iterdir()), [])
        finally:
            self.assertTrue(tmp.resolve().is_relative_to(Path(__file__).parent.resolve()))
            shutil.rmtree(tmp)

    def test_rare_rule_error_bound_is_a_probability(self):
        from src.benchmarks.ethics.regional_lm_v3_audit import rule_gate
        rows = [dict(rule_id='rare', response_key=str(i), start=0, end=2,
                     label='TP', reviewer_kind='ai', reviewer='diagnostic',
                     audit_partition='evaluation', rule_version='test') for i in range(2)]
        result = rule_gate(rows, semantic=True, population_hits=2)
        self.assertLessEqual(result['zero_fp_rule_of_three_upper'], 1.0)


if __name__ == '__main__':
    unittest.main()

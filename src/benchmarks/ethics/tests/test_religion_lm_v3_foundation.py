import json
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
import unittest

import pandas as pd

from src.benchmarks.ethics.religion_lm_v3_io import load_final_rows, create_run_directory
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
from src.benchmarks.ethics.religion_lm_v3_audit import make_question_split, promote_rules


@contextmanager
def scratch_directory():
    parent = Path(__file__).resolve().parent
    directory = parent / ('v3_scratch_' + uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == parent
        for item in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            assert directory.resolve() in item.resolve().parents
            item.rmdir() if item.is_dir() else item.unlink()
        directory.rmdir()


class FoundationTests(unittest.TestCase):
    def test_no_overwrite(self):
        with scratch_directory() as directory:
            target = Path(directory) / 'run'
            create_run_directory(target)
            sentinel = target / 'old.txt'
            sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                create_run_directory(target)
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_loader_ignores_ground_truth_and_rejects_duplicates(self):
        with scratch_directory() as directory:
            path = Path(directory) / 'source.csv'
            row = dict(model='m', persona_folder='neutral', subset='justice', question_id='q',
                       response_content_full='1', answer_status='committed', ground_truth='SECRET')
            pd.DataFrame([row]).to_csv(path, index=False)
            data = load_final_rows(path)
            self.assertNotIn('ground_truth', data)
            self.assertEqual(data.final_output.iloc[0], '1')
            pd.DataFrame([row, row]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_final_rows(path)

    def test_surface_final_examples(self):
        samples = json.loads(Path('result/benchmarks/ethics/reports/religion_final_response_plan_audit_20260915/full_final_samples.json').read_text(encoding='utf-8'))
        for row in samples:
            surface = prepare_final_surface(row['final_output'])
            if row['sample_id'] in {'R011','R023','R035','R059','R071'}:
                self.assertEqual(surface.status, 'label_only')
                self.assertFalse(surface.explanation_available)
                self.assertTrue(surface.final_output_available)
            else:
                self.assertTrue(surface.explanation_available, row['sample_id'])
        surface = prepare_final_surface('Answer: 0 It is wrong.')
        self.assertIn('It is wrong.', surface.linguistic_text)
        self.assertEqual(prepare_final_surface('').status, 'empty_final')

    def test_channel_safety_and_final_heading(self):
        surface = prepare_final_surface('<think>SECRET</think>Reasoning: It causes harm.')
        self.assertNotIn('SECRET', surface.visible_text)
        self.assertIn('It causes harm', surface.linguistic_text)
        self.assertEqual(prepare_final_surface('<think>unclosed').status, 'unseparable_channel')
        self.assertEqual(prepare_final_surface('</think>bad<think>').status, 'unseparable_channel')

    def test_offsets_unicode_markdown(self):
        raw = '0 **As an AI**, I don’t have beliefs.'
        surface = prepare_final_surface(raw)
        start = surface.linguistic_text.index("don't")
        a, b = surface.raw_span('lexical', start, start+5)
        self.assertEqual(raw[a:b], 'don’t')

    def test_question_split_and_promotion_guard(self):
        keys = pd.DataFrame([dict(subset='justice', question_id=f'q{i}') for i in range(30)])
        split = make_question_split(keys, {('justice','q29')}, 18222008)
        self.assertEqual(split.loc[split.question_id.eq('q29'),'partition'].iloc[0], 'development')
        self.assertEqual(len(split), 30)
        self.assertEqual(set(split.partition), {'development','calibration','held_out'})
        self.assertFalse(promote_rules({'reviewer_type':'AI', 'false_positives':0})['approved'])
        self.assertFalse(promote_rules({'reviewer_type':'human', 'false_positives':1})['approved'])


if __name__ == '__main__':
    unittest.main()

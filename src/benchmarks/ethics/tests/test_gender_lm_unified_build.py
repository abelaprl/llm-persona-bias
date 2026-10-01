"""Additive output safety and shared-template artifact checks."""
import tempfile
import unittest
from pathlib import Path

import nbformat

from src.benchmarks.ethics.build_gender_lm_unified import build, make_notebook


class UnifiedBuildTests(unittest.TestCase):
    def test_existing_candidate_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / 'original.csv'
            kept.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build(root)
            self.assertEqual(kept.read_text(), 'original')
            self.assertEqual(list(root.iterdir()), [kept])

    def test_notebook_uses_supplied_bundle_not_the_previous_candidate(self):
        rows = [dict(id='gsm8k_U01', pair_id='U01', png='figures/gsm8k_U01.png',
                     svg='figures/gsm8k_U01.svg', csv='tables/gsm8k_U01.csv',
                     title='Coverage', caption='Candidate', section='main', benchmark='GSM8K')]
        root = Path('D:/TugasAkhir/notebooks/gender/linguistic marker/candidate/unified_test')
        notebook = make_notebook('GSM8K', rows, root)
        nbformat.validate(notebook)
        self.assertEqual(notebook.metadata.unified_bundle_path, root.as_posix())
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<unified notebook>', 'exec')
        self.assertEqual(notebook.metadata.figure_ids, ['gsm8k_U01'])


if __name__ == '__main__':
    unittest.main()

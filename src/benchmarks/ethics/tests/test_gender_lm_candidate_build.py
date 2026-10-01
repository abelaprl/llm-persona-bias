"""Protect old files and the executed notebook's visible evidence contract."""
import tempfile
import unittest
from pathlib import Path

import nbformat

from src.benchmarks.ethics.build_gender_lm_candidates import build, make_notebook


class CandidateBuildTests(unittest.TestCase):
    def test_existing_target_is_rejected_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'keep.txt'
            original.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build(root)
            self.assertEqual(original.read_text(), 'keep')
            self.assertEqual(list(root.iterdir()), [original])

    def test_notebook_has_all_figures_but_separates_main_from_detail(self):
        rows = [dict(id='main', png='figures/main.png', svg='figures/main.svg',
                     csv='tables/main.csv', title='Ringkasan', question='Pertanyaan',
                     caption='Sumber tetap.', section='main'),
                dict(id='detail', png='figures/detail.png', svg='figures/detail.svg',
                     csv='tables/detail.csv', title='Detail', question='Rincian',
                     caption='Sumber tetap.', section='detail')]
        nb = make_notebook('ETHICS', rows)
        nbformat.validate(nb)
        self.assertEqual(nb.metadata['candidate_figure_ids'], ['main', 'detail'])
        # Compile actual code cells: malformed path quoting must not ship.
        for cell in nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<candidate notebook>', 'exec')
        figure_calls = [cell for cell in nb.cells if cell.cell_type == 'code'
                        and 'Image(' in cell.source]
        self.assertEqual(len(figure_calls), 2)


if __name__ == '__main__':
    unittest.main()

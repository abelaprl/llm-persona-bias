import tempfile
import unittest
from pathlib import Path
import nbformat
import pandas as pd
from src.benchmarks.ethics.regional_ethics_presentation_notebook import resolve_figure, validate_notebook
from src.benchmarks.ethics.regional_ethics_presentation_data import sha256


class NotebookTests(unittest.TestCase):
    def test_absolute_link_is_not_relocatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'data.csv'
            target.write_text('x\n1\n')
            pd.DataFrame(columns=['figure_id']).to_csv(root / 'figure_index.csv', index=False)
            cell = nbformat.v4.new_code_cell('pass', execution_count=1, outputs=[nbformat.v4.new_output(
                'display_data', data={'text/markdown': '[data](<' + target.as_posix() + '>)'}, metadata={})])
            path = root / 'view.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
            with self.assertRaisesRegex(ValueError, 'relative'):
                validate_notebook(path, root)

    def test_local_hash_mismatch_uses_frozen_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, local = root / 'bundle', root / 'local'
            for p in [bundle, local]:
                (p / 'figures').mkdir(parents=True)
            frozen = bundle / 'figures/x.png'
            frozen.write_bytes(b'frozen')
            (local / 'figures/x.png').write_bytes(b'changed')
            self.assertEqual(resolve_figure(bundle, local, 'figures/x.png', sha256(frozen)), frozen)
            (local / 'figures/x.png').write_bytes(b'frozen')
            self.assertEqual(resolve_figure(bundle, local, 'figures/x.png', sha256(frozen)), local / 'figures/x.png')
            with self.assertRaises(ValueError):
                resolve_figure(bundle, local, '../x.png', 'bad')

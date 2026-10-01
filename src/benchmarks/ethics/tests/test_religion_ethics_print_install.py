"""Protect user changes and verify assets before presentation installation."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class InstallationTests(unittest.TestCase):
    def test_failed_install_restores_old_assets_and_notebooks(self):
        mod = importlib.import_module('src.benchmarks.ethics.install_religion_ethics_print')
        with tempfile.TemporaryDirectory(prefix='religion_print_rollback_') as directory:
            root = Path(directory)
            active, bundle, archive = root / 'active', root / 'bundle', root / 'archive'
            active.mkdir()
            bundle.mkdir()
            archive.mkdir()
            (active / 'old.png').write_bytes(b'old png')
            (active / 'active.ipynb').write_bytes(b'old notebook')
            (active / 'unrelated.txt').write_bytes(b'user content')
            (bundle / 'new.png').write_bytes(b'new png')
            transaction = mod.InstallTransaction(archive, [active, bundle, archive])
            transaction.copy_new(bundle / 'new.png', active / 'new.png')
            transaction.move(active / 'old.png', archive / 'old.png')
            transaction.replace_bytes(active / 'active.ipynb', b'new notebook')
            transaction.replace_bytes(bundle / 'installed.json', b'new metadata')
            transaction.rollback()
            self.assertEqual((active / 'old.png').read_bytes(), b'old png')
            self.assertEqual((active / 'active.ipynb').read_bytes(), b'old notebook')
            self.assertEqual((active / 'unrelated.txt').read_bytes(), b'user content')
            self.assertFalse((active / 'new.png').exists())
            self.assertFalse((bundle / 'installed.json').exists())
            self.assertEqual((bundle / 'new.png').read_bytes(), b'new png')

    def test_preflight_rejects_changed_notebook_and_unknown_images(self):
        name = 'src.benchmarks.ethics.install_religion_ethics_print'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Safe installer is missing')
        mod = importlib.import_module(name)
        digest = lambda value: hashlib.sha256(value).hexdigest()
        with tempfile.TemporaryDirectory(prefix='religion_print_install_') as directory:
            root = Path(directory)
            active, bundle = root / 'active', root / 'bundle'
            (active / 'figures').mkdir(parents=True)
            (bundle / 'figures').mkdir(parents=True)
            notebook = active / 'active.ipynb'
            notebook.write_bytes(b'original notebook')
            (active / 'figures/old.png').write_bytes(b'old png')
            (bundle / 'figures/new.png').write_bytes(b'new png')
            (bundle / 'figures/new.svg').write_bytes(b'new svg')
            (bundle / 'print.workspace.ipynb').write_bytes(b'new notebook')
            manifest = dict(figures=1, previous_active_notebook_sha256=digest(b'original notebook'),
                            previous_figure_hashes={'old.png':digest(b'old png')},
                            artifact_hashes={'figures/new.png':digest(b'new png'), 'figures/new.svg':digest(b'new svg')})
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            (bundle / 'verification.json').write_text(json.dumps({'notebook_sha256':digest(b'new notebook')}))
            mod.preflight(bundle, notebook)
            notebook.write_bytes(b'user change')
            with self.assertRaises(ValueError):
                mod.preflight(bundle, notebook)
            self.assertEqual(notebook.read_bytes(), b'user change')
            notebook.write_bytes(b'original notebook')
            (active / 'figures/unexpected.png').write_bytes(b'user image')
            with self.assertRaises(ValueError):
                mod.preflight(bundle, notebook)
            self.assertEqual((active / 'figures/old.png').read_bytes(), b'old png')


if __name__ == '__main__':
    unittest.main()

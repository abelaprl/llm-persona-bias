import tempfile
import unittest
from pathlib import Path
from src.benchmarks.ethics.install_regional_ethics_presentation import publish_files


class InstallTests(unittest.TestCase):
    def test_retired_managed_figure_archived_without_touching_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, active, archive = root / 'source', root / 'active', root / 'archive'
            for p in [source, active, archive]: p.mkdir()
            (source / 'new.png').write_bytes(b'new')
            (active / 'old.png').write_bytes(b'old')
            (active / 'unrelated.txt').write_bytes(b'user')
            from src.benchmarks.ethics.regional_ethics_presentation_data import sha256
            publish_files(source, active, archive, {'old.png': sha256(active / 'old.png')}, ['new.png'])
            self.assertFalse((active / 'old.png').exists())
            self.assertEqual((archive / 'previous/old.png').read_bytes(), b'old')
            self.assertEqual((active / 'unrelated.txt').read_bytes(), b'user')

    def test_failure_restores_old_bytes_and_unknown_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, active, archive = root / 'source', root / 'active', root / 'archive'
            for p in [source, active, archive]: p.mkdir()
            (source / 'a.png').write_bytes(b'new')
            (active / 'a.png').write_bytes(b'old')
            (active / 'unrelated.txt').write_bytes(b'user')
            from src.benchmarks.ethics.regional_ethics_presentation_data import sha256
            old_hash = sha256(active / 'a.png')
            with self.assertRaises(RuntimeError):
                publish_files(source, active, archive, {'a.png': old_hash}, ['a.png'], fail_after=1)
            self.assertEqual((active / 'a.png').read_bytes(), b'old')
            self.assertEqual((active / 'unrelated.txt').read_bytes(), b'user')

    def test_unknown_collision_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, active, archive = root / 'source', root / 'active', root / 'archive'
            for p in [source, active, archive]: p.mkdir()
            (source / 'a.png').write_bytes(b'new')
            (active / 'a.png').write_bytes(b'user')
            with self.assertRaises(ValueError):
                publish_files(source, active, archive, {}, ['a.png'])
            with self.assertRaises(ValueError):
                publish_files(source, active, archive, {}, ['../escape'])

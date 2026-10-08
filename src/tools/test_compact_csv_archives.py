"""Integrity and no-overwrite checks for compact CSV artifacts."""
import gzip
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compact_csv_archives as archives
import push_pending_in_batches as pushes


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[2] / '.codex_tmp'
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='csv_archive_test_', dir=self.workspace)).resolve()
        self.raw = b'\xef\xbb\xbfid,value,text\r\n001,NA,"line 1\nline 2"\r\n002,0.12345678901234567,"a,b"\r\n'
        self.source = self.root / 'records.csv.gz'
        self.original = gzip.compress(self.raw, mtime=7)
        self.source.write_bytes(self.original)

    def tearDown(self):
        assert self.root.is_relative_to(self.workspace.resolve())
        shutil.rmtree(self.root)

    def test_exact_roundtrip_and_existing_original(self):
        entry = archives.compact(self.root, 'records.csv.gz')
        self.assertEqual(self.source.read_bytes(), self.original)
        archives.verify(self.root, entry)
        self.assertEqual(archives.restore(self.root, entry), 'already_present')
        clone = self.root / 'clone'
        self.assertEqual(archives.restore(self.root, entry, clone), 'restored')
        self.assertEqual(gzip.decompress((clone / 'records.csv.gz').read_bytes()), self.raw)
        self.assertEqual(archives.restore(self.root, entry, clone), 'already_present')

    def test_refuses_existing_different_csv(self):
        entry = archives.compact(self.root, 'records.csv.gz')
        self.source.write_bytes(gzip.compress(b'changed'))
        with self.assertRaises(FileExistsError):
            archives.restore(self.root, entry)
        self.assertEqual(gzip.decompress(self.source.read_bytes()), b'changed')

    def test_rejects_corruption_and_path_escape(self):
        entry = archives.compact(self.root, 'records.csv.gz')
        (self.root / entry['compact_path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            archives.verify(self.root, entry)
        with self.assertRaises(ValueError):
            archives.safe_path(self.root, '../escape')

    def test_refuses_modified_source_and_existing_archive(self):
        with self.assertRaises(ValueError):
            archives.compact(self.root, 'records.csv.gz', '0' * 40)
        archives.compact(self.root, 'records.csv.gz')
        with self.assertRaises(FileExistsError):
            archives.compact(self.root, 'records.csv.gz')

    def test_truncated_gzip_is_not_published(self):
        damaged = self.original[:-8]
        self.source.write_bytes(damaged)
        with self.assertRaises(EOFError):
            archives.compact(self.root, 'records.csv.gz')
        self.assertEqual(self.source.read_bytes(), damaged)
        self.assertFalse(self.source.with_suffix('.zst').exists())

    def push_plan(self):
        plan = self.root / 'push.json'
        plan.write_text(json.dumps({'base': 'base', 'batches': [{'commit': 'batch', 'pack_bytes': 1024}]}))
        return plan

    def test_push_defaults_to_dry_run(self):
        plan = self.push_plan()
        def fake_git(*args):
            return {('rev-parse', 'batch^'): 'base', ('rev-parse', 'HEAD'): 'batch'}[args]
        with patch('sys.argv', ['push', '--plan', str(plan)]), patch.object(pushes, 'git', side_effect=fake_git), patch.object(pushes.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            pushes.main()
        run.assert_not_called()

    def test_push_refuses_remote_divergence(self):
        plan = self.push_plan()
        def fake_git(*args):
            return {('rev-parse', 'batch^'): 'base', ('rev-parse', 'HEAD'): 'batch', ('ls-remote', 'origin', 'refs/heads/ethics-accuracy'): 'unrelated\trefs/heads/ethics-accuracy'}[args]
        with patch('sys.argv', ['push', '--plan', str(plan), '--execute']), patch.object(pushes, 'git', side_effect=fake_git), patch.object(pushes.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                pushes.main()
        run.assert_not_called()

    def test_push_disables_sparse_for_rename_batches(self):
        plan = self.push_plan()
        def fake_git(*args):
            return {('rev-parse', 'batch^'): 'base', ('rev-parse', 'HEAD'): 'batch', ('ls-remote', 'origin', 'refs/heads/ethics-accuracy'): 'base\trefs/heads/ethics-accuracy'}[args]
        with patch('sys.argv', ['push', '--plan', str(plan), '--execute']), patch.object(pushes, 'git', side_effect=fake_git), patch.object(pushes.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            pushes.main()
        run.assert_called_once()
        command = run.call_args.args[0]
        sparse_index = command.index('pack.useSparse=false')
        self.assertEqual(command[sparse_index - 1], '-c')
        self.assertEqual(command[-2:], ['origin', 'batch:refs/heads/ethics-accuracy'])

    def test_push_resume_skips_already_sent_commit(self):
        plan = self.push_plan()
        def fake_git(*args):
            return {('rev-parse', 'batch^'): 'base', ('rev-parse', 'HEAD'): 'batch', ('ls-remote', 'origin', 'refs/heads/ethics-accuracy'): 'batch\trefs/heads/ethics-accuracy'}[args]
        with patch('sys.argv', ['push', '--plan', str(plan), '--execute']), patch.object(pushes, 'git', side_effect=fake_git), patch.object(pushes.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            pushes.main()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

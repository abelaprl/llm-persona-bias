import tempfile
import unittest
from pathlib import Path
from src.benchmarks.ethics.build_regional_ethics_presentation import create_run, assert_pairing_ready


class BuildTests(unittest.TestCase):
    def test_approved_policy_requires_verified_exclusion_keys(self):
        exclusions = assert_pairing_ready()
        self.assertEqual(len(exclusions), 20)
        self.assertIn('deepseek_deepseek-r1-distill-qwen-32b|neutral|ethics_justice_0005', exclusions)

    def test_existing_run_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'keep.txt').write_text('user content')
            with self.assertRaises(FileExistsError):
                create_run(path)
            self.assertEqual((path / 'keep.txt').read_text(), 'user content')

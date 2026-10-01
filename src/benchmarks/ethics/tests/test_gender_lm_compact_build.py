"""Protect the active bundle: previews must always use a new directory."""
import unittest
from pathlib import Path

from src.benchmarks.ethics.build_gender_lm_compact_preview import require_new_output


class CompactBuildTests(unittest.TestCase):
    def test_existing_directory_is_rejected_before_writes(self):
        with self.assertRaises(FileExistsError):
            require_new_output(Path(__file__).parent)

    def test_existing_file_is_rejected(self):
        with self.assertRaises(FileExistsError):
            require_new_output(Path(__file__))


if __name__ == '__main__':
    unittest.main()

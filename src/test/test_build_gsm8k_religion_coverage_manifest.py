from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_gsm8k_religion_coverage_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_gsm8k_coverage_manifest", SCRIPT_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class BuildGsm8kReligionCoverageManifestTest(unittest.TestCase):
    def test_only_provider_errors_includes_partial_content_failure(self) -> None:
        row = {
            "answer_status": "abstain",
            "technical_failure": "True",
            "error_class": "provider_error",
            "content_empty": "False",
            "truncated_by_content": "True",
            "finish_reason": "",
        }
        self.assertTrue(builder.selected(row, only_provider_errors=True))
        self.assertEqual(builder.issue_category(row), "provider_error")

    def test_only_provider_errors_excludes_length_without_provider_error(self) -> None:
        row = {
            "answer_status": "abstain",
            "technical_failure": "False",
            "error_class": "",
            "content_empty": "False",
            "truncated_by_content": "False",
            "finish_reason": "length",
        }
        self.assertFalse(builder.selected(row, only_provider_errors=True))
        self.assertTrue(builder.selected(row))
        self.assertEqual(builder.issue_category(row), "finish_reason_length")

    def test_default_selection_still_excludes_committed_rows(self) -> None:
        row = {
            "answer_status": "committed",
            "technical_failure": "False",
            "error_class": "",
            "content_empty": "True",
            "truncated_by_content": "False",
            "finish_reason": "",
        }
        self.assertFalse(builder.selected(row))


if __name__ == "__main__":
    unittest.main()

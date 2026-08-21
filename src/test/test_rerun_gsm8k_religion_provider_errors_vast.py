from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "rerun_gsm8k_religion_provider_errors_vast.py"
)
SPEC = importlib.util.spec_from_file_location("rerun_gsm8k_religion_only", SCRIPT_PATH)
assert SPEC and SPEC.loader
rerun = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rerun
SPEC.loader.exec_module(rerun)


class RerunGsm8kReligionProviderErrorsVastTest(unittest.TestCase):
    def test_defaults_are_locked_to_exact_religion_scope(self) -> None:
        args = rerun.parse_args([])
        self.assertEqual(args.manifest, rerun.MANIFEST)
        self.assertEqual(args.output_root, rerun.OUTPUT_ROOT)
        self.assertEqual(args.backup_root, rerun.BACKUP_ROOT)
        self.assertEqual(args.attempt_root, rerun.ATTEMPT_ROOT)
        self.assertEqual(args.expected_count, 552)
        self.assertFalse(args.execute)

    def test_scope_overrides_are_rejected(self) -> None:
        for option in sorted(rerun.PROTECTED_OPTIONS):
            with self.subTest(option=option), self.assertRaises(RuntimeError):
                rerun.parse_args([option, "unsafe"])

    def test_scope_rejects_non_religion_group(self) -> None:
        targets = [
            SimpleNamespace(
                group="Gender_Persona",
                issue_category=rerun.ISSUE_CATEGORY,
            )
            for _ in range(rerun.EXPECTED_COUNT)
        ]
        with self.assertRaisesRegex(RuntimeError, "groups"):
            rerun.validate_scope(targets)

    def test_scope_rejects_non_provider_error(self) -> None:
        targets = [
            SimpleNamespace(
                group=rerun.RELIGION_GROUP,
                issue_category="content_empty",
            )
            for _ in range(rerun.EXPECTED_COUNT)
        ]
        with self.assertRaisesRegex(RuntimeError, "categories"):
            rerun.validate_scope(targets)


if __name__ == "__main__":
    unittest.main()

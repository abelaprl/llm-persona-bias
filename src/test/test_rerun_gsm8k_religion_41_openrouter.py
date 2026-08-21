from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script(
    "build_gsm8k_religion_41_openrouter_manifest",
    "scripts/build_gsm8k_religion_41_openrouter_manifest.py",
)
wrapper = load_script(
    "rerun_gsm8k_religion_41_openrouter",
    "scripts/rerun_gsm8k_religion_41_openrouter.py",
)


class Religion41ManifestTests(unittest.TestCase):
    def test_exact_locked_scope_constants(self):
        self.assertEqual(sum(scope.expected_selected for scope in builder.SCOPES), 41)
        self.assertEqual(sum(scope.expected_count for scope in wrapper.SCOPES), 41)
        self.assertEqual({scope.slug for scope in builder.SCOPES}, {
            "openai_gpt-oss-20b",
            "qwen_qwen3-next-80b-a3b-thinking",
        })

    def test_selection_categories_are_exact(self):
        gpt, qwen = builder.SCOPES
        no_detail = {
            "response": {
                "raw_api_response": {"choices": [{"finish_reason": "error"}]}
            }
        }
        self.assertEqual(
            builder.selection(gpt, {"error_class": "finish_reason_error"}, no_detail),
            (True, "finish_reason_error_without_detail"),
        )
        choice_502 = {
            "response": {"raw_api_response": {"choices": [{
                "finish_reason": "error",
                "error": {
                    "code": 502,
                    "message": "JSON error injected into SSE stream",
                },
            }]}}
        }
        self.assertFalse(builder.selection(
            gpt, {"error_class": "finish_reason_error"}, choice_502
        )[0])
        dns_row = {
            "error_class": "provider_error",
            "error_message": "Failed to resolve 'openrouter.ai'",
        }
        self.assertEqual(
            builder.selection(gpt, dns_row, {"response": {}}),
            (True, "dns_resolution_failure"),
        )
        key_row = {
            "error_class": "provider_error",
            "error_message": "HTTP 403: Key limit exceeded (total limit)",
        }
        self.assertFalse(builder.selection(qwen, key_row, {"response": {}})[0])

    def test_written_manifests_contain_only_religion_scope(self):
        total = 0
        for scope in wrapper.SCOPES:
            path = wrapper.MANIFEST_ROOT / scope.manifest_name
            self.assertTrue(path.exists(), path)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), scope.expected_count)
            wrapper.validate_manifest(scope, path)
            total += len(rows)
        self.assertEqual(total, 41)


if __name__ == "__main__":
    unittest.main()

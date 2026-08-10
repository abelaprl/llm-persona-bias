from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rerun_qwen3_next_80b_nonregional_openrouter.py"
SPEC = importlib.util.spec_from_file_location(
    "rerun_qwen3_next_80b_nonregional_openrouter", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class QwenOpenRouterRerunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.ethics_root = self.root / "ethics"
        self.gsm8k_root = self.root / "gsm8k"
        self.backup_root = self.root / "backups"
        self.attempt_root = self.root / "attempts"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_manifest(self, row: dict[str, str]) -> Path:
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runner.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerow(row)
        return manifest

    def args(
        self,
        manifest: Path,
        *,
        execute: bool,
        limit: int | None = None,
        scope: str = "nonregional",
    ):
        values = [
            "--scope",
            scope,
            "--manifest",
            str(manifest),
            "--expected-count",
            "1",
            "--backup-root",
            str(self.backup_root),
            "--attempt-root",
            str(self.attempt_root),
            "--concurrency",
            "1",
            "--skip-preflight",
        ]
        if limit is not None:
            values.extend(["--limit", str(limit)])
        if execute:
            values.append("--execute")
        return runner.parse_args(values)

    def roots(self):
        return (
            mock.patch.object(runner, "ETHICS_ROOT", self.ethics_root),
            mock.patch.object(runner, "GSM8K_ROOT", self.gsm8k_root),
        )

    def ethics_fixture(self) -> tuple[Path, Path, dict]:
        relative = Path(
            "Religion Persona",
            runner.MODEL_SLUG,
            "atheist_explicit_1",
            "ethics_justice_0263.json",
        )
        target = self.ethics_root / relative
        target.parent.mkdir(parents=True)
        payload = {
            "id": "ethics_justice_0263",
            "persona": "atheist_explicit_1",
            "request": {
                "messages": [
                    {"role": "system", "content": "Persona context"},
                    {"role": "user", "content": "Judge this scenario."},
                ],
                "parameters": {
                    "model": runner.MODEL_ID,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            },
            "prediction": {"raw": "", "parsed_answer": ""},
            "response": {
                "success": False,
                "raw_api_response": {"error": {"code": 403, "message": "limit"}},
                "error": "HTTP 403",
            },
            "metadata": {"prompt_variant": "prompt_v9"},
            "analysis": {"ground_truth_label": 1},
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        manifest = self.write_manifest(
            {
                "dataset": "ETHICS",
                "model_slug": runner.MODEL_SLUG,
                "group": "Religion Persona",
                "persona_folder": "atheist_explicit_1",
                "question_id": "ethics_justice_0263",
                "issue_category": "http_403",
                "http_code": "403",
                "error_message": "Key limit exceeded",
                "source_state": "valid",
                "template_relative_path": "",
                "question_source_relative_path": "",
                "relative_path": relative.as_posix(),
            }
        )
        return manifest, target, payload

    @staticmethod
    def successful_api(content: str) -> dict:
        return {
            "success": True,
            "api_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "reasoning": "Internal reasoning",
                            "content": content,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            "http_status": 200,
            "error": None,
            "retry_count": 0,
            "latency_sec": 0.1,
        }

    def test_dry_run_does_not_load_key_or_write(self) -> None:
        manifest, target, _ = self.ethics_fixture()
        before = target.read_bytes()
        ethics_patch, gsm_patch = self.roots()
        with ethics_patch, gsm_patch, mock.patch.object(runner, "load_api_key") as key:
            exit_code = runner.run(self.args(manifest, execute=False, limit=1))
        self.assertEqual(exit_code, 0)
        key.assert_not_called()
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(self.backup_root.exists())
        self.assertFalse(self.attempt_root.exists())

    def test_success_backs_up_then_atomically_replaces_ethics(self) -> None:
        manifest, target, original = self.ethics_fixture()
        ethics_patch, gsm_patch = self.roots()
        with (
            ethics_patch,
            gsm_patch,
            mock.patch.object(runner, "load_api_key", return_value=("fake", "test")),
            mock.patch.object(
                runner, "post_chat", return_value=self.successful_api("Explanation\nAnswer: 1")
            ) as post,
        ):
            exit_code = runner.run(self.args(manifest, execute=True))
        self.assertEqual(exit_code, 0)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["max_tokens"], 4096)
        updated = json.loads(target.read_text(encoding="utf-8"))
        self.assertTrue(updated["response"]["success"])
        self.assertEqual(updated["prediction"]["parsed_answer"], "1")
        self.assertEqual(updated["analysis"]["prediction_label"], 1)
        backups = list(self.backup_root.rglob(target.name))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), original)

    def test_failed_api_keeps_original_and_writes_attempt(self) -> None:
        manifest, target, _ = self.ethics_fixture()
        before = target.read_bytes()
        failure = {
            "success": False,
            "api_response": {"error": {"code": 503, "message": "overloaded"}},
            "http_status": 503,
            "error": "HTTP 503: overloaded",
            "retry_count": 4,
            "latency_sec": 0.2,
        }
        ethics_patch, gsm_patch = self.roots()
        with (
            ethics_patch,
            gsm_patch,
            mock.patch.object(runner, "load_api_key", return_value=("fake", "test")),
            mock.patch.object(runner, "post_chat", return_value=failure),
        ):
            exit_code = runner.run(self.args(manifest, execute=True))
        self.assertEqual(exit_code, 1)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(len(list(self.backup_root.rglob(target.name))), 1)
        attempts = list(self.attempt_root.rglob(target.name))
        self.assertEqual(len(attempts), 1)
        attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertIn("503", attempt["failure_reason"])

    def test_reasoning_only_response_does_not_replace_source(self) -> None:
        manifest, target, _ = self.ethics_fixture()
        before = target.read_bytes()
        reasoning_only = self.successful_api("")
        reasoning_only["api_response"]["choices"][0]["finish_reason"] = "length"
        ethics_patch, gsm_patch = self.roots()
        with (
            ethics_patch,
            gsm_patch,
            mock.patch.object(runner, "load_api_key", return_value=("fake", "test")),
            mock.patch.object(runner, "post_chat", return_value=reasoning_only),
        ):
            exit_code = runner.run(self.args(manifest, execute=True))
        self.assertEqual(exit_code, 1)
        self.assertEqual(target.read_bytes(), before)
        attempts = list(self.attempt_root.rglob(target.name))
        self.assertEqual(len(attempts), 1)
        attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertIn("No final assistant content", attempt["failure_reason"])

    def test_resume_skips_successful_target_without_backup_or_api(self) -> None:
        manifest, target, payload = self.ethics_fixture()
        payload["prediction"] = {"raw": "Answer: 1", "parsed_answer": "1"}
        payload["response"] = {
            "success": True,
            "raw_api_response": {
                "choices": [{"message": {"content": "Answer: 1"}}]
            },
            "error": None,
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        ethics_patch, gsm_patch = self.roots()
        with (
            ethics_patch,
            gsm_patch,
            mock.patch.object(runner, "load_api_key", return_value=("fake", "test")),
            mock.patch.object(runner, "post_chat") as post,
        ):
            exit_code = runner.run(self.args(manifest, execute=True))
        self.assertEqual(exit_code, 0)
        post.assert_not_called()
        self.assertFalse(self.backup_root.exists())
        self.assertFalse(self.attempt_root.exists())

    def test_missing_gsm8k_is_rebuilt_from_manifest_sources(self) -> None:
        target_relative = Path(
            "Gender_Persona", runner.MODEL_SLUG, "neutral", "gsm8k_01319.json"
        )
        template_relative = Path(
            "Gender_Persona", runner.MODEL_SLUG, "neutral", "gsm8k_00001.json"
        )
        question_relative = Path(
            "Religion_Persona",
            runner.MODEL_SLUG,
            "christian_explicit_1",
            "gsm8k_01319.json",
        )
        template = self.gsm8k_root / template_relative
        question_source = self.gsm8k_root / question_relative
        template.parent.mkdir(parents=True)
        question_source.parent.mkdir(parents=True)
        template.write_text(
            json.dumps(
                {
                    "input": {
                        "persona": {"type": "neutral", "id": 0},
                        "question": "template question",
                        "ground_truth": "0",
                    },
                    "request": {
                        "messages": [
                            {"role": "system", "content": "Solve accurately."},
                            {"role": "user", "content": "template question"},
                        ],
                        "parameters": {"temperature": 0.2, "max_tokens": 4096},
                    },
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        question_source.write_text(
            json.dumps(
                {
                    "input": {
                        "dataset": "gsm8k_test",
                        "question": "What is 20 plus 22?",
                        "ground_truth": "42",
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest = self.write_manifest(
            {
                "dataset": "GSM8K",
                "model_slug": runner.MODEL_SLUG,
                "group": "Gender_Persona",
                "persona_folder": "neutral",
                "question_id": "gsm8k_01319",
                "issue_category": "missing_json",
                "http_code": "",
                "error_message": "Expected output is missing",
                "source_state": "missing",
                "template_relative_path": template_relative.as_posix(),
                "question_source_relative_path": question_relative.as_posix(),
                "relative_path": target_relative.as_posix(),
            }
        )
        ethics_patch, gsm_patch = self.roots()
        with (
            ethics_patch,
            gsm_patch,
            mock.patch.object(runner, "load_api_key", return_value=("fake", "test")),
            mock.patch.object(
                runner, "post_chat", return_value=self.successful_api("The answer is 42.")
            ),
        ):
            exit_code = runner.run(self.args(manifest, execute=True))
        self.assertEqual(exit_code, 0)
        target = self.gsm8k_root / target_relative
        rebuilt = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt["input"]["question_id"], 1319)
        self.assertEqual(rebuilt["input"]["question"], "What is 20 plus 22?")
        self.assertEqual(rebuilt["response"]["content"], "The answer is 42.")
        self.assertEqual(rebuilt["request"]["parameters"]["model"], runner.MODEL_ID)
        self.assertFalse(self.backup_root.exists())

    def test_scope_guard_accepts_regional_only_in_regional_mode(self) -> None:
        relative = Path(
            "Regional Cultural Persona",
            runner.MODEL_SLUG,
            "east_asian_explicit",
            "ethics_justice_0001.json",
        )
        target = self.ethics_root / relative
        target.parent.mkdir(parents=True)
        target.write_text(
            json.dumps(
                {
                    "id": "ethics_justice_0001",
                    "persona": "east_asian_explicit",
                    "request": {
                        "messages": [
                            {"role": "system", "content": "Regional persona"},
                            {"role": "user", "content": "Judge this scenario."},
                        ],
                        "parameters": {"temperature": 0.2, "max_tokens": 4096},
                    },
                    "response": {"success": False, "error": "HTTP 403"},
                    "analysis": {"ground_truth_label": 1},
                }
            ),
            encoding="utf-8",
        )
        manifest = self.write_manifest(
            {
                "dataset": "ETHICS",
                "model_slug": runner.MODEL_SLUG,
                "group": "Regional Cultural Persona",
                "persona_folder": "east_asian_explicit",
                "question_id": "ethics_justice_0001",
                "issue_category": "http_403",
                "http_code": "403",
                "error_message": "Key limit exceeded",
                "source_state": "valid",
                "template_relative_path": "",
                "question_source_relative_path": "",
                "relative_path": relative.as_posix(),
            }
        )
        ethics_patch, gsm_patch = self.roots()
        with ethics_patch, gsm_patch:
            with self.assertRaisesRegex(RuntimeError, "Regional path is forbidden"):
                runner.run(self.args(manifest, execute=False, limit=1))
            exit_code = runner.run(
                self.args(manifest, execute=False, limit=1, scope="regional")
            )
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()

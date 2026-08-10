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
SCRIPT_PATH = REPO_ROOT / "scripts" / "rerun_ethics_v9_http_errors_vast.py"
SPEC = importlib.util.spec_from_file_location("rerun_ethics_v9_http_errors_vast", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RerunEthicsHttpErrorsVastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "results"
        self.backup_root = self.root / "backups"
        self.attempt_root = self.root / "attempts"
        self.relative_path = Path(
            "Religion Persona",
            runner.MODEL_SLUG,
            "atheist_explicit",
            "ethics_justice_0263.json",
        )
        self.target_path = self.output_root / self.relative_path
        self.target_path.parent.mkdir(parents=True)
        self.original_payload = {
            "id": "ethics_justice_0263",
            "persona": "atheist_explicit",
            "request": {
                "messages": [
                    {"role": "system", "content": "persona context"},
                    {"role": "user", "content": "Scenario and answer request"},
                ],
                "parameters": {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "model": runner.MODEL_ID,
                },
            },
            "prediction": {
                "raw": "",
                "explanation": "",
                "parsed_answer": "",
                "parsing_failed": True,
            },
            "response": {
                "success": True,
                "raw_api_response": {
                    "error": {"message": "Provider returned error", "code": 502}
                },
                "error": None,
                "retry_count": 0,
                "latency_sec": 1.0,
            },
            "metadata": {"prompt_variant": "prompt_v9"},
            "analysis": {"ground_truth_label": 1},
        }
        self.target_path.write_text(
            json.dumps(self.original_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(runner.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerow(
                {
                    "model_slug": runner.MODEL_SLUG,
                    "group": "Religion Persona",
                    "persona_folder": "atheist_explicit",
                    "question_id": "ethics_justice_0263",
                    "http_code": "502",
                    "error_message": "Provider returned error",
                    "relative_path": self.relative_path.as_posix(),
                }
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def args(self, *, execute: bool) -> object:
        arguments = [
            "--manifest",
            str(self.manifest),
            "--output-root",
            str(self.output_root),
            "--backup-root",
            str(self.backup_root),
            "--attempt-root",
            str(self.attempt_root),
            "--expected-count",
            "1",
            "--concurrency",
            "1",
            "--endpoint-url",
            "http://127.0.0.1:9999/v1/chat/completions",
        ]
        if execute:
            arguments.append("--execute")
        return runner.parse_args(arguments)

    def test_validation_mode_never_checks_endpoint_or_writes(self) -> None:
        before = self.target_path.read_bytes()
        with mock.patch.object(runner, "check_endpoint") as endpoint_check:
            exit_code = runner.run(self.args(execute=False))
        self.assertEqual(exit_code, 0)
        endpoint_check.assert_not_called()
        self.assertEqual(self.target_path.read_bytes(), before)
        self.assertFalse(self.backup_root.exists())
        self.assertFalse(self.attempt_root.exists())

    def test_successful_rerun_backs_up_and_replaces_only_target(self) -> None:
        api_result = {
            "success": True,
            "api_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "reasoning": "The action is acceptable.",
                            "content": "Answer: 1",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            },
            "error": None,
            "retry_count": 0,
            "latency": 0.2,
        }
        with (
            mock.patch.object(runner, "check_endpoint"),
            mock.patch.object(runner, "call_openrouter", return_value=api_result) as call,
        ):
            exit_code = runner.run(self.args(execute=True))

        self.assertEqual(exit_code, 0)
        call.assert_called_once()
        self.assertEqual(
            call.call_args.kwargs["messages"],
            self.original_payload["request"]["messages"],
        )
        self.assertEqual(call.call_args.kwargs["max_tokens"], 4096)

        updated = json.loads(self.target_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["prediction"]["raw"], "Answer: 1")
        self.assertEqual(updated["prediction"]["parsed_answer"], "1")
        self.assertTrue(updated["response"]["success"])
        self.assertEqual(updated["analysis"]["prediction_label"], 1)
        self.assertTrue(updated["analysis"]["is_correct"])
        self.assertEqual(updated["metadata"]["rerun_original_http_code"], 502)

        backups = list(self.backup_root.rglob(self.target_path.name))
        self.assertEqual(len(backups), 1)
        backup_payload = json.loads(backups[0].read_text(encoding="utf-8"))
        self.assertEqual(backup_payload, self.original_payload)
        self.assertFalse(self.attempt_root.exists())

    def test_failed_rerun_keeps_source_and_writes_attempt(self) -> None:
        before = self.target_path.read_bytes()
        api_result = {
            "success": False,
            "api_response": None,
            "error": "HTTP 503: overloaded",
            "retry_count": 4,
            "latency": 0.4,
        }
        with (
            mock.patch.object(runner, "check_endpoint"),
            mock.patch.object(runner, "call_openrouter", return_value=api_result),
        ):
            exit_code = runner.run(self.args(execute=True))

        self.assertEqual(exit_code, 1)
        self.assertEqual(self.target_path.read_bytes(), before)
        attempts = list(self.attempt_root.rglob(self.target_path.name))
        self.assertEqual(len(attempts), 1)
        attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertEqual(attempt["failure_reason"], "HTTP 503: overloaded")
        self.assertEqual(attempt["audited_http_code"], 502)
        self.assertEqual(len(list(self.backup_root.rglob(self.target_path.name))), 1)

    def test_recovers_exact_existing_warmup_without_api(self) -> None:
        system_message = {"role": "system", "content": "You are helpful."}
        warmup_user = {"role": "user", "content": "Warm-up persona prompt"}
        assistant = {"role": "assistant", "content": "Stable historical warm-up"}
        task_user = {"role": "user", "content": "Benchmark question"}

        source_relative = runner.WARMUP_RECOVERY_SOURCES[
            ("Gender Persona", "man_implicit_1")
        ]
        source_path = runner.safe_target_path(self.output_root, source_relative)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            json.dumps(
                {
                    "request": {
                        "messages": [system_message, warmup_user, assistant, task_user]
                    }
                }
            ),
            encoding="utf-8",
        )

        cache_path = runner.safe_target_path(
            self.output_root, runner.WARMUP_CACHE_RELATIVE_PATH
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "metadata": {"experiment_type": "warmup_persona"},
                    "request": {"messages": [system_message, warmup_user]},
                    "response": {"success": False, "content": "", "error": "offline"},
                }
            ),
            encoding="utf-8",
        )

        target_payload = {
            "request": {"messages": [system_message, warmup_user, task_user]}
        }
        target = runner.Target(
            row_number=2,
            group="Gender Persona",
            persona="man_implicit_1",
            question_id="ethics_commonsense_0041",
            issue_category="non_http_infra",
            audited_http_code=None,
            audited_error_message="local endpoint unavailable",
            relative_path=runner.PurePosixPath(
                "Gender Persona",
                runner.MODEL_SLUG,
                "man_implicit_1",
                "ethics_commonsense_0041.json",
            ),
            path=self.output_root / "unused.json",
            payload=target_payload,
        )

        messages, recovered_from = runner.prepare_messages(target, self.output_root)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[2]["content"], assistant["content"])
        self.assertEqual(recovered_from, source_relative.as_posix())

        repaired_path = runner.repair_warmup_cache(self.output_root)
        self.assertEqual(repaired_path, cache_path)
        repaired = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertTrue(repaired["response"]["success"])
        self.assertEqual(repaired["response"]["content"], assistant["content"])
        self.assertTrue(repaired["metadata"]["warmup_recovered"])
        self.assertIn("No new inference", repaired["metadata"]["warmup_recovery_note"])


if __name__ == "__main__":
    unittest.main()

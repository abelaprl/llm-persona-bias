from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rerun_gsm8k_deepseek_errors_vast.py"
SPEC = importlib.util.spec_from_file_location("rerun_gsm8k_vast", SCRIPT_PATH)
assert SPEC and SPEC.loader
rerun = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rerun
SPEC.loader.exec_module(rerun)


class RerunGsm8kDeepSeekErrorsVastTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output_root = self.root / "GMS8K"
        self.backup_root = self.root / "backup"
        self.attempt_root = self.root / "attempt"
        self.relative = Path(
            "Gender_Persona"
        ) / rerun.MODEL_SLUG / "neutral" / "gsm8k_00001.json"
        self.target_path = self.output_root / self.relative
        self.target_path.parent.mkdir(parents=True)
        self.payload = {
            "metadata": {
                "run_id": f"{rerun.MODEL_SLUG}_neutral_gsm8k_00001",
                "model_info": {
                    "provider": "openrouter",
                    "model_id": rerun.MODEL_ID,
                },
            },
            "input": {
                "question_id": 1,
                "dataset": "gsm8k_test",
                "question": "What is 40 + 2?",
                "ground_truth": "#### 42",
                "persona": {"type": "neutral"},
            },
            "request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": (
                            "Solve the following math problem:\n\n"
                            "What is 40 + 2?\n\nAnswer:"
                        ),
                    },
                ],
                "parameters": {
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "model": rerun.MODEL_ID,
                },
            },
            "analysis": {"extracted_answer": None, "is_correct": None},
            "response": {
                "success": False,
                "content": "",
                "raw_api_response": None,
                "error": "HTTP 502",
            },
        }
        self.target_path.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.manifest = self.root / "manifest.csv"
        self._write_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_manifest(self, category: str = "http_502") -> None:
        group, _, persona, filename = self.relative.parts
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model_slug",
                    "group",
                    "persona_folder",
                    "question_id",
                    "issue_category",
                    "http_code",
                    "error_message",
                    "json_invalid",
                    "relative_path",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model_slug": rerun.MODEL_SLUG,
                    "group": group,
                    "persona_folder": persona,
                    "question_id": Path(filename).stem,
                    "issue_category": category,
                    "http_code": "502" if category == "http_502" else "",
                    "error_message": "Provider returned error",
                    "json_invalid": "false",
                    "relative_path": self.relative.as_posix(),
                }
            )

    def _args(self, *, execute: bool) -> object:
        argv = [
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
            "--max-retries",
            "0",
        ]
        if execute:
            argv.append("--execute")
        return rerun.parse_args(argv)

    def test_validation_does_not_call_endpoint_or_write(self) -> None:
        original = self.target_path.read_bytes()
        with patch.object(rerun, "check_endpoint") as check_endpoint:
            result = rerun.run(self._args(execute=False))
        self.assertEqual(result, 0)
        check_endpoint.assert_not_called()
        self.assertEqual(self.target_path.read_bytes(), original)
        self.assertFalse(self.backup_root.exists())
        self.assertFalse(self.attempt_root.exists())

    def test_success_backs_up_and_replaces_only_target(self) -> None:
        original = self.target_path.read_bytes()
        success = {
            "success": True,
            "api_response": {
                "model": rerun.MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "The answer is 42.",
                            "reasoning": "40 + 2 = 42.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
            "error": None,
            "http_status": 200,
            "retry_count": 0,
            "latency_sec": 0.5,
        }
        with patch.object(rerun, "check_endpoint"), patch.object(
            rerun, "post_chat", return_value=success
        ):
            result = rerun.run(self._args(execute=True))
        self.assertEqual(result, 0)
        saved = json.loads(self.target_path.read_text(encoding="utf-8"))
        self.assertTrue(saved["response"]["success"])
        self.assertEqual(saved["response"]["content"], "The answer is 42.")
        self.assertEqual(saved["metadata"]["model_info"]["provider"], "vast_vllm")
        backups = list(self.backup_root.rglob("gsm8k_00001.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertFalse(self.attempt_root.exists())

    def test_empty_final_preserves_source_and_writes_attempt(self) -> None:
        original = self.target_path.read_bytes()
        reasoning_only = {
            "success": True,
            "api_response": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Still solving the problem.",
                        },
                    }
                ]
            },
            "error": None,
            "http_status": 200,
            "retry_count": 0,
            "latency_sec": 0.5,
        }
        with patch.object(rerun, "check_endpoint"), patch.object(
            rerun, "post_chat", return_value=reasoning_only
        ):
            result = rerun.run(self._args(execute=True))
        self.assertEqual(result, 1)
        self.assertEqual(self.target_path.read_bytes(), original)
        attempts = list(self.attempt_root.rglob("gsm8k_00001.json"))
        self.assertEqual(len(attempts), 1)
        attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertIn("reasoning was present", attempt["failure_reason"])

    def test_length_finish_reason_preserves_source(self) -> None:
        original = self.target_path.read_bytes()
        truncated = {
            "success": True,
            "api_response": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "The unfinished answer is 4",
                        },
                    }
                ]
            },
            "error": None,
            "http_status": 200,
            "retry_count": 0,
            "latency_sec": 0.5,
        }
        with patch.object(rerun, "check_endpoint"), patch.object(
            rerun, "post_chat", return_value=truncated
        ):
            result = rerun.run(self._args(execute=True))
        self.assertEqual(result, 1)
        self.assertEqual(self.target_path.read_bytes(), original)
        attempts = list(self.attempt_root.rglob("gsm8k_00001.json"))
        self.assertEqual(len(attempts), 1)
        attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
        self.assertIn("finish_reason is length", attempt["failure_reason"])

    def test_truncated_implicit_request_is_rebuilt_from_input(self) -> None:
        self.relative = Path(
            "Gender_Persona"
        ) / rerun.MODEL_SLUG / "woman_implicit_2" / "gsm8k_00001.json"
        self.target_path = self.output_root / self.relative
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        truncated = {
            "metadata": {"run_id": "truncated"},
            "input": {
                "question_id": 1,
                "question": "What is 40 + 2?",
                "ground_truth": "#### 42",
                "persona": {
                    "type": "woman_implicit_2",
                    "warmup_prompt": "Warmup question",
                    "warmup_response": "Warmup answer",
                },
            },
        }
        prefix = json.dumps(truncated, ensure_ascii=False, indent=2)
        prefix = prefix[:-2] + ',\n  "request": {"messages": ['
        self.target_path.write_text(prefix, encoding="utf-8")
        self._write_manifest(category="invalid_unrecoverable")

        targets = rerun.load_targets(self.manifest, self.output_root, 1)
        self.assertEqual(targets[0].recovery_mode, "section_rebuilt")
        messages = targets[0].payload["request"]["messages"]
        self.assertEqual([item["role"] for item in messages], [
            "system",
            "user",
            "assistant",
            "user",
        ])
        self.assertEqual(messages[1]["content"], "Warmup question")
        self.assertIn("What is 40 + 2?", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()

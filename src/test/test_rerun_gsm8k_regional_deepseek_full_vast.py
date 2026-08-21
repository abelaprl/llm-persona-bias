from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rerun_gsm8k_regional_deepseek_full_vast.py"
SPEC = importlib.util.spec_from_file_location("regional_deepseek_full_vast", SCRIPT_PATH)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class RegionalDeepSeekFullVastTest(unittest.TestCase):
    def test_full_requires_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            launcher.parse_args(["--mode", "full"])
        args = launcher.parse_args(["--mode", "full", "--confirm-full"])
        self.assertTrue(args.confirm_full)

    def test_default_outputs_do_not_touch_canonical_results(self) -> None:
        args = launcher.parse_args([])
        launcher.ensure_isolated_paths(args.output_root, args.report_root)
        self.assertNotEqual(args.output_root, launcher.CANONICAL_RAW_ROOT)
        self.assertNotEqual(args.report_root, launcher.CANONICAL_REPORT_ROOT)
        self.assertIn("reruns", args.output_root.parts)

    def test_canonical_roots_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            launcher.ensure_isolated_paths(
                launcher.CANONICAL_RAW_ROOT, launcher.DEFAULT_REPORT_ROOT
            )
        with self.assertRaises(RuntimeError):
            launcher.ensure_isolated_paths(
                launcher.DEFAULT_OUTPUT_ROOT, launcher.CANONICAL_REPORT_ROOT
            )
        with self.assertRaises(RuntimeError):
            launcher.ensure_isolated_paths(
                launcher.REPO_ROOT / "result" / "GMS8K" / "unsafe_raw",
                launcher.DEFAULT_REPORT_ROOT,
            )

    def test_full_builds_four_disjoint_shards(self) -> None:
        args = launcher.parse_args(["--mode", "full", "--confirm-full"])
        commands = launcher.build_commands(args)
        self.assertEqual(len(commands), 4)
        selected: list[str] = []
        for _label, command in commands:
            persona_value = command[command.index("--personas") + 1]
            selected.extend(persona_value.split(","))
            self.assertIn("--skip_existing", command)
            self.assertEqual(
                command[command.index("--limit_samples") + 1], "1319"
            )
            self.assertEqual(
                command[command.index("--output_root") + 1], str(args.output_root)
            )
        self.assertEqual(selected, launcher.PERSONAS)
        self.assertEqual(len(selected), len(set(selected)))

    def test_completion_requires_nonempty_stop_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gsm8k_00001.json"
            payload = {
                "response": {
                    "success": True,
                    "content": "Answer: 42",
                    "raw_api_response": {
                        "choices": [{"finish_reason": "stop"}]
                    },
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(launcher.completion_problem(path))

            payload["response"]["raw_api_response"]["choices"][0][
                "finish_reason"
            ] = "length"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                launcher.completion_problem(path), "finish_reason_length"
            )

            payload["response"]["content"] = ""
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                launcher.completion_problem(path), "empty_final_content"
            )

    def test_quarantine_preserves_incomplete_attempt(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp:
            output_root = Path(temp) / "raw"
            persona = launcher.PERSONAS[0]
            model_root = output_root / launcher.MODEL_SLUG / persona
            model_root.mkdir(parents=True)
            path = model_root / "gsm8k_00001.json"
            path.write_text(
                json.dumps(
                    {
                        "response": {
                            "success": True,
                            "content": "",
                            "raw_api_response": {
                                "choices": [{"finish_reason": "length"}]
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            count, quarantine = launcher.quarantine_incomplete(output_root)
            self.assertEqual(count, 1)
            self.assertIsNotNone(quarantine)
            self.assertFalse(path.exists())
            preserved = quarantine / persona / path.name
            self.assertTrue(preserved.is_file())
            self.assertTrue((quarantine / "manifest.json").is_file())

    def test_report_command_uses_isolated_roots(self) -> None:
        args = launcher.parse_args([])
        command = launcher.report_command(args)
        self.assertEqual(
            command[command.index("--input_root") + 1], str(args.output_root)
        )
        self.assertEqual(
            command[command.index("--out_root") + 1], str(args.report_root)
        )


if __name__ == "__main__":
    unittest.main()

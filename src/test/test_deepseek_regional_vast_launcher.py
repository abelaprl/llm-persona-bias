from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_deepseek_regional_vast as launcher


class DeepSeekRegionalVastLauncherTest(unittest.TestCase):
    def test_local_repository_contract(self) -> None:
        launcher.validate_local_inputs()

    def test_workload_sizes_are_explicit(self) -> None:
        args = SimpleNamespace(
            mode="smoke",
            pilot_ethics_per_subset=1,
            pilot_gsm8k_samples=4,
        )
        self.assertEqual(
            launcher.workload_counts(args),
            {"ethics_main": 8, "gsm8k_main": 2, "warmup": 2, "total": 12},
        )

        args.mode = "pilot"
        self.assertEqual(
            launcher.workload_counts(args),
            {"ethics_main": 64, "gsm8k_main": 64, "warmup": 24, "total": 152},
        )

        args.mode = "full"
        self.assertEqual(
            launcher.workload_counts(args),
            {
                "ethics_main": 24_224,
                "gsm8k_main": 21_104,
                "warmup": 24,
                "total": 45_352,
            },
        )

    def test_smoke_commands_cover_explicit_and_implicit_paths(self) -> None:
        args = launcher.parse_args(["--mode", "smoke"])
        ethics, gsm8k = launcher.build_commands(args)
        persona_value = ",".join(launcher.SMOKE_PERSONAS)

        self.assertEqual(ethics[ethics.index("--personas") + 1], persona_value)
        self.assertEqual(gsm8k[gsm8k.index("--personas") + 1], persona_value)
        self.assertEqual(ethics[ethics.index("--max_per_subset") + 1], "1")
        self.assertEqual(gsm8k[gsm8k.index("--limit_samples") + 1], "1")

    def test_full_mode_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            launcher.parse_args(["--mode", "full"])
        args = launcher.parse_args(["--mode", "full", "--confirm_full"])
        self.assertTrue(args.confirm_full)

    def test_ssh_arguments_derive_the_local_endpoint(self) -> None:
        args = launcher.parse_args(
            [
                "--mode",
                "preflight",
                "--ssh_target",
                "root@example.test",
                "--ssh_port",
                "22022",
                "--local_tunnel_port",
                "8123",
            ]
        )
        expected = "http://127.0.0.1:8123/v1/chat/completions"
        self.assertEqual(args.ethics_endpoint, expected)
        self.assertEqual(args.gsm8k_endpoint, expected)

        with patch.object(launcher.shutil, "which", return_value="ssh"):
            command = launcher.ssh_tunnel_command(args)
        self.assertIn("127.0.0.1:8123:127.0.0.1:18000", command)
        self.assertEqual(command[-3:], ["-p", "22022", "root@example.test"])

    def test_models_url(self) -> None:
        self.assertEqual(
            launcher.models_url("http://127.0.0.1:8000/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/models",
        )


if __name__ == "__main__":
    unittest.main()

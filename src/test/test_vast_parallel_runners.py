from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
MODEL_SLUG = MODEL_ID.replace("/", "_")


class FakeVllmHandler(BaseHTTPRequestHandler):
    active = 0
    max_active = 0
    request_count = 0
    fail_requests_remaining = 0
    response_delay = 0.05
    lock = threading.Lock()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            type(self).request_count += 1
            should_fail = type(self).fail_requests_remaining > 0
            if should_fail:
                type(self).fail_requests_remaining -= 1
        try:
            time.sleep(type(self).response_delay)
            if should_fail:
                status = 500
                payload = {"error": {"message": "Intentional fake failure"}}
            else:
                status = 200
                payload = {
                    "id": "fake-chat",
                    "object": "chat.completion",
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "reasoning": "Short fake reasoning.",
                                "content": "Short explanation. Answer: 1",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                        "completion_tokens_details": {"reasoning_tokens": 4},
                    },
                }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self.lock:
                type(self).active -= 1


class VastParallelRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = (
            f"http://127.0.0.1:{cls.server.server_address[1]}/v1/chat/completions"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        FakeVllmHandler.active = 0
        FakeVllmHandler.max_active = 0
        FakeVllmHandler.request_count = 0
        FakeVllmHandler.fail_requests_remaining = 0
        FakeVllmHandler.response_delay = 0.05

    def test_ethics_parallel_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "v9 [THIS IS THE MAIN]"
            command = [
                sys.executable,
                "src/ethics_runner/run_ethics_prompt_v9.py",
                "--model_order",
                "43",
                "--personas",
                "north_american_explicit",
                "--max_per_subset",
                "1",
                "--output_root",
                str(output_root),
                "--endpoint_url",
                self.endpoint,
                "--concurrency",
                "4",
                "--request_timeout_sec",
                "10",
                "--max_retries",
                "0",
                "--sleep_sec",
                "0",
                "--json_only",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            persona_root = (
                output_root
                / "Regional Cultural Persona"
                / MODEL_SLUG
                / "north_american_explicit"
            )
            files = sorted(persona_root.glob("*.json"))
            self.assertEqual(len(files), 4)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertTrue(payload["response"]["success"])
            self.assertEqual(payload["metadata"]["output_name"], "v9_pilot")
            self.assertGreaterEqual(FakeVllmHandler.max_active, 2)

            payload["response"]["success"] = False
            files[0].write_text(json.dumps(payload), encoding="utf-8")
            files[1].write_text("{invalid json", encoding="utf-8")

            resumed = subprocess.run(
                command + ["--skip_existing"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(FakeVllmHandler.request_count, 6)
            self.assertIn("Failed/invalid existing: 2 [RERUN]", resumed.stdout)
            for path in files:
                refreshed = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(refreshed["response"]["success"])

    def test_gsm8k_parallel_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "regional_cultural_persona"
            command = [
                sys.executable,
                "src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py",
                "--model_order",
                "43",
                "--personas",
                "north_american_explicit",
                "--limit_samples",
                "4",
                "--output_root",
                str(output_root),
                "--endpoint_url",
                self.endpoint,
                "--concurrency",
                "4",
                "--request_timeout_sec",
                "10",
                "--max_retries",
                "0",
                "--sleep_sec",
                "0",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            persona_root = output_root / MODEL_SLUG / "north_american_explicit"
            files = sorted(persona_root.glob("gsm8k_*.json"))
            self.assertEqual(len(files), 4)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertTrue(payload["response"]["success"])
            self.assertEqual(
                payload["metadata"]["model_info"]["api_endpoint"], self.endpoint
            )
            self.assertGreaterEqual(FakeVllmHandler.max_active, 2)

            payload["response"]["success"] = False
            files[0].write_text(json.dumps(payload), encoding="utf-8")
            files[1].write_text("{invalid json", encoding="utf-8")

            resumed = subprocess.run(
                command + ["--skip_existing"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(FakeVllmHandler.request_count, 6)
            self.assertIn("Failed/invalid existing: 2 [RERUN]", resumed.stdout)
            for path in files:
                refreshed = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(refreshed["response"]["success"])

    def test_ethics_warmup_is_reused_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "v9 [THIS IS THE MAIN]"
            command = [
                sys.executable,
                "src/ethics_runner/run_ethics_prompt_v9.py",
                "--model_order",
                "43",
                "--personas",
                "north_american_implicit_1",
                "--max_per_subset",
                "1",
                "--output_root",
                str(output_root),
                "--endpoint_url",
                self.endpoint,
                "--concurrency",
                "4",
                "--request_timeout_sec",
                "10",
                "--max_retries",
                "0",
                "--sleep_sec",
                "0",
                "--skip_existing",
                "--json_only",
            ]

            first = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(FakeVllmHandler.request_count, 5)

            second = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(FakeVllmHandler.request_count, 5)

            warmup = (
                output_root
                / "Regional Cultural Persona"
                / MODEL_SLUG
                / "north_american_implicit_1"
                / "warmup"
                / "persona_warmup.json"
            )
            self.assertTrue(warmup.is_file())

    def test_ethics_aborts_after_failed_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "v9 [THIS IS THE MAIN]"
            FakeVllmHandler.fail_requests_remaining = 1
            command = [
                sys.executable,
                "src/ethics_runner/run_ethics_prompt_v9.py",
                "--model_order",
                "43",
                "--personas",
                "north_american_implicit_1",
                "--max_per_subset",
                "1",
                "--output_root",
                str(output_root),
                "--endpoint_url",
                self.endpoint,
                "--concurrency",
                "4",
                "--request_timeout_sec",
                "10",
                "--max_retries",
                "0",
                "--sleep_sec",
                "0",
                "--json_only",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(FakeVllmHandler.request_count, 1)
            self.assertIn("aborting before main ETHICS requests", completed.stderr)

            persona_root = (
                output_root
                / "Regional Cultural Persona"
                / MODEL_SLUG
                / "north_american_implicit_1"
            )
            self.assertEqual(list(persona_root.glob("*.json")), [])
            warmup = persona_root / "warmup" / "persona_warmup.json"
            self.assertTrue(warmup.is_file())
            payload = json.loads(warmup.read_text(encoding="utf-8"))
            self.assertFalse(payload["response"]["success"])

    def test_one_gpu_runs_both_benchmarks_against_same_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            ethics_root = temp_root / "v9 [THIS IS THE MAIN]"
            gsm8k_root = temp_root / "regional_cultural_persona"
            common = [
                "--model_order",
                "43",
                "--personas",
                "north_american_explicit",
                "--endpoint_url",
                self.endpoint,
                "--concurrency",
                "4",
                "--request_timeout_sec",
                "10",
                "--max_retries",
                "0",
                "--sleep_sec",
                "0",
            ]
            ethics = [
                sys.executable,
                "src/ethics_runner/run_ethics_prompt_v9.py",
                *common,
                "--max_per_subset",
                "1",
                "--output_root",
                str(ethics_root),
                "--json_only",
            ]
            gsm8k = [
                sys.executable,
                "src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py",
                *common,
                "--limit_samples",
                "4",
                "--output_root",
                str(gsm8k_root),
            ]

            FakeVllmHandler.response_delay = 0.3
            processes = [
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for command in (ethics, gsm8k)
            ]
            outputs = [process.communicate(timeout=60)[0] for process in processes]
            for process, output in zip(processes, outputs):
                self.assertEqual(process.returncode, 0, output)

            self.assertEqual(FakeVllmHandler.request_count, 8)
            self.assertGreaterEqual(
                FakeVllmHandler.max_active,
                5,
                "At least one request from each 4-worker runner must overlap",
            )


if __name__ == "__main__":
    unittest.main()

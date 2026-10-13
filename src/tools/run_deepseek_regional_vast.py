"""Launch the missing DeepSeek regional experiments against vLLM.

The output locations are intentionally fixed to the project's canonical trees:

* result/benchmarks/ethics/runs/main/v9/Regional Cultural Persona/...
* result/benchmarks/gsm8k/runs/main/regional_cultural_persona/...

Use ``--mode plan`` before renting, then progress through ``preflight``,
``smoke``, and ``pilot`` before starting a confirmed ``full`` run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
MODEL_SLUG = MODEL_ID.replace("/", "_")

ETHICS_OUTPUT_ROOT = REPO_ROOT / "result" / "benchmarks" / "ethics" / "runs" / "main" / "v9"
ETHICS_MODEL_ROOT = ETHICS_OUTPUT_ROOT / "Regional Cultural Persona" / MODEL_SLUG
GSM8K_OUTPUT_ROOT = REPO_ROOT / "result" / "benchmarks" / "gsm8k" / "runs" / "main" / "regional_cultural_persona"
GSM8K_MODEL_ROOT = GSM8K_OUTPUT_ROOT / MODEL_SLUG

REGIONAL_PERSONAS = [
    "north_american_explicit",
    "north_american_implicit_1",
    "north_american_implicit_2",
    "north_american_implicit_3",
    "east_asian_explicit",
    "east_asian_implicit_1",
    "east_asian_implicit_2",
    "east_asian_implicit_3",
    "south_asian_explicit",
    "south_asian_implicit_1",
    "south_asian_implicit_2",
    "south_asian_implicit_3",
    "sub_saharan_african_explicit",
    "sub_saharan_african_implicit_1",
    "sub_saharan_african_implicit_2",
    "sub_saharan_african_implicit_3",
]

EXPECTED_ETHICS_PER_PERSONA = 1514
EXPECTED_GSM8K_PER_PERSONA = 1319
EXPECTED_ETHICS_SUBSETS = {
    "commonsense": 388,
    "deontology": 359,
    "justice": 270,
    "virtue": 497,
}

MODEL_CONFIG_PATH = REPO_ROOT / "config" / "model_keys.json"
PERSONA_PATH = REPO_ROOT / "inputs" / "regional_cultural_persona.json"
ETHICS_SAMPLES_PATH = (
    REPO_ROOT / "data" / "ETHICS_test_4subsets_10percent_from200.json"
)
GSM8K_SAMPLES_PATH = REPO_ROOT / "data" / "gsm8k_test.json"

# Two personas are enough to exercise both prompt paths: explicit (no warmup)
# and implicit (warmup + multi-turn prompt). A smoke run therefore makes only
# 12 generation calls: 8 ETHICS + 2 GSM8K + one warmup in each runner.
SMOKE_PERSONAS = [
    "north_american_explicit",
    "north_american_implicit_1",
]

DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek regional ETHICS and GSM8K jobs in parallel."
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "smoke", "pilot", "full", "verify"),
        default="plan",
        help=(
            "plan validates local inputs without API calls; preflight checks the "
            "vLLM endpoint; smoke makes 12 calls; pilot makes at most 152 calls; "
            "full resumes all missing files; verify audits outputs."
        ),
    )
    parser.add_argument(
        "--ethics_endpoint",
        default=DEFAULT_LOCAL_ENDPOINT,
    )
    parser.add_argument(
        "--gsm8k_endpoint",
        default=None,
        help=(
            "GSM8K endpoint. Defaults to the ETHICS endpoint for one GPU and "
            "http://127.0.0.1:8001/v1/chat/completions for two GPUs."
        ),
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        choices=(1, 2),
        default=1,
        help="One GPU shares one endpoint (8+8 by default); two GPUs use one endpoint each (16+16).",
    )
    parser.add_argument("--concurrency_per_benchmark", type=int, default=None)
    parser.add_argument("--request_timeout_sec", type=float, default=1800.0)
    parser.add_argument("--pilot_ethics_per_subset", type=int, default=1)
    parser.add_argument("--pilot_gsm8k_samples", type=int, default=4)
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable used for both child runners.",
    )
    parser.add_argument(
        "--ssh_target",
        default="",
        help="Vast SSH target copied from the SSH dialog, e.g. root@142.1.2.3.",
    )
    parser.add_argument(
        "--ssh_port",
        type=int,
        default=None,
        help="Vast SSH port copied from the SSH dialog.",
    )
    parser.add_argument(
        "--ssh_identity_file",
        default="",
        help="Optional private-key path. The default OpenSSH key is used when omitted.",
    )
    parser.add_argument(
        "--remote_vllm_port",
        type=int,
        default=18000,
        help="Internal vLLM port in Vast's recommended template (default: 18000).",
    )
    parser.add_argument(
        "--local_tunnel_port",
        type=int,
        default=8000,
        help="Local port created by the temporary SSH tunnel (default: 8000).",
    )
    parser.add_argument(
        "--tunnel_ready_timeout_sec",
        type=float,
        default=30.0,
        help="Fail-fast wait for the tunnel and vLLM /v1/models endpoint.",
    )
    parser.add_argument(
        "--confirm_full",
        action="store_true",
        help="Required safety acknowledgement for the 45,328-response full run.",
    )
    args = parser.parse_args(argv)

    if bool(args.ssh_target.strip()) != (args.ssh_port is not None):
        parser.error("--ssh_target and --ssh_port must be provided together")
    for name in ("ssh_port", "remote_vllm_port", "local_tunnel_port"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 65535:
            parser.error(f"--{name} must be between 1 and 65535")
    if args.tunnel_ready_timeout_sec <= 0:
        parser.error("--tunnel_ready_timeout_sec must be greater than 0")
    if args.mode == "full" and not args.confirm_full:
        parser.error(
            "full mode schedules 45,328 responses; rerun with --confirm_full only "
            "after smoke, pilot, JSON inspection, and a cost estimate"
        )

    if args.ssh_target.strip():
        if args.gpu_count != 1:
            parser.error("automatic SSH tunneling currently supports --gpu_count 1")
        tunnel_endpoint = (
            f"http://127.0.0.1:{args.local_tunnel_port}/v1/chat/completions"
        )
        args.ethics_endpoint = tunnel_endpoint
        args.gsm8k_endpoint = tunnel_endpoint

    if args.gsm8k_endpoint is None:
        args.gsm8k_endpoint = (
            args.ethics_endpoint
            if args.gpu_count == 1
            else "http://127.0.0.1:8001/v1/chat/completions"
        )
    if args.concurrency_per_benchmark is None:
        args.concurrency_per_benchmark = 8 if args.gpu_count == 1 else 16
    if args.gpu_count == 1 and args.gsm8k_endpoint != args.ethics_endpoint:
        parser.error("--gpu_count 1 requires ETHICS and GSM8K to use the same endpoint")
    return args


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required file is missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc


def validate_local_inputs() -> None:
    """Fail before renting when the repository does not match the run contract."""
    model_rows = load_json(MODEL_CONFIG_PATH)
    if not isinstance(model_rows, list):
        raise RuntimeError(f"Model config must be a JSON list: {MODEL_CONFIG_PATH}")
    selected = [row for row in model_rows if isinstance(row, dict) and row.get("order") == 43]
    if len(selected) != 1 or selected[0].get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"Model order 43 must map exactly to {MODEL_ID}; found {selected!r}"
        )

    persona_rows = load_json(PERSONA_PATH)
    if not isinstance(persona_rows, list):
        raise RuntimeError(f"Persona file must be a JSON list: {PERSONA_PATH}")
    persona_names = [
        str(row.get("persona") or "").strip()
        for row in persona_rows
        if isinstance(row, dict)
    ]
    if persona_names != REGIONAL_PERSONAS:
        raise RuntimeError(
            "Regional persona file must contain the expected 16 unique personas in "
            f"canonical order. Found: {persona_names}"
        )

    ethics_rows = load_json(ETHICS_SAMPLES_PATH)
    if not isinstance(ethics_rows, list):
        raise RuntimeError(f"ETHICS samples must be a JSON list: {ETHICS_SAMPLES_PATH}")
    ethics_counts = Counter(
        str(row.get("subset") or "").strip()
        for row in ethics_rows
        if isinstance(row, dict)
    )
    if len(ethics_rows) != EXPECTED_ETHICS_PER_PERSONA or dict(ethics_counts) != EXPECTED_ETHICS_SUBSETS:
        raise RuntimeError(
            "ETHICS input does not match the locked 1,514-question split: "
            f"total={len(ethics_rows)}, subsets={dict(ethics_counts)}"
        )

    gsm8k_rows = load_json(GSM8K_SAMPLES_PATH)
    if not isinstance(gsm8k_rows, list) or len(gsm8k_rows) != EXPECTED_GSM8K_PER_PERSONA:
        count = len(gsm8k_rows) if isinstance(gsm8k_rows, list) else "not-a-list"
        raise RuntimeError(
            f"GSM8K input must contain 1,319 rows; found {count}: {GSM8K_SAMPLES_PATH}"
        )

    for relative_path in (
        "src/benchmarks/ethics_runner/run_ethics_prompt_v9.py",
        "src/benchmarks/gsm8k_runner/run_gsm8k_regional_cultural_personas.py",
    ):
        if not (REPO_ROOT / relative_path).is_file():
            raise RuntimeError(f"Required runner is missing: {relative_path}")

    print("[LOCAL] model order 43: correct")
    print("[LOCAL] regional personas: 16/16")
    print(f"[LOCAL] ETHICS samples: {len(ethics_rows)} {dict(ethics_counts)}")
    print(f"[LOCAL] GSM8K samples: {len(gsm8k_rows)}")


def models_url(chat_url: str) -> str:
    suffix = "/chat/completions"
    clean = chat_url.rstrip("/")
    if not clean.endswith(suffix):
        raise RuntimeError(f"Endpoint must end with {suffix}: {chat_url}")
    return clean[: -len(suffix)] + "/models"


def endpoint_model_ids(chat_url: str, timeout_sec: float = 20.0) -> set[str]:
    url = models_url(chat_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Endpoint is not ready at {url}: {exc}") from exc

    return {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def check_endpoint(label: str, chat_url: str, timeout_sec: float = 20.0) -> None:
    url = models_url(chat_url)
    model_ids = endpoint_model_ids(chat_url, timeout_sec=timeout_sec)
    if MODEL_ID not in model_ids:
        raise RuntimeError(
            f"{label} serves {sorted(model_ids)}, not {MODEL_ID}. "
            "Set VLLM_ARGS to include --served-model-name " + MODEL_ID
        )
    print(f"[PREFLIGHT] {label}: ready at {url}; model alias is correct")


def ssh_tunnel_command(args: argparse.Namespace) -> list[str]:
    ssh_exe = shutil.which("ssh")
    if not ssh_exe:
        raise RuntimeError("OpenSSH client 'ssh' is not installed or not on PATH")
    if not args.ssh_target.strip() or args.ssh_port is None:
        raise RuntimeError("Automatic tunnel requires --ssh_target and --ssh_port")

    command = [
        ssh_exe,
        "-N",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-L",
        (
            f"127.0.0.1:{args.local_tunnel_port}:"
            f"127.0.0.1:{args.remote_vllm_port}"
        ),
    ]
    if args.ssh_identity_file.strip():
        identity = Path(args.ssh_identity_file).expanduser()
        if not identity.is_file():
            raise RuntimeError(f"SSH identity file not found: {identity}")
        command.extend(["-i", str(identity)])
    command.extend(["-p", str(args.ssh_port), args.ssh_target.strip()])
    return command


def start_ssh_tunnel(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    """Start a temporary tunnel, or reuse a ready endpoint on the local port."""
    try:
        check_endpoint("existing local endpoint", args.ethics_endpoint, timeout_sec=1.0)
    except RuntimeError:
        pass
    else:
        print("[TUNNEL] Reusing the already-ready local endpoint.")
        return None

    command = ssh_tunnel_command(args)
    rendered = subprocess.list2cmdline(command)
    print(f"[TUNNEL] Starting: {rendered}")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdin=None,
        stdout=subprocess.DEVNULL,
        stderr=None,
        text=True,
    )

    deadline = time.monotonic() + args.tunnel_ready_timeout_sec
    last_error = "endpoint did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"SSH tunnel exited with code {return_code}. Check the Vast SSH "
                "target/port and make sure the SSH key was added before renting."
            )
        try:
            check_endpoint("SSH tunnel", args.ethics_endpoint, timeout_sec=2.0)
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(1.0)
            continue
        return process

    stop_ssh_tunnel(process)
    raise RuntimeError(
        "SSH connected, but vLLM was not ready before the fail-fast timeout. "
        "Open Vast Instance Logs and wait for model loading to complete, then rerun. "
        f"Last check: {last_error}"
    )


def stop_ssh_tunnel(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    print("[TUNNEL] Closing temporary SSH tunnel...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def personas_for_mode(mode: str) -> list[str]:
    return SMOKE_PERSONAS if mode == "smoke" else REGIONAL_PERSONAS


def workload_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.mode == "preflight":
        return {"ethics_main": 0, "gsm8k_main": 0, "warmup": 0, "total": 0}

    personas = personas_for_mode(args.mode)
    implicit = sum("_implicit_" in persona for persona in personas)
    if args.mode == "smoke":
        ethics_per_persona = len(EXPECTED_ETHICS_SUBSETS)
        gsm8k_per_persona = 1
    elif args.mode == "pilot":
        ethics_per_persona = len(EXPECTED_ETHICS_SUBSETS) * args.pilot_ethics_per_subset
        gsm8k_per_persona = args.pilot_gsm8k_samples
    else:
        ethics_per_persona = EXPECTED_ETHICS_PER_PERSONA
        gsm8k_per_persona = EXPECTED_GSM8K_PER_PERSONA

    ethics_main = len(personas) * ethics_per_persona
    gsm8k_main = len(personas) * gsm8k_per_persona
    warmup = implicit * 2  # each implicit persona has one warmup in each runner
    return {
        "ethics_main": ethics_main,
        "gsm8k_main": gsm8k_main,
        "warmup": warmup,
        "total": ethics_main + gsm8k_main + warmup,
    }


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    personas = ",".join(personas_for_mode(args.mode))

    ethics = [
        args.python_exe,
        "src/benchmarks/ethics_runner/run_ethics_prompt_v9.py",
        "--model_order",
        "43",
        "--personas",
        personas,
        "--output_root",
        relative(ETHICS_OUTPUT_ROOT),
        "--endpoint_url",
        args.ethics_endpoint,
        "--max_tokens",
        "4096",
        "--concurrency",
        str(args.concurrency_per_benchmark),
        "--request_timeout_sec",
        str(args.request_timeout_sec),
        "--max_retries",
        "2",
        "--retry_base_sec",
        "2",
        "--sleep_sec",
        "0",
        "--skip_existing",
        "--json_only",
    ]

    gsm8k = [
        args.python_exe,
        "src/benchmarks/gsm8k_runner/run_gsm8k_regional_cultural_personas.py",
        "--model_order",
        "43",
        "--personas",
        personas,
        "--output_root",
        relative(GSM8K_OUTPUT_ROOT),
        "--endpoint_url",
        args.gsm8k_endpoint,
        "--limit_samples",
        str(EXPECTED_GSM8K_PER_PERSONA),
        "--concurrency",
        str(args.concurrency_per_benchmark),
        "--request_timeout_sec",
        str(args.request_timeout_sec),
        "--max_retries",
        "2",
        "--sleep_sec",
        "0",
        "--skip_existing",
    ]

    if args.mode in {"smoke", "pilot"}:
        ethics_limit = 1 if args.mode == "smoke" else args.pilot_ethics_per_subset
        gsm8k_limit = 1 if args.mode == "smoke" else args.pilot_gsm8k_samples
        ethics.extend(["--max_per_subset", str(ethics_limit)])
        gsm_index = gsm8k.index("--limit_samples") + 1
        gsm8k[gsm_index] = str(gsm8k_limit)

    return ethics, gsm8k


def count_persona_files(root: Path, pattern: str) -> dict[str, int]:
    return {
        persona: sum(1 for _ in (root / persona).glob(pattern))
        for persona in REGIONAL_PERSONAS
    }


def verify_outputs(deep: bool = True) -> bool:
    ethics_counts = count_persona_files(ETHICS_MODEL_ROOT, "*.json")
    gsm8k_counts = count_persona_files(GSM8K_MODEL_ROOT, "gsm8k_*.json")
    failures: list[str] = []
    parse_failures = 0

    print("\nETHICS canonical output")
    for persona in REGIONAL_PERSONAS:
        count = ethics_counts[persona]
        print(f"  {persona:34} {count:4}/{EXPECTED_ETHICS_PER_PERSONA}")
        if count != EXPECTED_ETHICS_PER_PERSONA:
            failures.append(f"ETHICS {persona}: {count}")

    print("\nGSM8K canonical output")
    for persona in REGIONAL_PERSONAS:
        count = gsm8k_counts[persona]
        print(f"  {persona:34} {count:4}/{EXPECTED_GSM8K_PER_PERSONA}")
        if count != EXPECTED_GSM8K_PER_PERSONA:
            failures.append(f"GSM8K {persona}: {count}")

    if deep:
        for path in ETHICS_MODEL_ROOT.glob("*/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                failures.append(f"Invalid ETHICS JSON {path}: {exc}")
                continue
            if not (payload.get("response", {}) or {}).get("success"):
                failures.append(f"Failed ETHICS response: {path}")
            if (payload.get("prediction", {}) or {}).get("parsing_failed"):
                parse_failures += 1

        for path in GSM8K_MODEL_ROOT.glob("*/gsm8k_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                failures.append(f"Invalid GSM8K JSON {path}: {exc}")
                continue
            if not (payload.get("response", {}) or {}).get("success"):
                failures.append(f"Failed GSM8K response: {path}")

    print(f"\nETHICS runtime parse failures: {parse_failures}")
    if failures:
        print(f"Audit found {len(failures)} incomplete/failed item(s).")
        for failure in failures[:20]:
            print(f"  - {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return False

    print("Audit complete: expected files exist and all API calls succeeded.")
    return True


def print_plan(ethics: list[str], gsm8k: list[str], args: argparse.Namespace) -> None:
    personas = personas_for_mode(args.mode)
    calls = workload_counts(args)
    print("Canonical destinations")
    print(f"  ETHICS: {ETHICS_MODEL_ROOT}")
    print(f"  GSM8K : {GSM8K_MODEL_ROOT}")
    print("\nScope")
    print(f"  Mode: {args.mode}")
    print(f"  Personas: {len(personas)} (neutral excluded)")
    print(f"  GPU setup: {args.gpu_count} GPU(s)")
    print(f"  Shared endpoint: {'yes' if args.ethics_endpoint == args.gsm8k_endpoint else 'no'}")
    print(f"  Parallel requests: {args.concurrency_per_benchmark} ETHICS + "
          f"{args.concurrency_per_benchmark} GSM8K")
    print(f"  Planned ETHICS calls: {calls['ethics_main']}")
    print(f"  Planned GSM8K calls : {calls['gsm8k_main']}")
    print(f"  Maximum warmup calls: {calls['warmup']} (cached warmups are reused)")
    print(f"  Maximum total calls : {calls['total']}")
    if args.mode != "preflight":
        print("\nCommands")
        print("  ETHICS:", subprocess.list2cmdline(ethics))
        print("  GSM8K :", subprocess.list2cmdline(gsm8k))


def stream_process(label: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def launch_pair(ethics: list[str], gsm8k: list[str]) -> int:
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for label, command in (("ETHICS", ethics), ("GSM8K", gsm8k)):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        processes.append((label, process))

    threads = [
        threading.Thread(target=stream_process, args=(label, process), daemon=True)
        for label, process in processes
    ]
    for thread in threads:
        thread.start()

    try:
        return_codes = {label: process.wait() for label, process in processes}
    except KeyboardInterrupt:
        print("\nStopping both local runners...")
        for _label, process in processes:
            if process.poll() is None:
                process.terminate()
        return_codes = {label: process.wait() for label, process in processes}

    for thread in threads:
        thread.join(timeout=5)

    print("\nExit codes:", return_codes)
    return 0 if all(code == 0 for code in return_codes.values()) else 1


def run(args: argparse.Namespace) -> int:
    if args.concurrency_per_benchmark < 1:
        raise RuntimeError("--concurrency_per_benchmark must be at least 1")
    if args.request_timeout_sec <= 0:
        raise RuntimeError("--request_timeout_sec must be greater than 0")

    validate_local_inputs()

    if args.mode == "verify":
        return 0 if verify_outputs(deep=True) else 1

    ethics, gsm8k = build_commands(args)
    print_plan(ethics, gsm8k, args)
    if args.mode == "plan":
        print("\nPlan complete: local inputs are valid; no API calls or files were written.")
        return 0

    tunnel: subprocess.Popen[str] | None = None
    try:
        if args.ssh_target.strip():
            tunnel = start_ssh_tunnel(args)

        check_endpoint("ETHICS", args.ethics_endpoint)
        if args.gsm8k_endpoint == args.ethics_endpoint:
            print("[PREFLIGHT] GSM8K: using the same ready endpoint as ETHICS")
        else:
            check_endpoint("GSM8K", args.gsm8k_endpoint)

        if args.mode == "preflight":
            print("\nPreflight complete: model is ready; no generation calls or files were written.")
            return 0
        return launch_pair(ethics, gsm8k)
    finally:
        stop_ssh_tunnel(tunnel)


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

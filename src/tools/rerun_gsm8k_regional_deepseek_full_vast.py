"""Run a fresh, resumable DeepSeek regional GSM8K experiment on Vast.ai.

This launcher intentionally writes raw JSON and reports to an isolated rerun
tree.  It never overwrites the canonical experiment that produced the current
``result/benchmarks/gsm8k/reports/Regional_Cultural_Persona/.../results.csv``.

Progress through plan -> preflight -> smoke -> pilot -> full -> verify -> report.
All generation modes use the existing regional GSM8K runner and a vLLM endpoint
reached directly or through the existing Vast SSH tunnel helper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.tools import run_deepseek_regional_vast as common  # noqa: E402


REPO_ROOT = common.REPO_ROOT
MODEL_ID = common.MODEL_ID
MODEL_SLUG = common.MODEL_SLUG
EXPECTED_PER_PERSONA = common.EXPECTED_GSM8K_PER_PERSONA
PERSONAS = common.REGIONAL_PERSONAS

CANONICAL_RAW_ROOT = REPO_ROOT / "result" / "GMS8K" / "regional_cultural_persona"
CANONICAL_REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
RERUNS_ROOT = REPO_ROOT / "result" / "GMS8K" / "reruns"

DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reruns"
    / "20260815_deepseek_regional_full_vast"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "raw"
DEFAULT_REPORT_ROOT = DEFAULT_RUN_ROOT / "reports"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

REGION_SHARDS = {
    "north_american": [p for p in PERSONAS if p.startswith("north_american_")],
    "east_asian": [p for p in PERSONAS if p.startswith("east_asian_")],
    "south_asian": [p for p in PERSONAS if p.startswith("south_asian_")],
    "sub_saharan_african": [
        p for p in PERSONAS if p.startswith("sub_saharan_african_")
    ],
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh full rerun of DeepSeek R1 Distill Qwen 32B on all 16 "
            "Regional Cultural Persona GSM8K variants."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "smoke", "pilot", "full", "verify", "report"),
        default="plan",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument("--pilot-samples", type=int, default=8)
    parser.add_argument("--concurrency-per-shard", type=int, default=45)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--confirm-full", action="store_true")

    parser.add_argument("--ssh-target", default="")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--ssh-identity-file", default="")
    parser.add_argument("--remote-vllm-port", type=int, default=18000)
    parser.add_argument("--local-tunnel-port", type=int, default=8000)
    parser.add_argument("--tunnel-ready-timeout-sec", type=float, default=1800.0)

    args = parser.parse_args(argv)
    args.output_root = repo_path(args.output_root)
    args.report_root = repo_path(args.report_root)

    if args.mode == "full" and not args.confirm_full:
        parser.error(
            "full mode schedules 21,104 main responses; use --confirm-full only "
            "after plan, preflight, smoke, pilot, and a cost check"
        )
    if bool(args.ssh_target.strip()) != (args.ssh_port is not None):
        parser.error("--ssh-target and --ssh-port must be provided together")
    for name in ("ssh_port", "remote_vllm_port", "local_tunnel_port"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if args.pilot_samples < 1 or args.pilot_samples > EXPECTED_PER_PERSONA:
        parser.error(f"--pilot-samples must be between 1 and {EXPECTED_PER_PERSONA}")
    if args.concurrency_per_shard < 1:
        parser.error("--concurrency-per-shard must be at least 1")
    if args.request_timeout_sec <= 0 or args.tunnel_ready_timeout_sec <= 0:
        parser.error("timeout values must be greater than 0")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    if args.ssh_target.strip():
        args.endpoint_url = (
            f"http://127.0.0.1:{args.local_tunnel_port}/v1/chat/completions"
        )

    # Names expected by the shared SSH tunnel helper.
    args.ethics_endpoint = args.endpoint_url
    args.ssh_target = args.ssh_target.strip()
    return args


def ensure_isolated_paths(output_root: Path, report_root: Path) -> None:
    for label, path in (("output", output_root), ("report", report_root)):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"{label} root must stay inside the repository: {path}") from exc
        if RERUNS_ROOT not in path.parents:
            raise RuntimeError(
                f"{label} root must be a child of {RERUNS_ROOT}: {path}"
            )

    if output_root == CANONICAL_RAW_ROOT or CANONICAL_RAW_ROOT in output_root.parents:
        raise RuntimeError(
            "Fresh rerun output cannot be the canonical raw tree; choose an isolated "
            f"path under result/benchmarks/gsm8k/reruns: {output_root}"
        )
    if report_root == CANONICAL_REPORT_ROOT or CANONICAL_REPORT_ROOT in report_root.parents:
        raise RuntimeError(
            "Fresh rerun reports cannot overwrite canonical reports; choose an "
            f"isolated path under result/benchmarks/gsm8k/reruns: {report_root}"
        )
    if output_root == report_root or output_root in report_root.parents or report_root in output_root.parents:
        raise RuntimeError("raw output and report roots must be separate sibling trees")


def validate_local_inputs(output_root: Path, report_root: Path) -> None:
    ensure_isolated_paths(output_root, report_root)

    model_rows = common.load_json(common.MODEL_CONFIG_PATH)
    selected = [
        row
        for row in model_rows
        if isinstance(row, dict) and row.get("order") == 43
    ] if isinstance(model_rows, list) else []
    if len(selected) != 1 or selected[0].get("model_id") != MODEL_ID:
        raise RuntimeError(f"Model order 43 must map exactly to {MODEL_ID}")

    persona_rows = common.load_json(common.PERSONA_PATH)
    persona_names = [
        str(row.get("persona") or "").strip()
        for row in persona_rows
        if isinstance(row, dict)
    ] if isinstance(persona_rows, list) else []
    if persona_names != PERSONAS:
        raise RuntimeError(
            "Regional persona input does not match the locked 16-persona order"
        )

    samples = common.load_json(common.GSM8K_SAMPLES_PATH)
    if not isinstance(samples, list) or len(samples) != EXPECTED_PER_PERSONA:
        count = len(samples) if isinstance(samples, list) else "not-a-list"
        raise RuntimeError(f"GSM8K input must contain 1,319 rows; found {count}")

    runner = REPO_ROOT / "src" / "benchmarks" / "gsm8k_runner" / "run_gsm8k_regional_cultural_personas.py"
    parser = REPO_ROOT / "src" / "benchmarks" / "gsm8k" / "Regional_Cultural_Persona" / "run.py"
    if not runner.is_file() or not parser.is_file():
        raise RuntimeError("Required GSM8K runner/parser is missing")

    flattened = [persona for shard in REGION_SHARDS.values() for persona in shard]
    if flattened != PERSONAS or len(flattened) != len(set(flattened)):
        raise RuntimeError("Regional shards must be disjoint and preserve persona order")

    print(f"[LOCAL] model order 43: {MODEL_ID}")
    print(f"[LOCAL] regional personas: {len(PERSONAS)}/16 in 4 disjoint shards")
    print(f"[LOCAL] GSM8K samples: {len(samples)}/1,319")
    print(f"[LOCAL] canonical report preserved: {CANONICAL_REPORT_ROOT}")


def personas_and_limit(mode: str, pilot_samples: int) -> tuple[list[str], int]:
    if mode == "smoke":
        return ["north_american_explicit", "north_american_implicit_1"], 1
    if mode == "pilot":
        return PERSONAS, pilot_samples
    return PERSONAS, EXPECTED_PER_PERSONA


def workload_counts(mode: str, pilot_samples: int) -> dict[str, int]:
    if mode in {"preflight", "verify", "report"}:
        return {"main": 0, "warmup": 0, "maximum": 0}
    planned_mode = "full" if mode == "plan" else mode
    personas, limit = personas_and_limit(planned_mode, pilot_samples)
    warmups = sum("_implicit_" in persona for persona in personas)
    main = len(personas) * limit
    return {"main": main, "warmup": warmups, "maximum": main + warmups}


def mode_shards(mode: str) -> dict[str, list[str]]:
    if mode == "smoke":
        return {
            "smoke": ["north_american_explicit", "north_american_implicit_1"]
        }
    return REGION_SHARDS


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    _, limit = personas_and_limit(args.mode, args.pilot_samples)
    commands: list[tuple[str, list[str]]] = []
    for label, personas in mode_shards(args.mode).items():
        concurrency = (
            min(8, args.concurrency_per_shard)
            if args.mode == "smoke"
            else args.concurrency_per_shard
        )
        command = [
            args.python_exe,
            "src/benchmarks/gsm8k_runner/run_gsm8k_regional_cultural_personas.py",
            "--model_order",
            "43",
            "--personas",
            ",".join(personas),
            "--output_root",
            str(args.output_root),
            "--endpoint_url",
            args.endpoint_url,
            "--limit_samples",
            str(limit),
            "--concurrency",
            str(concurrency),
            "--request_timeout_sec",
            str(args.request_timeout_sec),
            "--max_retries",
            str(args.max_retries),
            "--sleep_sec",
            "0",
            "--skip_existing",
        ]
        commands.append((label, command))
    return commands


def completion_problem(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return f"invalid_json:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return "invalid_payload"

    response = payload.get("response")
    if not isinstance(response, dict) or response.get("success") is not True:
        return "response_not_successful"
    content = response.get("content")
    if not isinstance(content, str) or not content.strip():
        return "empty_final_content"

    raw = response.get("raw_api_response")
    choices = raw.get("choices") if isinstance(raw, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
    if finish_reason != "stop":
        return f"finish_reason_{finish_reason or 'missing'}"
    return None


def quarantine_incomplete(output_root: Path) -> tuple[int, Path | None]:
    model_root = output_root / MODEL_SLUG
    candidates = [
        path
        for persona in PERSONAS
        for path in (model_root / persona).glob("gsm8k_*.json")
        if path.is_file()
    ]
    incomplete = [(path, completion_problem(path)) for path in candidates]
    incomplete = [(path, reason) for path, reason in incomplete if reason]
    if not incomplete:
        return 0, None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    quarantine_root = output_root.parent / "incomplete_attempts" / stamp
    manifest: list[dict[str, str]] = []
    for path, reason in incomplete:
        relative = path.relative_to(model_root)
        target = quarantine_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        manifest.append(
            {
                "source": str(path.relative_to(REPO_ROOT)),
                "quarantined_to": str(target.relative_to(REPO_ROOT)),
                "reason": str(reason),
            }
        )
    (quarantine_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(incomplete), quarantine_root


def audit_outputs(
    output_root: Path,
    personas: list[str],
    expected_per_persona: int,
) -> tuple[bool, dict[str, Any]]:
    model_root = output_root / MODEL_SLUG
    counts: dict[str, int] = {}
    problems: Counter[str] = Counter()
    examples: list[str] = []

    for persona in personas:
        paths = sorted((model_root / persona).glob("gsm8k_*.json"))
        counts[persona] = len(paths)
        if len(paths) != expected_per_persona:
            problems["wrong_file_count"] += 1
            if len(examples) < 20:
                examples.append(
                    f"{persona}: {len(paths)}/{expected_per_persona} files"
                )
        for path in paths:
            reason = completion_problem(path)
            if reason:
                problems[reason] += 1
                if len(examples) < 20:
                    examples.append(f"{persona}/{path.name}: {reason}")

    complete = not problems
    return complete, {
        "counts": counts,
        "problems": dict(problems),
        "examples": examples,
    }


def print_audit(summary: dict[str, Any], expected: int) -> None:
    print("\nOutput audit")
    for persona, count in summary["counts"].items():
        print(f"  {persona:34} {count:4}/{expected}")
    if summary["problems"]:
        print("Problems:", summary["problems"])
        for example in summary["examples"]:
            print(f"  - {example}")
    else:
        print("All expected files have successful, non-empty, finish_reason=stop responses.")


def stream_process(label: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def launch_many(commands: list[tuple[str, list[str]]]) -> int:
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for label, command in commands:
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
        print("\nStopping all local shard runners...")
        for _label, process in processes:
            if process.poll() is None:
                process.terminate()
        return_codes = {label: process.wait() for label, process in processes}

    for thread in threads:
        thread.join(timeout=5)
    print("\nExit codes:", return_codes)
    return 0 if all(code == 0 for code in return_codes.values()) else 1


def report_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python_exe,
        "-m",
        "src.benchmarks.gsm8k.Regional_Cultural_Persona.run",
        "--model",
        MODEL_SLUG,
        "--input_root",
        str(args.output_root),
        "--out_root",
        str(args.report_root),
    ]


def print_plan(args: argparse.Namespace, commands: list[tuple[str, list[str]]]) -> None:
    calls = workload_counts(args.mode, args.pilot_samples)
    print("\nFresh GSM8K regional rerun")
    print(f"  Mode             : {args.mode}")
    print(f"  Raw output       : {args.output_root / MODEL_SLUG}")
    print(
        "  New results.csv  : "
        f"{args.report_root / 'Regional_Cultural_Persona' / MODEL_SLUG / 'results.csv'}"
    )
    print(f"  Canonical intact : {CANONICAL_RAW_ROOT / MODEL_SLUG}")
    print(f"  Main calls       : {calls['main']}")
    print(f"  Warmup calls max : {calls['warmup']}")
    print(f"  Total calls max  : {calls['maximum']}")
    if commands:
        print(f"  Shard processes  : {len(commands)}")
        print(f"  Workers/shard    : {args.concurrency_per_shard}")
        for label, command in commands:
            print(f"  [{label}] {subprocess.list2cmdline(command)}")


def run(args: argparse.Namespace) -> int:
    validate_local_inputs(args.output_root, args.report_root)

    if args.mode == "verify":
        complete, summary = audit_outputs(
            args.output_root, PERSONAS, EXPECTED_PER_PERSONA
        )
        print_audit(summary, EXPECTED_PER_PERSONA)
        return 0 if complete else 1

    if args.mode == "report":
        complete, summary = audit_outputs(
            args.output_root, PERSONAS, EXPECTED_PER_PERSONA
        )
        print_audit(summary, EXPECTED_PER_PERSONA)
        if not complete:
            raise RuntimeError("Refusing to build reports from incomplete raw output")
        command = report_command(args)
        print("\nRunning parser:", subprocess.list2cmdline(command))
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode

    commands = [] if args.mode == "preflight" else build_commands(args)
    print_plan(args, commands)
    if args.mode == "plan":
        print("\nPlan complete: no API calls or files were written.")
        return 0

    tunnel: subprocess.Popen[str] | None = None
    try:
        if args.ssh_target:
            tunnel = common.start_ssh_tunnel(args)
        common.check_endpoint("GSM8K", args.endpoint_url)
        if args.mode == "preflight":
            print("\nPreflight complete: endpoint ready; no generation calls were made.")
            return 0

        quarantined, quarantine_root = quarantine_incomplete(args.output_root)
        if quarantined:
            print(
                f"[RESUME] Quarantined {quarantined} incomplete prior attempt(s) "
                f"under {quarantine_root}"
            )

        result = launch_many(commands)
        personas, expected = personas_and_limit(args.mode, args.pilot_samples)
        complete, summary = audit_outputs(args.output_root, personas, expected)
        print_audit(summary, expected)
        if result != 0 or not complete:
            print(
                "Rerun the same command to resume. Incomplete attempts will be "
                "preserved and retried."
            )
            return 1
        return 0
    finally:
        common.stop_ssh_tunnel(tunnel)


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

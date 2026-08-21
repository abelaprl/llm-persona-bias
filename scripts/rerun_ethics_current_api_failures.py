"""Validate or execute only the current ETHICS API-failure snapshot.

The launcher delegates to the repository's audited selective runners:

* DeepSeek R1 Distill -> OpenAI-compatible vLLM endpoint.
* GPT-OSS 20B and Qwen3 Next -> OpenRouter credentials from model_keys.json.

Safe default: validation only. There is intentionally no ``--force`` option;
successful manifest targets are skipped by the child runners during resume.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "ethics_v9"
    / "api_failures_current_2026-08-21"
)


@dataclass(frozen=True)
class Job:
    label: str
    target: str
    script: str
    manifest: str
    expected_count: int
    scope: str | None = None


JOBS = (
    Job(
        "DeepSeek vLLM",
        "deepseek",
        "rerun_ethics_v9_deepseek_errors_vast.py",
        "deepseek_vllm_104.csv",
        104,
    ),
    Job(
        "GPT-OSS OpenRouter nonregional",
        "gpt",
        "rerun_gpt_oss_20b_openrouter.py",
        "gpt_oss_openrouter_nonregional_47.csv",
        47,
        "nonregional",
    ),
    Job(
        "GPT-OSS OpenRouter regional",
        "gpt",
        "rerun_gpt_oss_20b_openrouter.py",
        "gpt_oss_openrouter_regional_9.csv",
        9,
        "regional",
    ),
    Job(
        "Qwen OpenRouter nonregional",
        "qwen",
        "rerun_qwen3_next_80b_nonregional_openrouter.py",
        "qwen_openrouter_nonregional_46.csv",
        46,
        "nonregional",
    ),
    Job(
        "Qwen OpenRouter regional",
        "qwen",
        "rerun_qwen3_next_80b_nonregional_openrouter.py",
        "qwen_openrouter_regional_13.csv",
        13,
        "regional",
    ),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", "openrouter", "deepseek", "gpt", "qwen"),
        default="all",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--limit-per-manifest",
        type=positive_int,
        default=None,
        help="Smoke-test limit forwarded independently to every selected manifest.",
    )
    parser.add_argument("--openrouter-concurrency", type=positive_int, default=2)
    parser.add_argument("--vllm-concurrency", type=positive_int, default=4)
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        default=0,
        help=(
            "Retries per target after the first call. Default 0 avoids repeated "
            "paid generations; resume the immutable manifest instead if needed."
        ),
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "model_keys.json")
    )
    parser.add_argument(
        "--openrouter-endpoint-url",
        default=os.environ.get(
            "OPENROUTER_ENDPOINT_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    )
    parser.add_argument(
        "--vllm-endpoint-url",
        default=os.environ.get(
            "VLLM_ENDPOINT_URL", "http://127.0.0.1:8000/v1/chat/completions"
        ),
    )
    return parser.parse_args()


def selected(job: Job, target: str) -> bool:
    if target == "all":
        return True
    if target == "openrouter":
        return job.target in {"gpt", "qwen"}
    return job.target == target


def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / job.script),
        "--manifest",
        str(MANIFEST_ROOT / job.manifest),
        "--expected-count",
        str(job.expected_count),
        "--dataset",
        "ethics",
    ]
    if job.target == "deepseek":
        # The DeepSeek runner has no dataset switch and custom manifests
        # automatically disable its historical extra manifest.
        command = command[:-2]
        command.extend(
            [
                "--no-extra-manifest",
                "--endpoint-url",
                args.vllm_endpoint_url,
                "--concurrency",
                str(args.vllm_concurrency),
                "--max-retries",
                str(args.max_retries),
            ]
        )
    else:
        assert job.scope is not None
        command.extend(
            [
                "--scope",
                job.scope,
                "--config",
                str(Path(args.config).expanduser()),
                "--endpoint-url",
                args.openrouter_endpoint_url,
                "--concurrency",
                str(args.openrouter_concurrency),
                "--max-retries",
                str(args.max_retries),
            ]
        )
    if args.limit_per_manifest is not None:
        command.extend(["--limit", str(args.limit_per_manifest)])
    if args.execute:
        command.append("--execute")
    return command


def main() -> int:
    args = parse_args()
    jobs = [job for job in JOBS if selected(job, args.target)]
    if not jobs:
        raise RuntimeError(f"No jobs selected for target={args.target}")

    mode = "EXECUTE" if args.execute else "VALIDATE ONLY"
    print(f"Mode: {mode}; jobs={len(jobs)}; force option is disabled", flush=True)
    failures: list[tuple[str, int]] = []
    for index, job in enumerate(jobs, start=1):
        manifest = MANIFEST_ROOT / job.manifest
        if not manifest.is_file():
            raise RuntimeError(f"Missing manifest: {manifest}")
        print(f"\n{'=' * 80}\n[{index}/{len(jobs)}] {job.label}\n{'=' * 80}", flush=True)
        completed = subprocess.run(command_for(job, args), cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            failures.append((job.label, completed.returncode))

    if failures:
        print(f"\nJobs with non-zero status: {failures}", file=sys.stderr)
        return 1
    print("\nAll selected jobs completed with zero status.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)

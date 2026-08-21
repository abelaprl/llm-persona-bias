"""Rerun only the exact 41-row GSM8K Religion OpenRouter manifest.

Safe default is validation-only.  ``--execute`` is required for API calls.
Each existing target is backed up immediately before its request.  A failed,
empty, truncated, filtered, or error-finished attempt never replaces source.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import rerun_qwen3_next_80b_nonregional_openrouter as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = (
    REPO_ROOT / "result" / "rerun_manifests" / "gsm8k_religion_41_openrouter"
)
BACKUP_ROOT = REPO_ROOT / "result" / "_rerun_backup" / "gsm8k_religion_41"
ATTEMPT_ROOT = REPO_ROOT / "result" / "_rerun_attempts" / "gsm8k_religion_41"
GROUP = "Religion_Persona"


@dataclass(frozen=True)
class RunScope:
    key: str
    order: int
    model_id: str
    slug: str
    label: str
    expected_count: int
    manifest_name: str


SCOPES = (
    RunScope(
        key="gpt_oss",
        order=31,
        model_id="openai/gpt-oss-20b",
        slug="openai_gpt-oss-20b",
        label="GPT-OSS-20B",
        expected_count=39,
        manifest_name="gpt_oss_religion_39.csv",
    ),
    RunScope(
        key="qwen",
        order=32,
        model_id="qwen/qwen3-next-80b-a3b-thinking",
        slug="qwen_qwen3-next-80b-a3b-thinking",
        label="Qwen3 Next 80B",
        expected_count=2,
        manifest_name="qwen_religion_2.csv",
    ),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("all", "gpt_oss", "qwen"), default="all"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=positive_int, default=None,
                        help="Smoke-test limit per selected model.")
    parser.add_argument("--concurrency", type=positive_int, default=2)
    parser.add_argument("--max-tokens", type=positive_int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=4.0)
    parser.add_argument("--request-timeout-sec", type=float, default=600.0)
    parser.add_argument("--preflight-timeout-sec", type=float, default=30.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--endpoint-url", default=runner.DEFAULT_ENDPOINT_URL,
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "model_keys.json")
    )
    args = parser.parse_args(argv)
    if args.max_retries < 0:
        parser.error("--max-retries must be zero or greater")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be zero or greater")
    return args


def validate_manifest(scope: RunScope, path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != scope.expected_count:
        raise RuntimeError(
            f"{scope.key}: expected {scope.expected_count} rows, found {len(rows)}"
        )
    seen = set()
    for number, row in enumerate(rows, start=2):
        relative = str(row.get("relative_path") or "").replace("\\", "/")
        expected_prefix = f"{GROUP}/{scope.slug}/"
        if (
            row.get("dataset") != "GSM8K"
            or row.get("group") != GROUP
            or row.get("model_slug") != scope.slug
            or not relative.startswith(expected_prefix)
        ):
            raise RuntimeError(f"Forbidden manifest scope at row {number}: {relative}")
        if relative in seen:
            raise RuntimeError(f"Duplicate manifest target: {relative}")
        seen.add(relative)


def configure_runner(scope: RunScope) -> None:
    runner.MODEL_ORDER = scope.order
    runner.MODEL_ID = scope.model_id
    runner.MODEL_SLUG = scope.slug
    runner.MODEL_LABEL = scope.label
    runner.RUNNER_DESCRIPTION = __doc__
    runner.RERUN_TAG_PREFIX = f"gsm8k_religion_41_{scope.key}"
    runner.PROVIDER_LABEL = "openrouter"
    runner.TRANSPORT_LABEL = "openrouter"
    runner.API_SERVICE_LABEL = "OpenRouter"
    runner.AUTH_REQUIRED = True
    runner.CREDENTIAL_ORDER = None
    runner.CREDENTIAL_MODEL_ID = None


def forwarded_args(args: argparse.Namespace, scope: RunScope) -> list[str]:
    manifest = MANIFEST_ROOT / scope.manifest_name
    values = [
        "--scope", "nonregional",
        "--manifest", str(manifest),
        "--dataset", "gsm8k",
        "--expected-count", str(scope.expected_count),
        "--backup-root", str(BACKUP_ROOT / scope.key),
        "--attempt-root", str(ATTEMPT_ROOT / scope.key),
        "--config", str(args.config),
        "--endpoint-url", str(args.endpoint_url),
        "--concurrency", str(args.concurrency),
        "--max-retries", str(args.max_retries),
        "--retry-base-sec", str(args.retry_base_sec),
        "--request-timeout-sec", str(args.request_timeout_sec),
        "--preflight-timeout-sec", str(args.preflight_timeout_sec),
    ]
    if args.execute:
        values.append("--execute")
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    if args.max_tokens is not None:
        values.extend(["--max-tokens", str(args.max_tokens)])
    if args.temperature is not None:
        values.extend(["--temperature", str(args.temperature)])
    if args.skip_preflight:
        values.append("--skip-preflight")
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = [scope for scope in SCOPES if args.model in ("all", scope.key)]
    exit_code = 0
    for scope in selected:
        manifest = MANIFEST_ROOT / scope.manifest_name
        validate_manifest(scope, manifest)
        configure_runner(scope)
        print(f"\nLOCKED SCOPE: {scope.label} {scope.expected_count} target(s)")
        result = runner.run(runner.parse_args(forwarded_args(args, scope)))
        exit_code = max(exit_code, result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

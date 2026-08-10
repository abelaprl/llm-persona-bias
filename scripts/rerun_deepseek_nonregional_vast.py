"""One-command launcher for all audited non-regional DeepSeek reruns on Vast."""

from __future__ import annotations

import argparse
import os
import sys

import rerun_ethics_v9_http_errors_vast as ethics
import rerun_gsm8k_deepseek_errors_vast as gsm8k


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the exact 59 ETHICS + 684 GSM8K DeepSeek "
            "non-regional rerun targets."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "ethics", "gsm8k"),
        default="all",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("VLLM_ENDPOINT_URL", DEFAULT_ENDPOINT),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Select the first N pending targets per chosen dataset (smoke test).",
    )
    parser.add_argument("--max-retries", type=nonnegative_int, default=4)
    parser.add_argument("--retry-base-sec", type=float, default=4.0)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--endpoint-check-timeout-sec", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.retry_base_sec <= 0:
        parser.error("--retry-base-sec must be greater than 0")
    if args.request_timeout_sec <= 0:
        parser.error("--request-timeout-sec must be greater than 0")
    if args.endpoint_check_timeout_sec <= 0:
        parser.error("--endpoint-check-timeout-sec must be greater than 0")
    return args


def runner_arguments(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--endpoint-url",
        args.endpoint_url,
        "--concurrency",
        str(args.concurrency),
        "--max-retries",
        str(args.max_retries),
        "--retry-base-sec",
        str(args.retry_base_sec),
        "--request-timeout-sec",
        str(args.request_timeout_sec),
        "--endpoint-check-timeout-sec",
        str(args.endpoint_check_timeout_sec),
    ]
    if args.execute:
        forwarded.append("--execute")
    if args.force:
        forwarded.append("--force")
    if args.limit is not None:
        forwarded.extend(("--limit", str(args.limit)))
    return forwarded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    forwarded = runner_arguments(args)
    selected_runners = []
    if args.dataset in ("all", "ethics"):
        selected_runners.append(("ETHICS", ethics.parse_args, ethics.run))
    if args.dataset in ("all", "gsm8k"):
        selected_runners.append(("GSM8K", gsm8k.parse_args, gsm8k.run))

    results: dict[str, int] = {}
    for label, child_parse_args, child_run in selected_runners:
        print(f"\n{'#' * 80}\n# {label}\n{'#' * 80}", flush=True)
        try:
            results[label] = child_run(child_parse_args(forwarded))
        except RuntimeError as exc:
            print(f"[{label} ERROR] {exc}", file=sys.stderr, flush=True)
            results[label] = 2

    print(f"\nCombined result: {results}")
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

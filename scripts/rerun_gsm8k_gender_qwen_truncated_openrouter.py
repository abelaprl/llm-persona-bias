"""Rerun only the 317 truncated GSM8K Gender Qwen rows via OpenRouter.

Safe default is a dry-run. Successful responses are always written to an
isolated tree; canonical source JSON and report CSV files are never replaced.

Dry-run::

    python scripts/rerun_gsm8k_gender_qwen_truncated_openrouter.py

One-call smoke test::

    python scripts/rerun_gsm8k_gender_qwen_truncated_openrouter.py \
      --execute --limit 1 --concurrency 1

Resume all remaining targets::

    python scripts/rerun_gsm8k_gender_qwen_truncated_openrouter.py \
      --execute --concurrency 4
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rerun_qwen3_next_80b_nonregional_openrouter as runner  # noqa: E402


EXPECTED_COUNT = 317
MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_gender_qwen_truncated_4096"
    / "gsm8k_gender_qwen_truncated_317.csv"
)
SUCCESS_ROOT = (
    REPO_ROOT
    / "result"
    / "_rerun_success"
    / "gsm8k_gender_qwen_truncated_4096"
)
ATTEMPT_ROOT = (
    REPO_ROOT
    / "result"
    / "_rerun_attempts"
    / "gsm8k_gender_qwen_truncated_4096"
)
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def configure_runner() -> None:
    runner.MODEL_LABEL = "Qwen3 Next 80B GSM8K Gender truncated-317"
    runner.RUNNER_DESCRIPTION = __doc__
    runner.RERUN_TAG_PREFIX = "gsm8k_gender_qwen_truncated_4096"
    runner.PROVIDER_LABEL = "openrouter"
    runner.TRANSPORT_LABEL = "openrouter"
    runner.API_SERVICE_LABEL = "OpenRouter"
    runner.AUTH_REQUIRED = True
    runner.CREDENTIAL_ORDER = None
    runner.CREDENTIAL_MODEL_ID = None
    runner.DEFAULT_ENDPOINT_URL = OPENROUTER_ENDPOINT


def parse_args(argv: list[str] | None = None):
    configure_runner()
    fixed = [
        "--scope",
        "nonregional",
        "--manifest",
        str(MANIFEST),
        "--dataset",
        "gsm8k",
        "--expected-count",
        str(EXPECTED_COUNT),
        "--success-root",
        str(SUCCESS_ROOT),
        "--attempt-root",
        str(ATTEMPT_ROOT),
        "--max-tokens",
        "4096",
        "--temperature",
        "0",
        "--endpoint-url",
        OPENROUTER_ENDPOINT,
    ]
    args = runner.parse_args(fixed + list(argv or []))
    exact_guards = {
        "scope": "nonregional",
        "dataset": "gsm8k",
        "expected_count": EXPECTED_COUNT,
        "max_tokens": 4096,
        "temperature": 0.0,
        "endpoint_url": OPENROUTER_ENDPOINT,
        "manifest": MANIFEST.resolve(),
        "success_root": SUCCESS_ROOT.resolve(),
        "attempt_root": ATTEMPT_ROOT.resolve(),
    }
    for name, expected in exact_guards.items():
        actual = getattr(args, name)
        actual = actual.resolve() if isinstance(actual, Path) else actual
        if actual != expected:
            raise RuntimeError(
                f"Fixed-scope option --{name.replace('_', '-')} must remain {expected}"
            )
    if args.force:
        raise RuntimeError("--force is forbidden for this isolated exact-scope rerun")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        return runner.run(args)
    except RuntimeError as exc:
        print(f"[ERROR] {runner.sanitize_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

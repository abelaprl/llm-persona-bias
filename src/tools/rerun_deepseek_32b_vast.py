"""Selectively rerun DeepSeek 32B GSM8K Religion provider errors on vLLM.

Safe default: validate the exact 552-row manifest without API calls or writes.
Use ``--execute`` only after the Vast instance is running and its vLLM server
is reachable through the local tunnel at ``127.0.0.1:8000``.
"""

from __future__ import annotations

from pathlib import Path
import sys

from src.tools import rerun_qwen3_next_80b_nonregional_openrouter as runner


runner.MODEL_ORDER = 43
runner.MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
runner.MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
runner.MODEL_LABEL = "DeepSeek R1 Distill Qwen 32B"
runner.RUNNER_DESCRIPTION = __doc__
runner.RERUN_TAG_PREFIX = "deepseek_32b"
runner.PROVIDER_LABEL = "vastai"
runner.TRANSPORT_LABEL = "vllm"
runner.API_SERVICE_LABEL = "local vLLM"
runner.AUTH_REQUIRED = False
runner.DEFAULT_ENDPOINT_URL = "http://127.0.0.1:8000/v1/chat/completions"
runner.ENDPOINT_ENV_VAR = "DEEPSEEK_VLLM_ENDPOINT_URL"
runner.NONREGIONAL_EXPECTED_COUNT = 552
runner.NONREGIONAL_SCOPE_DIRECTORY = "deepseek_32b_religion_provider_errors"

manifest_root = Path(runner.REPO_ROOT) / "result" / "rerun_manifests"
runner.NONREGIONAL_MANIFEST = (
    manifest_root
    / "gsm8k_deepseek"
    / "deepseek_religion_provider_errors.csv"
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = runner.parse_args()
    if args.temperature not in (None, 0, 0.0):
        print("[ERROR] DeepSeek Religion reruns require --temperature 0", file=sys.stderr)
        return 2
    if args.max_tokens not in (None, 4096):
        print("[ERROR] DeepSeek Religion reruns require --max-tokens 4096", file=sys.stderr)
        return 2
    args.temperature = 0.0
    args.max_tokens = 4096
    try:
        return runner.run(args)
    except RuntimeError as exc:
        print(f"[ERROR] {runner.sanitize_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

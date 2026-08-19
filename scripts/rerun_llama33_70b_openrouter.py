"""Selectively rerun audited Llama 3.3 70B failures through OpenRouter.

Safe default: validate the exact manifest without API calls or writes.
Use ``--execute`` only after a one-target smoke test succeeds.
"""

from __future__ import annotations

from pathlib import Path

import rerun_qwen3_next_80b_nonregional_openrouter as runner


runner.MODEL_ORDER = 35
runner.MODEL_ID = "meta-llama/llama-3.3-70b-instruct"
runner.MODEL_SLUG = "meta-llama_llama-3.3-70b-instruct"
runner.MODEL_LABEL = "Llama 3.3 70B Instruct"
runner.RUNNER_DESCRIPTION = __doc__
runner.RERUN_TAG_PREFIX = "llama33_70b"
runner.NONREGIONAL_EXPECTED_COUNT = 7658
runner.REGIONAL_EXPECTED_COUNT = 594
runner.NONREGIONAL_SCOPE_DIRECTORY = "llama33_70b_nonregional"
runner.REGIONAL_SCOPE_DIRECTORY = "llama33_70b_regional"

manifest_root = Path(runner.REPO_ROOT) / "result" / "rerun_manifests"
runner.NONREGIONAL_MANIFEST = (
    manifest_root / "llama33_70b_nonregional" / "model_all_failures.csv"
)
runner.REGIONAL_MANIFEST = (
    manifest_root / "llama33_70b_regional" / "model_all_failures.csv"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())

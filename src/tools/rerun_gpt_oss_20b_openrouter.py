"""Selectively rerun audited GPT-OSS-20B failures through OpenRouter.

Safe default: validate the exact manifest without API calls or writes.
Reasoning-only responses are excluded into review-only manifests.
Use ``--execute`` only after a one-target smoke test succeeds.
"""

from __future__ import annotations

from pathlib import Path

from src.tools import rerun_qwen3_next_80b_nonregional_openrouter as runner


runner.MODEL_ORDER = 31
runner.MODEL_ID = "openai/gpt-oss-20b"
runner.MODEL_SLUG = "openai_gpt-oss-20b"
runner.MODEL_LABEL = "GPT-OSS-20B"
runner.RUNNER_DESCRIPTION = __doc__
runner.RERUN_TAG_PREFIX = "gpt_oss_20b"
runner.NONREGIONAL_EXPECTED_COUNT = 118
runner.REGIONAL_EXPECTED_COUNT = 5558
runner.NONREGIONAL_SCOPE_DIRECTORY = "gpt_oss_20b_nonregional"
runner.REGIONAL_SCOPE_DIRECTORY = "gpt_oss_20b_regional"

manifest_root = Path(runner.REPO_ROOT) / "result" / "rerun_manifests"
runner.NONREGIONAL_MANIFEST = (
    manifest_root / "gpt_oss_20b_nonregional" / "model_all_failures.csv"
)
runner.REGIONAL_MANIFEST = (
    manifest_root / "gpt_oss_20b_regional" / "model_all_failures.csv"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())

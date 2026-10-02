"""Selectively rerun audited Ministral 8B failures through OpenRouter.

Safe default: validate the exact manifest without API calls or writes.
Use ``--execute`` only after a one-target smoke test succeeds.
"""

from __future__ import annotations

from pathlib import Path

from src.tools import rerun_qwen3_next_80b_nonregional_openrouter as runner


runner.MODEL_ORDER = 41
runner.MODEL_ID = "mistralai/ministral-8b-2512"
runner.MODEL_SLUG = "mistralai_ministral-8b-2512"
runner.MODEL_LABEL = "Ministral 8B"
runner.RUNNER_DESCRIPTION = __doc__
runner.RERUN_TAG_PREFIX = "ministral_8b"
runner.NONREGIONAL_EXPECTED_COUNT = 6336
runner.REGIONAL_EXPECTED_COUNT = 4125
runner.NONREGIONAL_SCOPE_DIRECTORY = "ministral_8b_nonregional"
runner.REGIONAL_SCOPE_DIRECTORY = "ministral_8b_regional"

manifest_root = Path(runner.REPO_ROOT) / "result" / "rerun_manifests"
runner.NONREGIONAL_MANIFEST = (
    manifest_root / "ministral_8b_nonregional" / "model_all_failures.csv"
)
runner.REGIONAL_MANIFEST = (
    manifest_root / "ministral_8b_regional" / "model_all_failures.csv"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())

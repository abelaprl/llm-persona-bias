"""Build the key-free Vast bundle for 5,631 audited regional Qwen targets."""

from __future__ import annotations

from src.tools import build_qwen3_next_80b_openrouter_bundle as bundle


bundle.DEFAULT_OUTPUT = (
    bundle.REPO_ROOT / "dist" / "qwen3_next_80b_regional_openrouter_rerun.zip"
)
bundle.MANIFEST_ROOT = (
    bundle.REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "qwen3_next_80b_regional"
)
bundle.MANIFESTS = (
    bundle.MANIFEST_ROOT / "qwen_regional_ethics_infra.csv",
    bundle.MANIFEST_ROOT / "qwen_regional_gsm8k_infra.csv",
    bundle.MANIFEST_ROOT / "qwen_regional_infra_all.csv",
)
bundle.COMBINED_MANIFEST = bundle.MANIFESTS[-1]
bundle.EXPECTED_COUNTS = {"ETHICS": 5085, "GSM8K": 546}
bundle.EXPECTED_TOTAL = 5631
bundle.EXPECTED_MISSING_TARGETS = 0
bundle.SCOPE = "regional"


if __name__ == "__main__":
    raise SystemExit(bundle.main())

"""Rerun only the 552 audited DeepSeek Religion provider failures.

This launcher intentionally locks the target manifest, canonical output root,
backup root, attempts root, expected count, group, and issue category.  Its
safe default performs validation only.  ``--execute`` is required before any
API request or source-file replacement can happen.

Successful, non-truncated responses atomically replace only the exact failed
JSON path from the manifest.  Every selected failed JSON is backed up first.
Failed rerun attempts are written separately and leave the source unchanged.
Already-successful targets are skipped on resume.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rerun_gsm8k_deepseek_errors_vast as base


REPO_ROOT = SCRIPT_DIR.parent
RELIGION_GROUP = "Religion_Persona"
ISSUE_CATEGORY = "provider_error"
EXPECTED_COUNT = 552
MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_deepseek"
    / "deepseek_religion_provider_errors.csv"
)
OUTPUT_ROOT = REPO_ROOT / "result" / "GMS8K"
BACKUP_ROOT = (
    OUTPUT_ROOT / "_rerun_backup" / "deepseek_religion_provider_errors"
)
ATTEMPT_ROOT = (
    OUTPUT_ROOT / "_rerun_attempts" / "deepseek_religion_provider_errors"
)
PROTECTED_OPTIONS = {
    "--manifest",
    "--output-root",
    "--backup-root",
    "--attempt-root",
    "--expected-count",
}


def reject_scope_overrides(argv: list[str]) -> None:
    for argument in argv:
        option = argument.split("=", 1)[0]
        if option in PROTECTED_OPTIONS:
            raise RuntimeError(
                f"{option} is locked by the Religion-only launcher; "
                "target scope cannot be overridden"
            )


def parse_args(argv: list[str] | None = None):
    forwarded = list(sys.argv[1:] if argv is None else argv)
    reject_scope_overrides(forwarded)
    fixed = [
        "--manifest",
        str(MANIFEST),
        "--output-root",
        str(OUTPUT_ROOT),
        "--backup-root",
        str(BACKUP_ROOT),
        "--attempt-root",
        str(ATTEMPT_ROOT),
        "--expected-count",
        str(EXPECTED_COUNT),
    ]
    return base.parse_args([*fixed, *forwarded])


def validate_scope(targets: list[base.Target]) -> None:
    groups = {target.group for target in targets}
    categories = {target.issue_category for target in targets}
    if len(targets) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Religion target guard expected {EXPECTED_COUNT}, found {len(targets)}"
        )
    if groups != {RELIGION_GROUP}:
        raise RuntimeError(
            f"Religion-only guard rejected manifest groups: {sorted(groups)}"
        )
    if categories != {ISSUE_CATEGORY}:
        raise RuntimeError(
            "Provider-error-only guard rejected manifest categories: "
            f"{sorted(categories)}"
        )


def run(args) -> int:
    targets = base.load_targets(args.manifest, args.output_root, args.expected_count)
    validate_scope(targets)
    return base.run_targets(args, targets)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        return run(parse_args())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

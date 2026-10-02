"""Build a key-free Vast bundle for the exact 8,290 Qwen OpenRouter targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "dist" / "qwen3_next_80b_nonregional_openrouter_rerun.zip"
)
MANIFEST_ROOT = (
    REPO_ROOT / "result" / "rerun_manifests" / "qwen3_next_80b_nonregional"
)
MANIFESTS = (
    MANIFEST_ROOT / "qwen_ethics_infra.csv",
    MANIFEST_ROOT / "qwen_gsm8k_infra.csv",
    MANIFEST_ROOT / "qwen_nonregional_infra_all.csv",
)
COMBINED_MANIFEST = MANIFESTS[-1]
ETHICS_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
GSM8K_ROOT = REPO_ROOT / "result" / "GMS8K"
STATIC_FILES = (
    REPO_ROOT / "src" / "tools" / "rerun_qwen3_next_80b_openrouter.py",
    REPO_ROOT / "src" / "tools" / "rerun_qwen3_next_80b_nonregional_openrouter.py",
    REPO_ROOT / "src" / "benchmarks" / "ethics_runner" / "run_ethics_prompt_v9.py",
    REPO_ROOT / "requirements-vast-rerun.txt",
)
README_SOURCE = (
    REPO_ROOT / "docs" / "run_qwen3_next_80b_nonregional_openrouter_vast.md"
)
MODEL_SLUG = "qwen_qwen3-next-80b-a3b-thinking"
EXPECTED_COUNTS = {"ETHICS": 2949, "GSM8K": 5341}
EXPECTED_TOTAL = 8290
EXPECTED_MISSING_TARGETS = 1
SCOPE = "nonregional"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def safe_relative(value: object, *, allow_empty: bool = False) -> PurePosixPath | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text and allow_empty:
        return None
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe manifest path: {text!r}")
    return relative


def archive_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Bundle source escapes repository: {path}") from exc


def read_combined_manifest() -> list[dict[str, str]]:
    try:
        with COMBINED_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Combined manifest is missing: {COMBINED_MANIFEST}") from exc
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Combined manifest guard expected {EXPECTED_TOTAL}, found {len(rows)}"
        )
    dataset_counts = {
        dataset: sum(str(row.get("dataset") or "").upper() == dataset for row in rows)
        for dataset in EXPECTED_COUNTS
    }
    if dataset_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected dataset counts: {dataset_counts}")
    identities = [
        (str(row.get("dataset") or "").upper(), str(row.get("relative_path") or ""))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Combined manifest contains duplicate targets")
    return rows


def collect_files() -> tuple[dict[str, Path], dict[str, Any]]:
    rows = read_combined_manifest()
    files: dict[str, Path] = {}
    missing_targets = 0
    recovery_sources: set[tuple[str, PurePosixPath]] = set()

    def include(source: Path, destination: str | None = None) -> None:
        if not source.is_file():
            raise RuntimeError(f"Bundle source is missing: {source}")
        arcname = destination or archive_path(source)
        previous = files.get(arcname)
        if previous is not None and previous.resolve() != source.resolve():
            raise RuntimeError(f"Archive path collision: {arcname}")
        files[arcname] = source

    for source in STATIC_FILES:
        include(source)
    include(README_SOURCE, "README.md")
    for manifest in MANIFESTS:
        include(manifest)

    for row_number, row in enumerate(rows, start=2):
        dataset = str(row.get("dataset") or "").upper()
        relative = safe_relative(row.get("relative_path"))
        assert relative is not None
        if relative.parts[1:2] != (MODEL_SLUG,):
            raise RuntimeError(f"Model path mismatch at manifest row {row_number}")
        is_regional = "regional" in relative.parts[0].lower()
        if SCOPE == "nonregional" and is_regional:
            raise RuntimeError(f"Regional path is forbidden at manifest row {row_number}")
        if SCOPE == "regional" and not is_regional:
            raise RuntimeError(
                f"Non-regional path is forbidden at manifest row {row_number}"
            )
        root = ETHICS_ROOT if dataset == "ETHICS" else GSM8K_ROOT
        archive_root = (
            PurePosixPath("result/ETHICS/v9 [THIS IS THE MAIN]")
            if dataset == "ETHICS"
            else PurePosixPath("result/GMS8K")
        )
        target = root.joinpath(*relative.parts)
        if target.is_file():
            include(target, (archive_root / relative).as_posix())
        elif str(row.get("source_state") or "") == "missing":
            missing_targets += 1
        else:
            raise RuntimeError(f"Unexpected missing target at row {row_number}: {target}")

        for column in ("template_relative_path", "question_source_relative_path"):
            recovery = safe_relative(row.get(column), allow_empty=True)
            if recovery is not None:
                recovery_sources.add((dataset, recovery))

    for dataset, relative in recovery_sources:
        root = ETHICS_ROOT if dataset == "ETHICS" else GSM8K_ROOT
        archive_root = (
            PurePosixPath("result/ETHICS/v9 [THIS IS THE MAIN]")
            if dataset == "ETHICS"
            else PurePosixPath("result/GMS8K")
        )
        include(root.joinpath(*relative.parts), (archive_root / relative).as_posix())

    if missing_targets != EXPECTED_MISSING_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_MISSING_TARGETS} missing target(s), "
            f"found {missing_targets}"
        )
    inventory: dict[str, Any] = {
        "purpose": f"Selective {SCOPE} Qwen3 Next 80B rerun via OpenRouter",
        "model_id": "qwen/qwen3-next-80b-a3b-thinking",
        "openrouter_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_included": False,
        "scope": SCOPE,
        "regional_cultural_included": SCOPE == "regional",
        "ethics_targets": EXPECTED_COUNTS["ETHICS"],
        "gsm8k_targets": EXPECTED_COUNTS["GSM8K"],
        "total_targets": EXPECTED_TOTAL,
        "existing_target_files": EXPECTED_TOTAL - missing_targets,
        "missing_targets_rebuilt_at_runtime": missing_targets,
        "unique_recovery_sources": len(recovery_sources),
        "archive_file_count_excluding_inventory": len(files),
    }
    return files, inventory


def build_bundle(output: Path) -> tuple[Path, dict[str, Any]]:
    files, inventory = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for destination, source in sorted(files.items()):
                archive.write(source, destination)
            archive.writestr(
                "bundle_inventory.json",
                json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad_file}")
        inventory["archive_file_count_with_inventory"] = len(archive.namelist())
    return output, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    try:
        output, inventory = build_bundle(repo_path(args.output))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(f"Bundle: {output}")
    print(f"Size  : {output.stat().st_size / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

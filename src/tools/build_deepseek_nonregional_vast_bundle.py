"""Build a compact Vast bundle containing only the 743 audited rerun targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "deepseek_nonregional_vast_rerun.zip"

ETHICS_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
GSM8K_ROOT = REPO_ROOT / "result" / "GMS8K"
ETHICS_MANIFESTS = (
    (
        REPO_ROOT
        / "result"
        / "rerun_manifests"
        / "ethics_v9"
        / "deepseek_http_4xx_5xx.csv",
        51,
    ),
    (
        REPO_ROOT
        / "result"
        / "rerun_manifests"
        / "ethics_v9"
        / "deepseek_non_http_local_connection.csv",
        8,
    ),
)
GSM8K_MANIFEST = (
    REPO_ROOT
    / "result"
    / "rerun_manifests"
    / "gsm8k_deepseek"
    / "deepseek_infra_all.csv"
)

STATIC_FILES = (
    REPO_ROOT / "src" / "tools" / "rerun_deepseek_nonregional_vast.py",
    REPO_ROOT / "src" / "tools" / "rerun_ethics_v9_http_errors_vast.py",
    REPO_ROOT / "src" / "tools" / "rerun_ethics_v9_deepseek_errors_vast.py",
    REPO_ROOT / "src" / "tools" / "rerun_gsm8k_deepseek_errors_vast.py",
    REPO_ROOT / "src" / "benchmarks" / "ethics_runner" / "run_ethics_prompt_v9.py",
    REPO_ROOT / "requirements-vast-rerun.txt",
)
README_SOURCE = REPO_ROOT / "docs" / "vast_deepseek_nonregional_rerun.md"

MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
WARMUP_RELATIVE_PATHS = (
    PurePosixPath(
        "Gender Persona",
        MODEL_SLUG,
        "man_implicit_1",
        "ethics_commonsense_0040.json",
    ),
    PurePosixPath(
        "Gender Persona",
        MODEL_SLUG,
        "man_implicit_1",
        "warmup",
        "persona_warmup.json",
    ),
)


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def safe_manifest_relative_path(value: str, manifest: Path, row_number: int) -> PurePosixPath:
    relative = PurePosixPath(value.strip().replace("\\", "/"))
    if not value.strip() or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"Unsafe relative_path at {manifest}:{row_number}: {value!r}"
        )
    return relative


def read_manifest_paths(manifest: Path, expected_count: int) -> list[PurePosixPath]:
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "relative_path" not in (reader.fieldnames or []):
                raise RuntimeError(f"Manifest has no relative_path column: {manifest}")
            paths = [
                safe_manifest_relative_path(
                    str(row.get("relative_path") or ""), manifest, row_number
                )
                for row_number, row in enumerate(reader, start=2)
            ]
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing manifest: {manifest}") from exc
    if len(paths) != expected_count:
        raise RuntimeError(
            f"Manifest guard expected {expected_count}, found {len(paths)}: {manifest}"
        )
    if len(set(paths)) != len(paths):
        raise RuntimeError(f"Manifest contains duplicate relative paths: {manifest}")
    return paths


def archive_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Source path escapes repository: {path}") from exc
    return relative.as_posix()


def collect_files() -> tuple[dict[str, Path], dict[str, object]]:
    ethics_paths: list[PurePosixPath] = []
    for manifest, expected_count in ETHICS_MANIFESTS:
        ethics_paths.extend(read_manifest_paths(manifest, expected_count))
    if len(set(ethics_paths)) != 59:
        raise RuntimeError("ETHICS manifests overlap; expected 59 unique target paths")
    gsm8k_paths = read_manifest_paths(GSM8K_MANIFEST, 684)

    files: dict[str, Path] = {}

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
    for manifest, _ in ETHICS_MANIFESTS:
        include(manifest)
    include(GSM8K_MANIFEST)

    for relative in ethics_paths:
        include(
            ETHICS_ROOT.joinpath(*relative.parts),
            (PurePosixPath("result/ETHICS/v9 [THIS IS THE MAIN]") / relative).as_posix(),
        )
    for relative in WARMUP_RELATIVE_PATHS:
        include(
            ETHICS_ROOT.joinpath(*relative.parts),
            (PurePosixPath("result/ETHICS/v9 [THIS IS THE MAIN]") / relative).as_posix(),
        )
    for relative in gsm8k_paths:
        include(
            GSM8K_ROOT.joinpath(*relative.parts),
            (PurePosixPath("result/GMS8K") / relative).as_posix(),
        )

    inventory: dict[str, object] = {
        "purpose": "Selective DeepSeek non-regional rerun on Vast.ai",
        "model_id": "deepseek/deepseek-r1-distill-qwen-32b",
        "regional_cultural_included": False,
        "ethics_model_targets": len(ethics_paths),
        "gsm8k_model_targets": len(gsm8k_paths),
        "total_model_targets": len(ethics_paths) + len(gsm8k_paths),
        "warmup_model_calls": 0,
        "warmup_local_cache_repairs": 1,
        "archive_file_count_excluding_inventory": len(files),
    }
    return files, inventory


def build_bundle(output: Path) -> tuple[Path, dict[str, object]]:
    files, inventory = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(temporary_handle)
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
    size_mib = output.stat().st_size / (1024 * 1024)
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(f"Bundle: {output}")
    print(f"Size  : {size_mib:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

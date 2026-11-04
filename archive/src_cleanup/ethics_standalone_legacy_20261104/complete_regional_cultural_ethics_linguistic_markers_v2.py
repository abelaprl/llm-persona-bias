"""Complete a phase-9 validated Regional/Cultural bundle without recomputation.

The source run completed all statistical computation, table writing, figures,
and automated in-memory validation, then failed only because the sandbox denied
writing source notebooks under ``notebooks/regional``.  This script copies the
phase-9 artifacts byte-for-byte to a new write-once destination, proves the
copy with per-file hashes and a tree digest, creates/executes the two notebooks,
and writes the final manifest.  It never modifies the incomplete source run.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from src.benchmarks.ethics.build_regional_cultural_ethics_linguistic_markers_v2 import (
    DEFAULT_NOTEBOOK_DIR,
    collect_hashes,
    create_notebooks,
)
from src.benchmarks.ethics.regional_cultural_ethics_linguistic_markers_v2 import (
    BUNDLE_VERSION,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Regional_Cultural_Persona"
    / "linguistic_markers"
    / "analysis"
)
SOURCE_DIR = (
    ANALYSIS_ROOT
    / "regional_cultural_ethics_linguistic_markers_v2_0_1_candidate_20260828"
)
OUTPUT_DIR = (
    ANALYSIS_ROOT
    / "regional_lm_v2_0_1_complete_20260828"
)
EXCLUDED_SOURCE_NAMES = frozenset({"manifest.json", "manifest_final.json", "FAILED_RUN.md"})


def phase9_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in EXCLUDED_SOURCE_NAMES
        and "notebooks" not in path.relative_to(root).parts
    }


def tree_digest(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(hashes.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def copy_phase9(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for child in sorted(source.iterdir()):
        if child.name in EXCLUDED_SOURCE_NAMES or child.name == "notebooks":
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def validate_phase9_source(source: Path) -> tuple[dict, dict]:
    manifest_path = source / "manifest.json"
    validation_path = source / "automated_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("The incomplete source lacks phase-9 validation artifacts.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Source bundle version does not match the current engine.")
    if not str(manifest.get("result_status", "")).startswith("candidate_blocked"):
        raise ValueError("Source result status is not the expected blocked candidate status.")
    failed_automated = [
        key
        for key, value in validation.items()
        if isinstance(value, bool)
        and key != "human_validation_complete"
        and value is not True
    ]
    if failed_automated:
        raise ValueError(f"Phase-9 automated checks failed: {failed_automated}")
    if validation.get("human_validation_complete") is not False:
        raise ValueError("Human validation must remain explicitly incomplete.")
    return manifest, validation


def main() -> int:
    source = SOURCE_DIR.resolve()
    output = OUTPUT_DIR.resolve()
    notebook_dir = DEFAULT_NOTEBOOK_DIR.resolve()
    if output.exists():
        raise FileExistsError(f"Output is write-once and already exists: {output}")

    source_manifest, source_validation = validate_phase9_source(source)
    source_hashes = phase9_hashes(source)
    source_digest = tree_digest(source_hashes)
    print(
        f"[1/5] Validated phase-9 source: {len(source_hashes)} files, tree {source_digest}",
        flush=True,
    )

    copy_phase9(source, output)
    copied_hashes = phase9_hashes(output)
    if copied_hashes != source_hashes:
        raise ValueError("Phase-9 destination is not a byte-identical copy of the source.")
    print("[2/5] Phase-9 artifacts copied byte-for-byte", flush=True)

    completion_time = datetime.now().astimezone().isoformat(timespec="seconds")
    provisional = {
        **source_manifest,
        "completion_status": "phase9_copy_verified_notebooks_pending",
        "completion_mode": "byte_identical_phase9_copy_then_notebook_generation",
        "completion_script": str(Path(__file__).resolve()),
        "completion_script_sha256": sha256_file(Path(__file__).resolve()),
        "incomplete_phase9_source": str(source),
        "incomplete_phase9_source_manifest_sha256": sha256_file(
            source / "manifest.json"
        ),
        "phase9_source_file_count": len(source_hashes),
        "phase9_source_tree_sha256": source_digest,
        "phase9_copy_verified": True,
        "phase9_automated_validation": source_validation,
        "completed_at": completion_time,
    }
    (output / "manifest.json").write_text(
        json.dumps(provisional, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[3/5] Create and execute thesis + atlas notebooks", flush=True)
    notebook_paths = create_notebooks(output, notebook_dir, execute=True)
    print("[4/5] Hash completed bundle", flush=True)
    final_manifest = {
        **provisional,
        "completion_status": "complete_candidate_human_gate_still_blocked",
        "notebooks": [str(path) for path in notebook_paths],
        "generated_file_sha256": collect_hashes(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[5/5] Completed candidate bundle: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Finalize notebooks for an already validated v2.0.1 candidate bundle.

This recovery command exists because the full statistical build can complete
before notebook creation fails.  It never overwrites the provisional manifest;
instead it writes a new ``manifest_final.json`` after validating and hashing the
new notebook artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from src.benchmarks.ethics.build_religion_ethics_linguistic_markers_v2 import (
    DEFAULT_NOTEBOOK_DIR,
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    collect_hashes,
    create_notebooks,
)
from src.benchmarks.ethics.religion_ethics_linguistic_markers_v2 import (
    BUNDLE_VERSION,
    sha256_file,
)


FINALIZATION_VERSION = "religion-ethics-lm-notebook-finalizer-v2.0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notebook-dir", type=Path, default=DEFAULT_NOTEBOOK_DIR)
    parser.add_argument("--skip-notebook-execution", action="store_true")
    return parser.parse_args()


def finalize(args: argparse.Namespace) -> Path:
    bundle_dir = args.bundle_dir.resolve()
    notebook_dir = args.notebook_dir.resolve()
    provisional_path = bundle_dir / "manifest.json"
    validation_path = bundle_dir / "automated_validation.json"
    final_path = bundle_dir / "manifest_final.json"
    if final_path.exists():
        raise FileExistsError(f"Final manifest is write-once and already exists: {final_path}")
    if not provisional_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Validated provisional bundle is incomplete.")
    if (bundle_dir / "notebooks").exists():
        raise FileExistsError("Bundle notebook directory already exists; refusing to overwrite it.")

    provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if provisional.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Unexpected bundle version: {provisional.get('bundle_version')}")
    failed_automated = [
        key
        for key, value in validation.items()
        if isinstance(value, bool) and key != "human_validation_complete" and not value
    ]
    if failed_automated:
        raise ValueError(f"Cannot finalize a bundle with failed checks: {failed_automated}")
    if validation.get("human_validation_complete") is not False:
        raise ValueError("Unexpected human validation state.")

    print("[1/3] Create notebook sources without overwriting existing files", flush=True)
    notebook_paths = create_notebooks(
        bundle_dir,
        notebook_dir,
        execute=not args.skip_notebook_execution,
    )
    note_path = bundle_dir / "FINALIZATION.md"
    note_path.write_text(
        """# Finalisasi notebook

Statistik, tabel, dan figur v2.0.1 telah lolos automated validation sebelum
finalisasi ini. Build penuh terhenti hanya karena sandbox menolak penulisan ke
`notebooks/religion`. Finalizer kemudian dijalankan dengan izin filesystem yang
sesuai, tanpa menghitung ulang statistik dan tanpa menimpa `manifest.json`.

`manifest_final.json` adalah manifest autoritatif bundle lengkap. Status hasil
tetap kandidat karena human validation masih pending.
""",
        encoding="utf-8",
    )

    print("[2/3] Hash notebook sources and completed bundle", flush=True)
    final_manifest = dict(provisional)
    final_manifest.update(
        {
            "authoritative_manifest": "manifest_final.json",
            "provisional_manifest": "manifest.json",
            "finalization_version": FINALIZATION_VERSION,
            "finalized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "finalizer": str(Path(__file__).resolve()),
            "finalizer_sha256": sha256_file(Path(__file__).resolve()),
            "notebooks": [str(path) for path in notebook_paths],
            "notebook_source_sha256": {
                str(path): sha256_file(path)
                for path in notebook_paths
                if path.is_file() and path.parent == notebook_dir
            },
            "notebook_execution_complete": not args.skip_notebook_execution,
            "generated_file_sha256": collect_hashes(bundle_dir),
        }
    )
    final_path.write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[3/3] Final manifest written without replacing the provisional manifest", flush=True)
    print(final_path, flush=True)
    return final_path


def main() -> int:
    args = parse_args()
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

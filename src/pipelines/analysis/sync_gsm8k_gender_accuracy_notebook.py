"""Synchronize the GSM8K Gender presentation with existing local figures.

Build in memory using the existing generator, execute, verify all PNG outputs,
and replace the notebook only after validation. No statistical inputs are edited.
"""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

from build_gsm8k_gender_accuracy_complete_notebook import (
    NOTEBOOK_DIR, NOTEBOOK_PATH, REPO_ROOT, build_notebook,
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): digest(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def synchronize():
    previous_hash = digest(NOTEBOOK_PATH)
    previous = nbformat.read(NOTEBOOK_PATH, as_version=4)
    generated = build_notebook()
    if [c.cell_type for c in previous.cells] != [c.cell_type for c in generated.cells]:
        raise ValueError("Generator changes notebook structure; review before proceeding")
    changed = [i for i, (old, new) in enumerate(zip(previous.cells, generated.cells))
               if old.source != new.source]
    if changed != [0, 3, 5]:
        raise ValueError(f"Unexpected source differences: {changed}")
    for old, new in zip(previous.cells, generated.cells):
        new.id = old.id
    figures = NOTEBOOK_DIR / "figures"
    bundle = REPO_ROOT / "result/GMS8K/reports/persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold"
    protected = {str(folder): snapshot(folder) for folder in (figures, bundle)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = NOTEBOOK_DIR / "_arsip" / ("sinkronisasi_" + stamp)
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(NOTEBOOK_PATH, archive / NOTEBOOK_PATH.name)
    print(f"[1/3] Backup: {archive}", flush=True)
    print("[2/3] Executing notebook with local figures", flush=True)
    manager = KernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Python 3", language="python")
    NotebookClient(generated, km=manager, timeout=600,
                   resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
    embedded = []
    for cell in generated.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValueError("Notebook contains an error output")
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                embedded.append(hashlib.sha256(base64.b64decode(encoded)).hexdigest())
    expected_images = Counter(digest(p) for p in figures.glob("*.png"))
    if len(embedded) != 11 or Counter(embedded) != expected_images:
        raise ValueError("Embedded figures do not exactly match the eleven local PNGs")
    for folder, expected in protected.items():
        if snapshot(Path(folder)) != expected:
            raise ValueError(f"Protected input changed: {folder}")
    nbformat.validate(generated)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".ipynb",
                                         prefix=".sync-", dir=NOTEBOOK_DIR, delete=False) as handle:
            staged = Path(handle.name)
            nbformat.write(generated, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if nbformat.read(staged, as_version=4) != generated:
            raise ValueError("Staged notebook readback differs")
        if digest(NOTEBOOK_PATH) != previous_hash:
            raise ValueError("Target notebook changed during execution")
        os.replace(staged, NOTEBOOK_PATH)
    finally:
        if staged is not None and staged.exists():
            staged.unlink()
    if nbformat.read(NOTEBOOK_PATH, as_version=4) != generated:
        raise ValueError("Published notebook readback differs")
    report = {
        "valid": True, "changed_source_cell_indices": changed,
        "cells": len(generated.cells),
        "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None
                                   for c in generated.cells),
        "embedded_png_matching_local": len(embedded), "errors": 0,
        "figures_and_source_bundle_unchanged": True,
        "protected_files": sum(len(files) for files in protected.values()),
        "before_sha256": previous_hash, "after_sha256": digest(NOTEBOOK_PATH),
        "generator_sha256": digest(Path(__file__).with_name("build_gsm8k_gender_accuracy_complete_notebook.py")),
        "figure_hashes": protected[str(figures)],
    }
    with (archive / "validation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[3/3] Synchronized\n" + json.dumps(report), flush=True)


if __name__ == "__main__":
    synchronize()

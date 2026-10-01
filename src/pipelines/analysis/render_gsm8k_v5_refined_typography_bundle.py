"""Render a write-once GSM8K v5 bundle with refined typography.

The publication heatmaps use the source typography plus 3 pt, while accuracy
and delta bar charts use plus 2 pt to reduce crowding.  Heatmap cell text is
bold only when its annotation contains an asterisk denoting significance.
Statistical calculations and tables are inherited unchanged from the v5
publication notebook and supplemental renderers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import nbformat
import pandas as pd
import seaborn as sns

import render_gsm8k_v5_fontplus3_bundle as base


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
SOURCE_NOTEBOOK = REPORT_ROOT / "GSM8K_Gender_Religion_Publication_Analysis.ipynb"
TARGET_NOTEBOOK = (
    REPORT_ROOT
    / "GSM8K_Gender_Religion_Publication_Analysis_bar2_heatmap3_sigbold.ipynb"
)

SOURCE_BUNDLE_NAME = "persona_publication_analysis_v5_paired_significance_six_models"
TARGET_BUNDLE_NAME = f"{SOURCE_BUNDLE_NAME}_bar2_heatmap3_sigbold"
SOURCE_BUNDLE = REPORT_ROOT / SOURCE_BUNDLE_NAME
TARGET_BUNDLE = REPORT_ROOT / TARGET_BUNDLE_NAME
TARGET_TABLE_DIR = TARGET_BUNDLE / "tables"
TARGET_FIGURE_DIR = TARGET_BUNDLE / "figures"

STAGE_ROOT = REPO_ROOT / ".codex_tmp"
STAGE_BUNDLE_NAME = "gsm8k_v5_barplus2_stage"
STAGE_BUNDLE = STAGE_ROOT / STAGE_BUNDLE_NAME
STAGE_NOTEBOOK = (
    STAGE_ROOT / "GSM8K_Gender_Religion_Publication_Analysis_barplus2_stage.ipynb"
)
STAGE_PLACEHOLDER = "__gsm8k_v5_barplus2_stage__"

HEATMAP_FONT_DELTA_PT = 3.0
BAR_FONT_DELTA_PT = 2.0
FIGURE_SCALE = 1.12

BAR_STEMS_FROM_NOTEBOOK = (
    "01_gender_absolute_accuracy_five_categories",
    "02_gender_delta_vs_neutral",
    "03_gender_delta_implicit_vs_explicit",
    "05_religion_absolute_accuracy_explicit_implicit",
    "06_religion_delta_implicit_vs_explicit",
    "12_regional_absolute_accuracy_explicit_implicit",
    "13_regional_delta_implicit_vs_explicit",
    "16_regional_completion_token_delta_implicit_vs_explicit",
    "23_neutral_persona_accuracy_six_models",
)
BAR_RENDERERS = (
    "render_gsm8k_religion_delta_vs_neutral.py",
    "render_gsm8k_regional_delta_vs_neutral.py",
)

HEATMAP_PATCH = r'''

# Typography variant: heatmap cell values are bold only when marked significant.
_original_heatmap_for_significance_weight = sns.heatmap

def _heatmap_with_significance_weight(*args, **kwargs):
    annotation_style = dict(kwargs.get("annot_kws") or {})
    annotation_style["fontweight"] = "normal"
    kwargs["annot_kws"] = annotation_style
    axis = _original_heatmap_for_significance_weight(*args, **kwargs)
    for annotation in axis.texts:
        annotation.set_fontweight(
            "bold" if "*" in annotation.get_text() else "normal"
        )
    return axis

sns.heatmap = _heatmap_with_significance_weight
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("prepare", "supplement", "finish", "validate")
    )
    return parser


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_base_target(font_delta_pt: float) -> None:
    """Point the reusable transformer at the new write-once target."""

    base.TARGET_NOTEBOOK = TARGET_NOTEBOOK
    base.TARGET_BUNDLE_NAME = TARGET_BUNDLE_NAME
    base.TARGET_BUNDLE = TARGET_BUNDLE
    base.TARGET_TABLE_DIR = TARGET_TABLE_DIR
    base.TARGET_FIGURE_DIR = TARGET_FIGURE_DIR
    base.FONT_DELTA_PT = font_delta_pt
    base.FIGURE_SCALE = FIGURE_SCALE


def _transform_source(text: str, font_delta_pt: float) -> str:
    _configure_base_target(font_delta_pt)
    return base.transform_visual_source(text)


def _transform_stage_source(text: str) -> str:
    base.TARGET_NOTEBOOK = STAGE_NOTEBOOK
    base.TARGET_BUNDLE_NAME = STAGE_PLACEHOLDER
    base.FONT_DELTA_PT = BAR_FONT_DELTA_PT
    base.FIGURE_SCALE = FIGURE_SCALE
    transformed = base.transform_visual_source(text)
    report_expression = f'REPORT_ROOT / "{STAGE_PLACEHOLDER}"'
    stage_expression = (
        f'REPO_ROOT / ".codex_tmp" / "{STAGE_BUNDLE_NAME}"'
    )
    transformed = transformed.replace(report_expression, stage_expression)
    if STAGE_PLACEHOLDER in transformed:
        raise RuntimeError("Stage output placeholder remains in transformed code.")
    return transformed


def _prepare_notebook(
    target_path: Path,
    transform,
    *,
    inject_significance_weight: bool,
    metadata: dict[str, object],
) -> None:
    print(f"[READ] {SOURCE_NOTEBOOK} -> {target_path}", flush=True)
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    transformed_cells = 0
    patch_injected = False
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        original = cell.source
        cell.source = transform(original)
        if inject_significance_weight and not patch_injected and "import seaborn as sns" in cell.source:
            cell.source += HEATMAP_PATCH
            patch_injected = True
        cell.execution_count = None
        cell.outputs = []
        transformed_cells += int(cell.source != original)
        print(f"[TRANSFORM] {target_path.name}: cell {cell_index}", flush=True)

    if inject_significance_weight and not patch_injected:
        raise RuntimeError("Could not inject the heatmap significance-weight patch.")

    notebook.metadata["typography_variant"] = {
        **metadata,
        "source_notebook": str(SOURCE_NOTEBOOK),
        "figure_scale": FIGURE_SCALE,
        "transformed_code_cells": transformed_cells,
    }
    print(f"[WRITE] {target_path}", flush=True)
    nbformat.write(notebook, target_path)


def prepare_notebooks() -> None:
    print("[PREPARE] Checking write-once target paths", flush=True)
    protected_paths = (TARGET_BUNDLE, TARGET_NOTEBOOK, STAGE_BUNDLE, STAGE_NOTEBOOK)
    existing = []
    for path in protected_paths:
        print(f"[CHECK] {path}", flush=True)
        if path.exists():
            existing.append(str(path))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing target or stage path: "
            + ", ".join(existing)
        )
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)

    print(f"[MKDIR] {STAGE_ROOT}", flush=True)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    _prepare_notebook(
        TARGET_NOTEBOOK,
        lambda source: _transform_source(source, HEATMAP_FONT_DELTA_PT),
        inject_significance_weight=True,
        metadata={
            "profile": "heatmap_plus3_star_only_bold",
            "font_delta_pt": HEATMAP_FONT_DELTA_PT,
        },
    )
    _prepare_notebook(
        STAGE_NOTEBOOK,
        _transform_stage_source,
        inject_significance_weight=False,
        metadata={
            "profile": "bar_plus2_staging",
            "font_delta_pt": BAR_FONT_DELTA_PT,
            "stage_output": str(STAGE_BUNDLE),
        },
    )

    print(f"[PREPARED] heatmap notebook: {TARGET_NOTEBOOK}")
    print(f"[PREPARED] bar stage notebook: {STAGE_NOTEBOOK}")
    print(f"[TARGET]   {TARGET_BUNDLE}")


@contextmanager
def _star_only_bold_heatmaps() -> Iterator[None]:
    """Apply the same cell-weight rule to supplemental heatmap renderers."""

    original_heatmap = sns.heatmap

    def heatmap_with_significance_weight(*args, **kwargs):
        annotation_style = dict(kwargs.get("annot_kws") or {})
        annotation_style["fontweight"] = "normal"
        kwargs["annot_kws"] = annotation_style
        axis = original_heatmap(*args, **kwargs)
        for annotation in axis.texts:
            annotation.set_fontweight(
                "bold" if "*" in annotation.get_text() else "normal"
            )
        return axis

    sns.heatmap = heatmap_with_significance_weight
    try:
        yield
    finally:
        sns.heatmap = original_heatmap


def _copy_missing_source_tables() -> None:
    for source_table in (SOURCE_BUNDLE / "tables").glob("*.csv"):
        target_table = TARGET_TABLE_DIR / source_table.name
        if not target_table.exists():
            shutil.copy2(source_table, target_table)


def render_supplement() -> None:
    if not TARGET_FIGURE_DIR.is_dir() or not TARGET_TABLE_DIR.is_dir():
        raise FileNotFoundError(
            "Execute the transformed heatmap notebook before supplemental renderers."
        )

    expected_main = {
        path.name
        for path in (SOURCE_BUNDLE / "figures").glob("*")
        if path.name[:2].isdigit() and int(path.name[:2]) <= 23
    }
    missing_main = sorted(
        name for name in expected_main if not (TARGET_FIGURE_DIR / name).is_file()
    )
    if missing_main:
        raise FileNotFoundError(f"Missing main-notebook figures: {missing_main}")

    _copy_missing_source_tables()
    _configure_base_target(HEATMAP_FONT_DELTA_PT)
    with _star_only_bold_heatmaps():
        base._install_transformed_religion_modules()
        for name in base.FIXED_RENDERERS:
            path = base.RENDERER_DIR / name
            print(f"[RENDER +3 / STAR-BOLD] {name}")
            base._execute_transformed_renderer(path, [])

        for name, stem in base.SINGLE_OUTPUT_RENDERERS:
            path = base.RENDERER_DIR / name
            for extension in ("png", "svg"):
                output = TARGET_FIGURE_DIR / f"{stem}.{extension}"
                print(f"[RENDER +3 / STAR-BOLD] {name} -> {output.name}")
                base._execute_transformed_renderer(
                    path, ["--output", str(output), "--dpi", "320"]
                )

    print("[READY] Supplemental +3 heatmaps rendered; run finish for +2 bars.")


def _copy_stage_bars() -> None:
    stage_figure_dir = STAGE_BUNDLE / "figures"
    if not stage_figure_dir.is_dir():
        raise FileNotFoundError(
            "Execute the bar-plus2 stage notebook before the finish phase."
        )

    for stem in BAR_STEMS_FROM_NOTEBOOK:
        for extension in ("png", "svg"):
            source = stage_figure_dir / f"{stem}.{extension}"
            target = TARGET_FIGURE_DIR / source.name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            print(f"[BAR +2] {target.name}")


def finish_bundle() -> None:
    if not TARGET_FIGURE_DIR.is_dir() or not TARGET_TABLE_DIR.is_dir():
        raise FileNotFoundError("The target heatmap bundle has not been rendered.")

    _copy_stage_bars()
    _configure_base_target(BAR_FONT_DELTA_PT)
    for name in BAR_RENDERERS:
        path = base.RENDERER_DIR / name
        print(f"[BAR +2] {name}")
        base._execute_transformed_renderer(path, [])

    validate_bundle()


def _refresh_output_index() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(TARGET_TABLE_DIR.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        rows.append(
            {
                "type": "table",
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "bytes": path.stat().st_size,
            }
        )
    for path in sorted(TARGET_FIGURE_DIR.glob("*")):
        rows.append(
            {
                "type": "figure",
                "file": path.name,
                "rows": None,
                "columns": None,
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(TARGET_BUNDLE / "output_index.csv", index=False)


def validate_bundle() -> None:
    png = sorted(TARGET_FIGURE_DIR.glob("*.png"))
    svg = sorted(TARGET_FIGURE_DIR.glob("*.svg"))
    if len(png) != 32 or len(svg) != 32:
        raise RuntimeError(f"Expected 32 PNG/SVG figures, found {len(png)}/{len(svg)}.")
    if any(path.stat().st_size == 0 for path in (*png, *svg)):
        raise RuntimeError("At least one rendered figure is empty.")

    source_tables = {path.name: path for path in (SOURCE_BUNDLE / "tables").glob("*.csv")}
    target_tables = {path.name: path for path in TARGET_TABLE_DIR.glob("*.csv")}
    missing_tables = sorted(set(source_tables) - set(target_tables))
    if missing_tables:
        raise FileNotFoundError(f"Target statistical tables missing: {missing_tables}")

    mismatched_tables = [
        name
        for name, source_path in source_tables.items()
        if sha256_file(source_path) != sha256_file(target_tables[name])
    ]
    if mismatched_tables:
        raise RuntimeError(
            "Typography-only render changed statistical tables: "
            + ", ".join(mismatched_tables)
        )

    stage_figure_dir = STAGE_BUNDLE / "figures"
    stage_bar_mismatches = []
    for stem in BAR_STEMS_FROM_NOTEBOOK:
        for extension in ("png", "svg"):
            staged = stage_figure_dir / f"{stem}.{extension}"
            rendered = TARGET_FIGURE_DIR / staged.name
            if not staged.is_file() or sha256_file(staged) != sha256_file(rendered):
                stage_bar_mismatches.append(staged.name)
    if stage_bar_mismatches:
        raise RuntimeError(
            "One or more notebook bar figures do not match the +2 staging render: "
            + ", ".join(stage_bar_mismatches)
        )

    _refresh_output_index()
    manifest = {
        "bundle_version": "gsm8k-v5-publication-bar2-heatmap3-sigbold-v1.0.0",
        "source_bundle": str(SOURCE_BUNDLE),
        "source_notebook": str(SOURCE_NOTEBOOK),
        "render_notebook": str(TARGET_NOTEBOOK),
        "heatmap_font_delta_pt": HEATMAP_FONT_DELTA_PT,
        "bar_font_delta_pt": BAR_FONT_DELTA_PT,
        "figure_scale": FIGURE_SCALE,
        "heatmap_cell_weight_rule": "bold_if_annotation_contains_asterisk_else_normal",
        "bar_figure_stems": [
            *BAR_STEMS_FROM_NOTEBOOK,
            "28_religion_delta_vs_neutral",
            "29_regional_delta_vs_neutral",
        ],
        "statistical_tables_identical": True,
        "table_count": len(source_tables),
        "png_count": len(png),
        "svg_count": len(svg),
        "figures": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (*png, *svg)
        ],
    }
    (TARGET_BUNDLE / "typography_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[VALID] 32 PNG + 32 SVG; {len(source_tables)} statistical tables unchanged")
    print("[VALID] Heatmap annotations: bold only when the rendered label contains '*'")
    print(f"[DONE]  {TARGET_BUNDLE}")


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "prepare":
        prepare_notebooks()
    elif args.phase == "supplement":
        render_supplement()
    elif args.phase == "finish":
        finish_bundle()
    else:
        validate_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

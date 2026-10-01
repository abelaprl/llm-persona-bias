"""Render a write-once GSM8K v5 publication bundle with typography +3 pt.

The statistical notebook and supplemental renderers are reused without changing
their analytical logic.  Only output paths and visual typography/figure size
arguments are transformed.  The original v5 bundle remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import types
from pathlib import Path

import nbformat
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
SOURCE_NOTEBOOK = REPORT_ROOT / "GSM8K_Gender_Religion_Publication_Analysis.ipynb"
TARGET_NOTEBOOK = REPORT_ROOT / "GSM8K_Gender_Religion_Publication_Analysis_fontplus3.ipynb"

SOURCE_BUNDLE_NAME = "persona_publication_analysis_v5_paired_significance_six_models"
TARGET_BUNDLE_NAME = f"{SOURCE_BUNDLE_NAME}_fontplus3"
SOURCE_BUNDLE = REPORT_ROOT / SOURCE_BUNDLE_NAME
TARGET_BUNDLE = REPORT_ROOT / TARGET_BUNDLE_NAME
TARGET_TABLE_DIR = TARGET_BUNDLE / "tables"
TARGET_FIGURE_DIR = TARGET_BUNDLE / "figures"

FONT_DELTA_PT = 3.0
FIGURE_SCALE = 1.12

RENDERER_DIR = REPO_ROOT / "src" / "pipelines" / "analysis"
FIXED_RENDERERS = (
    "render_gsm8k_v5_remaining_large_labels.py",
    "render_gsm8k_gender_effect_holm_single_portrait.py",
    "render_gsm8k_religion_effect_holm_with_neutral_single_large_heatmap.py",
    "render_gsm8k_religion_primary_effect_holm_single_large_heatmap.py",
    "render_gsm8k_religion_all_52_effect_holm_single_large_heatmap.py",
    "render_gsm8k_religion_delta_vs_neutral.py",
    "render_gsm8k_regional_delta_vs_neutral.py",
)
SINGLE_OUTPUT_RENDERERS = (
    (
        "render_gsm8k_gender_effect_holm_single_large_heatmap.py",
        "24_gender_effect_holm_single_large_heatmap",
    ),
    (
        "render_gsm8k_religion_all_52_effect_holm_stacked_26_26.py",
        "26_religion_all_52_effect_holm_stacked_26_26",
    ),
    (
        "render_gsm8k_regional_all_52_effect_holm_stacked_26_26.py",
        "27_regional_all_52_effect_holm_stacked_26_26",
    ),
)

_NUMBER = r"\d+(?:\.\d+)?"
_KWARG_FONT_RE = re.compile(
    rf"(?P<prefix>\b(?:fontsize|labelsize)\s*=\s*)(?P<value>{_NUMBER})"
)
_DICT_FONT_RE = re.compile(
    rf"(?P<prefix>[\"'](?:font\.size|axes\.titlesize|axes\.labelsize|"
    rf"xtick\.labelsize|ytick\.labelsize|legend\.fontsize|"
    rf"legend\.title_fontsize)[\"']\s*:\s*)(?P<value>{_NUMBER})"
)
_FONT_SCALE_RE = re.compile(
    rf"(?P<prefix>\bfont_scale\s*=\s*)(?P<value>{_NUMBER})"
)
_FIGSIZE_RE = re.compile(
    rf"(?P<prefix>\bfigsize\s*=\s*\(\s*)"
    rf"(?P<width>{_NUMBER})(?P<middle>\s*,\s*)"
    rf"(?P<height>{_NUMBER})(?P<suffix>\s*\))"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "supplement", "validate"))
    return parser


def _format_number(value: float) -> str:
    rounded = round(value, 3)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _increase_numeric_match(match: re.Match[str], delta: float) -> str:
    return match.group("prefix") + _format_number(float(match.group("value")) + delta)


def _scale_figsize_match(match: re.Match[str]) -> str:
    width = _format_number(float(match.group("width")) * FIGURE_SCALE)
    height = _format_number(float(match.group("height")) * FIGURE_SCALE)
    return (
        match.group("prefix")
        + width
        + match.group("middle")
        + height
        + match.group("suffix")
    )


def transform_visual_source(text: str) -> str:
    """Redirect output and increase every numeric typography argument."""

    transformed = text.replace(SOURCE_BUNDLE_NAME, TARGET_BUNDLE_NAME)
    transformed = transformed.replace(SOURCE_NOTEBOOK.name, TARGET_NOTEBOOK.name)
    transformed = _KWARG_FONT_RE.sub(
        lambda match: _increase_numeric_match(match, FONT_DELTA_PT), transformed
    )
    transformed = _DICT_FONT_RE.sub(
        lambda match: _increase_numeric_match(match, FONT_DELTA_PT), transformed
    )
    transformed = _FONT_SCALE_RE.sub(
        lambda match: _increase_numeric_match(match, FONT_DELTA_PT / 12.0),
        transformed,
    )
    transformed = _FIGSIZE_RE.sub(_scale_figsize_match, transformed)
    return transformed


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_notebook() -> None:
    if TARGET_BUNDLE.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {TARGET_BUNDLE}")
    if TARGET_NOTEBOOK.exists():
        raise FileExistsError(f"Refusing to overwrite existing notebook: {TARGET_NOTEBOOK}")
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)

    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    transformed_cells = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        original = cell.source
        cell.source = transform_visual_source(original)
        cell.execution_count = None
        cell.outputs = []
        transformed_cells += int(cell.source != original)

    combined_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    if TARGET_BUNDLE_NAME not in combined_source:
        raise RuntimeError("Target output bundle was not injected into the notebook.")
    source_bundle_literal = re.compile(
        rf"(?P<quote>[\"']){re.escape(SOURCE_BUNDLE_NAME)}(?P=quote)"
    )
    if source_bundle_literal.search(combined_source):
        raise RuntimeError("Source output bundle remains in transformed notebook code.")

    notebook.metadata["font_variant"] = {
        "source_notebook": str(SOURCE_NOTEBOOK),
        "font_delta_pt": FONT_DELTA_PT,
        "figure_scale": FIGURE_SCALE,
        "transformed_code_cells": transformed_cells,
    }
    nbformat.write(notebook, TARGET_NOTEBOOK)
    print(f"[PREPARED] {TARGET_NOTEBOOK}")
    print(f"[TARGET]   {TARGET_BUNDLE}")


def _execute_transformed_renderer(script_path: Path, argv: list[str]) -> None:
    source = transform_visual_source(script_path.read_text(encoding="utf-8"))
    if script_path.name == "render_gsm8k_v5_remaining_large_labels.py":
        source = source.replace(
            'assert len(list(FIGURE_DIR.glob("*.png"))) == 23',
            'assert len(list(FIGURE_DIR.glob("*.png"))) >= 23',
        ).replace(
            'assert len(list(FIGURE_DIR.glob("*.svg"))) == 23',
            'assert len(list(FIGURE_DIR.glob("*.svg"))) >= 23',
        )
    source_bundle_literal = re.compile(
        rf"(?P<quote>[\"']){re.escape(SOURCE_BUNDLE_NAME)}(?P=quote)"
    )
    if source_bundle_literal.search(source):
        raise RuntimeError(f"Source bundle remains in renderer: {script_path.name}")

    previous_argv = sys.argv
    try:
        sys.argv = [str(script_path), *argv]
        namespace = {
            "__file__": str(script_path),
            "__name__": "__main__",
            "__package__": None,
        }
        exec(compile(source, str(script_path), "exec"), namespace)
    finally:
        sys.argv = previous_argv


def _install_transformed_religion_modules() -> None:
    """Install transformed helper modules used by chained Religion renderers."""

    module_names = (
        "render_gsm8k_religion_primary_effect_holm_single_large_heatmap",
        "render_gsm8k_religion_effect_holm_with_neutral_single_large_heatmap",
        "render_gsm8k_religion_all_52_effect_holm_single_large_heatmap",
    )
    for module_name in module_names:
        script_path = RENDERER_DIR / f"{module_name}.py"
        source = transform_visual_source(script_path.read_text(encoding="utf-8"))
        module = types.ModuleType(module_name)
        module.__file__ = str(script_path)
        module.__package__ = None
        sys.modules[module_name] = module
        exec(compile(source, str(script_path), "exec"), module.__dict__)


def render_supplement() -> None:
    if not TARGET_FIGURE_DIR.is_dir() or not TARGET_TABLE_DIR.is_dir():
        raise FileNotFoundError(
            "Execute the transformed publication notebook before supplemental renderers."
        )
    source_main = [
        path.name
        for path in (SOURCE_BUNDLE / "figures").glob("*")
        if path.name[:2].isdigit() and int(path.name[:2]) <= 23
    ]
    missing_main = [name for name in source_main if not (TARGET_FIGURE_DIR / name).is_file()]
    if missing_main:
        raise FileNotFoundError(f"Missing main notebook figures: {missing_main}")

    for source_table in (SOURCE_BUNDLE / "tables").glob("*.csv"):
        target_table = TARGET_TABLE_DIR / source_table.name
        if not target_table.exists():
            shutil.copy2(source_table, target_table)

    _install_transformed_religion_modules()

    for name in FIXED_RENDERERS:
        path = RENDERER_DIR / name
        print(f"[RENDER] {name}")
        _execute_transformed_renderer(path, [])

    for name, stem in SINGLE_OUTPUT_RENDERERS:
        path = RENDERER_DIR / name
        for extension in ("png", "svg"):
            output = TARGET_FIGURE_DIR / f"{stem}.{extension}"
            print(f"[RENDER] {name} -> {output.name}")
            _execute_transformed_renderer(
                path,
                ["--output", str(output), "--dpi", "320"],
            )

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

    _refresh_output_index()
    manifest = {
        "bundle_version": "gsm8k-v5-publication-fontplus3-v1.0.0",
        "source_bundle": str(SOURCE_BUNDLE),
        "source_notebook": str(SOURCE_NOTEBOOK),
        "render_notebook": str(TARGET_NOTEBOOK),
        "font_delta_pt": FONT_DELTA_PT,
        "figure_scale": FIGURE_SCALE,
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
    (TARGET_BUNDLE / "font_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[VALID] 32 PNG + 32 SVG; {len(source_tables)} statistical tables unchanged")
    print(f"[DONE]  {TARGET_BUNDLE}")


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "prepare":
        prepare_notebook()
    elif args.phase == "supplement":
        render_supplement()
    else:
        validate_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

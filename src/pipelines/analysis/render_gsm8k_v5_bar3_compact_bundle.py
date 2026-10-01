"""Build a write-once GSM8K bundle with visibly larger compact bar charts.

Heatmaps are inherited from the validated star-only-bold variant. Accuracy and
delta bar charts are replaced by +3 pt renders on the original canvas size so
the typography remains visibly larger when the whole PNG is fit to a window.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

import render_gsm8k_v5_fontplus3_bundle as base


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
ORIGINAL_BUNDLE_NAME = "persona_publication_analysis_v5_paired_significance_six_models"
SOURCE_VARIANT_NAME = f"{ORIGINAL_BUNDLE_NAME}_bar2_heatmap3_sigbold"
TARGET_VARIANT_NAME = f"{ORIGINAL_BUNDLE_NAME}_bar3_compact_heatmap3_sigbold"

ORIGINAL_BUNDLE = REPORT_ROOT / ORIGINAL_BUNDLE_NAME
SOURCE_VARIANT = REPORT_ROOT / SOURCE_VARIANT_NAME
TARGET_VARIANT = REPORT_ROOT / TARGET_VARIANT_NAME
TARGET_TABLE_DIR = TARGET_VARIANT / "tables"
TARGET_FIGURE_DIR = TARGET_VARIANT / "figures"
BAR_STAGE_DIR = (
    REPO_ROOT / ".codex_tmp" / "gsm8k_v5_barplus3_compact_stage" / "figures"
)

BAR_STEMS = (
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
DELTA_RENDERERS = (
    "render_gsm8k_religion_delta_vs_neutral.py",
    "render_gsm8k_regional_delta_vs_neutral.py",
)


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_compact_bars() -> None:
    for stem in BAR_STEMS:
        for extension in ("png", "svg"):
            source = BAR_STAGE_DIR / f"{stem}.{extension}"
            target = TARGET_FIGURE_DIR / source.name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            print(f"[BAR +3 / COMPACT] {target.name}")


def _render_delta_vs_neutral() -> None:
    base.TARGET_NOTEBOOK = REPORT_ROOT / "GSM8K_bar3_compact_visual_only.ipynb"
    base.TARGET_BUNDLE_NAME = TARGET_VARIANT_NAME
    base.TARGET_BUNDLE = TARGET_VARIANT
    base.TARGET_TABLE_DIR = TARGET_TABLE_DIR
    base.TARGET_FIGURE_DIR = TARGET_FIGURE_DIR
    base.FONT_DELTA_PT = 3.0
    base.FIGURE_SCALE = 1.0
    for name in DELTA_RENDERERS:
        print(f"[BAR +3 / COMPACT] {name}")
        base._execute_transformed_renderer(base.RENDERER_DIR / name, [])


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
    pd.DataFrame(rows).to_csv(TARGET_VARIANT / "output_index.csv", index=False)


def _validate_and_manifest() -> None:
    png = sorted(TARGET_FIGURE_DIR.glob("*.png"))
    svg = sorted(TARGET_FIGURE_DIR.glob("*.svg"))
    if len(png) != 32 or len(svg) != 32:
        raise RuntimeError(f"Expected 32 PNG/SVG figures, found {len(png)}/{len(svg)}.")
    if any(path.stat().st_size == 0 for path in (*png, *svg)):
        raise RuntimeError("At least one figure is empty.")

    original_tables = {
        path.name: path for path in (ORIGINAL_BUNDLE / "tables").glob("*.csv")
    }
    target_tables = {path.name: path for path in TARGET_TABLE_DIR.glob("*.csv")}
    missing = sorted(set(original_tables) - set(target_tables))
    mismatched = [
        name
        for name, source in original_tables.items()
        if name in target_tables and sha256_file(source) != sha256_file(target_tables[name])
    ]
    if missing or mismatched:
        raise RuntimeError(f"Table validation failed; missing={missing}, mismatched={mismatched}")

    _refresh_output_index()
    manifest = {
        "bundle_version": "gsm8k-v5-bar3-compact-heatmap3-sigbold-v1.0.0",
        "original_bundle": str(ORIGINAL_BUNDLE),
        "inherited_heatmap_variant": str(SOURCE_VARIANT),
        "bar_font_delta_pt": 3.0,
        "bar_figure_scale": 1.0,
        "delta_annotations": "vertical, 21 pt for figures 02/03/06/13/16",
        "heatmap_font_delta_pt": 3.0,
        "heatmap_cell_weight_rule": "bold_if_annotation_contains_asterisk_else_normal",
        "figure16_metric": "relative completion-token change; not accuracy",
        "figure16_formula": "100 * (implicit_avg - explicit) / explicit",
        "statistical_tables_identical": True,
        "table_count": len(original_tables),
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
    (TARGET_VARIANT / "typography_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[VALID] 32 PNG + 32 SVG; {len(original_tables)} tables hash-identical")


def main() -> int:
    if TARGET_VARIANT.exists():
        raise FileExistsError(f"Refusing to overwrite target: {TARGET_VARIANT}")
    if not SOURCE_VARIANT.is_dir():
        raise FileNotFoundError(SOURCE_VARIANT)
    if not BAR_STAGE_DIR.is_dir():
        raise FileNotFoundError(BAR_STAGE_DIR)

    shutil.copytree(SOURCE_VARIANT, TARGET_VARIANT)
    _copy_compact_bars()
    _render_delta_vs_neutral()
    _validate_and_manifest()
    print(f"[DONE] {TARGET_VARIANT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

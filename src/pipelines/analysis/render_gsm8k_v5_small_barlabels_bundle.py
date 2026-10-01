"""Build a write-once GSM8K bundle with smaller bar-value annotations."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "result" / "GMS8K" / "reports"
ORIGINAL_NAME = "persona_publication_analysis_v5_paired_significance_six_models"
SOURCE_NAME = f"{ORIGINAL_NAME}_bar3_compact_heatmap3_sigbold"
TARGET_NAME = f"{ORIGINAL_NAME}_bar3_compact_smalllabels_heatmap3_sigbold"

ORIGINAL = REPORT_ROOT / ORIGINAL_NAME
SOURCE = REPORT_ROOT / SOURCE_NAME
TARGET = REPORT_ROOT / TARGET_NAME
TARGET_TABLES = TARGET / "tables"
TARGET_FIGURES = TARGET / "figures"
STAGE_FIGURES = (
    REPO_ROOT
    / ".codex_tmp"
    / "gsm8k_v5_bar3_compact_smalllabels_stage"
    / "figures"
)

REPLACED_STEMS = (
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


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def replace_bar_figures() -> None:
    for stem in REPLACED_STEMS:
        for extension in ("png", "svg"):
            source = STAGE_FIGURES / f"{stem}.{extension}"
            target = TARGET_FIGURES / source.name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            print(f"[SMALL BAR LABEL] {target.name}")


def refresh_output_index() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(TARGET_TABLES.glob("*.csv")):
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
    for path in sorted(TARGET_FIGURES.glob("*")):
        rows.append(
            {
                "type": "figure",
                "file": path.name,
                "rows": None,
                "columns": None,
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(TARGET / "output_index.csv", index=False)


def validate_and_manifest() -> None:
    png = sorted(TARGET_FIGURES.glob("*.png"))
    svg = sorted(TARGET_FIGURES.glob("*.svg"))
    if len(png) != 32 or len(svg) != 32:
        raise RuntimeError(f"Expected 32 PNG/SVG, found {len(png)}/{len(svg)}.")
    if any(path.stat().st_size == 0 for path in (*png, *svg)):
        raise RuntimeError("At least one figure is empty.")

    original_tables = {path.name: path for path in (ORIGINAL / "tables").glob("*.csv")}
    target_tables = {path.name: path for path in TARGET_TABLES.glob("*.csv")}
    mismatched = [
        name
        for name, source in original_tables.items()
        if name not in target_tables or sha256_file(source) != sha256_file(target_tables[name])
    ]
    if mismatched:
        raise RuntimeError(f"Statistical table mismatch: {mismatched}")

    refresh_output_index()
    manifest = {
        "bundle_version": "gsm8k-v5-bar3-compact-smalllabels-heatmap3-sigbold-v1.0.0",
        "original_bundle": str(ORIGINAL),
        "source_visual_bundle": str(SOURCE),
        "bar_axis_title_legend_font_delta_pt": 3.0,
        "accuracy_bar_value_annotation_pt": 17.0,
        "delta_bar_value_annotation_pt": 16.0,
        "delta_annotation_orientation": "vertical",
        "heatmaps_unchanged_from_source_visual_bundle": True,
        "statistical_tables_identical": True,
        "table_count": len(original_tables),
        "png_count": len(png),
        "svg_count": len(svg),
        "replaced_bar_stems": list(REPLACED_STEMS),
        "figures": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (*png, *svg)
        ],
    }
    (TARGET / "typography_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[VALID] 32 PNG + 32 SVG; {len(original_tables)} tables hash-identical")


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite target: {TARGET}")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if not STAGE_FIGURES.is_dir():
        raise FileNotFoundError(STAGE_FIGURES)
    shutil.copytree(SOURCE, TARGET)
    replace_bar_figures()
    validate_and_manifest()
    print(f"[DONE] {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

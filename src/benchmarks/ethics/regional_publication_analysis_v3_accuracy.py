"""Regional Cultural Persona wrapper for the shared v3 accuracy engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import religion_publication_analysis_v3_accuracy as engine


REGION_ORDER = [
    "east_asian",
    "north_american",
    "south_asian",
    "sub_saharan_african",
]
REGION_LABELS = {
    "neutral": "Netral",
    "east_asian": "East Asian",
    "north_american": "North American",
    "south_asian": "South Asian",
    "sub_saharan_african": "Sub-Saharan African",
}


def configure_engine() -> None:
    engine.ANALYSIS_VERSION = "ethics-regional-v3-accuracy-paired-holm-20260822"
    engine.DOMAIN_SLUG = "regional"
    engine.DOMAIN_TITLE = "Regional Cultural Persona"
    engine.PAIRWISE_TITLE = "Antarwilayah Budaya"
    engine.OUTPUT_GROUP_DIR = "Regional_Cultural_Persona"
    engine.RELIGION_ORDER = REGION_ORDER
    engine.RELIGION_LABELS = REGION_LABELS
    engine.PERSONA_ORDER = ["neutral"] + [
        engine.persona_name(region, scenario)
        for region in REGION_ORDER
        for scenario in ["E", "I1", "I2", "I3"]
    ]
    engine.PERSONA_META = {"neutral": ("neutral", "N")}
    engine.PERSONA_LABELS = {"neutral": "Netral"}
    for region in REGION_ORDER:
        for scenario in ["E", "I1", "I2", "I3"]:
            persona = engine.persona_name(region, scenario)
            engine.PERSONA_META[persona] = (region, scenario)
            engine.PERSONA_LABELS[persona] = f"{REGION_LABELS[region]} {scenario}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ETHICS Regional Cultural Persona v3 visualizations."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tables_only and args.figures_only:
        raise SystemExit("--tables-only dan --figures-only tidak dapat dipakai bersamaan")
    configure_engine()
    repo_root = engine.base.find_repo_root()
    input_path = (args.input or engine.default_input_path(repo_root)).resolve()
    output_dir = (args.output or engine.default_output_dir(repo_root)).resolve()
    if args.figures_only:
        print(f"[RESUME] Output: {output_dir}")
        rendered = engine.render_missing_figures(output_dir)
        print(f"[RESUME] File gambar baru: {len(rendered)}")
        validation = engine.finalize_existing_output(input_path, output_dir)
        print(
            f"[DONE] Resume Regional selesai | png={validation['png_files']} | "
            f"svg={validation['svg_files']} | valid={validation['valid']}"
        )
    else:
        engine.run_analysis(
            input_path=input_path,
            output_dir=output_dir,
            tables_only=args.tables_only,
        )


if __name__ == "__main__":
    main()

"""Render the standardized combined ETHICS Religion accuracy bundle.

This is a presentation-only adapter over the validated Religion v3 bundle. It
reuses the Regional renderer because both domains have four identities, one
explicit condition, three implicit conditions, and one neutral condition.
No response is reparsed or rescored and no inferential statistic is recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import render_ethics_regional_overall_standardized as engine


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BUNDLE = (
    REPO_ROOT
    / "result"
    / "ETHICS"
    / "reports"
    / "Religion_Persona"
    / "_combined"
    / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "religion" / "ethics accuracy analysis"

IDENTITY_ORDER = ["islam", "christian", "jewish", "atheist"]
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Kristen",
    "jewish": "Yahudi",
    "atheist": "Ateis",
}
IDENTITY_SHORT_LABELS = {
    "islam": "IS",
    "christian": "KR",
    "jewish": "YH",
    "atheist": "AT",
}


def _category_specs() -> list[tuple[str, str, list[str]]]:
    specs: list[tuple[str, str, list[str]]] = []
    for identity in IDENTITY_ORDER:
        label = IDENTITY_LABELS[identity]
        specs.append((f"{label} Explicit", f"{label} E", [f"{identity}_explicit"]))
        specs.append(
            (
                f"{label} Implicit Avg",
                f"{label} I",
                [f"{identity}_implicit_1", f"{identity}_implicit_2", f"{identity}_implicit_3"],
            )
        )
    specs.append(("Neutral", "Neutral", ["neutral"]))
    return specs


def configure_engine() -> None:
    specs = _category_specs()
    engine.SOURCE_BUNDLE = SOURCE_BUNDLE
    engine.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    engine.BUNDLE_VERSION = "ethics-religion-accuracy-combined-standardized-v1"
    engine.DOMAIN_TITLE = "Religion"
    engine.DOMAIN_SLUG = "religion"
    engine.EXPECTED_ANALYSIS_VERSION = "ethics-religion-v3-accuracy-paired-holm-20260822"
    engine.SCENARIO_FAMILY = "scenario_within_religion"
    engine.PAIRWISE_FAMILY = "religion_pair_within_scenario"
    engine.PAIRWISE_TABLE_FILENAME = "mcnemar_religion_pairwise_comparisons.csv"
    engine.COCHRAN_TABLE_FILENAME = "cochran_q_by_subset_model_religion.csv"
    engine.SOURCE_GROUP_PATH = "Religion_Persona"
    engine.NOTEBOOK_BUILDER_NAME = "build_ethics_religion_combined_accuracy_notebook.py"
    engine.NOTEBOOK_FILE_NAME = "ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb"
    engine.PARENT_SCOPE = "combined_ethics_religion"
    engine.REGION_ORDER = IDENTITY_ORDER
    engine.REGION_LABELS = IDENTITY_LABELS
    engine.REGION_SHORT_LABELS = IDENTITY_SHORT_LABELS
    engine.CATEGORY_SPECS = specs
    engine.CATEGORY_ORDER = [item[0] for item in specs]
    engine.CATEGORY_SHORT_LABELS = {item[0]: item[1] for item in specs}
    # Religion figures use the shared visual system with every text element
    # increased by exactly one point relative to the Regional defaults.
    engine.ANNOTATION_SIZE = 15
    engine.DELTA_ANNOTATION_SIZE = 16
    engine.TITLE_SIZE = 23
    engine.STATUS_ANNOTATION_SIZE = 15
    engine.STATUS_FIGSIZE = (32, 21)
    engine.STATUS_WSPACE = 0.24
    engine.STATUS_XTICK_SIZE = 11
    engine.REASON_TITLE_SIZE = 29
    engine.REASON_AXIS_LABEL_SIZE = 23
    engine.REASON_TICK_SIZE = 18
    engine.AXES_LABEL_SIZE = 18
    engine.XTICK_LABEL_SIZE = 14
    engine.YTICK_LABEL_SIZE = 16
    engine.LEGEND_SIZE = 14
    engine.LEGEND_TITLE_SIZE = 15
    engine.MODEL_TITLE_SIZE = 17
    engine.REASON_XTICK_SIZE = 18
    engine.REASON_ANNOTATION_SIZE = 21
    engine.STAT_FONT_SIZE_OFFSET = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=SOURCE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    configure_engine()
    engine.render(args.source_bundle, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

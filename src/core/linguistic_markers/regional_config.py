"""Configuration for the source-aligned GSM8K regional-cultural pipeline."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from .config import MAIN_MODELS, OPENING_CATEGORIES, PLANNED_CONTRASTS, RANDOM_SEED


PIPELINE_VERSION = "regional-gsm8k-lm-v1.1.0-candidate"
REPO_ROOT = Path(__file__).resolve().parents[3]

GSM8K_REGIONAL_PARSED_CSV = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Regional_Cultural_Persona"
    / "_combined"
    / "all_models_results_full.csv"
)
GSM8K_NEUTRAL_PARSED_CSV = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Gender_Persona"
    / "publication_source_aligned_v1_20260817"
    / "_combined"
    / "all_models_results_full.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Regional_Cultural_Persona"
    / "linguistic_markers"
    / PIPELINE_VERSION
)

REGIONAL_IDENTITIES = (
    "north_american",
    "east_asian",
    "south_asian",
    "sub_saharan_african",
)
REGIONAL_PERSONAS = (
    "neutral",
    *tuple(
        f"{identity}_{condition}"
        for identity in REGIONAL_IDENTITIES
        for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3")
    ),
)
IDENTITY_CONDITIONS = {
    identity: {
        "neutral": "neutral",
        "explicit": f"{identity}_explicit",
        "implicit_1": f"{identity}_implicit_1",
        "implicit_2": f"{identity}_implicit_2",
        "implicit_3": f"{identity}_implicit_3",
    }
    for identity in REGIONAL_IDENTITIES
}
DIRECT_IDENTITY_PAIRS = tuple(combinations(REGIONAL_IDENTITIES, 2))
DIRECT_CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
EXPECTED_QUESTIONS = 1319

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DIRECT_CONDITIONS",
    "DIRECT_IDENTITY_PAIRS",
    "EXPECTED_QUESTIONS",
    "GSM8K_NEUTRAL_PARSED_CSV",
    "GSM8K_REGIONAL_PARSED_CSV",
    "IDENTITY_CONDITIONS",
    "MAIN_MODELS",
    "OPENING_CATEGORIES",
    "PIPELINE_VERSION",
    "PLANNED_CONTRASTS",
    "RANDOM_SEED",
    "REGIONAL_IDENTITIES",
    "REGIONAL_PERSONAS",
    "REPO_ROOT",
]

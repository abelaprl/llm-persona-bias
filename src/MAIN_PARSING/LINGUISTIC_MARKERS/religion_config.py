"""Configuration for the source-aligned GSM8K religion marker pipeline."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from .config import MAIN_MODELS, OPENING_CATEGORIES, PLANNED_CONTRASTS, RANDOM_SEED


PIPELINE_VERSION = "religion-gsm8k-lm-v1.3.0-candidate"
REPO_ROOT = Path(__file__).resolve().parents[3]

GSM8K_RELIGION_PARSED_CSV = (
    REPO_ROOT
    / "result"
    / "GMS8K"
    / "reports"
    / "Religion_Persona"
    / "publication_source_aligned_v1_20260817"
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
    / "Religion_Persona"
    / "linguistic_markers"
    / PIPELINE_VERSION
)

RELIGION_IDENTITIES = ("islam", "christian", "jewish", "atheist")
RELIGION_PERSONAS = (
    "neutral",
    "islam_explicit",
    "islam_implicit_1",
    "islam_implicit_2",
    "islam_implicit_3",
    "christian_explicit",
    "christian_implicit_1",
    "christian_implicit_2",
    "christian_implicit_3",
    "jewish_explicit",
    "jewish_implicit_1",
    "jewish_implicit_2",
    "jewish_implicit_3",
    "atheist_explicit",
    "atheist_implicit_1",
    "atheist_implicit_2",
    "atheist_implicit_3",
)

IDENTITY_CONDITIONS = {
    identity: {
        "neutral": "neutral",
        "explicit": f"{identity}_explicit",
        "implicit_1": f"{identity}_implicit_1",
        "implicit_2": f"{identity}_implicit_2",
        "implicit_3": f"{identity}_implicit_3",
    }
    for identity in RELIGION_IDENTITIES
}

# All six planned identity pairs are retained. Direct differences are always
# left minus right in this fixed order; no Abrahamic identities are pooled.
DIRECT_IDENTITY_PAIRS = tuple(combinations(RELIGION_IDENTITIES, 2))
DIRECT_CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
EXPECTED_QUESTIONS = 1319

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DIRECT_CONDITIONS",
    "DIRECT_IDENTITY_PAIRS",
    "EXPECTED_QUESTIONS",
    "GSM8K_NEUTRAL_PARSED_CSV",
    "GSM8K_RELIGION_PARSED_CSV",
    "IDENTITY_CONDITIONS",
    "MAIN_MODELS",
    "OPENING_CATEGORIES",
    "PIPELINE_VERSION",
    "PLANNED_CONTRASTS",
    "RANDOM_SEED",
    "RELIGION_IDENTITIES",
    "RELIGION_PERSONAS",
    "REPO_ROOT",
]

"""Configuration shared by the gender linguistic-marker pipeline."""

from pathlib import Path


PIPELINE_VERSION = "gender-lm-v1.2.1-candidate"
RANDOM_SEED = 18222008

REPO_ROOT = Path(__file__).resolve().parents[3]
GSM8K_GENDER_ROOT = REPO_ROOT / "result" / "GMS8K" / "Gender_Persona"
ETHICS_MAIN_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"
ETHICS_PARSED_ROOT = REPO_ROOT / "result" / "reports" / "v9_main"
GSM8K_PARSED_CSV = (
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
    REPO_ROOT / "result" / "linguistic_markers" / PIPELINE_VERSION
)

MAIN_MODELS = (
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)

GENDER_PERSONAS = (
    "neutral",
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
)

IDENTITY_CONDITIONS = {
    "man": {
        "neutral": "neutral",
        "explicit": "man_explicit_1",
        "implicit_1": "man_implicit_1",
        "implicit_2": "man_implicit_2",
        "implicit_3": "man_implicit_3",
    },
    "woman": {
        "neutral": "neutral",
        "explicit": "woman_explicit_1",
        "implicit_1": "woman_implicit_1",
        "implicit_2": "woman_implicit_2",
        "implicit_3": "woman_implicit_3",
    },
}

# The subtraction direction follows Bab IV: persona minus comparator.
PLANNED_CONTRASTS = (
    ("explicit_vs_neutral", "explicit", "neutral"),
    ("implicit_1_vs_neutral", "implicit_1", "neutral"),
    ("implicit_2_vs_neutral", "implicit_2", "neutral"),
    ("implicit_3_vs_neutral", "implicit_3", "neutral"),
    ("implicit_1_vs_explicit", "implicit_1", "explicit"),
    ("implicit_2_vs_explicit", "implicit_2", "explicit"),
    ("implicit_3_vs_explicit", "implicit_3", "explicit"),
)

OPENING_CATEGORIES = (
    "deliberative",
    "refusal",
    "answer_first",
    "document_heading",
    "step_first",
    "bullet_first",
    "algebraic",
    "instructional",
    "problem_restatement",
    "contextual",
    "narrative",
)

EXPECTED_QUESTIONS = {"gsm8k": 1319, "ethics": 1514}

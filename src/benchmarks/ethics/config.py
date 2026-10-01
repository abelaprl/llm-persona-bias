"""Shared configuration for the ETHICS v9 answer-scoring pipeline.

Every path, mapping, and column list used by the pipeline lives here so that a
reviewer can audit the assumptions in one place.

See docs/plan_parsing_ethics.md for the evidence behind each mapping and
AGENT.md for the rules this pipeline must not violate.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Root of the authoritative v9 result tree. The folder name contains spaces
#: and brackets on purpose - it is the directory the experiments actually wrote
#: to. Do not "clean up" this name without migrating the data.
RESULT_ROOT = REPO_ROOT / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]"

#: Intermediate per-persona parse cache (one JSON object per line).
PARSE_CACHE_ROOT = REPO_ROOT / "result" / "parsing" / "v9_main"

#: Final CSV reports. Deliberately a NEW path: the old reports under
#: result/reports/v9_pilot/ must stay intact for comparison (AGENT.md 2.7).
REPORT_ROOT = REPO_ROOT / "result" / "reports" / "v9_main"

#: Benchmark items with their reference labels.
SAMPLES_FILE = REPO_ROOT / "data" / "ETHICS_test_4subsets_10percent_from200.json"


# --------------------------------------------------------------------------
# Experiment layout
# --------------------------------------------------------------------------

#: Folder name on disk -> short slug used in output filenames.
GROUPS: dict[str, str] = {
    "Gender Persona": "gender",
    "Religion Persona": "religion",
    "Regional Cultural Persona": "regional",
    "Neutral Persona": "neutral",
}

MODELS: tuple[str, ...] = (
    "deepseek_deepseek-r1-distill-qwen-32b",
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)

SUBSETS: tuple[str, ...] = ("commonsense", "deontology", "justice", "virtue")


# --------------------------------------------------------------------------
# Scoring semantics
# --------------------------------------------------------------------------

STANCE_WRONG = "wrong"
STANCE_ACCEPTABLE = "acceptable"

#: What the dataset label MEANS, per subset, expressed as a moral stance.
#:
#: ETHICS labels are NOT uniform across subsets. Verified directly from
#: data/ETHICS_*.json - see docs/plan_parsing_ethics.md finding T1:
#:
#:   commonsense : "I covered up the truth ..."   -> 1   (the act is WRONG)
#:                 "I exposed the truth ..."      -> 0   (acceptable)
#:   deontology  : 1 = the excuse is reasonable
#:   justice     : 1 = the statement is justified
#:   virtue      : 1 = the [SEP] trait fits the character
#:
#: The previous pipeline treated 1 as "acceptable" for every subset, which
#: inverted every commonsense score.
GT_STANCE: dict[str, dict[int, str]] = {
    "commonsense": {1: STANCE_WRONG, 0: STANCE_ACCEPTABLE},
    "deontology": {1: STANCE_ACCEPTABLE, 0: STANCE_WRONG},
    "justice": {1: STANCE_ACCEPTABLE, 0: STANCE_WRONG},
    "virtue": {1: STANCE_ACCEPTABLE, 0: STANCE_WRONG},
}

#: Whether accuracy on this subset may be interpreted as "the model's ability
#: to make a moral judgement".
#:
#:   deontology -> False: the `excuse` field was never sent to the model, yet
#:                 the label grades that excuse (finding T2).
#:   virtue     -> False: the label grades whether a trait word fits the
#:                 character, but the prompt asks right/wrong (finding T3).
#:
#: Both subsets are still parsed and still used for cross-persona comparison,
#: because the defect is constant across personas.
CONSTRUCT_VALID: dict[str, bool] = {
    "commonsense": True,
    "justice": True,
    "deontology": False,
    "virtue": False,
}

#: Digit -> stance, for the two conventions observed in the corpus.
#: The v9 prompt never defines what 0 and 1 mean (finding T4), so the mapping
#: has to be recovered per response.
CONVENTION_STANDARD = "standard"   # 0 = wrong, 1 = right
CONVENTION_INVERTED = "inverted"   # 0 = right, 1 = wrong (e.g. AITA framing)

DIGIT_STANCE: dict[str, dict[str, str]] = {
    CONVENTION_STANDARD: {"0": STANCE_WRONG, "1": STANCE_ACCEPTABLE},
    CONVENTION_INVERTED: {"0": STANCE_ACCEPTABLE, "1": STANCE_WRONG},
}

#: Used when the model does not declare its own mapping. Measured on 9,123
#: self-glossing responses in Gender Persona: 0 = wrong dominates in every
#: model (71%-97%).
DEFAULT_CONVENTION = CONVENTION_STANDARD


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

#: Columns carried over from result/reports/v9_pilot/ so existing notebooks
#: keep working. Order matters - keep it stable.
LEGACY_COLUMNS: tuple[str, ...] = (
    "model",
    "persona_folder",
    "file",
    "question_id",
    "dataset",
    "ground_truth",
    "response_content_full",
    "extracted_answer",
    "answer_status",
    "parse_source",
    "is_correct",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "truncated",
    "text_tail",
)

#: New provenance columns. Every one of them exists so a specific decision can
#: be audited or reversed later.
NEW_COLUMNS: tuple[str, ...] = (
    "subset",
    "persona_group",
    "extracted_stance",
    "gt_stance",
    "digit_convention",
    "candidate_answer",
    "candidate_source",
    "behavior_category",
    "abstain_reason",
    "extraction_confidence",
    "numeric_semantic_conflict",
    "response_flags",
    "construct_valid",
)

CSV_COLUMNS: tuple[str, ...] = LEGACY_COLUMNS + NEW_COLUMNS

SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "persona_group",
    "persona",
    "subset",
    "construct_valid",
    "total",
    "committed",
    "abstain",
    "missing",
    "commit_rate",
    "abstain_rate",
    "missing_rate",
    "n_correct",
    "accuracy_committed",
    "accuracy_committed_excl_conflict",
    "n_conflict",
    "pct_answer_wrong",
    "pct_answer_acceptable",
)

#: Number of trailing characters kept in the `text_tail` column.
TEXT_TAIL_CHARS = 200

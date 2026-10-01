"""Core analysis engine for the candidate ETHICS Religion marker bundle v2.

The module never extracts a verdict and never reads ground truth while creating
linguistic-marker values.  It joins the frozen marker surface to canonical
scoring only after marker extraction, then performs paired analyses.  The
source registry is still awaiting two independent human reviewers; therefore
every result produced here remains a candidate and cannot be cited as final.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import MonteCarloMethod, chi2, chi2_contingency, friedmanchisquare

from src.core.linguistic_markers.statistics import (
    _bootstrap_ci,
    _mcnemar_exact,
    _paired_numeric_test,
    _stable_seed,
    holm_adjust,
    paired_stuart_maxwell,
)


BUNDLE_VERSION = "religion-ethics-linguistic-markers-v2.0.2-candidate"
EXPECTED_PIPELINE_VERSION = "ethics-religion-lm-v1.1.10-candidate"
EXPECTED_INPUT_SHA256 = (
    "c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a"
)
EXPECTED_ROWS = 154_428
EXPECTED_QUESTIONS = 1_514
RANDOM_SEED = 18_222_008

MODELS = (
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
)
MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
IDENTITIES = ("islam", "christian", "jewish", "atheist")
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Kristen",
    "jewish": "Yahudi",
    "atheist": "Ateis",
}
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = {
    "neutral": "N",
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
SUBSETS = ("commonsense", "deontology", "justice", "virtue")
CONSTRUCT_VALID_SUBSETS = ("commonsense", "justice")
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

IDENTITY_CONDITIONS = {
    identity: {
        "neutral": "neutral",
        "explicit": f"{identity}_explicit",
        "implicit_1": f"{identity}_implicit_1",
        "implicit_2": f"{identity}_implicit_2",
        "implicit_3": f"{identity}_implicit_3",
    }
    for identity in IDENTITIES
}
PERSONAS = (
    "neutral",
    *(
        f"{identity}_{condition}"
        for identity in IDENTITIES
        for condition in CONDITIONS
    ),
)
DIRECT_IDENTITY_PAIRS = tuple(combinations(IDENTITIES, 2))

# All differences are left minus right.  The family names encode the
# pre-registered multiplicity correction required by Bab IV.
WITHIN_CONTRASTS = (
    ("explicit_vs_neutral", "explicit", "neutral", "neutral_sensitivity_4"),
    ("implicit_1_vs_neutral", "implicit_1", "neutral", "neutral_sensitivity_4"),
    ("implicit_2_vs_neutral", "implicit_2", "neutral", "neutral_sensitivity_4"),
    ("implicit_3_vs_neutral", "implicit_3", "neutral", "neutral_sensitivity_4"),
    ("implicit_1_vs_explicit", "implicit_1", "explicit", "primary_implicit_explicit_3"),
    ("implicit_2_vs_explicit", "implicit_2", "explicit", "primary_implicit_explicit_3"),
    ("implicit_3_vs_explicit", "implicit_3", "explicit", "primary_implicit_explicit_3"),
)
PRIMARY_WITHIN_CONTRASTS = tuple(
    contrast for contrast in WITHIN_CONTRASTS if contrast[3] == "primary_implicit_explicit_3"
)

SURFACE_OUTCOMES = ("word_count", "avg_sentence_length", "ttr_300")
CORE_MARKERS = (
    "organization_explicit_step",
    "organization_bullet",
    "organization_section_header",
    "stance_hedge",
    "stance_booster",
    "stance_assumption",
    "stance_self_correction",
    "connective_causal",
    "connective_contrastive",
    "connective_conditional",
    "self_reference_first_person",
    "self_reference_inclusive_we",
    "engagement_reader_directive",
    "stance_contextual_hedge",
    "stance_distancing",
    "connective_additive",
    "connective_concessive",
    "affect_empathy_validation",
)


@dataclass(frozen=True)
class OutcomeSpec:
    """One pre-defined analysis outcome and its single inferential scale."""

    outcome_id: str
    outcome_tier: str
    outcome_family: str
    value_col: str
    raw_col: str
    outcome_scale: str
    registered_measurement: str
    analysis_role: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return (
        values.astype("string")
        .str.strip()
        .str.casefold()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def nullable_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")
    return (
        values.astype("string")
        .str.strip()
        .str.casefold()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    m = len(order)
    running = 1.0
    for reverse_rank, position in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        running = min(running, p[position] * m / rank)
        adjusted[position] = min(1.0, running)
    return adjusted


def build_outcome_dictionary(registry: pd.DataFrame) -> tuple[pd.DataFrame, tuple[OutcomeSpec, ...]]:
    required = {
        "marker_id",
        "family",
        "measurement",
        "analysis_role",
        "registry_scope",
        "operational_definition",
        "theoretical_source",
        "validation_status",
    }
    missing = required - set(registry.columns)
    if missing:
        raise KeyError(f"Marker registry misses columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    specs: list[OutcomeSpec] = []
    surface_meta = {
        "word_count": ("text_length", "Jumlah token kata pada penjelasan visible."),
        "avg_sentence_length": (
            "text_length",
            "Rata-rata jumlah kata per kalimat pada penjelasan visible.",
        ),
        "ttr_300": (
            "lexical_diversity",
            "Type-token ratio yang dibatasi pada maksimum 300 token kata.",
        ),
    }
    for outcome_id in SURFACE_OUTCOMES:
        family, definition = surface_meta[outcome_id]
        spec = OutcomeSpec(
            outcome_id=outcome_id,
            outcome_tier="core_confirmatory",
            outcome_family=family,
            value_col=outcome_id,
            raw_col="",
            outcome_scale="numeric",
            registered_measurement="surface",
            analysis_role="primary",
        )
        specs.append(spec)
        rows.append(
            {
                **spec.__dict__,
                "operational_definition": definition,
                "theoretical_source": "Surface metric; pipeline operationalization",
                "validation_status": "derived_from_frozen_visible_surface",
                "registry_scope": "surface",
                "density_unit": "not_applicable",
                "raw_count_reported": False,
            }
        )

    registry_ids = set(registry["marker_id"].astype(str))
    absent_core = set(CORE_MARKERS) - registry_ids
    if absent_core:
        raise ValueError(f"Core marker absent from registry: {sorted(absent_core)}")

    for row in registry.itertuples(index=False):
        marker_id = str(row.marker_id)
        if marker_id in CORE_MARKERS:
            tier = "core_confirmatory"
        elif str(row.registry_scope) == "religion_specific":
            tier = "religion_specific"
        else:
            tier = "exploratory_general"
        measurement = str(row.measurement)
        scale = "binary" if measurement == "binary" else "density_per_100_words"
        spec = OutcomeSpec(
            outcome_id=marker_id,
            outcome_tier=tier,
            outcome_family=str(row.family),
            value_col=(
                f"{marker_id}__present"
                if measurement == "binary"
                else f"{marker_id}__density"
            ),
            raw_col=f"{marker_id}__count",
            outcome_scale=scale,
            registered_measurement=measurement,
            analysis_role=str(row.analysis_role),
        )
        specs.append(spec)
        rows.append(
            {
                **spec.__dict__,
                "operational_definition": str(row.operational_definition),
                "theoretical_source": str(row.theoretical_source),
                "validation_status": str(row.validation_status),
                "registry_scope": str(row.registry_scope),
                "density_unit": (
                    "occurrences_per_100_words"
                    if measurement != "binary"
                    else "response_presence_0_1"
                ),
                "raw_count_reported": True,
            }
        )

    dictionary = pd.DataFrame(rows)
    tier_counts = dictionary["outcome_tier"].value_counts().to_dict()
    expected = {
        "core_confirmatory": 21,
        "exploratory_general": 12,
        "religion_specific": 20,
    }
    if tier_counts != expected:
        raise ValueError(f"Unexpected outcome hierarchy: {tier_counts}; expected {expected}")
    if len(specs) != 53:
        raise ValueError(f"Expected 53 pre-defined outcomes; got {len(specs)}")
    return dictionary, tuple(specs)


def load_sources(
    marker_dir: Path,
    canonical_csv: Path,
    audit_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict[str, object]]:
    """Load and validate frozen marker and scoring sources without reparsing."""

    marker_manifest_path = marker_dir / "manifest.json"
    registry_path = marker_dir / "marker_registry.csv"
    response_path = marker_dir / "response_level_markers.csv.gz"
    source_paths = (
        marker_manifest_path,
        registry_path,
        response_path,
        canonical_csv,
        audit_manifest_path,
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    marker_manifest = json.loads(marker_manifest_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if marker_manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError(f"Unexpected marker pipeline: {marker_manifest.get('pipeline_version')}")
    if marker_manifest.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("Marker manifest does not point to the pinned canonical source.")
    canonical_hash = sha256_file(canonical_csv)
    if canonical_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Canonical SHA-256 changed: {canonical_hash}")
    if marker_manifest.get("inference_run") is not False:
        raise ValueError("Frozen marker source unexpectedly contains inference.")
    if audit_manifest.get("human_review_status") != "pending_two_independent_human_reviewers":
        raise ValueError("Unexpected human review state in audit manifest.")

    registry = pd.read_csv(registry_path, low_memory=False)
    outcome_dictionary, outcomes = build_outcome_dictionary(registry)
    marker_columns = [
        f"{marker_id}__{suffix}"
        for marker_id in registry["marker_id"].astype(str)
        for suffix in ("count", "density", "present")
    ]
    response_usecols = [
        "model_id",
        "persona",
        "identity",
        "condition",
        "condition_bucket",
        "question_id",
        "subset",
        "answer_status",
        "abstain_reason",
        "text_status",
        "text_available",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        "moral_polarity_initial",
        "moral_polarity_final",
        "moral_polarity_shift",
        "moral_polarity_direction",
        *marker_columns,
    ]
    frame = pd.read_csv(response_path, usecols=response_usecols, low_memory=False)
    frame = frame.rename(columns={"answer_status": "marker_answer_status"})
    frame["text_available"] = bool_series(frame["text_available"])
    numeric_columns = [
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "ttr_300",
        *marker_columns,
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    canonical_usecols = [
        "model",
        "persona_folder",
        "question_id",
        "dataset",
        "ground_truth",
        "extracted_answer",
        "answer_status",
        "parse_source",
        "candidate_answer",
        "candidate_source",
        "behavior_category",
        "abstain_reason",
        "extraction_confidence",
        "numeric_semantic_conflict",
        "response_flags",
        "is_correct",
        "extracted_stance",
        "construct_valid",
        "finish_reason",
        "truncated",
    ]
    scoring = pd.read_csv(canonical_csv, usecols=canonical_usecols, low_memory=False).rename(
        columns={
            "model": "model_id",
            "persona_folder": "persona",
            "abstain_reason": "scoring_abstain_reason",
        }
    )
    scoring["numeric_semantic_conflict"] = bool_series(
        scoring["numeric_semantic_conflict"]
    )
    scoring["is_correct"] = nullable_bool(scoring["is_correct"])
    scoring["construct_valid"] = nullable_bool(scoring["construct_valid"])
    scoring["truncated"] = bool_series(scoring["truncated"])

    key = ["model_id", "persona", "question_id"]
    if frame.duplicated(key).any() or scoring.duplicated(key).any():
        raise ValueError("Duplicate model/persona/question key in an input source.")
    frame = frame.merge(scoring, on=key, how="left", validate="one_to_one")
    # Consolidate the 150 marker columns before adding analysis provenance.
    frame = frame.copy()
    if len(frame) != EXPECTED_ROWS or frame[key].isna().any().any():
        raise ValueError("Merged response count or keys do not match the canonical corpus.")
    if not frame["marker_answer_status"].astype(str).equals(frame["answer_status"].astype(str)):
        raise ValueError("Marker and canonical answer statuses are not row-identical.")
    if set(frame["model_id"].unique()) != set(MODELS):
        raise ValueError("Unexpected model set.")
    if set(frame["persona"].unique()) != set(PERSONAS):
        raise ValueError("Unexpected persona set.")
    if set(frame["subset"].unique()) != set(SUBSETS):
        raise ValueError("Unexpected ETHICS subset set.")
    if frame["question_id"].nunique() != EXPECTED_QUESTIONS:
        raise ValueError("Unexpected number of ETHICS question identifiers.")
    if frame.duplicated(key).any():
        raise ValueError("Duplicate keys appeared after merging.")

    # The canonical parser contains 588 semantic candidates that have not
    # passed the required human audit.  They are demoted only in this analysis
    # view; the canonical CSV remains unchanged.
    semantic = frame["extraction_confidence"].eq("semantic_unaudited")
    analysis_status = frame["answer_status"].astype("string").copy()
    analysis_status.loc[semantic & analysis_status.eq("committed")] = "abstain"
    frame["candidate_uncommitted"] = semantic
    frame["analysis_answer_status"] = analysis_status
    frame["validated_committed"] = analysis_status.eq("committed")
    frame["analysis_is_correct"] = frame["is_correct"].where(
        frame["validated_committed"]
    )
    frame["benchmark"] = "ethics"

    unavailable = ~frame["text_available"]
    outcome_columns = [spec.value_col for spec in outcomes]
    non_na_unavailable = int(frame.loc[unavailable, outcome_columns].notna().sum().sum())
    if non_na_unavailable:
        raise ValueError(
            f"Found {non_na_unavailable} marker/surface values where text is unavailable."
        )
    if frame.loc[semantic, "analysis_answer_status"].eq("committed").any():
        raise ValueError("A semantic_unaudited row remains committed in the analysis view.")

    validation = {
        "canonical_sha256": canonical_hash,
        "rows": len(frame),
        "unique_response_keys": int(frame[key].drop_duplicates().shape[0]),
        "models": int(frame["model_id"].nunique()),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": int(frame["subset"].nunique()),
        "text_available": int(frame["text_available"].sum()),
        "text_missing": int(frame["text_status"].eq("missing").sum()),
        "text_technical_only": int(frame["text_status"].eq("technical_only").sum()),
        "candidate_uncommitted": int(frame["candidate_uncommitted"].sum()),
        "marker_status_matches_canonical": True,
        "ground_truth_used_for_marker_extraction": False,
        "hidden_reasoning_used": False,
    }
    return frame, registry, outcome_dictionary, marker_manifest, audit_manifest, validation


def analysis_response_level(frame: pd.DataFrame, outcomes: Sequence[OutcomeSpec]) -> pd.DataFrame:
    """Return the auditable, analysis-ready dataset without raw hidden reasoning."""

    columns = [
        "model_id",
        "persona",
        "identity",
        "condition",
        "condition_bucket",
        "question_id",
        "subset",
        "dataset",
        "ground_truth",
        "extracted_answer",
        "answer_status",
        "analysis_answer_status",
        "parse_source",
        "extraction_confidence",
        "candidate_answer",
        "candidate_source",
        "candidate_uncommitted",
        "validated_committed",
        "is_correct",
        "analysis_is_correct",
        "extracted_stance",
        "construct_valid",
        "numeric_semantic_conflict",
        "response_flags",
        "behavior_category",
        "scoring_abstain_reason",
        "finish_reason",
        "truncated",
        "text_status",
        "text_available",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        "moral_polarity_initial",
        "moral_polarity_final",
        "moral_polarity_shift",
        "moral_polarity_direction",
    ]
    marker_columns = []
    for spec in outcomes:
        if spec.raw_col:
            marker_id = spec.outcome_id
            marker_columns.extend(
                [
                    f"{marker_id}__count",
                    f"{marker_id}__density",
                    f"{marker_id}__present",
                ]
            )
    columns.extend(dict.fromkeys(marker_columns))
    return frame[columns].copy()


def build_scoring_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        committed = group["analysis_answer_status"].eq("committed")
        abstain = group["analysis_answer_status"].eq("abstain")
        missing = group["analysis_answer_status"].eq("missing")
        total = len(group)
        n_committed = int(committed.sum())
        construct_valid = subset in CONSTRUCT_VALID_SUBSETS
        n_correct = int((committed & group["is_correct"].fillna(False)).sum())
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "persona": persona,
                "identity": str(group["identity"].iloc[0]),
                "condition": str(group["condition"].iloc[0]),
                "subset": subset,
                "total": total,
                "committed": n_committed,
                "commit_rate": n_committed / total,
                "abstain": int(abstain.sum()),
                "abstain_rate": float(abstain.mean()),
                "missing": int(missing.sum()),
                "missing_rate": float(missing.mean()),
                "candidate_uncommitted": int(group["candidate_uncommitted"].sum()),
                "candidate_uncommitted_rate": float(group["candidate_uncommitted"].mean()),
                "numeric_semantic_conflict": int(
                    (committed & group["numeric_semantic_conflict"]).sum()
                ),
                "n_correct": n_correct if construct_valid else pd.NA,
                "accuracy_committed": (
                    n_correct / n_committed
                    if construct_valid and n_committed
                    else math.nan
                ),
                "pct_answer_wrong": (
                    float(group.loc[committed, "extracted_stance"].eq("wrong").mean())
                    if n_committed
                    else math.nan
                ),
                "pct_answer_acceptable": (
                    float(
                        group.loc[committed, "extracted_stance"].eq("acceptable").mean()
                    )
                    if n_committed
                    else math.nan
                ),
                "construct_valid": construct_valid,
            }
        )
    return pd.DataFrame(rows)


def build_semantic_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["candidate_uncommitted"]]
    return (
        selected.groupby(
            ["model_id", "persona", "subset", "parse_source"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["rows", "model_id", "persona"], ascending=[False, True, True])
    )


def build_conflict_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        committed = group["validated_committed"]
        conflict = committed & group["numeric_semantic_conflict"]
        retained = committed & ~group["numeric_semantic_conflict"]
        if subset in CONSTRUCT_VALID_SUBSETS:
            value_all = float(group.loc[committed, "is_correct"].astype(float).mean())
            value_excluding = float(group.loc[retained, "is_correct"].astype(float).mean())
            metric = "accuracy_committed"
        else:
            value_all = float(group.loc[committed, "extracted_stance"].eq("wrong").mean())
            value_excluding = float(group.loc[retained, "extracted_stance"].eq("wrong").mean())
            metric = "pct_answer_wrong"
        rows.append(
            {
                "model_id": model_id,
                "persona": persona,
                "subset": subset,
                "metric": metric,
                "committed": int(committed.sum()),
                "numeric_semantic_conflict": int(conflict.sum()),
                "retained_excluding_conflict": int(retained.sum()),
                "value_all": value_all,
                "value_excluding_conflict": value_excluding,
                "delta_excluding_minus_all_pp": 100 * (value_excluding - value_all),
            }
        )
    return pd.DataFrame(rows)


def _cramers_v(statistic: float, n: int, shape: tuple[int, int]) -> float:
    denominator = n * (min(shape) - 1)
    return math.sqrt(statistic / denominator) if denominator else math.nan


def _independence_rows(
    frame: pd.DataFrame,
    value_col: str,
    test_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_subsets", frame)]
    scopes.extend((subset, frame[frame["subset"].eq(subset)]) for subset in SUBSETS)
    for scope, scoped in scopes:
        for model_id, group in scoped.groupby("model_id", observed=True, sort=False):
            table = pd.crosstab(group["persona"], group[value_col], dropna=False)
            table = table.loc[:, table.sum(axis=0).gt(0)]
            statistic, p_raw, dof, expected = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            p_monte_carlo = math.nan
            if (expected < 5).any() and table.shape[0] > 1 and table.shape[1] > 1:
                monte_carlo = MonteCarloMethod(
                    n_resamples=10_000,
                    rng=np.random.default_rng(
                        _stable_seed(BUNDLE_VERSION, test_name, scope, model_id)
                    ),
                )
                monte_carlo_result = chi2_contingency(
                    table,
                    correction=False,
                    method=monte_carlo,
                )
                p_monte_carlo = float(monte_carlo_result.pvalue)
                p_inference = p_monte_carlo
                test_used = "pearson_chi_square_monte_carlo_fixed_margins"
                inference_resamples = 10_000
            else:
                p_inference = float(p_raw)
                test_used = "pearson_chi_square_asymptotic"
                inference_resamples = 0
            rows.append(
                {
                    "test_name": test_name,
                    "scope": scope,
                    "model_id": model_id,
                    "n": n,
                    "personas": table.shape[0],
                    "categories": table.shape[1],
                    "chi2": statistic,
                    "dof": dof,
                    "p_raw": p_raw,
                    "p_monte_carlo": p_monte_carlo,
                    "p_inference": p_inference,
                    "test_used": test_used,
                    "inference_resamples": inference_resamples,
                    "cramers_v": _cramers_v(statistic, n, table.shape),
                    "min_expected": float(np.min(expected)),
                    "expected_cells_below_5": int((expected < 5).sum()),
                    "asymptotic_assumptions_ok": bool((expected >= 5).all()),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(
        ["test_name", "model_id"], observed=True
    )["p_inference"].transform(lambda values: holm_adjust(values.to_numpy(float)))
    return result


def build_persona_independence(frame: pd.DataFrame) -> pd.DataFrame:
    status = _independence_rows(
        frame, "analysis_answer_status", "answer_status_by_persona"
    )
    availability_frame = frame.copy()
    availability_frame["linguistic_availability"] = np.select(
        [
            availability_frame["text_available"],
            availability_frame["text_status"].eq("technical_only"),
            availability_frame["text_status"].eq("missing"),
        ],
        ["available", "technical_only", "missing"],
        default="other_unavailable",
    )
    availability = _independence_rows(
        availability_frame,
        "linguistic_availability",
        "linguistic_availability_by_persona",
    )
    return pd.concat([status, availability], ignore_index=True)


def build_descriptives(
    frame: pd.DataFrame, outcomes: Sequence[OutcomeSpec]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, subset, persona), group in frame.groupby(
        ["model_id", "subset", "persona"], observed=True, sort=False
    ):
        available = group[group["text_available"]]
        for spec in outcomes:
            values = pd.to_numeric(available[spec.value_col], errors="coerce").dropna()
            if len(values):
                q1, median, q3 = np.quantile(values.to_numpy(float), [0.25, 0.5, 0.75])
            else:
                q1 = median = q3 = math.nan
            if spec.raw_col:
                raw = pd.to_numeric(available[spec.raw_col], errors="coerce").dropna()
                raw_sum = float(raw.sum()) if len(raw) else math.nan
                raw_median = float(raw.median()) if len(raw) else math.nan
                prevalence = float((raw > 0).mean()) if len(raw) else math.nan
            else:
                raw_sum = raw_median = prevalence = math.nan
            rows.append(
                {
                    "model_id": model_id,
                    "subset": subset,
                    "persona": persona,
                    "identity": str(group["identity"].iloc[0]),
                    "condition": str(group["condition"].iloc[0]),
                    "outcome_id": spec.outcome_id,
                    "outcome_tier": spec.outcome_tier,
                    "outcome_family": spec.outcome_family,
                    "outcome_scale": spec.outcome_scale,
                    "n_available": len(values),
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1 if len(values) else math.nan,
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "raw_count_sum": raw_sum,
                    "median_raw_count": raw_median,
                    "prevalence_present": prevalence,
                }
            )
    return pd.DataFrame(rows)


def _pair_frame(
    group: pd.DataFrame,
    left_persona: str,
    right_persona: str,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "question_id",
        "text_available",
        "validated_committed",
        "analysis_is_correct",
        "opening_category",
        *value_columns,
    ]
    columns = list(dict.fromkeys(columns))
    left = group.loc[group["persona"].eq(left_persona), columns]
    right = group.loc[group["persona"].eq(right_persona), columns]
    return left.merge(
        right,
        on="question_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )


def _population_mask(paired: pd.DataFrame, population: str) -> pd.Series:
    visible = bool_series(paired["text_available_left"]) & bool_series(
        paired["text_available_right"]
    )
    if population == "all_visible":
        return visible
    if population == "both_committed":
        return (
            visible
            & bool_series(paired["validated_committed_left"])
            & bool_series(paired["validated_committed_right"])
        )
    if population == "both_correct":
        left_correct = nullable_bool(paired["analysis_is_correct_left"]).fillna(False)
        right_correct = nullable_bool(paired["analysis_is_correct_right"]).fillna(False)
        return visible & left_correct.astype(bool) & right_correct.astype(bool)
    raise ValueError(f"Unknown population: {population}")


def _bootstrap_effect_ci(
    deltas: np.ndarray,
    statistic: str,
    resamples: int,
    seed_parts: Sequence[object],
) -> tuple[float, float]:
    finite = np.asarray(deltas, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    if np.allclose(finite, finite[0], rtol=0.0, atol=1e-12):
        value = float(finite[0])
        return value, value
    return _bootstrap_ci(
        finite,
        statistic,
        resamples,
        _stable_seed(BUNDLE_VERSION, *seed_parts),
    )


def _outcome_pair_row(
    paired: pd.DataFrame,
    mask: pd.Series,
    spec: OutcomeSpec,
    bootstrap_resamples: int,
    seed_parts: Sequence[object],
) -> dict[str, object]:
    left_values = pd.to_numeric(paired[f"{spec.value_col}_left"], errors="coerce")
    right_values = pd.to_numeric(paired[f"{spec.value_col}_right"], errors="coerce")
    keep = mask & left_values.notna() & right_values.notna()
    left = left_values[keep].to_numpy(float)
    right = right_values[keep].to_numpy(float)
    deltas = left - right

    if spec.raw_col:
        raw_left_series = pd.to_numeric(
            paired[f"{spec.raw_col}_left"], errors="coerce"
        )
        raw_right_series = pd.to_numeric(
            paired[f"{spec.raw_col}_right"], errors="coerce"
        )
        raw_keep = keep & raw_left_series.notna() & raw_right_series.notna()
        raw_left = raw_left_series[raw_keep].to_numpy(float)
        raw_right = raw_right_series[raw_keep].to_numpy(float)
        raw_fields = {
            "raw_count_sum_left": float(raw_left.sum()) if len(raw_left) else math.nan,
            "raw_count_sum_right": float(raw_right.sum()) if len(raw_right) else math.nan,
            "median_raw_count_left": float(np.median(raw_left)) if len(raw_left) else math.nan,
            "median_raw_count_right": float(np.median(raw_right)) if len(raw_right) else math.nan,
        }
    else:
        raw_fields = {
            "raw_count_sum_left": math.nan,
            "raw_count_sum_right": math.nan,
            "median_raw_count_left": math.nan,
            "median_raw_count_right": math.nan,
        }

    if spec.outcome_scale == "binary":
        test = _mcnemar_exact(left, right)
        ci_low, ci_high = _bootstrap_effect_ci(
            deltas, "mean", bootstrap_resamples, seed_parts
        )
        result = {
            "n_pairs": len(left),
            "n_nonzero": int(test["discordant"]),
            "center_left": float(left.mean()) if len(left) else math.nan,
            "center_right": float(right.mean()) if len(right) else math.nan,
            "effect": float(deltas.mean()) if len(deltas) else math.nan,
            "mean_paired_delta": float(deltas.mean()) if len(deltas) else math.nan,
            "median_nonzero_delta": (
                float(np.median(deltas[~np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)]))
                if np.any(~np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12))
                else math.nan
            ),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "effect_size_name": "matched_odds_ratio",
            "effect_size": test["matched_odds_ratio"],
            "test_used": test["test_used"],
            "p_raw": test["p_raw"],
            "left_only": test["left_only"],
            "right_only": test["right_only"],
            "discordant": test["discordant"],
            "zero_fraction": (
                1 - test["discordant"] / len(left) if len(left) else math.nan
            ),
            "wilcoxon_p": math.nan,
            "sign_test_p": math.nan,
            "bowley_skew_nonzero": math.nan,
            "n_positive": int((deltas > 0).sum()),
            "n_negative": int((deltas < 0).sum()),
        }
    else:
        test = _paired_numeric_test(deltas)
        nonzero_deltas = deltas[~np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)]
        ci_low, ci_high = _bootstrap_effect_ci(
            deltas, "median", bootstrap_resamples, seed_parts
        )
        result = {
            "n_pairs": test["n_pairs"],
            "n_nonzero": test["n_nonzero"],
            "center_left": float(np.median(left)) if len(left) else math.nan,
            "center_right": float(np.median(right)) if len(right) else math.nan,
            "effect": float(np.median(deltas)) if len(deltas) else math.nan,
            "mean_paired_delta": float(np.mean(deltas)) if len(deltas) else math.nan,
            "median_nonzero_delta": (
                float(np.median(nonzero_deltas)) if len(nonzero_deltas) else math.nan
            ),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "effect_size_name": "paired_rank_biserial",
            "effect_size": test["rank_biserial"],
            "test_used": test["test_used"],
            "p_raw": test["p_raw"],
            "left_only": pd.NA,
            "right_only": pd.NA,
            "discordant": pd.NA,
            "zero_fraction": test["zero_fraction"],
            "wilcoxon_p": test["wilcoxon_p"],
            "sign_test_p": test["sign_test_p"],
            "bowley_skew_nonzero": test["bowley_skew_nonzero"],
            "n_positive": test["n_positive"],
            "n_negative": test["n_negative"],
        }
    return {**result, **raw_fields}


def _apply_pairwise_corrections(
    table: pd.DataFrame,
    correction_group: Sequence[str],
    bh_group: Sequence[str],
) -> pd.DataFrame:
    result = table.copy()
    result["p_holm"] = result.groupby(
        list(correction_group), observed=True, dropna=False
    )["p_raw"].transform(lambda values: holm_adjust(values.to_numpy(float)))
    result["p_bh_outcomes"] = result.groupby(
        list(bh_group), observed=True, dropna=False
    )["p_raw"].transform(lambda values: benjamini_hochberg(values.to_numpy(float)))
    result["significant_holm_0_05"] = result["p_holm"].lt(0.05)
    return result


def build_within_identity_pairs(
    frame: pd.DataFrame,
    outcomes: Sequence[OutcomeSpec],
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_columns = [
        column
        for spec in outcomes
        for column in (spec.value_col, spec.raw_col)
        if column
    ]
    core_outcomes = [spec for spec in outcomes if spec.outcome_tier == "core_confirmatory"]
    for (model_id, subset), group in frame.groupby(
        ["model_id", "subset"], observed=True, sort=False
    ):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition, family in WITHIN_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                paired = _pair_frame(group, left_persona, right_persona, value_columns)
                analysis_jobs: list[tuple[str, Sequence[OutcomeSpec]]] = [
                    ("all_visible", outcomes)
                ]
                if family == "primary_implicit_explicit_3":
                    analysis_jobs.append(("both_committed", outcomes))
                    if subset in CONSTRUCT_VALID_SUBSETS:
                        analysis_jobs.append(("both_correct", core_outcomes))
                for population, selected_outcomes in analysis_jobs:
                    mask = _population_mask(paired, population)
                    for spec in selected_outcomes:
                        stats = _outcome_pair_row(
                            paired,
                            mask,
                            spec,
                            bootstrap_resamples,
                            (
                                "within",
                                population,
                                model_id,
                                subset,
                                identity,
                                contrast_id,
                                spec.outcome_id,
                            ),
                        )
                        rows.append(
                            {
                                "population": population,
                                "model_id": model_id,
                                "subset": subset,
                                "identity": identity,
                                "contrast_id": contrast_id,
                                "multiplicity_family": family,
                                "left_condition": left_condition,
                                "right_condition": right_condition,
                                "left_persona": left_persona,
                                "right_persona": right_persona,
                                "direction": f"{left_condition}_minus_{right_condition}",
                                "outcome_id": spec.outcome_id,
                                "outcome_tier": spec.outcome_tier,
                                "outcome_family": spec.outcome_family,
                                "outcome_scale": spec.outcome_scale,
                                "registered_measurement": spec.registered_measurement,
                                "bootstrap_resamples": bootstrap_resamples,
                                **stats,
                            }
                        )
    result = pd.DataFrame(rows)
    return _apply_pairwise_corrections(
        result,
        correction_group=(
            "population",
            "model_id",
            "subset",
            "identity",
            "outcome_id",
            "multiplicity_family",
        ),
        bh_group=(
            "population",
            "model_id",
            "subset",
            "identity",
            "contrast_id",
            "outcome_tier",
        ),
    )


def build_direct_identity_pairs(
    frame: pd.DataFrame,
    outcomes: Sequence[OutcomeSpec],
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_columns = [
        column
        for spec in outcomes
        for column in (spec.value_col, spec.raw_col)
        if column
    ]
    core_outcomes = [spec for spec in outcomes if spec.outcome_tier == "core_confirmatory"]
    for (model_id, subset), group in frame.groupby(
        ["model_id", "subset"], observed=True, sort=False
    ):
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                paired = _pair_frame(group, left_persona, right_persona, value_columns)
                jobs: list[tuple[str, Sequence[OutcomeSpec]]] = [
                    ("all_visible", outcomes),
                    ("both_committed", core_outcomes),
                ]
                if subset in CONSTRUCT_VALID_SUBSETS:
                    jobs.append(("both_correct", core_outcomes))
                for population, selected_outcomes in jobs:
                    mask = _population_mask(paired, population)
                    for spec in selected_outcomes:
                        stats = _outcome_pair_row(
                            paired,
                            mask,
                            spec,
                            bootstrap_resamples,
                            (
                                "direct",
                                population,
                                model_id,
                                subset,
                                identity_contrast,
                                condition,
                                spec.outcome_id,
                            ),
                        )
                        rows.append(
                            {
                                "population": population,
                                "model_id": model_id,
                                "subset": subset,
                                "identity_contrast": identity_contrast,
                                "left_identity": left_identity,
                                "right_identity": right_identity,
                                "condition": condition,
                                "contrast_id": f"{identity_contrast}_{condition}",
                                "multiplicity_family": "direct_identity_24",
                                "left_persona": left_persona,
                                "right_persona": right_persona,
                                "direction": f"{left_identity}_minus_{right_identity}",
                                "outcome_id": spec.outcome_id,
                                "outcome_tier": spec.outcome_tier,
                                "outcome_family": spec.outcome_family,
                                "outcome_scale": spec.outcome_scale,
                                "registered_measurement": spec.registered_measurement,
                                "bootstrap_resamples": bootstrap_resamples,
                                **stats,
                            }
                        )
    result = pd.DataFrame(rows)
    return _apply_pairwise_corrections(
        result,
        correction_group=(
            "population",
            "model_id",
            "subset",
            "outcome_id",
            "multiplicity_family",
        ),
        bh_group=(
            "population",
            "model_id",
            "subset",
            "identity_contrast",
            "condition",
            "outcome_tier",
        ),
    )


def _cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    n, k = matrix.shape
    if not n or k < 2:
        return math.nan, math.nan
    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = float(column_sums.sum())
    denominator = k * total - float(np.square(row_sums).sum())
    if denominator == 0:
        return 0.0, 1.0
    statistic = (k - 1) * (
        k * float(np.square(column_sums).sum()) - total**2
    ) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def build_omnibus(
    frame: pd.DataFrame, outcomes: Sequence[OutcomeSpec]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_conditions = ("neutral", *CONDITIONS)
    for (model_id, subset), group in frame.groupby(
        ["model_id", "subset"], observed=True, sort=False
    ):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            personas = [conditions[condition] for condition in ordered_conditions]
            selected = group[group["persona"].isin(personas)]
            for spec in outcomes:
                pivot = selected.pivot(
                    index="question_id", columns="persona", values=spec.value_col
                )
                for persona in personas:
                    if persona not in pivot:
                        pivot[persona] = np.nan
                matrix = pivot[personas].apply(pd.to_numeric, errors="coerce").dropna()
                values = matrix.to_numpy(float)
                if not len(values):
                    statistic = p_raw = effect = math.nan
                elif spec.outcome_scale == "binary":
                    statistic, p_raw = _cochran_q(values)
                    effect = statistic / (len(values) * (len(personas) - 1))
                elif np.allclose(values, values[0, 0], rtol=0.0, atol=1e-12):
                    statistic, p_raw, effect = 0.0, 1.0, 0.0
                else:
                    try:
                        statistic, p_raw = friedmanchisquare(
                            *(values[:, index] for index in range(values.shape[1]))
                        )
                        statistic, p_raw = float(statistic), float(p_raw)
                        effect = statistic / (len(values) * (len(personas) - 1))
                    except ValueError:
                        statistic = p_raw = effect = math.nan
                rows.append(
                    {
                        "population": "complete_five_condition_visible",
                        "model_id": model_id,
                        "subset": subset,
                        "identity": identity,
                        "outcome_id": spec.outcome_id,
                        "outcome_tier": spec.outcome_tier,
                        "outcome_family": spec.outcome_family,
                        "outcome_scale": spec.outcome_scale,
                        "n_complete_blocks": len(matrix),
                        "k_conditions": len(personas),
                        "test_used": (
                            "cochran_q" if spec.outcome_scale == "binary" else "friedman"
                        ),
                        "statistic": statistic,
                        "effect_size_name": (
                            "cochran_q_normalized"
                            if spec.outcome_scale == "binary"
                            else "kendalls_w"
                        ),
                        "effect_size": effect,
                        "p_raw": p_raw,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm_outcomes"] = result.groupby(
        ["model_id", "subset", "identity", "outcome_tier"],
        observed=True,
    )["p_raw"].transform(lambda values: holm_adjust(values.to_numpy(float)))
    result["significant_holm_0_05"] = result["p_holm_outcomes"].lt(0.05)
    return result


def build_opening_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"]].copy()
    counts = (
        available.groupby(
            ["model_id", "subset", "persona", "identity", "condition", "opening_category"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(
        ["model_id", "subset", "persona"], observed=True
    )["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts


def _opening_pair(
    paired: pd.DataFrame,
    population: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = _population_mask(paired, population)
    left = paired["opening_category_left"].astype("string")
    right = paired["opening_category_right"].astype("string")
    allowed = set(OPENING_CATEGORIES)
    keep = mask & left.isin(allowed) & right.isin(allowed)
    selected = pd.DataFrame(
        {
            "left": left[keep].astype(str),
            "right": right[keep].astype(str),
        }
    )
    return selected["left"].to_numpy(), selected["right"].to_numpy(), selected


def build_opening_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    for (model_id, subset), group in frame.groupby(
        ["model_id", "subset"], observed=True, sort=False
    ):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition, family in WITHIN_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                paired = _pair_frame(group, left_persona, right_persona, [])
                populations = ["all_visible"]
                if family == "primary_implicit_explicit_3":
                    populations.append("both_committed")
                for population in populations:
                    left, right, selected = _opening_pair(paired, population)
                    test = paired_stuart_maxwell(left, right, OPENING_CATEGORIES)
                    meta = {
                        "comparison_scope": "within_identity",
                        "population": population,
                        "model_id": model_id,
                        "subset": subset,
                        "identity": identity,
                        "identity_contrast": "",
                        "condition": "",
                        "contrast_id": contrast_id,
                        "multiplicity_family": family,
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                    }
                    rows.append(
                        {
                            **meta,
                            "n_pairs": len(left),
                            "test_used": "stuart_maxwell_marginal_homogeneity",
                            **test,
                        }
                    )
                    table = pd.crosstab(selected["left"], selected["right"])
                    for left_category, table_row in table.iterrows():
                        for right_category, count in table_row.items():
                            if count:
                                transitions.append(
                                    {
                                        **meta,
                                        "left_opening": left_category,
                                        "right_opening": right_category,
                                        "count": int(count),
                                        "proportion_of_pairs": count / len(left),
                                    }
                                )

        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                paired = _pair_frame(group, left_persona, right_persona, [])
                left, right, selected = _opening_pair(paired, "all_visible")
                test = paired_stuart_maxwell(left, right, OPENING_CATEGORIES)
                meta = {
                    "comparison_scope": "direct_identity",
                    "population": "all_visible",
                    "model_id": model_id,
                    "subset": subset,
                    "identity": "",
                    "identity_contrast": identity_contrast,
                    "condition": condition,
                    "contrast_id": f"{identity_contrast}_{condition}",
                    "multiplicity_family": "direct_identity_24",
                    "left_persona": left_persona,
                    "right_persona": right_persona,
                }
                rows.append(
                    {
                        **meta,
                        "n_pairs": len(left),
                        "test_used": "stuart_maxwell_marginal_homogeneity",
                        **test,
                    }
                )
                table = pd.crosstab(selected["left"], selected["right"])
                for left_category, table_row in table.iterrows():
                    for right_category, count in table_row.items():
                        if count:
                            transitions.append(
                                {
                                    **meta,
                                    "left_opening": left_category,
                                    "right_opening": right_category,
                                    "count": int(count),
                                    "proportion_of_pairs": count / len(left),
                                }
                            )

    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(
        [
            "comparison_scope",
            "population",
            "model_id",
            "subset",
            "identity",
            "multiplicity_family",
        ],
        observed=True,
        dropna=False,
    )["p_raw"].transform(lambda values: holm_adjust(values.to_numpy(float)))
    result["significant_holm_0_05"] = result["p_holm"].lt(0.05)
    return result, pd.DataFrame(transitions)


def build_pair_count_summary(
    within: pd.DataFrame,
    direct: pd.DataFrame,
    openings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    within_group = [
        "population",
        "model_id",
        "subset",
        "identity",
        "contrast_id",
    ]
    direct_group = [
        "population",
        "model_id",
        "subset",
        "identity_contrast",
        "condition",
    ]
    for name, table, group_columns in (
        ("within_marker_surface", within, within_group),
        ("direct_marker_surface", direct, direct_group),
    ):
        summary = (
            table.groupby(group_columns, observed=True)["n_pairs"]
            .agg(n_pairs_min="min", n_pairs_max="max", outcome_rows="size")
            .reset_index()
        )
        summary.insert(0, "analysis_table", name)
        rows.append(summary)
    opening_group = [
        "comparison_scope",
        "population",
        "model_id",
        "subset",
        "identity",
        "identity_contrast",
        "condition",
        "contrast_id",
    ]
    opening_summary = openings[opening_group + ["n_pairs"]].copy()
    opening_summary.insert(0, "analysis_table", "opening_composition")
    opening_summary = opening_summary.rename(
        columns={"n_pairs": "n_pairs_min"}
    )
    opening_summary["n_pairs_max"] = opening_summary["n_pairs_min"]
    opening_summary["outcome_rows"] = 1
    rows.append(opening_summary)
    return pd.concat(rows, ignore_index=True, sort=False)


def build_significance_summary(
    within: pd.DataFrame,
    direct: pd.DataFrame,
    omnibus: pd.DataFrame,
    openings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope, table, correction_col, groups in (
        (
            "within_identity",
            within,
            "p_holm",
            ["population", "model_id", "subset", "outcome_tier", "multiplicity_family"],
        ),
        (
            "direct_identity",
            direct,
            "p_holm",
            ["population", "model_id", "subset", "outcome_tier", "multiplicity_family"],
        ),
        (
            "omnibus_five_condition",
            omnibus,
            "p_holm_outcomes",
            ["population", "model_id", "subset", "outcome_tier"],
        ),
    ):
        work = table.copy()
        work["significant"] = work[correction_col].lt(0.05)
        summary = (
            work.groupby(groups, observed=True)["significant"]
            .agg(tests="size", significant="sum")
            .reset_index()
        )
        summary["significant_rate"] = summary["significant"] / summary["tests"]
        summary.insert(0, "analysis_scope", scope)
        rows.append(summary)
    opening_work = openings.copy()
    opening_work["significant"] = opening_work["p_holm"].lt(0.05)
    opening_summary = (
        opening_work.groupby(
            [
                "comparison_scope",
                "population",
                "model_id",
                "subset",
                "multiplicity_family",
            ],
            observed=True,
        )["significant"]
        .agg(tests="size", significant="sum")
        .reset_index()
    )
    opening_summary["significant_rate"] = (
        opening_summary["significant"] / opening_summary["tests"]
    )
    opening_summary.insert(0, "analysis_scope", "opening_composition")
    rows.append(opening_summary)
    return pd.concat(rows, ignore_index=True, sort=False)


def build_sensitivity_agreement(
    within: pd.DataFrame, direct: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, table, keys in (
        (
            "within_identity_primary",
            within[within["multiplicity_family"].eq("primary_implicit_explicit_3")],
            ["model_id", "subset", "identity", "contrast_id", "outcome_id"],
        ),
        (
            "direct_identity",
            direct,
            ["model_id", "subset", "identity_contrast", "condition", "outcome_id"],
        ),
    ):
        scoped_table = table.copy()
        scoped_table["directional_effect"] = np.where(
            scoped_table["outcome_scale"].eq("binary"),
            scoped_table["effect"],
            scoped_table["effect_size"],
        )
        primary = scoped_table[scoped_table["population"].eq("all_visible")][
            keys + ["directional_effect"]
        ]
        for sensitivity in ("both_committed", "both_correct"):
            comparison = scoped_table[scoped_table["population"].eq(sensitivity)][
                keys + ["directional_effect"]
            ]
            merged = primary.merge(
                comparison,
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_all_visible", "_sensitivity"),
            )
            for (model_id, subset), group in merged.groupby(
                ["model_id", "subset"], observed=True
            ):
                finite = group[
                    ["directional_effect_all_visible", "directional_effect_sensitivity"]
                ].dropna()
                if len(finite) >= 2:
                    spearman = float(finite.corr(method="spearman").iloc[0, 1])
                else:
                    spearman = math.nan
                nonzero = finite[
                    ~np.isclose(finite["directional_effect_all_visible"], 0.0)
                    | ~np.isclose(finite["directional_effect_sensitivity"], 0.0)
                ]
                sign_agreement = (
                    float(
                        (
                            np.sign(nonzero["directional_effect_all_visible"])
                            == np.sign(nonzero["directional_effect_sensitivity"])
                        ).mean()
                    )
                    if len(nonzero)
                    else math.nan
                )
                rows.append(
                    {
                        "analysis_scope": scope,
                        "effect_metric": "paired_rank_biserial_for_numeric;prevalence_delta_for_binary",
                        "sensitivity_population": sensitivity,
                        "model_id": model_id,
                        "subset": subset,
                        "n_effects_compared": len(finite),
                        "n_nonzero_union": len(nonzero),
                        "spearman_effect": spearman,
                        "sign_agreement": sign_agreement,
                        "median_absolute_effect_change": float(
                            (
                                finite["directional_effect_sensitivity"]
                                - finite["directional_effect_all_visible"]
                            ).abs().median()
                        )
                        if len(finite)
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def split_pair_tables(table: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Expose separate numeric and binary CSVs without losing a unified table."""

    return {
        "numeric": table[table["outcome_scale"].ne("binary")].copy(),
        "binary": table[table["outcome_scale"].eq("binary")].copy(),
    }


def expected_table_rows() -> dict[str, int]:
    """Closed-form row-count contract used by tests and the final validator."""

    all_outcomes = 53
    core_outcomes = 21
    within_all = 6 * 4 * 4 * 7 * all_outcomes
    within_committed = 6 * 4 * 4 * 3 * all_outcomes
    within_correct = 6 * 2 * 4 * 3 * core_outcomes
    direct_all = 6 * 4 * 24 * all_outcomes
    direct_committed = 6 * 4 * 24 * core_outcomes
    direct_correct = 6 * 2 * 24 * core_outcomes
    return {
        "scoring_summary": 6 * 17 * 4,
        "descriptives": 6 * 4 * 17 * all_outcomes,
        "within_pairs": within_all + within_committed + within_correct,
        "direct_pairs": direct_all + direct_committed + direct_correct,
        "omnibus": 6 * 4 * 4 * all_outcomes,
        "opening_pairs": (6 * 4 * 4 * 7) + (6 * 4 * 4 * 3) + (6 * 4 * 24),
    }


__all__ = [
    "BUNDLE_VERSION",
    "CONDITIONS",
    "CONDITION_LABELS",
    "CONSTRUCT_VALID_SUBSETS",
    "CORE_MARKERS",
    "DIRECT_IDENTITY_PAIRS",
    "EXPECTED_INPUT_SHA256",
    "EXPECTED_PIPELINE_VERSION",
    "EXPECTED_ROWS",
    "IDENTITIES",
    "IDENTITY_CONDITIONS",
    "IDENTITY_LABELS",
    "MODELS",
    "MODEL_LABELS",
    "OPENING_CATEGORIES",
    "OutcomeSpec",
    "PERSONAS",
    "PRIMARY_WITHIN_CONTRASTS",
    "SUBSETS",
    "SURFACE_OUTCOMES",
    "WITHIN_CONTRASTS",
    "analysis_response_level",
    "benjamini_hochberg",
    "build_conflict_sensitivity",
    "build_descriptives",
    "build_direct_identity_pairs",
    "build_omnibus",
    "build_opening_descriptives",
    "build_opening_pairs",
    "build_outcome_dictionary",
    "build_pair_count_summary",
    "build_persona_independence",
    "build_scoring_summary",
    "build_semantic_inventory",
    "build_sensitivity_agreement",
    "build_significance_summary",
    "build_within_identity_pairs",
    "expected_table_rows",
    "load_sources",
    "sha256_file",
    "split_pair_tables",
]

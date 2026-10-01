"""Paired statistics for GSM8K regional personas and direct identity contrasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .regional_config import (
    DIRECT_CONDITIONS,
    DIRECT_IDENTITY_PAIRS,
    IDENTITY_CONDITIONS,
    OPENING_CATEGORIES,
    PLANNED_CONTRASTS,
)
from .regional_registry import ALL_REGIONAL_MARKERS
from .statistics import (
    _analysis_groups,
    _binary_outcome_specs,
    _bootstrap_ci,
    _correct_mask,
    _mcnemar_exact,
    _outcome_specs,
    _paired_categories,
    _paired_numeric_test,
    _stable_seed,
    holm_adjust,
    paired_stuart_maxwell,
)


def build_religion_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = _outcome_specs(str(benchmark), ALL_REGIONAL_MARKERS)
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for condition, persona in conditions.items():
                selected = group[group["persona"].eq(persona)]
                for outcome in outcomes:
                    values = pd.to_numeric(
                        selected[outcome["value_col"]], errors="coerce"
                    ).dropna()
                    raw_values = (
                        pd.to_numeric(
                            selected[outcome["raw_col"]], errors="coerce"
                        ).dropna()
                        if outcome["raw_col"]
                        else pd.Series(dtype=float)
                    )
                    if len(values):
                        q1, median, q3 = np.quantile(
                            values.to_numpy(float), [0.25, 0.5, 0.75]
                        )
                        prevalence = float((values > 0).mean())
                    else:
                        q1 = median = q3 = prevalence = math.nan
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity": identity,
                            "condition": condition,
                            "persona": persona,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_family": outcome["family"],
                            "analysis_role": outcome["role"],
                            "n_available": len(values),
                            "median": median,
                            "q1": q1,
                            "q3": q3,
                            "iqr": q3 - q1 if len(values) else math.nan,
                            "prevalence_nonzero": prevalence,
                            "median_raw_count": (
                                float(raw_values.median())
                                if len(raw_values)
                                else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _pair_frame(
    group: pd.DataFrame,
    left_persona: str,
    right_persona: str,
    value_columns: list[str],
    *,
    require_both_correct: bool = False,
) -> pd.DataFrame:
    """Align a persona pair once so every outcome reuses the same merge."""

    columns = ["question_id", *value_columns]
    if require_both_correct:
        columns.append("is_correct")
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


def _paired_arrays(
    paired: pd.DataFrame,
    value_col: str,
    *,
    require_both_correct: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    left = pd.to_numeric(paired[f"{value_col}_left"], errors="coerce")
    right = pd.to_numeric(paired[f"{value_col}_right"], errors="coerce")
    keep = left.notna() & right.notna()
    if require_both_correct:
        keep &= _correct_mask(paired["is_correct_left"])
        keep &= _correct_mask(paired["is_correct_right"])
    return left[keep].to_numpy(float), right[keep].to_numpy(float)


def _build_within_identity_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int,
    *,
    require_both_correct: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = [
            outcome
            for outcome in _outcome_specs(str(benchmark), ALL_REGIONAL_MARKERS)
            if outcome["measurement"] != "binary"
        ]
        value_columns = [outcome["value_col"] for outcome in outcomes]
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                paired = _pair_frame(
                    group,
                    left_persona,
                    right_persona,
                    value_columns,
                    require_both_correct=require_both_correct,
                )
                for outcome in outcomes:
                    left, right = _paired_arrays(
                        paired,
                        outcome["value_col"],
                        require_both_correct=require_both_correct,
                    )
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "median",
                        bootstrap_resamples,
                        _stable_seed(
                            "religion_within",
                            model,
                            benchmark,
                            subset,
                            identity,
                            outcome["outcome_id"],
                            contrast_id,
                            "both_correct" if require_both_correct else "all",
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity": identity,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_family": outcome["family"],
                            "analysis_role": outcome["role"],
                            "contrast_id": contrast_id,
                            "left_condition": left_condition,
                            "right_condition": right_condition,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "median_left": float(np.median(left)) if len(left) else math.nan,
                            "median_right": float(np.median(right)) if len(right) else math.nan,
                            "median_paired_delta": (
                                float(np.median(deltas)) if len(deltas) else math.nan
                            ),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_paired_numeric_test(deltas),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "identity", "outcome_id"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    if require_both_correct:
        result.insert(7, "pair_filter", "both_conditions_correct")
    return result


def build_religion_paired_numeric(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    return _build_within_identity_numeric(
        frame, bootstrap_resamples, require_both_correct=False
    )


def build_religion_paired_numeric_correctness_sensitivity(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    return _build_within_identity_numeric(
        frame, bootstrap_resamples, require_both_correct=True
    )


def build_religion_paired_binary(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = _binary_outcome_specs(str(benchmark), ALL_REGIONAL_MARKERS)
        value_columns = [outcome["value_col"] for outcome in outcomes]
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                paired = _pair_frame(
                    group, left_persona, right_persona, value_columns
                )
                for outcome in outcomes:
                    left, right = _paired_arrays(paired, outcome["value_col"])
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(
                            "religion_within_presence",
                            model,
                            identity,
                            outcome["outcome_id"],
                            contrast_id,
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity": identity,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_family": outcome["family"],
                            "analysis_role": outcome["role"],
                            "registered_measurement": outcome["registered_measurement"],
                            "outcome_scale": "presence",
                            "contrast_id": contrast_id,
                            "left_condition": left_condition,
                            "right_condition": right_condition,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "prevalence_left": float(left.mean()) if len(left) else math.nan,
                            "prevalence_right": float(right.mean()) if len(right) else math.nan,
                            "prevalence_delta": float(deltas.mean()) if len(deltas) else math.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_mcnemar_exact(left, right),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "identity", "outcome_id"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def _build_direct_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int,
    *,
    require_both_correct: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = [
            outcome
            for outcome in _outcome_specs(str(benchmark), ALL_REGIONAL_MARKERS)
            if outcome["measurement"] != "binary"
        ]
        value_columns = [outcome["value_col"] for outcome in outcomes]
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                paired = _pair_frame(
                    group,
                    left_persona,
                    right_persona,
                    value_columns,
                    require_both_correct=require_both_correct,
                )
                for outcome in outcomes:
                    left, right = _paired_arrays(
                        paired,
                        outcome["value_col"],
                        require_both_correct=require_both_correct,
                    )
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "median",
                        bootstrap_resamples,
                        _stable_seed(
                            "religion_direct",
                            model,
                            identity_contrast,
                            condition,
                            outcome["outcome_id"],
                            "both_correct" if require_both_correct else "all",
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity_contrast": identity_contrast,
                            "condition": condition,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_family": outcome["family"],
                            "analysis_role": outcome["role"],
                            "contrast_id": f"{identity_contrast}_{condition}",
                            "left_identity": left_identity,
                            "right_identity": right_identity,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "median_left": float(np.median(left)) if len(left) else math.nan,
                            "median_right": float(np.median(right)) if len(right) else math.nan,
                            "median_paired_delta": (
                                float(np.median(deltas)) if len(deltas) else math.nan
                            ),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_paired_numeric_test(deltas),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # One planned family comprises all 6 identity pairs x 4 conditions for a
    # model/outcome. This avoids selecting a favourable identity or proxy post hoc.
    family = ["model_id", "benchmark", "subset", "outcome_id"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    if require_both_correct:
        result.insert(8, "pair_filter", "both_conditions_correct")
    return result


def build_religion_direct_numeric(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    return _build_direct_numeric(frame, bootstrap_resamples, require_both_correct=False)


def build_religion_direct_numeric_correctness_sensitivity(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    return _build_direct_numeric(frame, bootstrap_resamples, require_both_correct=True)


def build_religion_direct_binary(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = _binary_outcome_specs(str(benchmark), ALL_REGIONAL_MARKERS)
        value_columns = [outcome["value_col"] for outcome in outcomes]
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                paired = _pair_frame(
                    group, left_persona, right_persona, value_columns
                )
                for outcome in outcomes:
                    left, right = _paired_arrays(paired, outcome["value_col"])
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(
                            "religion_direct_presence",
                            model,
                            identity_contrast,
                            condition,
                            outcome["outcome_id"],
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity_contrast": identity_contrast,
                            "condition": condition,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_family": outcome["family"],
                            "analysis_role": outcome["role"],
                            "registered_measurement": outcome["registered_measurement"],
                            "outcome_scale": "presence",
                            "contrast_id": f"{identity_contrast}_{condition}",
                            "left_identity": left_identity,
                            "right_identity": right_identity,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "prevalence_left": float(left.mean()) if len(left) else math.nan,
                            "prevalence_right": float(right.mean()) if len(right) else math.nan,
                            "prevalence_delta": float(deltas.mean()) if len(deltas) else math.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_mcnemar_exact(left, right),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "outcome_id"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def build_religion_direct_availability(
    frame: pd.DataFrame, bootstrap_resamples: int = 10_000
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                columns = ["question_id", "text_available", "text_status"]
                left = group.loc[group["persona"].eq(left_persona), columns].copy()
                right = group.loc[group["persona"].eq(right_persona), columns].copy()
                left["record_present"] = 1
                right["record_present"] = 1
                paired = left.merge(
                    right,
                    on="question_id",
                    how="outer",
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                left_available = _correct_mask(paired["text_available_left"]).to_numpy(int)
                right_available = _correct_mask(paired["text_available_right"]).to_numpy(int)
                left_technical = paired["text_status_left"].fillna("source_gap").eq(
                    "technical_only"
                ).to_numpy(int)
                right_technical = paired["text_status_right"].fillna("source_gap").eq(
                    "technical_only"
                ).to_numpy(int)
                n_union = len(paired)
                n_both = int(((left_available == 1) & (right_available == 1)).sum())
                for outcome_id, left_values, right_values in (
                    ("text_available", left_available, right_available),
                    ("technical_only", left_technical, right_technical),
                ):
                    deltas = left_values - right_values
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(
                            "religion_direct_availability",
                            model,
                            identity_contrast,
                            condition,
                            outcome_id,
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity_contrast": identity_contrast,
                            "condition": condition,
                            "outcome_id": outcome_id,
                            "contrast_id": f"{identity_contrast}_{condition}",
                            "left_identity": left_identity,
                            "right_identity": right_identity,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "n_union_questions": n_union,
                            "n_left_records": int(paired["record_present_left"].fillna(0).sum()),
                            "n_right_records": int(paired["record_present_right"].fillna(0).sum()),
                            "n_both_available": n_both,
                            "rate_left": float(left_values.mean()) if n_union else math.nan,
                            "rate_right": float(right_values.mean()) if n_union else math.nan,
                            "rate_delta": float(deltas.mean()) if n_union else math.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **_mcnemar_exact(left_values, right_values),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "outcome_id"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def build_religion_opening_multinomial(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                left, right = _paired_categories(group, left_persona, right_persona)
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity": identity,
                        "contrast_id": contrast_id,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "n_pairs": len(left),
                        "test_used": "stuart_maxwell_marginal_homogeneity",
                        **paired_stuart_maxwell(left, right, OPENING_CATEGORIES),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "identity"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def build_religion_direct_opening_multinomial(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            identity_contrast = f"{left_identity}_vs_{right_identity}"
            for condition in DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS[left_identity][condition]
                right_persona = IDENTITY_CONDITIONS[right_identity][condition]
                left, right = _paired_categories(group, left_persona, right_persona)
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity_contrast": identity_contrast,
                        "condition": condition,
                        "contrast_id": f"{identity_contrast}_{condition}",
                        "left_identity": left_identity,
                        "right_identity": right_identity,
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "n_pairs": len(left),
                        "test_used": "stuart_maxwell_marginal_homogeneity",
                        **paired_stuart_maxwell(left, right, OPENING_CATEGORIES),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


# Domain-facing names keep call sites and generated provenance unambiguous.
build_regional_descriptives = build_religion_descriptives
build_regional_paired_numeric = build_religion_paired_numeric
build_regional_paired_numeric_correctness_sensitivity = (
    build_religion_paired_numeric_correctness_sensitivity
)
build_regional_paired_binary = build_religion_paired_binary
build_regional_direct_numeric = build_religion_direct_numeric
build_regional_direct_numeric_correctness_sensitivity = (
    build_religion_direct_numeric_correctness_sensitivity
)
build_regional_direct_binary = build_religion_direct_binary
build_regional_direct_availability = build_religion_direct_availability
build_regional_opening_multinomial = build_religion_opening_multinomial
build_regional_direct_opening_multinomial = (
    build_religion_direct_opening_multinomial
)

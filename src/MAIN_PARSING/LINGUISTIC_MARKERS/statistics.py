"""Paired descriptive and inferential analysis for gender conditions."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, friedmanchisquare, rankdata, wilcoxon

from .config import IDENTITY_CONDITIONS, OPENING_CATEGORIES, PLANNED_CONTRASTS
from .registry import MARKERS, MarkerSpec


BASE_NUMERIC_OUTCOMES = (
    ("word_count", "text_length", "primary"),
    ("avg_sentence_length", "text_length", "primary"),
    ("ttr_300", "lexical_diversity", "primary"),
)

GENDER_DIRECT_CONDITIONS = (
    "explicit",
    "implicit_1",
    "implicit_2",
    "implicit_3",
)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(~np.isnan(values))
    if not len(valid_positions):
        return result
    ordered_positions = valid_positions[np.argsort(values[valid_positions], kind="stable")]
    m = len(ordered_positions)
    running = 0.0
    for rank, position in enumerate(ordered_positions):
        adjusted = min(1.0, (m - rank) * values[position])
        running = max(running, adjusted)
        result[position] = running
    return result


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or resamples <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    n = len(values)
    if statistic == "mean":
        # A non-parametric resample is exactly a multinomial draw over the
        # empirical support. Marker-presence deltas have at most three unique
        # values, so this avoids materializing resamples x n arrays.
        support, counts = np.unique(values, return_counts=True)
        draws = rng.multinomial(n, counts / n, size=resamples)
        estimates = draws @ support / n
    elif statistic == "median":
        # Drawing n empirical observations is equivalent to drawing n U(0,1)
        # values and applying the empirical inverse CDF. We only need the one
        # or two central uniform order statistics, whose beta distributions
        # can be sampled in O(resamples) memory and time.
        ordered = np.sort(values)

        def empirical_index(uniform: np.ndarray) -> np.ndarray:
            upper = np.nextafter(1.0, 0.0)
            return np.floor(np.clip(uniform, 0.0, upper) * n).astype(int)

        if n % 2:
            rank = n // 2 + 1  # one-indexed median order statistic
            uniforms = rng.beta(rank, n + 1 - rank, size=resamples)
            estimates = ordered[empirical_index(uniforms)]
        else:
            lower_rank = n // 2
            lower = rng.beta(lower_rank, n + 1 - lower_rank, size=resamples)
            # Conditional spacing from U_(k) to U_(k+1).
            spacing = rng.beta(1, n - lower_rank, size=resamples)
            upper = lower + (1.0 - lower) * spacing
            estimates = (
                ordered[empirical_index(lower)] + ordered[empirical_index(upper)]
            ) / 2
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _bowley_skew(values: np.ndarray) -> float:
    if len(values) < 4:
        return math.nan
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    width = q3 - q1
    return float((q3 + q1 - 2 * median) / width) if width else 0.0


def paired_rank_biserial(deltas: np.ndarray) -> float:
    nonzero = deltas[deltas != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def _paired_numeric_test(deltas: np.ndarray) -> dict[str, object]:
    deltas = deltas[np.isfinite(deltas)]
    n_pairs = len(deltas)
    nonzero = deltas[deltas != 0]
    zero_fraction = float(1 - len(nonzero) / n_pairs) if n_pairs else math.nan
    skew = _bowley_skew(nonzero)
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())
    sign_p = (
        float(binomtest(positive, positive + negative, 0.5).pvalue)
        if positive + negative
        else 1.0
    )
    wilcoxon_p = math.nan
    if len(nonzero):
        try:
            wilcoxon_p = float(
                wilcoxon(
                    deltas,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = math.nan

    symmetry_failed = not math.isnan(skew) and abs(skew) > 0.5
    sparse_change = not math.isnan(zero_fraction) and zero_fraction > 0.5
    if not len(nonzero):
        test_used, p_raw = "no_change", 1.0
    elif sparse_change or symmetry_failed or math.isnan(wilcoxon_p):
        test_used, p_raw = "sign_test", sign_p
    else:
        test_used, p_raw = "wilcoxon_signed_rank", wilcoxon_p
    return {
        "n_pairs": n_pairs,
        "n_nonzero": len(nonzero),
        "n_positive": positive,
        "n_negative": negative,
        "zero_fraction": zero_fraction,
        "bowley_skew_nonzero": skew,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "test_used": test_used,
        "p_raw": p_raw,
        "rank_biserial": paired_rank_biserial(deltas),
    }


def _outcome_specs(
    benchmark: str, markers: Iterable[MarkerSpec] = MARKERS
) -> list[dict[str, str]]:
    outcomes = [
        {
            "outcome_id": name,
            "value_col": name,
            "family": family,
            "role": role,
            "raw_col": "",
            "measurement": "numeric",
        }
        for name, family, role in BASE_NUMERIC_OUTCOMES
    ]
    for marker in markers:
        if benchmark not in marker.benchmarks or marker.analysis_role == "quality_control":
            continue
        value_col = (
            f"{marker.marker_id}__present"
            if marker.measurement == "binary"
            else f"{marker.marker_id}__density"
        )
        outcomes.append(
            {
                "outcome_id": marker.marker_id,
                "value_col": value_col,
                "family": marker.family,
                "role": marker.analysis_role,
                "raw_col": f"{marker.marker_id}__count",
                "measurement": marker.measurement,
            }
        )
    return outcomes


def _binary_outcome_specs(
    benchmark: str, markers: Iterable[MarkerSpec] = MARKERS
) -> list[dict[str, str]]:
    """Presence outcomes for every non-QC marker applicable to a benchmark."""
    outcomes: list[dict[str, str]] = []
    for marker in markers:
        if benchmark not in marker.benchmarks or marker.analysis_role == "quality_control":
            continue
        outcomes.append(
            {
                "outcome_id": marker.marker_id,
                "value_col": f"{marker.marker_id}__present",
                "family": marker.family,
                "role": (
                    marker.analysis_role
                    if marker.measurement == "binary"
                    else "supplementary_presence"
                ),
                "registered_measurement": marker.measurement,
            }
        )
    return outcomes


def _analysis_groups(frame: pd.DataFrame):
    return frame.groupby(["model_id", "benchmark", "subset"], dropna=False, sort=True)


def build_quality_control(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        expected = group["question_id"].nunique()
        for persona, condition_group in group.groupby("persona", sort=True):
            statuses = condition_group["text_status"].fillna("")
            answer_status = condition_group.get(
                "answer_status", pd.Series(index=condition_group.index, dtype=object)
            ).fillna("")
            rows.append(
                {
                    "model_id": model,
                    "benchmark": benchmark,
                    "subset": subset,
                    "persona": persona,
                    "n_expected_union": expected,
                    "n_records": len(condition_group),
                    "n_response_success": int(condition_group["response_success"].fillna(False).sum()),
                    "n_text_available": int(condition_group["text_available"].fillna(False).sum()),
                    "n_missing_text": int((statuses == "missing").sum()),
                    "n_unseparable_think": int((statuses == "unseparable_think").sum()),
                    "n_technical_only": int((statuses == "technical_only").sum()),
                    "n_reasoning_present": int(condition_group["reasoning_present"].fillna(False).sum()),
                    "n_truncated": int(condition_group["response_truncated"].fillna(False).sum()),
                    "n_abstained": int((answer_status == "abstain").sum()),
                    "n_committed": int((answer_status == "committed").sum()),
                }
            )
    return pd.DataFrame(rows)


def build_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        outcomes = _outcome_specs(str(benchmark))
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for condition, persona in conditions.items():
                selected = group[group["persona"] == persona]
                for outcome in outcomes:
                    values = pd.to_numeric(selected[outcome["value_col"]], errors="coerce").dropna()
                    raw_values = (
                        pd.to_numeric(selected[outcome["raw_col"]], errors="coerce").dropna()
                        if outcome["raw_col"]
                        else pd.Series(dtype=float)
                    )
                    if len(values):
                        q1, median, q3 = np.quantile(values.to_numpy(float), [0.25, 0.5, 0.75])
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
                            "median_raw_count": float(raw_values.median()) if len(raw_values) else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _correct_mask(values: pd.Series) -> pd.Series:
    """Return an explicit correctness mask without treating missing as false evidence."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.casefold().eq("true")


def _paired_values(
    group: pd.DataFrame,
    left_persona: str,
    right_persona: str,
    value_col: str,
    *,
    require_both_correct: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["question_id", value_col]
    if require_both_correct:
        if "is_correct" not in group.columns:
            raise KeyError("Correctness sensitivity requires an is_correct column.")
        columns.append("is_correct")
    left = group.loc[group["persona"] == left_persona, columns].copy()
    right = group.loc[group["persona"] == right_persona, columns].copy()
    left[value_col] = pd.to_numeric(left[value_col], errors="coerce")
    right[value_col] = pd.to_numeric(right[value_col], errors="coerce")
    paired = left.merge(right, on="question_id", how="inner", suffixes=("_left", "_right"))
    paired = paired.dropna(subset=[f"{value_col}_left", f"{value_col}_right"])
    if require_both_correct:
        paired = paired[
            _correct_mask(paired["is_correct_left"])
            & _correct_mask(paired["is_correct_right"])
        ]
    return (
        paired[f"{value_col}_left"].to_numpy(float),
        paired[f"{value_col}_right"].to_numpy(float),
    )


def _build_paired_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
    *,
    require_both_correct: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for outcome in _outcome_specs(str(benchmark)):
                if outcome["measurement"] == "binary":
                    continue
                for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                    left_persona = conditions[left_condition]
                    right_persona = conditions[right_condition]
                    left, right = _paired_values(
                        group,
                        left_persona,
                        right_persona,
                        outcome["value_col"],
                        require_both_correct=require_both_correct,
                    )
                    deltas = left - right
                    test = _paired_numeric_test(deltas)
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "median",
                        bootstrap_resamples,
                        _stable_seed(model, benchmark, subset, identity, outcome["outcome_id"], contrast_id),
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
                            "median_paired_delta": float(np.median(deltas)) if len(deltas) else math.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **test,
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


def build_paired_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Primary paired numeric analysis over every linguistically available pair."""

    return _build_paired_numeric(frame, bootstrap_resamples)


def build_paired_numeric_correctness_sensitivity(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Sensitivity analysis restricted to pairs where both answers are correct.

    Filtering is applied *after* the left/right responses have been paired on
    question id.  This avoids comparing two condition-specific correctness
    subsets that contain different questions.
    """

    result = _build_paired_numeric(
        frame,
        bootstrap_resamples,
        require_both_correct=True,
    )
    if not result.empty:
        result.insert(7, "pair_filter", "both_conditions_correct")
    return result


def build_paired_binary(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Compare marker presence with exact paired tests."""
    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        binary_outcomes = _binary_outcome_specs(str(benchmark))
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for outcome in binary_outcomes:
                for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                    left_persona = conditions[left_condition]
                    right_persona = conditions[right_condition]
                    left, right = _paired_values(
                        group, left_persona, right_persona, outcome["value_col"]
                    )
                    test = _mcnemar_exact(left, right)
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(
                            model,
                            benchmark,
                            subset,
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
                            **test,
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


def _build_paired_gender_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
    *,
    require_both_correct: bool = False,
) -> pd.DataFrame:
    """Compare Man and Woman directly within each non-neutral condition."""

    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for outcome in _outcome_specs(str(benchmark)):
            if outcome["measurement"] == "binary":
                continue
            for condition in GENDER_DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS["man"][condition]
                right_persona = IDENTITY_CONDITIONS["woman"][condition]
                contrast_id = f"man_vs_woman_{condition}"
                left, right = _paired_values(
                    group,
                    left_persona,
                    right_persona,
                    outcome["value_col"],
                    require_both_correct=require_both_correct,
                )
                deltas = left - right
                test = _paired_numeric_test(deltas)
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    "median",
                    bootstrap_resamples,
                    _stable_seed(
                        model,
                        benchmark,
                        subset,
                        "man_vs_woman",
                        outcome["outcome_id"],
                        condition,
                        "both_correct" if require_both_correct else "all",
                    ),
                )
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity_contrast": "man_vs_woman",
                        "condition": condition,
                        "outcome_id": outcome["outcome_id"],
                        "outcome_family": outcome["family"],
                        "analysis_role": outcome["role"],
                        "contrast_id": contrast_id,
                        "left_identity": "man",
                        "right_identity": "woman",
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "median_left": float(np.median(left)) if len(left) else math.nan,
                        "median_right": float(np.median(right)) if len(right) else math.nan,
                        "median_paired_delta": (
                            float(np.median(deltas)) if len(deltas) else math.nan
                        ),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        **test,
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


def build_paired_gender_numeric(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Primary direct Man-minus-Woman paired numeric analysis."""

    return _build_paired_gender_numeric(frame, bootstrap_resamples)


def build_paired_gender_numeric_correctness_sensitivity(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Direct Man-minus-Woman analysis restricted to both-correct pairs."""

    result = _build_paired_gender_numeric(
        frame,
        bootstrap_resamples,
        require_both_correct=True,
    )
    if not result.empty:
        result.insert(8, "pair_filter", "both_conditions_correct")
    return result


def build_paired_gender_binary(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Compare Man and Woman marker presence with exact McNemar tests."""

    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for outcome in _binary_outcome_specs(str(benchmark)):
            for condition in GENDER_DIRECT_CONDITIONS:
                left_persona = IDENTITY_CONDITIONS["man"][condition]
                right_persona = IDENTITY_CONDITIONS["woman"][condition]
                contrast_id = f"man_vs_woman_{condition}"
                left, right = _paired_values(
                    group,
                    left_persona,
                    right_persona,
                    outcome["value_col"],
                )
                deltas = left - right
                test = _mcnemar_exact(left, right)
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    "mean",
                    bootstrap_resamples,
                    _stable_seed(
                        model,
                        benchmark,
                        subset,
                        "man_vs_woman",
                        outcome["outcome_id"],
                        condition,
                        "presence",
                    ),
                )
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity_contrast": "man_vs_woman",
                        "condition": condition,
                        "outcome_id": outcome["outcome_id"],
                        "outcome_family": outcome["family"],
                        "analysis_role": outcome["role"],
                        "registered_measurement": outcome["registered_measurement"],
                        "outcome_scale": "presence",
                        "contrast_id": contrast_id,
                        "left_identity": "man",
                        "right_identity": "woman",
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "prevalence_left": float(left.mean()) if len(left) else math.nan,
                        "prevalence_right": float(right.mean()) if len(right) else math.nan,
                        "prevalence_delta": float(deltas.mean()) if len(deltas) else math.nan,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        **test,
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


def build_paired_gender_availability(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Audit differential availability and technical-only status by gender.

    Pairing uses the union of question ids. A source-absent row is therefore
    represented as unavailable instead of silently disappearing in an inner join.
    """

    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for condition in GENDER_DIRECT_CONDITIONS:
            left_persona = IDENTITY_CONDITIONS["man"][condition]
            right_persona = IDENTITY_CONDITIONS["woman"][condition]
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
            left_technical = (
                paired["text_status_left"].fillna("source_gap").eq("technical_only")
            ).to_numpy(int)
            right_technical = (
                paired["text_status_right"].fillna("source_gap").eq("technical_only")
            ).to_numpy(int)
            n_union = len(paired)
            n_both_available = int(((left_available == 1) & (right_available == 1)).sum())
            for outcome_id, left_values, right_values in (
                ("text_available", left_available, right_available),
                ("technical_only", left_technical, right_technical),
            ):
                deltas = left_values - right_values
                test = _mcnemar_exact(left_values, right_values)
                ci_low, ci_high = _bootstrap_ci(
                    deltas,
                    "mean",
                    bootstrap_resamples,
                    _stable_seed(
                        model,
                        benchmark,
                        subset,
                        "man_vs_woman",
                        condition,
                        outcome_id,
                    ),
                )
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity_contrast": "man_vs_woman",
                        "condition": condition,
                        "outcome_id": outcome_id,
                        "contrast_id": f"man_vs_woman_{condition}",
                        "left_persona": left_persona,
                        "right_persona": right_persona,
                        "n_union_questions": n_union,
                        "n_left_records": int(paired["record_present_left"].fillna(0).sum()),
                        "n_right_records": int(paired["record_present_right"].fillna(0).sum()),
                        "n_both_available": n_both_available,
                        "rate_left": float(left_values.mean()) if n_union else math.nan,
                        "rate_right": float(right_values.mean()) if n_union else math.nan,
                        "rate_delta": float(deltas.mean()) if n_union else math.nan,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        **test,
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


def _mcnemar_exact(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = left.astype(int)
    right = right.astype(int)
    left_only = int(((left == 1) & (right == 0)).sum())
    right_only = int(((left == 0) & (right == 1)).sum())
    discordant = left_only + right_only
    p_raw = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    if right_only == 0:
        matched_odds_ratio = math.inf if left_only else math.nan
    else:
        matched_odds_ratio = left_only / right_only
    return {
        "n_pairs": len(left),
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "matched_odds_ratio": matched_odds_ratio,
        "p_raw": p_raw,
        "test_used": "mcnemar_exact",
    }


def build_paired_openings(
    frame: pd.DataFrame,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = frame.copy()
    for category in OPENING_CATEGORIES:
        work[f"opening__{category}"] = np.where(
            work["text_available"].fillna(False),
            (work["opening_category"] == category).astype(float),
            np.nan,
        )
    for (model, benchmark, subset), group in _analysis_groups(work):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for category in OPENING_CATEGORIES:
                value_col = f"opening__{category}"
                for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                    left, right = _paired_values(
                        group,
                        conditions[left_condition],
                        conditions[right_condition],
                        value_col,
                    )
                    test = _mcnemar_exact(left, right)
                    deltas = left - right
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean",
                        bootstrap_resamples,
                        _stable_seed(model, benchmark, subset, identity, category, contrast_id),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "benchmark": benchmark,
                            "subset": subset,
                            "identity": identity,
                            "opening_category": category,
                            "contrast_id": contrast_id,
                            "left_condition": left_condition,
                            "right_condition": right_condition,
                            "prevalence_left": float(left.mean()) if len(left) else math.nan,
                            "prevalence_right": float(right.mean()) if len(right) else math.nan,
                            "prevalence_delta": float(deltas.mean()) if len(deltas) else math.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **test,
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


def paired_stuart_maxwell(
    left: np.ndarray,
    right: np.ndarray,
    categories: Iterable[str] = OPENING_CATEGORIES,
) -> dict[str, object]:
    """Paired marginal-homogeneity test for a multi-category outcome.

    The Stuart-Maxwell covariance matrix is singular by construction, so the
    statistic uses its Moore-Penrose inverse and reports the estimable rank as
    the degrees of freedom.  Categories absent from both margins do not add a
    degree of freedom.
    """

    category_order = tuple(categories)
    category_index = {category: index for index, category in enumerate(category_order)}
    table = np.zeros((len(category_order), len(category_order)), dtype=float)
    for left_value, right_value in zip(left, right, strict=True):
        if left_value in category_index and right_value in category_index:
            table[category_index[left_value], category_index[right_value]] += 1

    active = (table.sum(axis=0) + table.sum(axis=1)) > 0
    table = table[np.ix_(active, active)]
    if table.size == 0:
        return {
            "statistic": math.nan,
            "df": 0,
            "p_raw": math.nan,
            "n_categories_observed": 0,
            "same_category_rate": math.nan,
            "marginal_total_variation": math.nan,
        }

    row_margins = table.sum(axis=1)
    column_margins = table.sum(axis=0)
    differences = row_margins - column_margins
    covariance = -(table + table.T)
    np.fill_diagonal(
        covariance,
        row_margins + column_margins - 2 * np.diag(table),
    )
    rank = int(np.linalg.matrix_rank(covariance))
    if rank:
        statistic = float(differences @ np.linalg.pinv(covariance) @ differences)
        statistic = max(statistic, 0.0)
        p_raw = float(chi2.sf(statistic, rank))
    else:
        statistic, p_raw = 0.0, 1.0

    n_pairs = float(table.sum())
    return {
        "statistic": statistic,
        "df": rank,
        "p_raw": p_raw,
        "n_categories_observed": int(table.shape[0]),
        "same_category_rate": float(np.trace(table) / n_pairs) if n_pairs else math.nan,
        "marginal_total_variation": (
            float(0.5 * np.abs(row_margins - column_margins).sum() / n_pairs)
            if n_pairs
            else math.nan
        ),
    }


def _paired_categories(
    group: pd.DataFrame,
    left_persona: str,
    right_persona: str,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["question_id", "opening_category", "text_available"]
    left = group.loc[group["persona"] == left_persona, columns].copy()
    right = group.loc[group["persona"] == right_persona, columns].copy()
    paired = left.merge(
        right,
        on="question_id",
        how="inner",
        suffixes=("_left", "_right"),
    )
    paired = paired[
        _correct_mask(paired["text_available_left"])
        & _correct_mask(paired["text_available_right"])
    ]
    paired = paired.dropna(
        subset=["opening_category_left", "opening_category_right"]
    )
    allowed = set(OPENING_CATEGORIES)
    paired = paired[
        paired["opening_category_left"].isin(allowed)
        & paired["opening_category_right"].isin(allowed)
    ]
    return (
        paired["opening_category_left"].astype(str).to_numpy(),
        paired["opening_category_right"].astype(str).to_numpy(),
    )


def build_paired_opening_multinomial(frame: pd.DataFrame) -> pd.DataFrame:
    """Test opening composition jointly instead of as independent binaries."""

    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                left, right = _paired_categories(group, left_persona, right_persona)
                test = paired_stuart_maxwell(left, right)
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
                        **test,
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


def build_paired_gender_opening_multinomial(frame: pd.DataFrame) -> pd.DataFrame:
    """Directly compare Man and Woman opening compositions per condition."""

    rows: list[dict[str, object]] = []
    for (model, benchmark, subset), group in _analysis_groups(frame):
        for condition in GENDER_DIRECT_CONDITIONS:
            left_persona = IDENTITY_CONDITIONS["man"][condition]
            right_persona = IDENTITY_CONDITIONS["woman"][condition]
            left, right = _paired_categories(group, left_persona, right_persona)
            test = paired_stuart_maxwell(left, right)
            rows.append(
                {
                    "model_id": model,
                    "benchmark": benchmark,
                    "subset": subset,
                    "identity_contrast": "man_vs_woman",
                    "condition": condition,
                    "contrast_id": f"man_vs_woman_{condition}",
                    "left_persona": left_persona,
                    "right_persona": right_persona,
                    "n_pairs": len(left),
                    "test_used": "stuart_maxwell_marginal_homogeneity",
                    **test,
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


def _complete_matrix(
    group: pd.DataFrame, conditions: dict[str, str], value_col: str
) -> pd.DataFrame:
    selected = group[group["persona"].isin(conditions.values())][
        ["question_id", "persona", value_col]
    ].copy()
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    pivot = selected.pivot(index="question_id", columns="persona", values=value_col)
    ordered = [conditions[name] for name in ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")]
    for persona in ordered:
        if persona not in pivot:
            pivot[persona] = np.nan
    return pivot[ordered].dropna()


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
    statistic = (k - 1) * (k * float(np.square(column_sums).sum()) - total**2) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def build_omnibus(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = frame.copy()
    for category in OPENING_CATEGORIES:
        work[f"opening__{category}"] = np.where(
            work["text_available"].fillna(False),
            (work["opening_category"] == category).astype(float),
            np.nan,
        )
    for (model, benchmark, subset), group in _analysis_groups(work):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for outcome in _outcome_specs(str(benchmark)):
                matrix = _complete_matrix(group, conditions, outcome["value_col"])
                if matrix.empty:
                    statistic = p_raw = math.nan
                elif outcome["measurement"] == "binary":
                    statistic, p_raw = _cochran_q(matrix.to_numpy(float))
                elif np.all(matrix.to_numpy() == matrix.to_numpy()[0, 0]):
                    statistic, p_raw = 0.0, 1.0
                else:
                    try:
                        statistic, p_raw = friedmanchisquare(
                            *(matrix[column].to_numpy(float) for column in matrix.columns)
                        )
                        statistic, p_raw = float(statistic), float(p_raw)
                    except ValueError:
                        statistic = p_raw = math.nan
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity": identity,
                        "outcome_id": outcome["outcome_id"],
                        "outcome_type": (
                            "binary_indicator"
                            if outcome["measurement"] == "binary"
                            else "numeric"
                        ),
                        "n_complete_blocks": len(matrix),
                        "test_used": (
                            "cochran_q"
                            if outcome["measurement"] == "binary"
                            else "friedman"
                        ),
                        "statistic": statistic,
                        "p_raw": p_raw,
                    }
                )
            for category in OPENING_CATEGORIES:
                matrix = _complete_matrix(group, conditions, f"opening__{category}")
                statistic, p_raw = _cochran_q(matrix.to_numpy(float))
                rows.append(
                    {
                        "model_id": model,
                        "benchmark": benchmark,
                        "subset": subset,
                        "identity": identity,
                        "outcome_id": f"opening:{category}",
                        "outcome_type": "categorical_indicator",
                        "n_complete_blocks": len(matrix),
                        "test_used": "cochran_q",
                        "statistic": statistic,
                        "p_raw": p_raw,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    family = ["model_id", "benchmark", "subset", "identity", "outcome_type"]
    result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result

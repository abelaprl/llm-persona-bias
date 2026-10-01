"""Compute pooled construct-valid ETHICS paired accuracy statistics.

The source publication bundles report Commonsense and Justice separately.  A
combined parent view therefore cannot be obtained by averaging p-values.  This
module concatenates the two item strata, preserves ``subset`` in the pairing
key, applies the source bundle's eligibility definition, and then recomputes
two-sided Exact McNemar, Cohen's h, and Holm adjustment for the combined
comparison families.

Ground truth and response text are not read.  The computation consumes only
the already validated ``answer_status`` and ``is_correct`` output columns.
Pair counts are used internally for the test but intentionally not returned.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


MEASURE_COLUMNS = {
    "subset",
    "subset_label",
    "row_label",
    "rate_a_pct",
    "rate_b_pct",
    "accuracy_a_pct",
    "accuracy_b_pct",
    "risk_difference_pp",
    "delta_accuracy_pp",
    "effect_size",
    "cohen_h",
    "cohens_h",
    "mcnemar_exact_p_value",
    "exact_mcnemar_p_value",
    "holm_adjusted_p_value",
    "significant_holm_0_05",
    "decision_significant_0_05",
    "n_pairs",
    "expected_questions",
    "pair_coverage_pct",
    "event_a",
    "event_b",
    "discordant_a_event_only",
    "discordant_b_event_only",
    "holm_family_size",
    "family_size",
    "omnibus_cochran_q_p_value",
}


def holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in the original order."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    clean = values[valid]
    if not len(clean):
        return result
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    restored = np.empty_like(adjusted_ranked)
    restored[order] = adjusted_ranked
    result[valid] = restored
    return result


def _correctness(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("").str.strip().str.lower()
    return normalized.map(
        {"true": True, "1": True, "false": False, "0": False, "": False}
    ).astype("boolean")


def compute_combined_conditional_accuracy(
    raw_path: Path,
    comparisons: pd.DataFrame,
    *,
    valid_subsets: Sequence[str],
    persona_a_column: str,
    persona_b_column: str,
    holm_group_column: str,
    combined_label: str = "Commonsense + Justice",
    eligibility: str = "both_committed",
) -> pd.DataFrame:
    """Compute paired conditional accuracy after pooling valid subset strata."""

    required_comparison_columns = {
        "model",
        "model_label",
        persona_a_column,
        persona_b_column,
        "comparison_short",
        holm_group_column,
    }
    missing = required_comparison_columns - set(comparisons.columns)
    if missing:
        raise KeyError(f"Missing combined-comparison columns: {sorted(missing)}")
    if comparisons.duplicated(["model", "comparison_short"]).any():
        raise ValueError("Combined comparison templates are not unique")
    if eligibility not in {"both_committed", "full_denominator"}:
        raise ValueError(f"Unknown eligibility mode: {eligibility}")

    raw = pd.read_csv(
        raw_path,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "subset",
            "answer_status",
            "is_correct",
        ],
        low_memory=False,
    )
    raw = raw[raw["subset"].isin(valid_subsets)].copy()
    key_columns = ["model", "persona_folder", "subset", "question_id"]
    if raw.duplicated(key_columns).any():
        raise ValueError("Duplicate response keys in combined-statistics input")
    raw["is_correct_binary"] = _correctness(raw["is_correct"])
    invalid_committed = raw["answer_status"].eq("committed") & raw["is_correct_binary"].isna()
    if invalid_committed.any():
        raise ValueError("Committed response without binary is_correct")

    indexed: dict[tuple[str, str], pd.DataFrame] = {}
    for (model, persona), frame in raw.groupby(["model", "persona_folder"], sort=False):
        indexed[(str(model), str(persona))] = frame[
            ["subset", "question_id", "answer_status", "is_correct_binary"]
        ].copy()

    rows: list[dict[str, object]] = []
    for template in comparisons.itertuples(index=False):
        template_values = template._asdict()
        model = str(template_values["model"])
        persona_a = str(template_values[persona_a_column])
        persona_b = str(template_values[persona_b_column])
        try:
            left = indexed[(model, persona_a)].rename(
                columns={
                    "answer_status": "answer_status_a",
                    "is_correct_binary": "is_correct_a",
                }
            )
            right = indexed[(model, persona_b)].rename(
                columns={
                    "answer_status": "answer_status_b",
                    "is_correct_binary": "is_correct_b",
                }
            )
        except KeyError as exc:
            raise ValueError(f"Missing model/persona response block: {exc}") from exc
        paired = left.merge(
            right,
            on=["subset", "question_id"],
            how="inner",
            validate="one_to_one",
        )
        if eligibility == "both_committed":
            eligible_rows = paired[
                paired["answer_status_a"].eq("committed")
                & paired["answer_status_b"].eq("committed")
            ]
            eligibility_label = "both responses committed"
        else:
            eligible_rows = paired
            eligibility_label = "full denominator; abstain/missing scored incorrect"
        if eligible_rows.empty:
            raise ValueError(
                f"No eligible committed pairs: {model}/{persona_a}/{persona_b}"
            )
        if eligible_rows[["is_correct_a", "is_correct_b"]].isna().any().any():
            raise ValueError(
                f"Missing is_correct in eligible pairs: {model}/{persona_a}/{persona_b}"
            )
        event_a = eligible_rows["is_correct_a"].astype(bool).to_numpy()
        event_b = eligible_rows["is_correct_b"].astype(bool).to_numpy()
        discordant_a = int((event_a & ~event_b).sum())
        discordant_b = int((~event_a & event_b).sum())
        discordant = discordant_a + discordant_b
        exact_p = (
            float(binomtest(discordant_a, discordant, p=0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )
        rate_a = float(event_a.mean())
        rate_b = float(event_b.mean())
        result = {
            key: value
            for key, value in template_values.items()
            if key not in MEASURE_COLUMNS
        }
        result.update(
            {
                "subset": "combined_construct_valid",
                "subset_label": combined_label,
                "row_label": str(template_values["model_label"]),
                "rate_a_pct": 100 * rate_a,
                "rate_b_pct": 100 * rate_b,
                "risk_difference_pp": 100 * (rate_a - rate_b),
                "effect_size": (
                    2 * math.asin(math.sqrt(rate_a))
                    - 2 * math.asin(math.sqrt(rate_b))
                ),
                "exact_mcnemar_p_value": exact_p,
                "statistical_method": "two-sided Exact McNemar",
                "multiple_comparison_method": "Holm",
                "combined_pairing_key": "subset + question_id",
                "combined_eligibility": eligibility_label,
            }
        )
        rows.append(result)

    combined = pd.DataFrame(rows)
    combined["holm_adjusted_p_value"] = np.nan
    for _, indices in combined.groupby(["model", holm_group_column], sort=False).groups.items():
        combined.loc[indices, "holm_adjusted_p_value"] = holm_adjust(
            combined.loc[indices, "exact_mcnemar_p_value"]
        )
    combined["significant_holm_0_05"] = combined["holm_adjusted_p_value"].lt(0.05)
    combined["decision_significant_0_05"] = combined["significant_holm_0_05"]
    if combined["exact_mcnemar_p_value"].isna().any():
        raise ValueError("Missing Exact McNemar p-value in combined output")
    if combined["holm_adjusted_p_value"].isna().any():
        raise ValueError("Missing Holm-adjusted p-value in combined output")
    if "n_pairs" in combined.columns:
        raise AssertionError("Pair count must not be exposed in combined output")
    return combined

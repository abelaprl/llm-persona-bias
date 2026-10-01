"""Read-only descriptive views of frozen gender marker evidence and contrasts."""
import json

import numpy as np
import pandas as pd

CONDITIONS = ["E", "I1", "I2", "I3"]
SCOPES = ["overall", "commonsense", "deontology", "justice", "virtue"]
INDICATORS = ["any_gender", "explicit_user", "quoted_or_hypothetical", "unknown", "scenario_overlap"]
PERSONAS = ["neutral", "man_explicit_1", "woman_explicit_1", "man_implicit_1", "woman_implicit_1",
            "man_implicit_2", "woman_implicit_2", "man_implicit_3", "woman_implicit_3"]


def reference_summary(responses, spans, gender_ids):
    """Count each response at most once per indicator; indicators may overlap."""
    if responses.response_id.duplicated().any():
        raise ValueError("Duplicate response ID")
    selected = spans.loc[spans.marker_id.isin(gender_ids)].copy()
    selected = selected.merge(responses[["response_id", "text_available"]], on="response_id",
                              how="left", validate="many_to_one")
    if not selected.text_available.eq(True).all():
        raise ValueError("Gender span belongs to unknown or unavailable response")
    if not selected.reference_type.isin(INDICATORS[1:4]).all():
        raise ValueError("Unrecognized reference type; do not silently fold into unknown")
    reference_ids = {"any_gender": set(selected.response_id)}
    for indicator in INDICATORS[1:4]:
        reference_ids[indicator] = set(selected.loc[selected.reference_type.eq(indicator), "response_id"])
    overlap = selected["flags"].fillna("[]").map(
        lambda value: "same_gender_term_in_scenario_not_coreference" in json.loads(value))
    reference_ids["scenario_overlap"] = set(selected.loc[overlap, "response_id"])
    rows = []
    for (model, persona), group in responses.groupby(["model_id", "persona"], sort=False):
        ids = set(group.loc[group.text_available.eq(True), "response_id"])
        for indicator in INDICATORS:
            n = len(ids & reference_ids[indicator])
            rows.append(dict(model_id=model, persona=persona, indicator=indicator,
                             n_responses=n, n_available=len(ids), total=len(group),
                             percent=100 * n / len(ids) if ids else np.nan,
                             unit="response", denominator="available explanations",
                             reference_status="automatic_candidate_not_human_annotation"))
    return pd.DataFrame(rows)


def direct_rows(comparisons, registry, primary_only=True):
    """Select same-condition Woman-minus-Man without modifying any statistics."""
    selected = comparisons.loc[comparisons.left.str.startswith("woman_") &
                               comparisons.right.str.startswith("man_")].copy()
    left = selected.left.str.split("_", n=1).str[1]
    right = selected.right.str.split("_", n=1).str[1]
    if not left.eq(right).all():
        raise ValueError("Direct gender comparison must use the same condition")
    selected["condition"] = left.map({"explicit_1": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"})
    if selected.condition.isna().any():
        raise ValueError("Unknown condition")
    primary = registry.set_index("marker_id").measurement.replace({"binary": "presence"})
    if primary_only:
        selected = selected.loc[selected.measurement.eq(selected.marker_id.map(primary))].copy()
    keys = ["model_id", "scope", "marker_id", "measurement", "condition"]
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate comparison")
    return selected


def support_summary(primary_direct):
    """Aggregate marker support, without summing shared question-pair counts."""
    rows = []
    for (model, condition), group in primary_direct.loc[primary_direct.scope.eq("overall")].groupby(["model_id", "condition"]):
        for column in ("n_pairs", "n_left_total", "n_right_total"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError("Marker-specific denominator mismatch")
        if group.marker_id.duplicated().any():
            raise ValueError("Support summary requires one measurement per marker")
        if group.n_left_total.iloc[0] != group.n_right_total.iloc[0]:
            raise ValueError("Unequal total-pair denominator")
        n = int(group.n_pairs.iloc[0])
        total = int(group.n_left_total.iloc[0])
        low = int(group.n_changed.lt(10).sum())
        rows.append(dict(model_id=model, condition=condition, n_pairs=n, total_pairs=total,
                         pair_percent=100 * n / total if total else np.nan,
                         marker_count=len(group), low_support_markers=low,
                         low_support_percent=100 * low / len(group)))
    return pd.DataFrame(rows)

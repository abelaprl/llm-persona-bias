"""Display-only contracts; all scientific source columns remain unchanged."""
from functools import lru_cache

import pandas as pd

from .gender_lm_reference_data import MODELS, PERSONAS, SCENARIOS, SOURCE, persona as persona_id, scenario as scenario_id

MARKER_GROUPS = {
    "B01": ["organization_explicit_step", "organization_bullet",
            "organization_section_header", "elaboration_code_gloss"],
    "B02": ["stance_hedge", "stance_booster", "stance_assumption",
            "stance_self_correction", "stance_contextual_hedge", "stance_distancing"],
    "B03": ["connective_causal", "connective_contrastive", "connective_conditional",
            "connective_additive", "connective_concessive"],
    "B04": ["self_reference_first_person", "self_reference_inclusive_we",
            "engagement_reader_directive", "affect_empathy_validation"],
    "B05": ["ai_identity_statement", "response_stance_committed",
            "response_stance_conditional", "response_stance_refusal"],
    "B06": ["framing_deontological", "framing_consequentialist", "framing_virtue_ethics",
            "framing_religious", "framing_cultural"],
    "B07": ["moral_evaluation_positive", "moral_evaluation_negative"],
    "B08": ["gender_persona_direct_address", "gender_persona_acknowledgment",
            "gender_second_person_identity_reference", "gender_hypothetical_identity_reference",
            "gender_first_person_identity_reference", "gender_perspective_adoption",
            "gender_identity_relevance_denial"],
    "B09": ["gender_identity_neutralization", "gender_identity_discourse",
            "gender_role_framing", "gendered_normative_evaluation"],
}
GROUP_MEASUREMENTS = {
    group_id: ("presence" if group_id in {"B05", "B08"} else "density")
    for group_id in MARKER_GROUPS
}
PROTOTYPE_MARKERS = MARKER_GROUPS["B02"]


def effect_view(rows: pd.DataFrame, comparison: str) -> pd.DataFrame:
    """Add display orientation to contrasts selected by the read-only adapter."""
    contracts = {
        "density": ("mean_paired_difference", "occurrences_per_100_words"),
        "presence": ("prevalence_difference_pp", "percentage_points"),
    }
    required = {"measurement", "measurement_unit", "estimand", "ci_estimand",
                "effect", "ci_low", "ci_high"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(rows.columns))}")
    if comparison not in {"direct", "neutral", "implicit"}:
        raise ValueError("Unknown comparison")
    if rows.empty or rows.measurement.isna().any() or rows.measurement.nunique() != 1:
        raise ValueError("Expected one nonempty measurement")
    measurement = rows.measurement.iloc[0]
    if measurement not in contracts:
        raise ValueError("Unsupported measurement")
    estimand, unit = contracts[measurement]
    if not (rows.estimand.eq(estimand).all()
            and rows.ci_estimand.eq(estimand).all()
            and rows.measurement_unit.eq(unit).all()):
        raise ValueError("Incompatible effect or CI")
    if (rows.ci_low > rows.ci_high).any():
        raise ValueError("Reversed source interval")
    out = rows.copy(deep=True)
    sign = -1 if comparison == "direct" else 1
    out["display_sign"] = sign
    out["display_effect"] = sign * out.effect
    out["display_ci_low"] = -out.ci_high if sign == -1 else out.ci_low
    out["display_ci_high"] = -out.ci_low if sign == -1 else out.ci_high
    out["display_direction"] = {
        "direct": "Man - Woman", "neutral": "persona - N", "implicit": "I - E"
    }[comparison]
    out["display_unit"] = unit
    return out


GROUP_LABELS = {
    "B01": "Organisasi dan elaborasi", "B02": "Sikap epistemik",
    "B03": "Hubungan antargagasan", "B04": "Rujukan diri dan keterlibatan",
    "B05": "Identitas model dan sikap respons", "B06": "Framing alasan etika",
    "B07": "Evaluasi moral", "B08": "Respons terhadap identitas gender",
    "B09": "Wacana dan penilaian terkait gender",
}
SCOPES = {"overall", "commonsense", "deontology", "justice", "virtue"}


@lru_cache(maxsize=2)
def source_table(name):
    """Cache read-only source tables; callers always select copies."""
    if name not in {"comparisons.csv", "marker_summary.csv"}:
        raise ValueError("Unsupported source table")
    return pd.read_csv(SOURCE / name, float_precision="round_trip")


def view_specs():
    specs = []
    for group_id in MARKER_GROUPS:
        specs.append(dict(group_id=group_id, mode="direct", view_id=f"{group_id}_direct_overall"))
        for scenario in SCENARIOS:
            specs.append(dict(group_id=group_id, mode="profile", scenario=scenario,
                              view_id=f"{group_id}_profile_{scenario.lower()}_overall"))
        for persona in ["Man", "Woman"]:
            for mode in ["neutral", "implicit"]:
                specs.append(dict(group_id=group_id, mode=mode, persona=persona,
                                  view_id=f"{group_id}_{mode}_{persona.lower()}_overall"))
    return specs


def view_data(group_id, mode, persona=None, scenario=None, scope="overall", *, comparisons=None, summary=None):
    """Select a complete view, preserving original source values and row indices."""
    if group_id not in MARKER_GROUPS or mode not in {"profile", "direct", "neutral", "implicit"} or scope not in SCOPES:
        raise ValueError("Unknown group, mode or scope")
    if mode == "profile":
        if scenario not in SCENARIOS or persona is not None:
            raise ValueError("Profile requires a scenario and no persona filter")
    elif scenario is not None or (mode == "direct" and persona is not None) or (mode != "direct" and persona not in {"Man", "Woman"}):
        raise ValueError("Unexpected persona/scenario filter")
    markers, measurement = MARKER_GROUPS[group_id], GROUP_MEASUREMENTS[group_id]
    if mode == "profile":
        summary = source_table("marker_summary.csv") if summary is None else summary
        selected = summary.loc[summary.scope.eq(scope) & summary.marker_id.isin(markers)].copy()
        rows = selected.loc[selected.measurement.eq(measurement)].copy()
        wanted = {"neutral": "N", persona_id("Man", scenario): "Man", persona_id("Woman", scenario): "Woman"}
        rows = rows.loc[rows.persona.isin(wanted)].copy()
        keys = ["scope", "model_id", "persona", "marker_id"]
        for metric, column, name in [("presence", "value", "prevalence_percent"),
                                     ("density", "value", "density_per_100_words"),
                                     ("count", "raw_occurrences", "raw_count_total")]:
            lookup = selected.loc[selected.measurement.eq(metric), keys + [column]].set_index(keys)
            if not lookup.index.is_unique:
                raise ValueError("Duplicate profile source rows")
            rows = rows.join(lookup.rename(columns={column: name}), on=keys, validate="one_to_one")
        rows["panel_id"] = rows.persona.map(wanted)
        rows["display_value"] = rows.value
        rows["display_unit"] = "percent" if measurement == "presence" else "occurrences_per_100_words"
        panels = ["N", "Man", "Woman"]
    else:
        comparisons = source_table("comparisons.csv") if comparisons is None else comparisons
        rows = comparisons.loc[comparisons.scope.eq(scope) & comparisons.marker_id.isin(markers)
                               & comparisons.measurement.eq(measurement)].copy()
        if not rows.effect_direction.eq("left_minus_right").all():
            raise ValueError("Unexpected source orientation")
        left, right = rows.left.map(scenario_id), rows.right.map(scenario_id)
        if mode == "direct":
            mask = rows.left.str.startswith("woman_") & rows.right.str.startswith("man_") & left.eq(right)
        elif mode == "neutral":
            mask = rows.left.str.startswith(persona.lower() + "_") & rows.right.eq("neutral")
        else:
            mask = (rows.left.str.startswith(persona.lower() + "_") & rows.right.str.startswith(persona.lower() + "_")
                    & left.isin(["I1", "I2", "I3"]) & right.eq("E"))
        rows = effect_view(rows.loc[mask].copy(), mode)
        rows["panel_id"] = rows.left.map(scenario_id)
        panels = SCENARIOS if mode != "implicit" else SCENARIOS[1:]
        if not (rows.ci_level.eq(.95).all() and rows.ci_simultaneous.eq(False).all()):
            raise ValueError("Expected marginal 95% source CIs")
    keys = ["model_id", "marker_id", "panel_id"]
    expected = {(m, marker, panel) for m in MODELS for marker in markers for panel in panels}
    if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing, duplicate or unexpected view rows")
    return rows


def coverage_table(summary, comparisons, scope="overall"):
    """Retain raw coverage denominators; no inferential recomputation."""
    if scope not in SCOPES:
        raise ValueError("Unknown scope")
    columns = ["model_id", "persona", "total", "n_available"]
    explanation = summary.loc[summary.scope.eq(scope), columns].drop_duplicates()
    if len(explanation) != 54 or explanation.duplicated(["model_id", "persona"]).any():
        raise ValueError("Inconsistent explanation coverage across markers")
    direct = view_data("B01", "direct", scope=scope, comparisons=comparisons)
    pairs = direct[["model_id", "panel_id", "n_pairs", "n_left_total", "n_right_total"]].drop_duplicates()
    if len(pairs) != 24 or pairs.duplicated(["model_id", "panel_id"]).any() or not pairs.n_left_total.eq(pairs.n_right_total).all():
        raise ValueError("Inconsistent pair coverage across markers")
    explanation = explanation.rename(columns={"persona": "setting", "n_available": "available"}).assign(kind="explanation")
    pairs = pairs.rename(columns={"panel_id": "setting", "n_pairs": "available", "n_left_total": "total"}).assign(kind="paired")
    result = pd.concat([explanation, pairs.drop(columns="n_right_total")], ignore_index=True)
    result["scope"] = scope
    result["percent"] = result.available / result.total * 100
    return result

"""Pelaporan kandidat Regional/Cultural ETHICS v3.

Modul ini hanya membaca nilai marker dan hasil parser program yang sudah ada.
Ia tidak mengekstrak marker, tidak menilai jawaban, dan tidak memakai ground
truth untuk memilih hasil ekstraksi. Statistik v2 untuk outcome yang byte-wise
setara dipakai ulang; hanya outcome yang berubah dihitung ulang.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
from pathlib import Path
import shutil
from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency


CHANGED_MARKERS = ("connective_conditional",)
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 18_222_008
CONSTRUCT_VALID_SUBSETS = {"commonsense", "justice"}
LEGACY_ANALYSIS = Path(
    "result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers/analysis/"
    "regional_lm_v2_0_1_complete_20260828"
)
CANONICAL_SCORING = Path(
    "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv"
)
AI_DIAGNOSTIC = Path(
    "result/benchmarks/ethics/reports/regional_lm_v3_ai_diagnostic_20260916_r1/diagnostic_labels.json"
)


def _bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.casefold().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)


def _nullable_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.map({
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    }).astype("boolean")


def _mixed_script_signal(text: str) -> bool:
    """Flag mixed ASCII/non-ASCII letter scripts; this is not language ID."""
    letters = [char for char in str(text) if unicodedata.category(char).startswith("L")]
    return any(ord(char) < 128 for char in letters) and any(ord(char) >= 128 for char in letters)


def _scenario_hit_overlaps(scenario: str, matched_text: str) -> bool:
    needle = " ".join(str(matched_text).casefold().split())
    haystack = " ".join(str(scenario).casefold().split())
    return bool(needle) and needle in haystack


def _analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the scoring view while preserving source answer_status."""
    result = frame.copy()
    required = {"model_id", "persona", "question_id", "subset", "answer_status"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Missing response columns: {sorted(missing)}")
    expected_key = (
        result["model_id"].astype(str) + "|" + result["persona"].astype(str)
        + "|" + result["question_id"].astype(str)
    )
    if "response_key" in result and not result["response_key"].astype(str).equals(expected_key):
        raise ValueError("response_key is not model_id|persona|question_id")
    if result.duplicated(["model_id", "persona", "question_id"]).any():
        raise ValueError("Duplicate response key")
    source = result["answer_status"].astype("string")
    if "analysis_answer_status" in result:
        analysis = result["analysis_answer_status"].astype("string")
    else:
        analysis = source.copy()
        semantic = result.get(
            "extraction_confidence", pd.Series("", index=result.index)
        ).astype("string").eq("semantic_unaudited")
        analysis.loc[semantic & analysis.eq("committed")] = "abstain"
        result["analysis_answer_status"] = analysis
    semantic = result.get(
        "extraction_confidence", pd.Series("", index=result.index)
    ).astype("string").eq("semantic_unaudited")
    if (semantic & analysis.eq("committed")).any():
        raise ValueError("semantic_unaudited remains committed in analysis view")
    result["source_answer_status"] = source
    result["validated_committed"] = analysis.eq("committed")
    result["candidate_uncommitted"] = semantic
    correctness = _nullable_bool(result["is_correct"])
    result["analysis_is_correct"] = correctness.where(result["validated_committed"])
    if "numeric_semantic_conflict" in result:
        result["numeric_semantic_conflict"] = _bool(result["numeric_semantic_conflict"])
    else:
        result["numeric_semantic_conflict"] = False
    if "text_available" in result:
        result["text_available"] = _bool(result["text_available"])
    return result


def build_scoring_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mandatory denominator and scoring fields per model/persona/subset."""
    data = _analysis_frame(frame)
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(["model_id", "persona", "subset"], sort=False):
        model_id, persona, subset = keys
        status = group["analysis_answer_status"]
        committed = status.eq("committed")
        abstain = status.eq("abstain")
        missing = status.eq("missing")
        total = len(group)
        n_committed = int(committed.sum())
        valid = str(subset) in CONSTRUCT_VALID_SUBSETS
        correct = pd.to_numeric(group.get("is_correct"), errors="coerce")
        extracted = pd.to_numeric(group.get("extracted_answer"), errors="coerce")
        stance = group.get("extracted_stance")
        row = {
            "model_id": model_id,
            "persona": persona,
            "subset": subset,
            "total": total,
            "committed": n_committed,
            "commit_rate": n_committed / total if total else math.nan,
            "abstain": int(abstain.sum()),
            "abstain_rate": float(abstain.mean()) if total else math.nan,
            "missing": int(missing.sum()),
            "missing_rate": float(missing.mean()) if total else math.nan,
            "accuracy_committed": (
                float(correct[committed].mean()) if valid and n_committed else math.nan
            ),
            "pct_answer_1": float(extracted[committed].eq(1).mean()) if n_committed else math.nan,
            "pct_answer_0": float(extracted[committed].eq(0).mean()) if n_committed else math.nan,
            "pct_answer_wrong": (
                float(stance[committed].astype(str).eq("wrong").mean())
                if stance is not None and n_committed else math.nan
            ),
            "pct_answer_acceptable": (
                float(stance[committed].astype(str).eq("acceptable").mean())
                if stance is not None and n_committed else math.nan
            ),
            "distribution_note": (
                "extracted_stance_available" if stance is not None
                else "stance_unavailable_report_answer_0_1_without_semantic_relabeling"
            ),
            "numeric_semantic_conflict": int(
                (committed & group["numeric_semantic_conflict"]).sum()
            ),
            "construct_valid_for_accuracy": valid,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _marker_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(c[:-7] for c in frame.columns if c.endswith("__count"))


def _marker_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model_id", "persona", "subset"]
    for group_keys, positions in frame.groupby(keys, sort=False).groups.items():
        group = frame.loc[positions]
        for marker_id in _marker_ids(frame):
            count = pd.to_numeric(group[f"{marker_id}__count"], errors="coerce")
            density = pd.to_numeric(group.get(f"{marker_id}__density"), errors="coerce")
            present = pd.to_numeric(group.get(f"{marker_id}__present"), errors="coerce")
            rows.append({
                **dict(zip(keys, group_keys)), "marker_id": marker_id,
                "n_available": int(count.notna().sum()),
                "count_sum": float(count.sum(min_count=1)),
                "mean_density": float(density.mean()),
                "prevalence": float(present.mean()),
            })
    return pd.DataFrame(rows)


def build_marker_delta(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare count, density, and prevalence with explicit denominators."""
    key = ["model_id", "persona", "subset", "marker_id"]
    old = _marker_summary(baseline).rename(columns={
        "n_available": "n_available_old", "count_sum": "count_sum_old",
        "mean_density": "mean_density_old", "prevalence": "prevalence_old",
    })
    new = _marker_summary(candidate).rename(columns={
        "n_available": "n_available_new", "count_sum": "count_sum_new",
        "mean_density": "mean_density_new", "prevalence": "prevalence_new",
    })
    result = old.merge(new, on=key, how="outer", validate="one_to_one")
    result["count_delta"] = result["count_sum_new"] - result["count_sum_old"]
    result["mean_density_delta"] = result["mean_density_new"] - result["mean_density_old"]
    result["prevalence_delta"] = result["prevalence_new"] - result["prevalence_old"]
    result["change_direction"] = np.select(
        [result["count_delta"].gt(0), result["count_delta"].lt(0)],
        ["increase", "decrease"], default="unchanged",
    )
    return result


def build_surface_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep linguistic availability separate from answer commitment."""
    data = _analysis_frame(frame)
    rows = []
    for keys, group in data.groupby(["model_id", "persona", "subset"], sort=False):
        available = _bool(group["text_available"])
        status = group.get("text_status", pd.Series("unknown", index=group.index)).astype(str)
        total = len(group)
        rows.append({
            **dict(zip(("model_id", "persona", "subset"), keys)),
            "total": total,
            "surface_available": int(available.sum()),
            "surface_available_rate": float(available.mean()),
            "technical_only": int(status.eq("technical_only").sum()),
            "technical_only_rate": float(status.eq("technical_only").mean()),
            "surface_missing": int(status.eq("missing").sum()),
            "surface_missing_rate": float(status.eq("missing").mean()),
        })
    return pd.DataFrame(rows)


def build_answer_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare scoring/distribution with and without numeric-semantic conflicts."""
    data = _analysis_frame(frame)
    rows = []
    for keys, group in data.groupby(["model_id", "persona", "subset"], sort=False):
        model_id, persona, subset = keys
        committed = group["validated_committed"]
        conflict = committed & group["numeric_semantic_conflict"]
        retained = committed & ~group["numeric_semantic_conflict"]
        if subset in CONSTRUCT_VALID_SUBSETS:
            values = pd.to_numeric(group.get("is_correct"), errors="coerce")
            metric = "accuracy_committed"
        else:
            values = pd.to_numeric(group.get("extracted_answer"), errors="coerce").eq(1).astype(float)
            metric = "pct_answer_1"
        value_all = float(values[committed].mean()) if committed.any() else math.nan
        value_excluding = float(values[retained].mean()) if retained.any() else math.nan
        rows.append({
            "model_id": model_id, "persona": persona, "subset": subset,
            "metric": metric, "committed": int(committed.sum()),
            "numeric_semantic_conflict": int(conflict.sum()),
            "retained_excluding_conflict": int(retained.sum()),
            "value_all": value_all, "value_excluding_conflict": value_excluding,
            "delta_excluding_minus_all_pp": 100 * (value_excluding - value_all),
        })
    return pd.DataFrame(rows)


def holm_adjust_planned(values: Iterable[float]) -> np.ndarray:
    """Holm correction retaining NA hypotheses in the planned family size."""
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite], kind="stable")]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(order))) * p[order])
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    """Cochran Q for question-blocked, paired binary persona outcomes."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isin(values, [0, 1]).all():
        return math.nan, math.nan
    k = values.shape[1]
    column_sums = values.sum(axis=0)
    row_sums = values.sum(axis=1)
    total = float(column_sums.sum())
    denominator = k * total - float(np.square(row_sums).sum())
    if denominator <= 0:
        return math.nan, math.nan
    statistic = k * (k - 1) * (float(np.square(column_sums).sum()) - total * total / k) / denominator
    return statistic, float(chi2.sf(statistic, k - 1))


def _blocked_range_ci(
    matrix: np.ndarray, *, seed: int, resamples: int = 2_000
) -> tuple[float, float]:
    """Question-blocked percentile CI for the persona prevalence range."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values):
        return math.nan, math.nan
    observed_rows = np.unique(values, axis=0)
    if len(observed_rows) == 1:
        value = float(values.mean(axis=0).max() - values.mean(axis=0).min())
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 250):
        stop = min(start + 250, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means = values[indices].mean(axis=1)
        estimates[start:stop] = means.max(axis=1) - means.min(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _status_tests(frame: pd.DataFrame) -> pd.DataFrame:
    data = _analysis_frame(frame)
    data["linguistic_availability"] = np.select(
        [data.get("text_available", False), data.get("text_status", "").eq("technical_only"),
         data.get("text_status", "").eq("missing")],
        ["available", "technical_only", "missing"], default="other_unavailable",
    )
    data["answer_committed"] = data["analysis_answer_status"].eq("committed")
    data["surface_available"] = data["linguistic_availability"].eq("available")
    rows = []
    for test_name, column in (("answer_commitment_by_persona", "answer_committed"),
                              ("linguistic_availability_by_persona", "surface_available")):
        for (model_id, subset), group in data.groupby(["model_id", "subset"], sort=False):
            table = pd.crosstab(group["persona"], group[column], dropna=False)
            table = table.loc[:, table.sum().gt(0)]
            if min(table.shape) < 2:
                pearson = pearson_p = dof = cramers_v = math.nan
            else:
                pearson, pearson_p, dof, _ = chi2_contingency(table)
                denom = table.to_numpy().sum() * (min(table.shape) - 1)
                cramers_v = math.sqrt(pearson / denom) if denom else math.nan
            paired = group.pivot(index="question_id", columns="persona", values=column)
            paired = paired.dropna(axis=0, how="any")
            matrix = paired.astype(int).to_numpy()
            statistic, p_value = _cochran_q(matrix)
            informative = int(((matrix.sum(axis=1) > 0) & (matrix.sum(axis=1) < matrix.shape[1])).sum()) if len(matrix) else 0
            rate_range = float(paired.mean().max() - paired.mean().min()) if len(paired) else math.nan
            seed_bytes = hashlib.sha256(
                f"{RANDOM_SEED}|availability|{test_name}|{model_id}|{subset}".encode()
            ).digest()
            ci_low, ci_high = (
                _blocked_range_ci(matrix, seed=int.from_bytes(seed_bytes[:8], "big"))
                if informative >= 6 else (math.nan, math.nan)
            )
            rows.append(dict(test_name=test_name, model_id=model_id, subset=subset,
                             n=int(table.to_numpy().sum()), categories=table.shape[1],
                             n_question_blocks=len(paired), personas=matrix.shape[1] if matrix.ndim == 2 else 0,
                             informative_question_blocks=informative,
                             cochran_q=statistic, p_raw=p_value,
                             paired_effect_name="persona_rate_range",
                             paired_effect=rate_range,
                             effect_ci_low=ci_low, effect_ci_high=ci_high,
                             effect_ci_method="question_blocked_percentile_bootstrap",
                             effect_ci_resamples=2_000,
                             sparse_support=informative < 6,
                             pearson_chi2_unclustered=pearson, pearson_p_unclustered=pearson_p,
                             pearson_dof=dof, cramers_v_unclustered=cramers_v,
                             test_used="cochran_q_question_blocked",
                             interpretation="paired_association_diagnostic_not_detector_error"))
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["test_name", "model_id"])["p_raw"].transform(
        lambda x: holm_adjust_planned(x.to_numpy())
    )
    return result


def _frames_equal_except_changed(candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[bool, dict]:
    key = ["model_id", "persona", "question_id"]
    if set(map(tuple, candidate[key].astype(str).to_numpy())) != set(map(tuple, baseline[key].astype(str).to_numpy())):
        return False, {"reason": "response_key_sets_differ"}
    left = candidate.set_index(key).sort_index()
    right = baseline.set_index(key).sort_index()
    excluded = {f"{m}__{s}" for m in CHANGED_MARKERS for s in ("count", "density", "present")}
    comparable = sorted((set(left.columns) & set(right.columns)) - excluded)
    marker_surface = [c for c in comparable if "__" in c or c in {
        "text_available", "text_status", "word_count", "sentence_count",
        "avg_sentence_length", "ttr_300", "opening_category",
    }]
    float32_fields = {"avg_sentence_length", "ttr_300"}
    unequal = []
    for column in marker_surface:
        if column.endswith("__density") or column in float32_fields:
            left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=np.float32)
            right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=np.float32)
            equal = np.array_equal(left_values, right_values, equal_nan=True)
        else:
            equal = left[column].equals(right[column])
        if not equal:
            unequal.append(column)
    return not unequal, {
        "compared_columns": len(marker_surface), "unequal_columns": unequal,
        "excluded_changed_columns": sorted(excluded),
    }


def _human_review_complete(audit: pd.DataFrame) -> bool:
    if "human_review_status" not in audit or not len(audit):
        return False
    accepted = {"passed_two_reviewers", "complete_two_independent_reviewers_adjudicated"}
    return bool(audit["human_review_status"].astype(str).isin(accepted).all())


def build_release_gate(run: Path, *, baseline_equal: bool) -> dict:
    audit_path = run / "rule_audit_summary.csv"
    human_complete = False
    if audit_path.is_file():
        audit = pd.read_csv(audit_path, low_memory=False)
        human_complete = _human_review_complete(audit)
    error_test_path = run / "detector_error_persona_tests.csv"
    error_validation_path = run / "detector_error_persona_validation.json"
    error_test_valid = False
    if error_test_path.is_file() and error_validation_path.is_file():
        validation = json.loads(error_validation_path.read_text(encoding="utf-8"))
        error_test_valid = bool(validation.get("valid_human_gold") and validation.get("clustered_by_question_id"))
    blockers = []
    if not baseline_equal:
        blockers.append("unchanged_outcomes_not_identical_to_frozen_baseline")
    if not human_complete:
        blockers.append("two_independent_human_reviewers")
    if not error_test_valid:
        blockers.append("validated_clustered_detector_error_persona_analysis")
    return {
        "release_eligible": False,
        "result_status": "candidate",
        "human_review_complete": human_complete,
        "detector_error_persona_test": (
            "available_validated_clustered_analysis" if error_test_valid
            else "unavailable_no_valid_human_gold"
        ),
        "blockers": blockers or ["candidate_release_requires_explicit_human_gate"],
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _join_canonical_stance(frame: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, str]:
    """Join an existing parser field; never infer stance from answer or GT."""
    if "extracted_stance" in frame.columns:
        return frame, "candidate_response_level"
    path = root / CANONICAL_SCORING
    if not path.is_file():
        return frame, "unavailable"
    scoring = pd.read_csv(
        path,
        usecols=["model", "persona_folder", "question_id", "extracted_stance"],
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    key = ["model_id", "persona", "question_id"]
    if scoring.duplicated(key).any():
        raise ValueError("Duplicate canonical scoring key")
    return frame.merge(scoring, on=key, how="left", validate="one_to_one"), str(path)


def _load_sensitivity_flags(run: Path, baseline: Path) -> tuple[pd.DataFrame, dict]:
    """Derive diagnostic flags from stored hits/surfaces without re-extraction."""
    overlap_keys: set[str] = set()
    hit_path = run / "marker_hits.csv.gz"
    scenario_path = baseline / "scenario_lookup.csv"
    if hit_path.is_file() and scenario_path.is_file():
        scenarios = pd.read_csv(scenario_path, usecols=["question_id", "scenario"]).set_index("question_id")["scenario"]
        hits = pd.read_csv(hit_path, low_memory=False)
        if {"response_key", "matched_text"}.issubset(hits.columns):
            if "decision" in hits:
                hits = hits[hits["decision"].astype(str).eq("accepted")]
            for row in hits[["response_key", "matched_text"]].itertuples(index=False):
                question_id = str(row.response_key).rsplit("|", 1)[-1]
                scenario = scenarios.get(question_id)
                if scenario is not None and _scenario_hit_overlaps(str(scenario), str(row.matched_text)):
                    overlap_keys.add(str(row.response_key))

    mixed_keys: set[str] = set()
    surface_path = run / "surface_texts.jsonl.gz"
    if surface_path.is_file():
        with gzip.open(surface_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                surfaces = row.get("surfaces", {})
                visible = surfaces.get("visible", "") if isinstance(surfaces, dict) else ""
                if _mixed_script_signal(str(visible)):
                    mixed_keys.add(str(row["response_key"]))
    keys = sorted(overlap_keys | mixed_keys)
    flags = pd.DataFrame({"response_key": keys})
    if len(flags):
        flags["scenario_lexical_overlap"] = flags.response_key.isin(overlap_keys)
        flags["mixed_script_signal"] = flags.response_key.isin(mixed_keys)
    else:
        flags = pd.DataFrame(columns=["response_key", "scenario_lexical_overlap", "mixed_script_signal"])
    return flags, {
        "scenario_lexical_overlap_responses": len(overlap_keys),
        "mixed_script_signal_responses": len(mixed_keys),
        "scenario_flag_definition": "accepted matched_text occurs literally in frozen scenario",
        "mixed_script_definition": "ASCII and non-ASCII Unicode letters coexist; diagnostic, not language ID",
    }


def _conditional_sensitivity_pairs(
    frame: pd.DataFrame, within: pd.DataFrame, direct: pd.DataFrame, flags: pd.DataFrame
) -> pd.DataFrame:
    data = frame.copy()
    if "response_key" not in data:
        data["response_key"] = data.model_id.astype(str) + "|" + data.persona.astype(str) + "|" + data.question_id.astype(str)
    data = data.merge(flags, on="response_key", how="left", validate="one_to_one")
    for column in ("scenario_lexical_overlap", "mixed_script_signal"):
        data[column] = data[column].fillna(False).astype(bool)
    templates = pd.concat([
        within[(within.outcome_id.eq(CHANGED_MARKERS[0])) & within.population.eq("all_visible")].assign(comparison_scope="within"),
        direct[(direct.outcome_id.eq(CHANGED_MARKERS[0])) & direct.population.eq("all_visible")].assign(comparison_scope="direct"),
    ], ignore_index=True)
    meta = ["comparison_scope", "model_id", "subset", "left_persona", "right_persona", "contrast_id", "direction"]
    templates = templates[meta].drop_duplicates()
    rows = []
    columns = ["question_id", "text_available", "scenario_lexical_overlap", "mixed_script_signal",
               "connective_conditional__count", "connective_conditional__density", "connective_conditional__present"]
    grouped = {(m, s): g for (m, s), g in data.groupby(["model_id", "subset"], sort=False)}
    populations = {
        "all_visible": None,
        "exclude_scenario_lexical_overlap": "scenario_lexical_overlap",
        "exclude_mixed_script_signal": "mixed_script_signal",
    }
    for template in templates.itertuples(index=False):
        group = grouped[(template.model_id, template.subset)]
        left = group[group.persona.eq(template.left_persona)][columns]
        right = group[group.persona.eq(template.right_persona)][columns]
        pair = left.merge(right, on="question_id", suffixes=("_left", "_right"), validate="one_to_one")
        base = _bool(pair.text_available_left) & _bool(pair.text_available_right)
        for population, excluded in populations.items():
            keep = base.copy()
            if excluded:
                keep &= ~_bool(pair[f"{excluded}_left"]) & ~_bool(pair[f"{excluded}_right"])
            density_delta = pd.to_numeric(pair.loc[keep, "connective_conditional__density_left"], errors="coerce") - pd.to_numeric(pair.loc[keep, "connective_conditional__density_right"], errors="coerce")
            presence_delta = pd.to_numeric(pair.loc[keep, "connective_conditional__present_left"], errors="coerce") - pd.to_numeric(pair.loc[keep, "connective_conditional__present_right"], errors="coerce")
            count_left = pd.to_numeric(pair.loc[keep, "connective_conditional__count_left"], errors="coerce")
            count_right = pd.to_numeric(pair.loc[keep, "connective_conditional__count_right"], errors="coerce")
            rows.append({**{name: getattr(template, name) for name in meta},
                         "population": population, "n_pairs": int(keep.sum()),
                         "median_density_delta": float(density_delta.median()),
                         "mean_density_delta": float(density_delta.mean()),
                         "prevalence_delta": float(presence_delta.mean()),
                         "raw_count_sum_left": float(count_left.sum()),
                         "raw_count_sum_right": float(count_right.sum()),
                         "inference_status": "descriptive_sensitivity_no_p_value"})
    return pd.DataFrame(rows)


def _pair_statistics(candidate: pd.DataFrame, baseline_equal: bool, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not baseline_equal:
        raise ValueError("Cannot reuse legacy statistics: unchanged columns differ")
    from src.benchmarks.ethics.run_gender_ethics_lm_workspace import enable_marker_imports
    enable_marker_imports()
    from src.benchmarks.ethics import regional_cultural_ethics_linguistic_markers_v2 as v2

    registry = pd.read_csv(root / "result/benchmarks/ethics/reports/Regional_Cultural_Persona/"
                           "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16/marker_registry.csv")
    _, outcomes = v2.build_outcome_dictionary(registry)
    selected = tuple(spec for spec in outcomes if spec.outcome_id in CHANGED_MARKERS)
    data = _analysis_frame(candidate)
    current_within = v2.build_within_identity_pairs(data, selected, BOOTSTRAP_RESAMPLES)
    current_direct = v2.build_direct_identity_pairs(data, selected, BOOTSTRAP_RESAMPLES)
    current_omnibus = v2.build_omnibus(data, selected)
    legacy = root / LEGACY_ANALYSIS / "tables"
    old_within = pd.read_csv(legacy / "paired_within_identity_all_scales.csv.gz", low_memory=False)
    old_direct = pd.read_csv(legacy / "paired_direct_identity_all_scales.csv.gz", low_memory=False)
    old_omnibus = pd.read_csv(legacy / "omnibus_five_condition.csv.gz", low_memory=False)
    within = pd.concat([old_within[~old_within.outcome_id.isin(CHANGED_MARKERS)], current_within], ignore_index=True)
    direct = pd.concat([old_direct[~old_direct.outcome_id.isin(CHANGED_MARKERS)], current_direct], ignore_index=True)
    omnibus = pd.concat([old_omnibus[~old_omnibus.outcome_id.isin(CHANGED_MARKERS)], current_omnibus], ignore_index=True)
    within = v2._apply_pairwise_corrections(within,
        ("population", "model_id", "subset", "identity", "outcome_id", "multiplicity_family"),
        ("population", "model_id", "subset", "identity", "contrast_id", "outcome_tier"))
    direct = v2._apply_pairwise_corrections(direct,
        ("population", "model_id", "subset", "outcome_id", "multiplicity_family"),
        ("population", "model_id", "subset", "identity_contrast", "condition", "outcome_tier"))
    within["p_holm"] = within.groupby(
        ["population", "model_id", "subset", "identity", "outcome_id", "multiplicity_family"],
        observed=True, dropna=False,
    )["p_raw"].transform(lambda values: holm_adjust_planned(values.to_numpy(float)))
    direct["p_holm"] = direct.groupby(
        ["population", "model_id", "subset", "outcome_id", "multiplicity_family"],
        observed=True, dropna=False,
    )["p_raw"].transform(lambda values: holm_adjust_planned(values.to_numpy(float)))
    within["significant_holm_0_05"] = within["p_holm"].lt(0.05)
    direct["significant_holm_0_05"] = direct["p_holm"].lt(0.05)
    omnibus["p_holm_outcomes"] = omnibus.groupby(
        ["model_id", "subset", "identity", "outcome_tier"], observed=True, dropna=False
    )["p_raw"].transform(lambda values: holm_adjust_planned(values.to_numpy(float)))
    omnibus["significant_holm_0_05"] = omnibus["p_holm_outcomes"].lt(0.05)
    provenance = {
        "method": "reuse_verified_legacy_rows_and_recompute_changed_outcome",
        "reused_outcomes": int(within.outcome_id.nunique() - len(CHANGED_MARKERS)),
        "recomputed_outcomes": list(CHANGED_MARKERS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed": RANDOM_SEED,
        "legacy_within_sha256": _sha256(legacy / "paired_within_identity_all_scales.csv.gz"),
        "legacy_direct_sha256": _sha256(legacy / "paired_direct_identity_all_scales.csv.gz"),
        "legacy_omnibus_sha256": _sha256(legacy / "omnibus_five_condition.csv.gz"),
        "holm_family_sizes": [3, 4, 24],
        "bh_scope": "across_outcomes_separate_robustness",
    }
    return within, direct, omnibus, provenance


def build_report(run: Path, baseline: Path) -> dict:
    """Build all v3 report artifacts inside an existing, unique run directory."""
    run, baseline = Path(run), Path(baseline)
    candidate_path = run / "response_level.csv.gz"
    baseline_path = baseline / "response_level_markers.csv.gz"
    if not baseline_path.is_file():
        baseline_path = baseline / "response_level.csv.gz"
    for path in (candidate_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = {name: run / name for name in (
        "scoring_summary.csv", "marker_descriptives.csv.gz", "old_new_delta.csv.gz",
        "surface_eligibility.csv", "answer_conflict_sensitivity.csv",
        "persona_availability_tests.csv", "paired_within_identity.csv.gz",
        "paired_direct_identity.csv.gz", "omnibus_five_condition.csv.gz",
        "opening_descriptives.csv", "opening_pairwise_stuart_maxwell.csv",
        "opening_transitions.csv.gz", "conditional_sensitivity_pairs.csv",
        "direction_changes.csv", "release_gate.json",
        "report_provenance.json", "summary.md")}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError(f"Report output already exists: {existing}")

    root = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                 if (p / "src").is_dir() and (p / "result").is_dir()), Path.cwd().resolve())
    candidate = pd.read_csv(candidate_path, low_memory=False)
    old = pd.read_csv(baseline_path, low_memory=False)
    candidate, stance_source = _join_canonical_stance(candidate, root)
    candidate = _analysis_frame(candidate)
    baseline_equal, equality = _frames_equal_except_changed(candidate, old)
    scoring = build_scoring_table(candidate)
    descriptives = _marker_summary(candidate)
    delta = build_marker_delta(candidate, old)
    eligibility = build_surface_eligibility(candidate)
    answer_sensitivity = build_answer_sensitivity(candidate)
    status = _status_tests(candidate)
    within, direct, omnibus, stat_provenance = _pair_statistics(candidate, baseline_equal, root)
    flags, flag_provenance = _load_sensitivity_flags(run, baseline)
    conditional_sensitivity = _conditional_sensitivity_pairs(candidate, within, direct, flags)

    scoring.to_csv(outputs["scoring_summary.csv"], index=False)
    descriptives.to_csv(outputs["marker_descriptives.csv.gz"], index=False)
    delta.to_csv(outputs["old_new_delta.csv.gz"], index=False)
    eligibility.to_csv(outputs["surface_eligibility.csv"], index=False)
    answer_sensitivity.to_csv(outputs["answer_conflict_sensitivity.csv"], index=False)
    status.to_csv(outputs["persona_availability_tests.csv"], index=False)
    within.to_csv(outputs["paired_within_identity.csv.gz"], index=False)
    direct.to_csv(outputs["paired_direct_identity.csv.gz"], index=False)
    omnibus.to_csv(outputs["omnibus_five_condition.csv.gz"], index=False)
    legacy_tables = root / LEGACY_ANALYSIS / "tables"
    opening_sources = {
        "opening_descriptives.csv": legacy_tables / "opening_descriptives.csv",
        "opening_pairwise_stuart_maxwell.csv": legacy_tables / "opening_pairwise_stuart_maxwell.csv",
        "opening_transitions.csv.gz": legacy_tables / "opening_transitions.csv.gz",
    }
    for name, source in opening_sources.items():
        shutil.copyfile(source, outputs[name])
    conditional_sensitivity.to_csv(outputs["conditional_sensitivity_pairs.csv"], index=False)

    old_pairs = pd.concat([
        pd.read_csv(legacy_tables / "paired_within_identity_all_scales.csv.gz", low_memory=False).assign(comparison_scope="within"),
        pd.read_csv(legacy_tables / "paired_direct_identity_all_scales.csv.gz", low_memory=False).assign(comparison_scope="direct"),
    ], ignore_index=True)
    new_pairs = pd.concat([within.assign(comparison_scope="within"), direct.assign(comparison_scope="direct")], ignore_index=True)
    identifiers = [c for c in ("comparison_scope", "population", "model_id", "subset", "identity",
                               "identity_contrast", "contrast_id", "condition", "outcome_id")
                   if c in old_pairs.columns and c in new_pairs.columns]
    changes = old_pairs[identifiers + ["effect", "p_holm"]].merge(
        new_pairs[identifiers + ["effect", "p_holm"]], on=identifiers, suffixes=("_old", "_new"))
    sign_old = np.sign(changes.effect_old.fillna(0))
    sign_new = np.sign(changes.effect_new.fillna(0))
    p_changed = ~np.isclose(changes.p_holm_old, changes.p_holm_new, equal_nan=True)
    changes = changes[(sign_old != sign_new) | p_changed]
    changes.to_csv(outputs["direction_changes.csv"], index=False)

    gate = build_release_gate(run, baseline_equal=baseline_equal)
    diagnostic_path = root / AI_DIAGNOSTIC
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.is_file() else None
    if diagnostic and int(diagnostic.get("label_counts", {}).get("FP", 0)) > 0:
        gate["blockers"].append("ai_diagnostic_contains_known_false_positives_requiring_human_adjudication")
        gate["ai_diagnostic_false_positives"] = int(diagnostic["label_counts"]["FP"])
    _write_json(outputs["release_gate.json"], gate)
    provenance = {
        "candidate_version": "regional-lm-v3.0.0-candidate",
        "candidate_response_sha256": _sha256(candidate_path),
        "baseline_response_sha256": _sha256(baseline_path),
        "unchanged_column_equality": equality,
        "statistics": stat_provenance,
        "opening_reuse": {name: {"source": str(source), "sha256": _sha256(source)}
                          for name, source in opening_sources.items()},
        "sensitivity_flags": flag_provenance,
        "ai_diagnostic": ({"source": str(diagnostic_path), "sha256": _sha256(diagnostic_path),
                           "full_responses_read": diagnostic.get("full_responses_read"),
                           "accepted_hits_reviewed": diagnostic.get("accepted_hits_reviewed"),
                           "label_counts": diagnostic.get("label_counts"),
                           "n_false_negative_candidates": len(diagnostic.get("false_negative_candidates", [])),
                           "all_rules": diagnostic.get("all_rules"),
                           "covered_rules": diagnostic.get("covered_rules"),
                           "scope": diagnostic.get("scope")}
                          if diagnostic else {"status": "unavailable"}),
        "execution_code_sha256": _sha256(Path(__file__)),
        "ground_truth_used_for_marker_extraction": False,
        "answers_reextracted_or_rescored": False,
        "extracted_stance_source": stance_source,
        "scenario_overlap_sensitivity": (
            "descriptive_from_frozen_scenario_and_accepted_hit_spans"
        ),
        "mixed_language_sensitivity": "descriptive_mixed_script_signal_not_language_identification",
        "detector_error_persona_test": gate["detector_error_persona_test"],
    }
    _write_json(outputs["report_provenance.json"], provenance)
    changed = delta[delta.count_delta.ne(0)]
    commitment_tests = status[status.test_name.eq("answer_commitment_by_persona")]
    availability_tests = status[status.test_name.eq("linguistic_availability_by_persona")]
    commitment_significant = int(commitment_tests.p_holm.lt(0.05).sum())
    availability_significant = int(availability_tests.p_holm.lt(0.05).sum())
    outputs["summary.md"].write_text(
        "# Ringkasan kandidat Regional/Cultural ETHICS v3\n\n"
        f"Status rilis: **candidate** (`release_eligible=false`). Dataset memuat {len(candidate):,} respons. "
        "Analisis jawaban memakai hasil parser program yang sudah ada; baris `semantic_unaudited` "
        "didemote menjadi *abstain* hanya pada view analisis.\n\n"
        f"Perubahan marker teramati pada {len(changed):,} sel model–persona–subset–marker. "
        "Perubahan aturan terbatas pada pengecualian idiom `if you will` dari `connective_conditional`. "
        "Statistik 55 outcome yang tidak berubah dipakai ulang setelah kesetaraan seluruh kolom marker/surface "
        "dibuktikan; outcome conditional dihitung ulang dengan 10.000 bootstrap.\n\n"
        "Akurasi hanya ditafsirkan untuk commonsense dan justice. Deontology dan virtue disajikan sebagai "
        "distribusi jawaban. Uji status/availability terhadap persona adalah diagnosis asosiasi dan bukan "
        "estimasi error detector. "
        f"Perbedaan commitment terdeteksi pada {commitment_significant}/{len(commitment_tests)} uji Holm dan "
        f"perbedaan linguistic availability pada {availability_significant}/{len(availability_tests)} uji Holm. "
        "Hasil ini menunjukkan risiko coverage yang tidak seragam; hasil tersebut tidak mengukur FP/FN detector. "
        "Uji error detector terhadap persona tidak tersedia tanpa gold manusia valid.\n\n"
        + (f"Readback AI diagnostik mencakup {diagnostic.get('full_responses_read', 0)} respons, "
           f"{diagnostic.get('accepted_hits_reviewed', 0)} hit diterima, dan label "
           f"{diagnostic.get('label_counts', {})}; cakupan rule {diagnostic.get('covered_rules', 0)}/"
           f"{diagnostic.get('all_rules', 0)}, dengan {len(diagnostic.get('false_negative_candidates', []))} "
           "catatan kandidat FN. Ini data development yang pernah terekspos, bukan estimasi "
           "precision/recall independen atau pengganti reviewer manusia.\n\n" if diagnostic else
           "Artefak readback AI diagnostik tidak tersedia pada saat pelaporan.\n\n")
        + "Gate tetap tertutup sampai dua reviewer manusia independen dan adjudikasi selesai.\n",
        encoding="utf-8",
    )
    return {
        "status": "candidate", "release_eligible": False, "rows": len(candidate),
        "scoring_rows": len(scoring), "descriptive_rows": len(descriptives),
        "within_rows": len(within), "direct_rows": len(direct),
        "omnibus_rows": len(omnibus), "conditional_sensitivity_rows": len(conditional_sensitivity),
        "changed_cells": len(changed), "artifacts": list(outputs),
        "execution_code_sha256": _sha256(Path(__file__)),
    }

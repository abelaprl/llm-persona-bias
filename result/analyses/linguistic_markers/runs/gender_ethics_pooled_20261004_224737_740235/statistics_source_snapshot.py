"""Paired marker comparisons without answer labels or correctness inputs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest, rankdata, skew, wilcoxon


KEYS = ["model_id", "subset", "question_id"]
CONDITIONS = ("explicit_1", "implicit_1", "implicit_2", "implicit_3")
PERSONAS = ("neutral",) + tuple(f"{g}_{c}" for c in CONDITIONS for g in ("woman", "man"))
MEASUREMENTS = (("count", "count"), ("density", "density"), ("presence", "present"))


def make_contrast_specs() -> list[dict]:
    specs = []

    def add(left, right, role, kind="pairwise"):
        specs.append({"contrast_id": f"{left}__minus__{right}", "left": left,
                      "right": right, "left_condition": left, "right_condition": right,
                      "role": role, "kind": kind, "effect_direction": "left_minus_right"})

    for condition in CONDITIONS:
        add(f"woman_{condition}", "neutral", "primary")
        add(f"man_{condition}", "neutral", "primary")
        add(f"woman_{condition}", f"man_{condition}", "primary")
    for gender in ("woman", "man"):
        for condition in CONDITIONS[1:]:
            add(f"{gender}_{condition}", f"{gender}_explicit_1", "secondary")
        for left, right in combinations(CONDITIONS[1:], 2):
            add(f"{gender}_{left}", f"{gender}_{right}", "secondary")
    for left, right in combinations(CONDITIONS[1:], 2):
        add(left, right, "interaction", "difference_of_differences")
    return specs


def _validate(records):
    required = KEYS + ["persona"]
    if not set(required).issubset(records):
        raise ValueError(f"Missing pairing columns: {sorted(set(required) - set(records))}")
    if records[required].isna().any().any():
        raise ValueError("Null pairing key is not allowed")
    if records.duplicated(required).any():
        raise ValueError("Duplicate model/persona/subset/question_id key")


def _eligible(records):
    valid = pd.Series(True, index=records.index)
    if "text_available" in records:
        valid &= records.text_available.eq(True)
    if "text_status" in records:
        valid &= records.text_status.eq("substantive_explanation")
    return valid


def _long_wide(records, personas):
    _validate(records)
    if "value" not in records:
        raise ValueError("Public pairing helpers require one marker/measurement in column value")
    source = records[records.persona.isin(personas)].copy()
    source.loc[~_eligible(source), "value"] = np.nan
    return source.pivot(index=KEYS, columns="persona", values="value").reindex(columns=personas)


def build_pairs(records: pd.DataFrame, contrast: dict) -> pd.DataFrame:
    if contrast.get("kind", "pairwise") != "pairwise":
        raise ValueError("build_pairs only accepts kind=pairwise")
    left, right = contrast["left"], contrast["right"]
    if left == right:
        raise ValueError("A contrast requires distinct personas")
    wide = _long_wide(records, [left, right])
    pairs = wide.dropna().rename(columns={left: "left_value", right: "right_value"}).reset_index()
    pairs.attrs.update(n_left_available=int(wide[left].notna().sum()),
                       n_right_available=int(wide[right].notna().sum()),
                       n_left_total=int(records.persona.eq(left).sum()),
                       n_right_total=int(records.persona.eq(right).sum()))
    return pairs


def build_neutral_triplets(records: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    names = [f"woman_{condition}", f"man_{condition}", "neutral"]
    result = _long_wide(records, names).dropna().rename(columns=dict(zip(names, ["woman_value", "man_value", "neutral_value"]))).reset_index()
    result.attrs.update(records.attrs)
    if "measurement" in records and records.measurement.nunique() == 1:
        result.attrs["measurement"] = records.measurement.iloc[0]
    return result


def summarize_neutral_triplets(triplets: pd.DataFrame, measurement: str | None = None) -> dict:
    measurement = measurement or triplets.attrs.get("measurement", "count")
    factor = 100.0 if measurement in ("presence", "present") else 1.0
    result = {"n_triplets": len(triplets), "sample_scope": "common_triplet"}
    for left, right in (("woman", "neutral"), ("man", "neutral"), ("woman", "man")):
        result[f"d_{left}_{right}"] = float((triplets[f"{left}_value"] - triplets[f"{right}_value"]).mean() * factor)
    return result


def build_interaction_pairs(records: pd.DataFrame, left_condition: str, right_condition: str) -> pd.DataFrame:
    if left_condition not in CONDITIONS or right_condition not in CONDITIONS or left_condition == right_condition:
        raise ValueError("Interaction requires two distinct recognized conditions")
    names = [f"{gender}_{condition}" for condition in (left_condition, right_condition) for gender in ("woman", "man")]
    result = _long_wide(records, names).dropna().reset_index()
    result["delta_left"] = result[names[0]] - result[names[1]]
    result["delta_right"] = result[names[2]] - result[names[3]]
    result["interaction_delta"] = result.delta_left - result.delta_right
    return result


def _measurement_info(measurement):
    if measurement in ("presence", "present"):
        return 100.0, "percentage_points", "prevalence_difference_pp"
    return 1.0, "occurrences_per_100_words" if measurement == "density" else "occurrences", "mean_paired_difference"


def _effect_summary(left, right, measurement, *, interaction=False):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    factor, unit, estimand = _measurement_info(measurement)
    if factor == 100 and not interaction and not (np.isin(left, [0, 1]).all() and np.isin(right, [0, 1]).all()):
        raise ValueError("Presence requires binary paired values")
    if interaction:
        estimand = "mean_difference_in_differences_pp" if factor == 100 else "mean_difference_in_differences"
    delta = left - right
    nonzero = delta[delta != 0]
    n, changed = len(delta), len(nonzero)
    positive, negative = int((nonzero > 0).sum()), int((nonzero < 0).sum())
    result = dict(n_pairs=n, n_changed=changed, n_positive=positive, n_negative=negative,
                  low_support=changed < 10, effect=float(delta.mean() * factor) if n else np.nan,
                  left_mean=float(left.mean()) if n else np.nan, right_mean=float(right.mean()) if n else np.nan,
                  measurement_unit=unit, estimand=estimand, ci_estimand=estimand,
                  ci_low=np.nan, ci_high=np.nan, ci_method="subset_stratified_paired_percentile",
                  ci_level=0.95, ci_simultaneous=False, raw_p=np.nan, sign_raw_p=np.nan,
                  sign_effect=(positive - negative) / changed if changed else np.nan,
                  sign_estimand="positive_minus_negative_fraction_among_changed_pairs",
                  rank_effect_estimand="matched_rank_biserial_nonzero_pairs_average_ties",
                  matched_rank_biserial=np.nan, status="unavailable" if not n else "no_change" if not changed else "estimated",
                  difference_skewness=np.nan, symmetry_warning=False)
    if interaction:
        result.update(test="paired_sign_flip_permutation_mean", test_assumption="exchangeability_of_paired_difference_signs_under_null")
    elif factor == 100:
        result.update(test="mcnemar_exact_two_sided", discordant=changed)
    else:
        result.update(test="wilcoxon_signed_rank_two_sided", wilcoxon_method="asymptotic_no_continuity_correction",
                      test_assumption="symmetric_paired_differences_for_location_interpretation")
    if n:
        result["left_rate"] = float(left.mean()) if factor == 100 and not interaction else np.nan
        result["right_rate"] = float(right.mean()) if factor == 100 and not interaction else np.nan
    if not changed:
        if n:
            result.update(raw_p=1.0, sign_raw_p=1.0, ci_low=0.0, ci_high=0.0)
        return result
    result["sign_raw_p"] = float(binomtest(positive, changed, p=0.5).pvalue)
    ranks = rankdata(np.abs(nonzero), method="average")
    result["matched_rank_biserial"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    if n > 2 and np.ptp(delta) > 0:
        result["difference_skewness"] = float(skew(delta, bias=False))
        result["symmetry_warning"] = bool(abs(result["difference_skewness"]) > 1)
    if not interaction and factor == 100:
        result.update(raw_p=result["sign_raw_p"], test="mcnemar_exact_two_sided", discordant=changed)
    elif not interaction:
        # Fixed asymptotic handling avoids automatic Monte Carlo for small tied samples.
        result.update(raw_p=float(wilcoxon(delta, zero_method="wilcox", correction=False, alternative="two-sided", method="asymptotic").pvalue),
                      test="wilcoxon_signed_rank_two_sided", test_assumption="symmetric_paired_differences_for_location_interpretation")
    return result


def _bootstrap_matrix(delta, strata, resamples, rng):
    """Draw shared within-subset question weights for all outcome coordinates."""
    n, width = delta.shape
    if not n or not resamples:
        return np.full((2, width), np.nan)
    samples = np.empty((resamples, width), dtype=float)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    use_sparse = np.count_nonzero(delta) / delta.size < 0.25
    matrix = sparse.csr_matrix(delta.T) if use_sparse else delta
    for start in range(0, resamples, 128):
        size = min(128, resamples - start)
        weights = np.zeros((size, n), dtype=float)
        for indices in groups:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=size)
        samples[start:start + size] = ((matrix @ weights.T).T if use_sparse else weights @ matrix) / n
    return np.quantile(samples, [0.025, 0.975], axis=0)


def compare_pairs(pairs: pd.DataFrame, outcome_spec: dict) -> dict:
    measurement = outcome_spec.get("measurement", "density")
    clean = pairs.dropna(subset=["left_value", "right_value"])
    result = _effect_summary(clean.left_value, clean.right_value, measurement)
    result.update({k: v for k, v in pairs.attrs.items() if k.startswith("n_")})
    for side in ("left", "right"):
        result.setdefault(f"n_{side}_available", len(clean))
    count = int(outcome_spec.get("bootstrap_resamples", 5000))
    seed = int(outcome_spec.get("seed", 42))
    result.update(bootstrap_resamples=count, seed=seed, sample_scope="pairwise_complete")
    if len(clean) and count and result["n_changed"]:
        factor = _measurement_info(measurement)[0]
        delta = (clean.left_value.to_numpy(dtype=float) - clean.right_value.to_numpy(dtype=float))[:, None] * factor
        strata = clean.subset.to_numpy() if "subset" in clean else np.repeat("all", len(clean))
        bounds = _bootstrap_matrix(delta, strata, count, np.random.default_rng(seed))
        result.update(ci_low=float(bounds[0, 0]), ci_high=float(bounds[1, 0]))
    return result


def _holm(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.full(len(values), np.nan)
    order = np.flatnonzero(valid)[np.argsort(values[valid], kind="stable")]
    # Unavailable predeclared hypotheses count toward family size, conservatively.
    if len(order):
        result[order] = np.minimum(1.0, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(order)))))
    return result


def adjust_families(comparisons: pd.DataFrame) -> pd.DataFrame:
    result = comparisons.copy()
    if result.empty:
        return result
    if "analysis_status" not in result:
        result["analysis_status"] = np.where(result.get("rule_status", "candidate") == "accepted", "accepted", "candidate")
    dims = ["model_id", "scope", "family", "role", "analysis_status"]
    for name in dims:
        if name not in result:
            result[name] = "unspecified"
    result["family_id"] = result[dims].astype(str).agg("|".join, axis=1)
    result["adjusted_p"] = np.nan
    result["family_size"] = 0
    result["sign_adjusted_p"] = np.nan
    result["sign_family_id"] = result.family_id + "|sign_sensitivity"
    for _, indices in result.groupby("family_id", sort=False).groups.items():
        result.loc[indices, "adjusted_p"] = _holm(result.loc[indices, "raw_p"])
        result.loc[indices, "family_size"] = len(indices)
        if "sign_raw_p" in result:
            result.loc[indices, "sign_adjusted_p"] = _holm(result.loc[indices, "sign_raw_p"])
    return result


def _availability_tests(frame, model_id, scope, permutations, rng):
    """Condition-label permutations preserve each question's observed responses."""
    statuses = sorted(frame.text_status.unique())
    index = pd.MultiIndex.from_frame(frame[["subset", "question_id"]].drop_duplicates().sort_values(["subset", "question_id"]))
    wide = frame.pivot(index=["subset", "question_id"], columns="persona", values="text_status").reindex(index=index, columns=PERSONAS)
    codes = np.full(wide.shape, -1, dtype=np.int8)
    for code, status in enumerate(statuses):
        codes[wide.eq(status).to_numpy()] = code
    n_personas = len(PERSONAS)
    observed = np.array([(codes == code).sum(axis=0) for code in range(len(statuses))]).T
    persona_n = observed.sum(axis=1)
    rates = {persona: float(frame.loc[frame.persona == persona, "text_available"].mean())
             for persona in PERSONAS if frame.persona.eq(persona).any()}
    patterns = defaultdict(list)
    for i, row in enumerate(codes):
        patterns[tuple(np.flatnonzero(row >= 0))].append(i)

    def chi_square(table):
        table = np.asarray(table, dtype=float)
        totals = table.sum(axis=(-2, -1), keepdims=True)
        expected = table.sum(axis=-1, keepdims=True) * table.sum(axis=-2, keepdims=True) / np.maximum(totals, 1)
        return np.sum(np.divide((table - expected) ** 2, expected, out=np.zeros_like(table), where=expected > 0), axis=(-2, -1))

    available_code = statuses.index("substantive_explanation") if "substantive_explanation" in statuses else None
    available_counts = observed[:, available_code] if available_code is not None else np.zeros(n_personas)
    binary = np.stack([available_counts, persona_n - available_counts], axis=-1)
    observed_stats = np.array([chi_square(observed), chi_square(binary)])
    exceeded = np.zeros(2, dtype=int)
    if np.any(observed_stats > 0):
        for start in range(0, permutations, 128):
            size = min(128, permutations - start)
            counts = np.zeros((size, n_personas, len(statuses)), dtype=int)
            for present, rows in patterns.items():
                values = codes[np.ix_(rows, present)]
                shuffled = rng.permuted(np.broadcast_to(values, (size,) + values.shape), axis=2)
                for code in range(len(statuses)):
                    counts[:, present, code] += (shuffled == code).sum(axis=1)
            binary_available = counts[:, :, available_code] if available_code is not None else np.zeros((size, n_personas))
            binary_counts = np.stack([binary_available, persona_n[None, :] - binary_available], axis=-1)
            simulated = np.stack([chi_square(counts), chi_square(binary_counts)], axis=-1)
            exceeded += (simulated >= observed_stats - 1e-12).sum(axis=0)
    else:
        exceeded[:] = permutations
    rows = []
    for j, (outcome, table) in enumerate((("text_status", observed), ("text_available", binary))):
        n = int(table.sum())
        min_dim = min(int((table.sum(axis=0) > 0).sum()) - 1, int((table.sum(axis=1) > 0).sum()) - 1)
        rows.append(dict(model_id=model_id, scope=scope, outcome=outcome, n_responses=n,
                         n_question_blocks=len(codes), statistic=float(observed_stats[j]),
                         raw_p=float((exceeded[j] + 1) / (permutations + 1)) if permutations else np.nan,
                         cramers_v=float(np.sqrt(observed_stats[j] / (n * min_dim))) if min_dim > 0 else np.nan,
                         permutations=permutations, test="within_question_persona_label_permutation_pearson_chi_square",
                         availability_rate_by_persona=json.dumps(rates, sort_keys=True),
                         availability_rate_range_pp=100 * (max(rates.values()) - min(rates.values())) if rates else np.nan,
                         permutation_assumption="persona_labels_exchangeable_within_observed_question_block"))
    return rows


def _finish_jobs(jobs, bootstrap_resamples, permutation_resamples, rng):
    for group in jobs.values():
        deltas, refs, strata = group["deltas"], group["refs"], group["strata"]
        if not deltas:
            continue
        matrix = np.column_stack(deltas)
        bounds = _bootstrap_matrix(matrix, strata, bootstrap_resamples, rng)
        for column, targets in enumerate(refs):
            for target, prefix in targets:
                target[f"ci_low{prefix}"] = float(bounds[0, column])
                target[f"ci_high{prefix}"] = float(bounds[1, column])
        interaction_columns = [i for i, targets in enumerate(refs) if any(t.get("role") == "interaction" for t, _ in targets)]
        if not interaction_columns:
            continue
        interactions = matrix[:, interaction_columns]
        observed = np.abs(interactions.sum(axis=0))
        exceeded = np.zeros(len(interaction_columns), dtype=int)
        for start in range(0, permutation_resamples, 256):
            size = min(256, permutation_resamples - start)
            signs = rng.integers(0, 2, size=(size, len(matrix)), dtype=np.int8) * 2 - 1
            simulated = np.abs(signs.astype(float) @ interactions)
            exceeded += (simulated >= observed - 1e-12).sum(axis=0)
        p = (exceeded + 1) / (permutation_resamples + 1)
        for k, column in enumerate(interaction_columns):
            for target, _ in refs[column]:
                if target.get("role") == "interaction":
                    target["raw_p"] = float(p[k]) if permutation_resamples else np.nan


def run_comparisons(records: pd.DataFrame, registry: list[dict], *, bootstrap_resamples: int = 5000,
                    seed: int = 42, include_subsets: bool = True,
                    availability_permutations: int = 10000, interaction_permutations: int = 10000) -> dict[str, pd.DataFrame]:
    """Analyze wide response features; candidate output is never promoted here."""
    _validate(records)
    if min(bootstrap_resamples, availability_permutations, interaction_permutations) < 0:
        raise ValueError("Resample counts must be nonnegative")
    unknown = set(records.persona) - set(PERSONAS)
    if unknown:
        raise ValueError(f"Unexpected personas: {sorted(unknown)}")
    frame = records.copy()
    if "text_available" not in frame or "text_status" not in frame:
        raise ValueError("text_available and text_status are required")
    if not frame.text_available.eq(frame.text_status.eq("substantive_explanation")).all():
        raise ValueError("text_available must equal substantive_explanation status")
    columns, outcomes = [], []
    for marker in registry:
        for measurement, suffix in MEASUREMENTS:
            column = f"{marker['marker_id']}__{suffix}"
            if column not in frame:
                raise ValueError(f"Missing marker measurement: {column}")
            columns.append(column)
            outcomes.append(dict(marker_id=marker["marker_id"], family=marker.get("family", "unspecified"),
                                 analysis_role=marker.get("analysis_role", marker.get("inferential_role", "descriptive")),
                                 rule_status=marker.get("rule_status", "candidate"), measurement=measurement))
    frame.loc[~_eligible(frame), columns] = np.nan
    comparisons, triplets, interactions, coverage, availability = [], [], [], [], []
    specifications = make_contrast_specs()
    for model_id, model_frame in frame.groupby("model_id", sort=True):
        model_seed = int.from_bytes(hashlib.sha256(f"{seed}|{model_id}".encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(model_seed)
        scopes = [("overall", model_frame)]
        if include_subsets:
            scopes += list(model_frame.groupby("subset", sort=True))
        for scope, scope_frame in scopes:
            availability_rows = _availability_tests(scope_frame, model_id, scope, availability_permutations, rng)
            for row in availability_rows:
                row.update(seed=seed, model_seed=f"0x{model_seed:016x}")
            availability.extend(availability_rows)
            key_index = pd.MultiIndex.from_frame(scope_frame[["subset", "question_id"]].drop_duplicates().sort_values(["subset", "question_id"]))
            strata = key_index.get_level_values("subset").to_numpy()
            arrays, source_n, source_present = {}, {}, {}
            for persona in PERSONAS:
                side = scope_frame[scope_frame.persona == persona]
                source_n[persona] = len(side)
                source_present[persona] = key_index.isin(side.set_index(["subset", "question_id"]).index)
                arrays[persona] = side.set_index(["subset", "question_id"])[columns].reindex(key_index).to_numpy(dtype=float)
            jobs = {}

            def enqueue(delta, mask, target, prefix=""):
                if not len(delta):
                    return
                if np.ptp(delta) == 0:
                    target[f"ci_low{prefix}"] = target[f"ci_high{prefix}"] = float(delta[0])
                    if target.get("role") == "interaction" and delta[0] != 0:
                        target["raw_p"] = min(1.0, 2.0 ** (1 - len(delta)))
                        target["test"] = "paired_sign_flip_exact_constant_difference"
                    return
                key = mask.tobytes()
                group = jobs.setdefault(key, {"strata": strata[mask], "deltas": [], "refs": [], "lookup": {}})
                fingerprint = delta.tobytes()
                if fingerprint in group["lookup"]:
                    group["refs"][group["lookup"][fingerprint]].append((target, prefix))
                else:
                    group["lookup"][fingerprint] = len(group["deltas"])
                    group["deltas"].append(delta)
                    group["refs"].append([(target, prefix)])

            for spec in specifications:
                interaction = spec["kind"] != "pairwise"
                if interaction:
                    names = [f"{g}_{c}" for c in (spec["left"], spec["right"]) for g in ("woman", "man")]
                    valid_left = np.isfinite(arrays[names[0]]) & np.isfinite(arrays[names[1]])
                    valid_right = np.isfinite(arrays[names[2]]) & np.isfinite(arrays[names[3]])
                    left_array, right_array = arrays[names[0]] - arrays[names[1]], arrays[names[2]] - arrays[names[3]]
                else:
                    names = [spec["left"], spec["right"]]
                    left_array, right_array = arrays[names[0]], arrays[names[1]]
                    valid_left, valid_right = np.isfinite(left_array), np.isfinite(right_array)
                for j, outcome in enumerate(outcomes):
                    mask = valid_left[:, j] & valid_right[:, j]
                    left, right = left_array[mask, j], right_array[mask, j]
                    result = dict(model_id=model_id, scope=scope, **outcome, **spec,
                                  **_effect_summary(left, right, outcome["measurement"], interaction=interaction))
                    subset_counts = {str(s): int(np.sum(strata[mask] == s)) for s in np.unique(strata)}
                    n_pairs = int(mask.sum())
                    result.update(sample_scope="common_quadruplet" if interaction else "pairwise_complete",
                                  n_left_available=int(valid_left[:, j].sum()), n_right_available=int(valid_right[:, j].sum()),
                                  n_left_total=int(np.sum(source_present[names[0]] & source_present[names[1]])) if interaction else source_n[names[0]],
                                  n_right_total=int(np.sum(source_present[names[2]] & source_present[names[3]])) if interaction else source_n[names[1]],
                                  n_pairs_by_subset=json.dumps(subset_counts, sort_keys=True),
                                  effective_subset_weights=json.dumps({s: n / n_pairs if n_pairs else None for s, n in subset_counts.items()}, sort_keys=True),
                                  bootstrap_resamples=bootstrap_resamples, seed=seed, model_seed=f"0x{model_seed:016x}",
                                  permutation_resamples=interaction_permutations if interaction else 0,
                                  overall_weighting="actual_eligible_question_composition")
                    for side in ("left", "right"):
                        total = result[f"n_{side}_total"]
                        result[f"{side}_availability_rate"] = result[f"n_{side}_available"] / total if total else np.nan
                        result[f"n_{side}_unpaired"] = result[f"n_{side}_available"] - n_pairs
                    enqueue((left - right) * _measurement_info(outcome["measurement"])[0], mask, result)
                    (interactions if interaction else comparisons).append(result)
                    coverage.append({k: v for k, v in result.items() if k in ("model_id", "scope", "marker_id", "measurement", "contrast_id", "role", "sample_scope", "n_pairs_by_subset", "effective_subset_weights") or k.startswith("n_") or k.endswith("availability_rate")})
            for condition in CONDITIONS:
                names = [f"woman_{condition}", f"man_{condition}", "neutral"]
                valid = np.logical_and.reduce([np.isfinite(arrays[name]) for name in names])
                for j, outcome in enumerate(outcomes):
                    mask = valid[:, j]
                    result = dict(model_id=model_id, scope=scope, condition=condition, **outcome,
                                  n_triplets=int(mask.sum()), sample_scope="common_triplet",
                                  bootstrap_resamples=bootstrap_resamples, seed=seed, model_seed=f"0x{model_seed:016x}", ci_method="subset_stratified_paired_percentile",
                                  ci_level=0.95, ci_simultaneous=False, effect_direction="left_minus_right")
                    factor, unit, estimand = _measurement_info(outcome["measurement"])
                    result.update(measurement_unit=unit, estimand=estimand, ci_estimand=estimand,
                                  n_triplets_by_subset=json.dumps({str(s): int(np.sum(strata[mask] == s)) for s in np.unique(strata)}, sort_keys=True))
                    for li, ri, suffix in ((0, 2, "woman_neutral"), (1, 2, "man_neutral"), (0, 1, "woman_man")):
                        delta = (arrays[names[li]][mask, j] - arrays[names[ri]][mask, j]) * factor
                        result[f"d_{suffix}"] = float(delta.mean()) if len(delta) else np.nan
                        result[f"ci_low_{suffix}"] = result[f"ci_high_{suffix}"] = np.nan
                        enqueue(delta, mask, result, "_" + suffix)
                    triplets.append(result)
            _finish_jobs(jobs, bootstrap_resamples, interaction_permutations, rng)
    availability_frame = pd.DataFrame(availability)
    if not availability_frame.empty:
        availability_frame["family_id"] = availability_frame.model_id.astype(str) + "|availability_all_scopes"
        availability_frame["adjusted_p"] = np.nan
        availability_frame["family_size"] = 0
        for _, indices in availability_frame.groupby("family_id", sort=False).groups.items():
            availability_frame.loc[indices, "adjusted_p"] = _holm(availability_frame.loc[indices, "raw_p"])
            availability_frame.loc[indices, "family_size"] = len(indices)
    return {"comparisons": adjust_families(pd.DataFrame(comparisons)),
            "neutral_triplet_comparisons": pd.DataFrame(triplets),
            "interaction_comparisons": adjust_families(pd.DataFrame(interactions)),
            "pairing_coverage": pd.DataFrame(coverage), "availability_tests": availability_frame}

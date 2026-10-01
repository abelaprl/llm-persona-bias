"""Blind audit packets and conservative, signed human-evidence gates.

Benchmark answer labels are never selected or evaluated here. Reviewer worksheets
refer to a deduplicated full-text corpus, and the selection key stays separate.
CI claims are deliberately stricter than the operational promotion decision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta

from .contracts import AuditDecision


KEYS = ["model_id", "persona", "subset", "question_id"]
REVIEW_FIELDS = ["reviewer_type", "reviewer_id", "reviewed_at", "full_response_read",
                 "notes", "adjudicated", "adjudicator_id", "adjudicated_at"]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_split(subset: str, question_id: str, seed: int = 42,
                 previously_read: Iterable | None = None) -> str:
    """Assign whole question clusters; previously inspected questions are discovery."""
    if previously_read is not None:
        excluded = set(tuple(v) if isinstance(v, list) else v for v in previously_read)
        if (str(subset), str(question_id)) in excluded or str(question_id) in excluded:
            return "discovery"
    bucket = int(_hash(f"{seed}|{subset}|{question_id}"), 16) % 100
    return "discovery" if bucket < 60 else "development" if bucket < 80 else "held_out"


def _registry_rules(registry, spans=None) -> pd.DataFrame:
    if isinstance(registry, dict):
        registry = registry.get("markers", [])
    rows = []
    for marker in registry or []:
        marker_id = marker["marker_id"]
        patterns = marker.get("patterns") or [{"rule_id": marker_id}]
        for i, pattern in enumerate(patterns):
            pattern = pattern if isinstance(pattern, dict) else {"pattern": pattern}
            rows.append({"marker_id": marker_id,
                         "rule_id": pattern.get("rule_id", f"{marker_id}.pattern_{i + 1}"),
                         "rule_kind": pattern.get("rule_kind", marker.get("rule_kind", "semantic")),
                         "rule_definition": marker.get("definition", marker.get("operational_definition",
                                               marker.get("description", ""))),
                         "rule_note": pattern.get("note", ""),
                         "reference_surface_name": {"lexical": "explanation_text", "structure": "structure_text", "visible": "visible_text"}.get(marker.get("surface", "lexical"), "explanation_text"),
                         "registry_rule_status": marker.get("rule_status", "candidate")})
    known = {r["rule_id"] for r in rows}
    if spans is not None and not spans.empty:
        for marker_id, rule_id in spans[["marker_id", "rule_id"]].drop_duplicates().itertuples(index=False, name=None):
            if rule_id not in known:
                rows.append({"marker_id": marker_id, "rule_id": rule_id, "rule_kind": "semantic",
                             "rule_definition": "", "rule_note": "", "registry_rule_status": "candidate"})
                known.add(rule_id)
    return pd.DataFrame(rows, columns=["marker_id", "rule_id", "rule_kind", "rule_definition", "rule_note", "reference_surface_name", "registry_rule_status"])


def _random_sample(frame: pd.DataFrame, n: int, seed: int, salt: str) -> pd.DataFrame:
    state = int(_hash(f"{seed}|{salt}")[:8], 16)
    return frame.sample(n=min(n, len(frame)), random_state=state).copy()


def _blank_review(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in REVIEW_FIELDS:
        frame[field] = "human" if field == "reviewer_type" else ""
    return frame


def sample_audit(records: pd.DataFrame, spans: pd.DataFrame, seed: int = 42,
                 registry=None, previously_read: Iterable | None = None,
                 recall_per_stratum: int = 6) -> dict[str, pd.DataFrame]:
    """Sample discovery, precision, and hit-independent recall; leave labels blank.

    Recall samples availability strata separately, then records their inclusion
    probabilities. Detector hits never influence recall selection. Precision uses
    development hits, excluding questions selected for full-response discovery.
    A small eligible development pool never silently reduces the promotion quota.
    """
    if recall_per_stratum < 3:
        raise ValueError("recall_per_stratum must be at least 3")
    allowed = KEYS + ["raw_text", "visible_text", "explanation_text", "structure_text", "text_status", "text_available", "word_count"]
    rec = records[[c for c in allowed if c in records]].copy()
    for key in KEYS:
        rec[key] = rec[key].astype(str)
    if rec.duplicated(KEYS).any():
        raise ValueError("Audit requires unique response keys")
    rec = rec.sort_values(KEYS).reset_index(drop=True)
    rec["response_id"] = [_hash("|".join(row))[:24] for row in rec[KEYS].itertuples(index=False, name=None)]
    exclusions = set(tuple(x) if isinstance(x, list) else x for x in (previously_read or []))
    split_cache = {(s, q): assign_split(s, q, seed, exclusions)
                   for s, q in rec[["subset", "question_id"]].drop_duplicates().itertuples(index=False, name=None)}
    rec["split"] = [split_cache[(s, q)] for s, q in rec[["subset", "question_id"]].itertuples(index=False, name=None)]
    span_cols = KEYS + ["marker_id", "rule_id", "start", "end", "text", "surface_name", "flags", "rule_status"]
    hits = spans.reindex(columns=span_cols).copy()
    for key in KEYS:
        hits[key] = hits[key].astype(str)
    hits = hits.merge(rec[KEYS + ["response_id", "split"]], on=KEYS, how="left", validate="many_to_one")
    if hits.response_id.isna().any():
        raise ValueError("A detector span has no corresponding response")
    hits = hits.sort_values(KEYS + ["rule_id", "start", "end"]).reset_index(drop=True)
    hits["span_id"] = [_hash(f"{r.response_id}|{r.rule_id}|{r.surface_name}|{r.start}|{r.end}")[:24]
                       for r in hits.itertuples()]
    if hits.span_id.duplicated().any():
        raise ValueError("Duplicate evidence spans cannot be counted as independent audit units")
    rules = _registry_rules(registry, hits)
    strata = rec[["model_id", "persona", "subset"]].drop_duplicates()
    coverage = strata.merge(rules[["marker_id", "rule_id"]], how="cross")
    counts = hits.groupby(["model_id", "persona", "subset", "marker_id", "rule_id"], observed=True).agg(
        total_hits=("span_id", "size"), response_hits=("response_id", "nunique"), question_hits=("question_id", "nunique")).reset_index()
    coverage = coverage.merge(counts, how="left").fillna({"total_hits": 0, "response_hits": 0, "question_hits": 0})
    for col in ("total_hits", "response_hits", "question_hits"):
        coverage[col] = coverage[col].astype(int)

    discovery = []
    for (model, rule), group in hits[hits.split == "discovery"].groupby(["model_id", "rule_id"], sort=True):
        group = group.drop_duplicates("response_id")
        chosen = _random_sample(group, 10, seed, f"discovery|{model}|{rule}")
        chosen["sampling_probability"] = len(chosen) / len(group)
        discovery.append(chosen)
    observed_groups = set(hits[["model_id", "rule_id"]].itertuples(index=False, name=None))
    discovery_group_counts = {(str(frame.model_id.iloc[0]), str(frame.rule_id.iloc[0])): len(frame)
                              for frame in discovery if len(frame)}
    for model, rule in sorted(observed_groups):
        needed = 10 - discovery_group_counts.get((str(model), str(rule)), 0)
        if needed > 0:
            group = hits[(hits.model_id == model) & (hits.rule_id == rule) & hits.split.eq("development")].drop_duplicates("response_id")
            chosen = _random_sample(group, needed, seed, f"discovery_supplement|{model}|{rule}")
            chosen["sampling_probability"] = len(chosen) / len(group) if len(group) else np.nan
            if len(chosen):
                discovery.append(chosen)
    discovery = pd.concat(discovery, ignore_index=True) if discovery else hits.iloc[:0].assign(sampling_probability=pd.Series(dtype=float))
    discovery["discovery_reader_type"] = ""
    discovery["discovery_reader_id"] = ""
    discovery["full_response_read"] = ""
    discovery["linguistic_notes"] = ""
    discovery_questions = set(discovery[["subset", "question_id"]].itertuples(index=False, name=None))
    eligible_mask = hits.split.eq("development") & pd.Series(
        [(s, q) not in discovery_questions for s, q in hits[["subset", "question_id"]].itertuples(index=False, name=None)], index=hits.index, dtype=bool)
    precision = []
    totals = hits.groupby("rule_id").size().to_dict()
    audit_coverage = []
    for rule in rules.to_dict("records"):
        group = hits.loc[eligible_mask & hits.rule_id.eq(rule["rule_id"])]
        quota = 50 if rule["rule_kind"] == "explicit" else 100
        total = int(totals.get(rule["rule_id"], 0))
        required = min(quota, total)
        chosen = _random_sample(group, required, seed, f"precision|{rule['rule_id']}")
        chosen["sampling_probability"] = len(chosen) / len(group) if len(group) else np.nan
        chosen["required_sample"] = required
        chosen["total_hits"] = total
        precision.append(chosen)
        audit_coverage.append({**rule, "total_hits": total, "eligible_audit_hits": len(group),
                               "required_sample": required, "sampled_precision": len(chosen),
                               "n_reviewed": 0, "false_positive": np.nan,
                               "audit_status": "pending_human_audit" if total else "no_hits"})
    precision = pd.concat(precision, ignore_index=True) if precision else hits.iloc[:0].copy()
    for col in ("sampling_probability", "required_sample", "total_hits"):
        if col not in precision:
            precision[col] = pd.Series(dtype=float)
    precision = precision.merge(rules, on=["marker_id", "rule_id"], how="left")
    precision["annotation_id"] = "precision_" + precision.span_id.astype(str)
    precision["annotation_type"] = "precision"
    precision["is_true_positive"] = ""
    precision["adjudicated_is_true_positive"] = ""
    public_cols = ["annotation_id", "annotation_type", "response_id", "marker_id", "rule_id",
                   "rule_kind", "rule_definition", "rule_note", "start", "end", "text", "surface_name",
                   "required_sample", "total_hits", "sampling_probability", "is_true_positive", "adjudicated_is_true_positive"]
    precision_public = _blank_review(precision.reindex(columns=public_cols))

    recall_parts = []
    available_rec = rec[rec.split != "held_out"]
    for name, group in available_rec.groupby(["model_id", "persona", "subset", "text_available"], sort=True, dropna=False):
        selected = _random_sample(group, recall_per_stratum, seed, f"recall|{name}")
        selected["sampling_probability"] = len(selected) / len(group)
        recall_parts.append(selected)
    recall = pd.concat(recall_parts, ignore_index=True) if recall_parts else rec.iloc[:0].assign(sampling_probability=pd.Series(dtype=float))
    recall["has_detector_hit"] = recall.response_id.isin(set(hits.response_id))
    marker_defs = rules[["marker_id", "rule_definition", "reference_surface_name"]].drop_duplicates("marker_id")
    recall_public = recall[["response_id", "sampling_probability"]].merge(marker_defs, how="cross")
    recall_public["annotation_id"] = "recall_" + recall_public.response_id + "_" + recall_public.marker_id
    recall_public["annotation_type"] = "recall"
    for field in ("reference_spans_json", "reference_count", "reference_positive", "reference_complete",
                  "detected_count", "matched_count", "adjudicated_reference_count", "adjudicated_detected_count",
                  "adjudicated_matched_count"):
        recall_public[field] = ""
    recall_public = _blank_review(recall_public)
    selected_ids = set(discovery.response_id) | set(precision.response_id) | set(recall.response_id)
    lookup_cols = ["response_id", "raw_text", "visible_text", "explanation_text", "structure_text", "text_status", "text_available", "word_count"]
    lookup = rec.loc[rec.response_id.isin(selected_ids), [c for c in lookup_cols if c in rec]].copy()
    selection_cols = KEYS + ["response_id", "split", "text_available", "text_status"]
    split_counts = rec.groupby(["subset", "split"]).agg(total_responses=("response_id", "size"), unique_questions=("question_id", "nunique")).reset_index()
    return {"precision_reviewer_1": precision_public, "precision_reviewer_2": precision_public.copy(),
            "recall_reviewer_1": recall_public, "recall_reviewer_2": recall_public.copy(),
            "corpus_lookup": lookup, "private_response_key": rec[selection_cols],
            "precision_selection": precision[KEYS + ["annotation_id", "response_id", "split", "span_id", "sampling_probability"]],
            "recall_selection": recall[selection_cols + ["has_detector_hit", "sampling_probability"]],
            "discovery_samples": discovery.drop(columns=["text"], errors="ignore"),
            "rule_coverage": coverage, "rule_audit_summary": pd.DataFrame(audit_coverage),
            "rule_definitions": rules, "split_counts": split_counts}


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    return None


def _nonempty(value) -> bool:
    return pd.notna(value) and bool(str(value).strip())


def _count(value):
    try:
        result = float(value)
        return int(result) if np.isfinite(result) and result >= 0 and result.is_integer() else None
    except (TypeError, ValueError):
        return None


def _resolve_annotations(annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve two independent signed human reviews; never treat blanks as negatives."""
    resolved, agreement = [], []
    required = {"annotation_id", "marker_id", "annotation_type", "reviewer_type", "reviewer_id", "reviewed_at", "full_response_read"}
    if not required <= set(annotations.columns):
        return pd.DataFrame(), pd.DataFrame()
    annotations = annotations.loc[annotations.reviewer_type.eq("human") & annotations.reviewer_id.map(_nonempty)
                                  & annotations.reviewed_at.map(_nonempty)
                                  & annotations.full_response_read.map(lambda x: _boolean(x) is True)]
    for annotation_id, group in annotations.groupby("annotation_id", sort=True):
        signed = group.loc[group.reviewer_type.eq("human") & group.reviewer_id.map(_nonempty)
                           & group.reviewed_at.map(_nonempty) & group.full_response_read.map(lambda x: _boolean(x) is True)]
        if signed.empty or signed.reviewer_id.nunique() < 2 or signed.reviewer_id.duplicated().any():
            continue
        if signed.marker_id.nunique() != 1 or signed.annotation_type.nunique() != 1:
            continue
        kind = signed.iloc[0].annotation_type
        if kind == "recall":
            if ("reference_complete" not in signed or not signed.reference_complete.map(_boolean).eq(True).all()
                    or ("text_available" in signed and signed.text_available.map(_boolean).eq(False).any())):
                continue
        cols = ["is_true_positive"] if kind == "precision" else ["reference_count", "detected_count", "matched_count"]
        if kind not in {"precision", "recall"} or not set(cols) <= set(signed.columns):
            continue
        parsed = []
        for item in signed.to_dict("records"):
            values = tuple((_boolean if kind == "precision" else _count)(item.get(c)) for c in cols)
            if any(v is None for v in values):
                continue
            if kind == "recall" and values[2] > min(values[:2]):
                continue
            parsed.append(values)
        if len(parsed) != len(signed):
            continue
        agrees = len(set(parsed)) == 1
        base = signed.iloc[0].to_dict()
        agreement.append({"annotation_id": annotation_id, "marker_id": base["marker_id"],
                          "annotation_type": kind, "agrees": agrees,
                          "reviewers": signed.reviewer_id.nunique(),
                          "presence_disagreement": (len({v[0] > 0 for v in parsed}) != 1) if kind == "recall" else not agrees})
        if not agrees:
            adjudications = []
            for item in signed.to_dict("records"):
                if (_boolean(item.get("adjudicated")) is True and _nonempty(item.get("adjudicator_id"))
                        and _nonempty(item.get("adjudicated_at"))):
                    values = tuple((_boolean if kind == "precision" else _count)(item.get("adjudicated_" + c)) for c in cols)
                    if all(v is not None for v in values) and (kind == "precision" or values[2] <= min(values[:2])):
                        adjudications.append(values)
            if not adjudications or len(set(adjudications)) != 1:
                continue
            values = adjudications[0]
        else:
            values = parsed[0]
        base.update(dict(zip(cols, values)))
        base["human_review_complete"] = True
        base["pre_adjudication_agreement"] = agrees
        resolved.append(base)
    return pd.DataFrame(resolved), pd.DataFrame(agreement)


def _exact_ci(success: int, total: int, alpha: float = .05) -> tuple[float, float]:
    if not total:
        return np.nan, np.nan
    return (0.0 if success == 0 else float(beta.ppf(alpha / 2, success, total - success + 1)),
            1.0 if success == total else float(beta.ppf(1 - alpha / 2, success + 1, total - success)))


def _cluster_estimate(frame, numerator, denominator):
    """Cluster-bootstrap ratio plus explicit zero-error uncertainty guard.

    The zero-error exact bound is a conservative cluster-success diagnostic,
    not a design-valid confidence bound for an arbitrarily weighted span ratio.
    Certification is therefore disabled unless each cluster is one Bernoulli unit.
    """
    if frame.empty:
        return {"point": np.nan, "low": np.nan, "high": np.nan, "clusters": 0,
                "success": 0, "total": 0, "independent_bernoulli": False}
    frame = frame.copy()
    frame["_n"] = numerator(frame).astype(float)
    frame["_d"] = denominator(frame).astype(float)
    success, total = int(frame._n.sum()), int(frame._d.sum())
    if not total:
        return {"point": np.nan, "low": np.nan, "high": np.nan, "clusters": 0,
                "success": success, "total": total, "independent_bernoulli": False}
    probabilities = pd.to_numeric(frame.get("sampling_probability", pd.Series(1.0, index=frame.index)), errors="coerce")
    valid_probability = probabilities.notna().all() and probabilities.gt(0).all() and probabilities.le(1).all()
    frame["_w"] = 1 / probabilities if valid_probability else 1.0
    frame["_wn"], frame["_wd"] = frame._n * frame._w, frame._d * frame._w
    cluster_available = {"subset", "question_id"} <= set(frame.columns)
    frame["_cluster"] = frame.subset.astype(str) + "|" + frame.question_id.astype(str) if cluster_available else frame.annotation_id.astype(str)
    grouped = frame.groupby("_cluster")[["_n", "_d", "_wn", "_wd"]].sum()
    positive = grouped[grouped._d > 0]
    ncluster = len(positive)
    point = float(frame._wn.sum() / frame._wd.sum())
    counts = positive[["_wn", "_wd"]].to_numpy()
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(1000):
        resample = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        samples.append(resample[0] / resample[1])
    low, high = np.quantile(samples, [.025, .975]).tolist()
    bernoulli = bool(cluster_available and valid_probability and probabilities.nunique() == 1 and positive._d.eq(1).all())
    if bernoulli:
        low, high = _exact_ci(success, total)
    elif success == total:
        low = min(low, _exact_ci(ncluster, ncluster)[0])
    elif success == 0:
        high = max(high, _exact_ci(0, ncluster)[1])
    return {"point": point, "low": low, "high": high, "clusters": ncluster,
            "success": success, "total": total, "independent_bernoulli": bernoulli}


def evaluate_audit(annotations: pd.DataFrame, registry=None) -> pd.DataFrame:
    """Summarize marker occurrence and response-presence evidence, not answer accuracy."""
    resolved, agreement = _resolve_annotations(annotations)
    marker_ids = set(_registry_rules(registry).marker_id)
    if "marker_id" in annotations:
        marker_ids.update(annotations.marker_id.dropna().astype(str))
    rows = []
    for marker in sorted(marker_ids):
        selected = resolved.loc[resolved.marker_id.eq(marker)] if not resolved.empty else resolved
        precision_rows = selected.loc[selected.annotation_type.eq("precision")] if not selected.empty else selected
        recall_rows = selected.loc[selected.annotation_type.eq("recall")] if not selected.empty else selected
        precision = _cluster_estimate(precision_rows, lambda x: x.is_true_positive.astype(int), lambda x: pd.Series(1, index=x.index))
        recall = _cluster_estimate(recall_rows, lambda x: x.matched_count, lambda x: x.reference_count)
        presence_precision = _cluster_estimate(recall_rows, lambda x: (x.reference_count.gt(0) & x.detected_count.gt(0)).astype(int), lambda x: x.detected_count.gt(0).astype(int))
        presence_recall = _cluster_estimate(recall_rows, lambda x: (x.reference_count.gt(0) & x.detected_count.gt(0)).astype(int), lambda x: x.reference_count.gt(0).astype(int))
        agrees = agreement[agreement.marker_id.eq(marker)] if not agreement.empty else agreement
        frozen = (not selected.empty and "registry_frozen" in selected and selected.registry_frozen.map(_boolean).eq(True).all())
        heldout = not selected.empty and "split" in selected and selected.split.eq("held_out").all()
        independent_reference = (not recall_rows.empty and "reference_independent" in recall_rows and recall_rows.reference_independent.map(_boolean).eq(True).all())
        expected_annotations = annotations.loc[annotations.marker_id.eq(marker), "annotation_id"].nunique() if {"marker_id", "annotation_id"} <= set(annotations) else 0
        all_complete = len(selected) == expected_annotations and expected_annotations > 0
        target = (precision["low"] >= .99 and recall["low"] >= .99 and precision["independent_bernoulli"]
                  and recall["independent_bernoulli"] and frozen and heldout and independent_reference and all_complete)
        row = {"marker_id": marker, "n_reviewed_precision": len(precision_rows), "n_reviewed_recall": len(recall_rows),
               "true_positive": precision["success"], "false_positive": precision["total"] - precision["success"],
               "reference_positive_count": recall["total"], "matched_reference_count": recall["success"],
               "false_negative": recall["total"] - recall["success"],
               "precision": precision["point"], "precision_ci_low": precision["low"], "precision_ci_high": precision["high"],
               "recall": recall["point"], "recall_ci_low": recall["low"], "recall_ci_high": recall["high"],
               "precision_question_clusters": precision["clusters"], "recall_positive_question_clusters": recall["clusters"],
               "presence_precision": presence_precision["point"], "presence_recall": presence_recall["point"],
               "n_double_reviewed": len(agrees),
               "pre_adjudication_agreement": float(agrees.agrees.mean()) if len(agrees) else np.nan,
               "prevalence_disagreement": float(agrees.presence_disagreement.mean()) if len(agrees) else np.nan,
               "precision_target": .99, "recall_target": .99,
               "all_annotations_complete": all_complete,
               "target_status": "demonstrated_individual" if target else "not_demonstrated",
               "simultaneous_41_marker_claim": False,
               "ci_method": "question_cluster_bootstrap_with_zero_error_exact_guard; exact_if_one_Bernoulli_per_cluster",
               "ci_limitation": "Cluster zero-error guard is diagnostic; certification requires independent Bernoulli units and frozen held-out human reference."}
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_audit_groups(annotations: pd.DataFrame, registry=None,
                          group_columns=("model_id", "persona", "subset")) -> pd.DataFrame:
    """Expose per-group estimates after the blind worksheets regain private keys."""
    if not set(group_columns) <= set(annotations):
        return pd.DataFrame()
    summaries = []
    for key, group in annotations.groupby(list(group_columns), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        summary = evaluate_audit(group, registry)
        for name, value in zip(group_columns, key):
            summary[name] = value
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def aggregate_audit_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep descriptive micro totals separate from equal-marker macro estimates.

    These pooled audit-sample estimates never replace individual marker gates or
    claim population weighting when rule-specific sampling probabilities differ.
    """
    rows = []
    for metric, success_col, total_cols in (
            ("precision", "true_positive", ["true_positive", "false_positive"]),
            ("recall", "matched_reference_count", ["reference_positive_count"])):
        eligible = summary[summary[metric].notna()]
        total = int(eligible[total_cols].sum().sum())
        rows.append({"metric": metric, "aggregation": "micro_audit_sample", "estimate": float(eligible[success_col].sum() / total) if total else np.nan,
                     "positive_denominator": total, "markers_with_estimate": len(eligible), "target_status": "not_a_per_marker_claim"})
        rows.append({"metric": metric, "aggregation": "macro_equal_marker", "estimate": float(eligible[metric].mean()) if len(eligible) else np.nan,
                     "positive_denominator": np.nan, "markers_with_estimate": len(eligible), "target_status": "not_a_per_marker_claim"})
    return pd.DataFrame(rows)


def promotion_decisions(audit: pd.DataFrame) -> list[AuditDecision]:
    """Promote each rule only from complete double human reviews, never a summary."""
    if "rule_id" not in audit:
        return []
    resolved, _ = _resolve_annotations(audit)
    decisions = []
    for rule_id, group in audit.dropna(subset=["rule_id"]).groupby("rule_id", sort=True):
        precision_group = group[group.annotation_type.eq("precision")] if "annotation_type" in group else group.iloc[:0]
        reviewed = (resolved[(resolved.rule_id == rule_id) & (resolved.annotation_type == "precision")]
                    if not resolved.empty and "rule_id" in resolved else pd.DataFrame())
        n = len(reviewed)
        fp = int((~reviewed.is_true_positive.astype(bool)).sum()) if n else 0
        kind = group.rule_kind.iloc[0] if "rule_kind" in group else "semantic"
        quota = 50 if kind == "explicit" else 100
        totals = pd.to_numeric(group.get("total_hits", pd.Series(dtype=float)), errors="coerce").dropna()
        required = min(quota, int(totals.max())) if len(totals) else quota
        if "required_sample" in group:
            requested = pd.to_numeric(group.required_sample, errors="coerce").dropna()
            if len(requested):
                required = max(required, int(requested.max()))
        all_complete = len(precision_group) > 0 and n == precision_group.annotation_id.nunique()
        if fp:
            status, reason = "rejected", "False positive found; revise this rule and draw a fresh audit sample."
        elif not all_complete:
            status, reason = "candidate", "Incomplete, unsigned, duplicate, or unresolved independent human annotations."
        elif n < required or required == 0:
            status, reason = "candidate", f"Insufficient reviewed hits: {n}/{required}."
        else:
            status, reason = "accepted", "Operational quota met with resolved double human review and zero false positives; this is not a 99% quality claim."
        if "rule_status" in group and group.rule_status.eq("revoked").any():
            status, reason = "revoked", "Rule explicitly revoked; other rule decisions remain independent."
        if "rule_version" in group and group.rule_version.nunique() > 1:
            status, reason = "candidate", "Mixed rule revisions require a new isolated promotion audit."
        decisions.append(AuditDecision(str(rule_id), status, n, fp, reason))
    return decisions


def build_audit_package(records: pd.DataFrame, spans: pd.DataFrame, registry,
                        output_dir: str | Path, seed: int = 42,
                        previously_read: Iterable | None = None) -> dict:
    """Write a new packet; never overwrite any earlier audit file."""
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Audit output must be new or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    frames = sample_audit(records, spans, seed, registry=registry, previously_read=previously_read)
    empty_annotations = pd.concat([frames["precision_reviewer_1"], frames["recall_reviewer_1"]], ignore_index=True)
    frames["marker_audit_summary"] = evaluate_audit(empty_annotations, registry=registry)
    frames["audit_aggregate_summary"] = aggregate_audit_summary(frames["marker_audit_summary"])
    files = {}
    for name, frame in frames.items():
        path = target / f"{name}.csv"
        with path.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        files[name] = {"path": path.name, "rows": len(frame)}
    metadata = {"status": "candidate_pending_independent_human_audit", "seed": seed,
                "human_annotations_completed": 0, "precision_target": .99, "recall_target": .99,
                "marker_count": len(frames["marker_audit_summary"]), "files": files,
                "audit_summary_path": str(target / "marker_audit_summary.csv"),
                "previously_read_exclusion_count": len(list(previously_read or [])),
                "held_out_locked": True, "automatic_promotion": False,
                "instructions": [
                    "Berikan tiap penilai hanya worksheet miliknya, rule_definitions.csv, dan corpus_lookup.csv; kunci privat tidak dibagikan.",
                    "Baca respons penuh lewat response_id; jangan memakai GT/correctness atau anotasi agen sebagai reference manusia.",
                    "Anotasi marker hanya pada reference_surface_name: explanation_text untuk leksikal, structure_text untuk struktur, atau visible_text sesuai registry. raw_text hanya untuk audit transformasi; hidden reasoning bukan bahan anotasi marker.",
                    "Isi reviewer_id, reviewed_at, full_response_read dan anotasi; template human tanpa tanda tangan bukan audit selesai.",
                    "Recall: tandai semua reference_spans_json dan reference_count sebelum melihat detektor; matched_count/detected_count diisi setelah reference dibekukan.",
                    "No-hit tidak berarti tidak ada konstruk; unavailable tidak boleh menjadi reference negatif linguistik.",
                    "Gabungkan dua worksheet dengan kunci privat sebelum evaluasi; disagreement memerlukan adjudikasi eksplisit dan bertanggal.",
                    "Held-out tetap terkunci sampai registry dibekukan; sampel development tidak membuktikan target held-out 99%.",
                    "CI cluster bootstrap pada zero-error memakai exact guard diagnostik; tanpa unit independen jangan mengklaim sertifikasi 99%.",
                    "Eksklusi soal yang telah dibaca harus lengkap dari manifest eksplorasi sebelumnya; default kosong bukan bukti held-out belum terpapar."]}
    with (target / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata

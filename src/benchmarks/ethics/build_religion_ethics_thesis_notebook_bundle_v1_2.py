"""Build the thesis-facing ETHICS Religion notebook bundle v1.2.

The marker registry is still a candidate.  This module therefore produces
paired descriptive marker tables only.  It does calculate the parser-status
chi-square audit because that test concerns measurement availability, not the
validity of a linguistic-marker dictionary.

All outputs are write-once.  The canonical response and marker bundles are
read-only and their SHA-256 hashes are pinned in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, rankdata


BUNDLE_VERSION = "religion-ethics-thesis-notebook-bundle-v1.2.0"
EXPECTED_PIPELINE_VERSION = "ethics-religion-lm-v1.1.10-candidate"
EXPECTED_INPUT_SHA256 = "c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a"

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
MODEL_COLORS = dict(
    zip(MODELS, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
)
IDENTITIES = ("islam", "christian", "jewish", "atheist")
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Kristen",
    "jewish": "Yahudi",
    "atheist": "Ateis",
}
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = {
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
SUBSETS = ("commonsense", "deontology", "justice", "virtue")
CONSTRUCT_VALID_SUBSETS = ("commonsense", "justice")
DIRECT_PAIRS = tuple(combinations(IDENTITIES, 2))
PAIR_KEYS = ("model_id", "subset", "question_id")

RELIGION_LABELS = {
    "religion_identity_islam": "Identitas Islam",
    "religion_identity_christian": "Identitas Kristen",
    "religion_identity_jewish": "Identitas Yahudi",
    "religion_identity_atheist": "Identitas ateis",
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Tuhan/deity",
    "religion_figure_reference": "Tokoh agama",
    "religion_scripture_reference": "Kitab/teks agama",
    "religion_practice_community": "Praktik/komunitas",
    "religion_afterlife_soteriology": "Akhirat/soteriologi",
    "religion_named_concept_reference": "Konsep agama khusus",
    "religion_formulaic_expression": "Formula agama",
    "religion_persona_direct_address": "Sapaan persona",
    "religion_perspective_adoption": "Adopsi perspektif",
    "religion_moral_authority_framing": "Otoritas moral agama",
    "religion_secular_humanist_framing": "Kerangka sekuler",
    "religion_pluralist_framing": "Kerangka pluralis",
    "religion_identity_irrelevance": "Identitas tidak relevan",
    "religion_blessing_invocation": "Doa/berkat",
    "religion_respect_acknowledgement": "Pengakuan/respek",
}
GENERAL_LABELS = {
    "organization_explicit_step": "Langkah eksplisit",
    "organization_bullet": "Bullet",
    "organization_section_header": "Section header",
    "stance_hedge": "Hedge",
    "stance_booster": "Booster",
    "stance_assumption": "Asumsi",
    "stance_self_correction": "Self-correction",
    "stance_contextual_hedge": "Hedge kontekstual",
    "stance_distancing": "Distancing",
    "connective_causal": "Konektif kausal",
    "connective_contrastive": "Konektif kontras",
    "connective_conditional": "Konektif kondisional",
    "connective_additive": "Konektif aditif",
    "connective_concessive": "Konektif konsesif",
    "self_reference_first_person": "Orang pertama",
    "self_reference_inclusive_we": "Inclusive we",
    "engagement_reader_directive": "Direktif pembaca",
    "affect_empathy_validation": "Empati/validasi",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_series(values: pd.Series) -> pd.Series:
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


def _nullable_bool(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.casefold()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )


def _holm_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def _rank_biserial(difference: np.ndarray) -> tuple[int, int, float]:
    valid = difference[np.isfinite(difference)]
    nonzero = valid[~np.isclose(valid, 0.0, rtol=0.0, atol=1e-12)]
    if not len(nonzero):
        return len(valid), 0, 0.0 if len(valid) else math.nan
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return len(valid), len(nonzero), (positive - negative) / float(ranks.sum())


def _load_sources(
    marker_dir: Path,
    canonical_csv: Path,
    audit_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    marker_manifest_path = marker_dir / "manifest.json"
    registry_path = marker_dir / "marker_registry.csv"
    response_path = marker_dir / "response_level_markers.csv.gz"
    for path in (marker_manifest_path, registry_path, response_path, canonical_csv, audit_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    marker_manifest = json.loads(marker_manifest_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if marker_manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError(f"Unexpected marker pipeline: {marker_manifest.get('pipeline_version')}")
    if marker_manifest.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("Marker input hash does not match the pinned canonical source.")
    if _sha256(canonical_csv) != EXPECTED_INPUT_SHA256:
        raise ValueError("Canonical CSV hash does not match the pinned source.")
    if marker_manifest.get("inference_run") is not False:
        raise ValueError("Candidate marker input unexpectedly contains inference.")

    registry = pd.read_csv(registry_path, low_memory=False)
    religion_markers = registry.loc[
        registry["marker_id"].str.startswith("religion_"), "marker_id"
    ].tolist()
    general_markers = registry.loc[
        ~registry["marker_id"].str.startswith("religion_"), "marker_id"
    ].tolist()
    if len(religion_markers) != 20 or len(general_markers) != 30:
        raise ValueError(
            f"Expected 20 Religion and 30 general markers; got {len(religion_markers)} and {len(general_markers)}."
        )

    marker_columns = [
        f"{marker}__{suffix}"
        for marker in (*religion_markers, *general_markers)
        for suffix in ("count", "density", "present")
    ]
    response_usecols = [
        "model_id",
        "persona",
        "identity",
        "condition",
        "question_id",
        "subset",
        "text_available",
        "word_count",
        *marker_columns,
    ]
    print(f"[1/8] Reading marker response surface: {response_path}")
    frame = pd.read_csv(response_path, usecols=response_usecols, low_memory=False)
    frame["text_available"] = _bool_series(frame["text_available"])
    numeric_columns = ["word_count", *marker_columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    canonical_usecols = [
        "model",
        "persona_folder",
        "question_id",
        "answer_status",
        "parse_source",
        "extraction_confidence",
        "numeric_semantic_conflict",
        "is_correct",
        "extracted_stance",
        "construct_valid",
    ]
    print(f"[2/8] Reading canonical scoring provenance: {canonical_csv}")
    scoring = pd.read_csv(canonical_csv, usecols=canonical_usecols, low_memory=False).rename(
        columns={"model": "model_id", "persona_folder": "persona"}
    )
    scoring["numeric_semantic_conflict"] = _bool_series(scoring["numeric_semantic_conflict"])
    scoring["is_correct"] = _nullable_bool(scoring["is_correct"])
    scoring["construct_valid"] = _nullable_bool(scoring["construct_valid"])

    key = ["model_id", "persona", "question_id"]
    if frame.duplicated(key).any() or scoring.duplicated(key).any():
        raise ValueError("Duplicate model/persona/question key in an input source.")
    frame = frame.merge(scoring, on=key, how="left", validate="one_to_one")
    if len(frame) != 154_428 or frame[key].isna().any().any():
        raise ValueError("Merged response count or keys do not match the canonical corpus.")
    if set(frame["model_id"].unique()) != set(MODELS):
        raise ValueError("Unexpected model set.")
    if set(frame["subset"].unique()) != set(SUBSETS):
        raise ValueError("Unexpected ETHICS subset set.")

    semantic = frame["extraction_confidence"].eq("semantic_unaudited")
    analysis_status = frame["answer_status"].astype("string").copy()
    analysis_status.loc[semantic & analysis_status.eq("committed")] = "abstain"
    frame["candidate_uncommitted"] = semantic
    frame["analysis_answer_status"] = analysis_status
    frame["validated_committed"] = analysis_status.eq("committed")
    frame["analysis_is_correct"] = frame["is_correct"].where(frame["validated_committed"])
    if frame.loc[semantic, "analysis_answer_status"].eq("committed").any():
        raise ValueError("A semantic_unaudited row remains committed in the analysis view.")
    return frame, registry, marker_manifest, audit_manifest


def _analysis_status_view(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_id",
        "persona",
        "identity",
        "condition",
        "question_id",
        "subset",
        "answer_status",
        "analysis_answer_status",
        "candidate_uncommitted",
        "validated_committed",
        "parse_source",
        "extraction_confidence",
        "numeric_semantic_conflict",
        "is_correct",
        "analysis_is_correct",
        "extracted_stance",
        "construct_valid",
        "text_available",
    ]
    return frame[columns].copy()


def _scoring_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        committed = group["analysis_answer_status"].eq("committed")
        abstain = group["analysis_answer_status"].eq("abstain")
        missing = group["analysis_answer_status"].eq("missing")
        total = len(group)
        n_committed = int(committed.sum())
        construct_valid = bool(group["construct_valid"].fillna(False).all())
        n_correct = int((committed & group["is_correct"].fillna(False)).sum())
        rows.append(
            {
                "model_id": model_id,
                "persona": persona,
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
                "numeric_semantic_conflict": int((committed & group["numeric_semantic_conflict"]).sum()),
                "n_correct": n_correct if construct_valid else pd.NA,
                "accuracy_committed": n_correct / n_committed if construct_valid and n_committed else math.nan,
                "pct_answer_wrong": float(group.loc[committed, "extracted_stance"].eq("wrong").mean())
                if n_committed
                else math.nan,
                "pct_answer_acceptable": float(
                    group.loc[committed, "extracted_stance"].eq("acceptable").mean()
                )
                if n_committed
                else math.nan,
                "construct_valid": construct_valid,
            }
        )
    return pd.DataFrame(rows)


def _semantic_inventory(frame: pd.DataFrame) -> pd.DataFrame:
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


def _conflict_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        committed = group["validated_committed"]
        conflict = committed & group["numeric_semantic_conflict"]
        retained = committed & ~group["numeric_semantic_conflict"]
        construct_valid = subset in CONSTRUCT_VALID_SUBSETS
        if construct_valid:
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


def _persona_independence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, scoped in (
        ("religion_only", frame[frame["identity"].isin(IDENTITIES)]),
        ("religion_plus_neutral", frame),
    ):
        for model_id, group in scoped.groupby("model_id", observed=True, sort=False):
            table = pd.crosstab(group["persona"], group["analysis_answer_status"])
            table = table.loc[:, table.sum(axis=0).gt(0)]
            statistic, p_raw, dof, _ = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            denominator = n * (min(table.shape) - 1)
            rows.append(
                {
                    "scope": scope,
                    "model_id": model_id,
                    "n": n,
                    "personas": table.shape[0],
                    "status_categories": table.shape[1],
                    "chi2": statistic,
                    "dof": dof,
                    "p_raw": p_raw,
                    "cramers_v": math.sqrt(statistic / denominator) if denominator else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby("scope", group_keys=False)["p_raw"].apply(_holm_adjust)
    result["significant_holm_0_05"] = result["p_holm"].lt(0.05)
    return result


def _scope_groups(frame: pd.DataFrame):
    yield "all", frame
    for subset in SUBSETS:
        yield subset, frame[frame["subset"].eq(subset)]


def _paired_table_rows(
    joined: pd.DataFrame,
    markers: list[str],
    registry_lookup: pd.DataFrame,
    comparison_meta: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lookup = registry_lookup.set_index("marker_id")
    for model_id, model_rows in joined.groupby("model_id", observed=True, sort=False):
        for subset, scoped in _scope_groups(model_rows):
            available = scoped[
                scoped["text_available_left"].fillna(False)
                & scoped["text_available_right"].fillna(False)
            ]
            for marker in markers:
                left_count = pd.to_numeric(available[f"{marker}__count_left"], errors="coerce").to_numpy(float)
                right_count = pd.to_numeric(available[f"{marker}__count_right"], errors="coerce").to_numpy(float)
                left_density = pd.to_numeric(
                    available[f"{marker}__density_left"], errors="coerce"
                ).to_numpy(float)
                right_density = pd.to_numeric(
                    available[f"{marker}__density_right"], errors="coerce"
                ).to_numpy(float)
                left_present = pd.to_numeric(
                    available[f"{marker}__present_left"], errors="coerce"
                ).to_numpy(float)
                right_present = pd.to_numeric(
                    available[f"{marker}__present_right"], errors="coerce"
                ).to_numpy(float)
                valid = (
                    np.isfinite(left_count)
                    & np.isfinite(right_count)
                    & np.isfinite(left_density)
                    & np.isfinite(right_density)
                    & np.isfinite(left_present)
                    & np.isfinite(right_present)
                )
                left_count = left_count[valid]
                right_count = right_count[valid]
                left_density = left_density[valid]
                right_density = right_density[valid]
                left_present = left_present[valid]
                right_present = right_present[valid]
                valid_rows = available.iloc[np.flatnonzero(valid)]
                density_difference = left_density - right_density
                n_pairs, n_nonzero, effect = _rank_biserial(density_difference)
                left_words = float(pd.to_numeric(valid_rows["word_count_left"], errors="coerce").sum())
                right_words = float(pd.to_numeric(valid_rows["word_count_right"], errors="coerce").sum())
                marker_meta = lookup.loc[marker]
                rows.append(
                    {
                        **comparison_meta,
                        "model_id": model_id,
                        "subset": subset,
                        "marker_id": marker,
                        "family": marker_meta["family"],
                        "analysis_role": marker_meta["analysis_role"],
                        "measurement": marker_meta["measurement"],
                        "n_pairs": n_pairs,
                        "n_nonzero_density_difference": n_nonzero,
                        "left_raw_occurrences": int(left_count.sum()),
                        "right_raw_occurrences": int(right_count.sum()),
                        "left_total_words": int(left_words),
                        "right_total_words": int(right_words),
                        "left_occurrences_per_100_words": 100 * left_count.sum() / left_words
                        if left_words
                        else math.nan,
                        "right_occurrences_per_100_words": 100 * right_count.sum() / right_words
                        if right_words
                        else math.nan,
                        "left_mean_response_density": float(np.mean(left_density)) if n_pairs else math.nan,
                        "right_mean_response_density": float(np.mean(right_density)) if n_pairs else math.nan,
                        "mean_response_density_delta": float(np.mean(density_difference))
                        if n_pairs
                        else math.nan,
                        "median_response_density_delta": float(np.median(density_difference))
                        if n_pairs
                        else math.nan,
                        "density_rank_biserial": effect,
                        "left_positive_responses": int(left_present.sum()),
                        "right_positive_responses": int(right_present.sum()),
                        "left_prevalence_pct": 100 * float(np.mean(left_present)) if n_pairs else math.nan,
                        "right_prevalence_pct": 100 * float(np.mean(right_present)) if n_pairs else math.nan,
                        "prevalence_delta_pp": 100 * float(np.mean(left_present - right_present))
                        if n_pairs
                        else math.nan,
                    }
                )
    return rows


def _paired_marker_tables(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    religion = registry.loc[registry["marker_id"].str.startswith("religion_"), "marker_id"].tolist()
    general = registry.loc[~registry["marker_id"].str.startswith("religion_"), "marker_id"].tolist()
    value_columns = [
        f"{marker}__{suffix}"
        for marker in (*religion, *general)
        for suffix in ("count", "density", "present")
    ]
    columns = [*PAIR_KEYS, "text_available", "word_count", *value_columns]
    neutral = frame[frame["condition"].eq("neutral")][columns]
    within_religion: list[dict[str, object]] = []
    within_general: list[dict[str, object]] = []
    direct_religion: list[dict[str, object]] = []
    direct_general: list[dict[str, object]] = []

    comparison_number = 0
    for identity in IDENTITIES:
        for condition in CONDITIONS:
            comparison_number += 1
            print(f"[3/8] Persona vs Neutral {comparison_number}/16: {identity} {condition}")
            left = frame[
                frame["identity"].eq(identity) & frame["condition"].eq(condition)
            ][columns]
            joined = left.merge(neutral, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            meta = {
                "comparison_type": "persona_vs_neutral",
                "identity": identity,
                "identity_contrast": pd.NA,
                "condition": condition,
                "left_label": f"{identity}_{condition}",
                "right_label": "neutral",
            }
            within_religion.extend(_paired_table_rows(joined, religion, registry, meta))
            within_general.extend(_paired_table_rows(joined, general, registry, meta))

    comparison_number = 0
    for condition in CONDITIONS:
        for left_identity, right_identity in DIRECT_PAIRS:
            comparison_number += 1
            contrast = f"{left_identity}_vs_{right_identity}"
            print(f"[4/8] Direct identity {comparison_number}/24: {contrast} {condition}")
            left = frame[
                frame["identity"].eq(left_identity) & frame["condition"].eq(condition)
            ][columns]
            right = frame[
                frame["identity"].eq(right_identity) & frame["condition"].eq(condition)
            ][columns]
            joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            meta = {
                "comparison_type": "direct_identity",
                "identity": pd.NA,
                "identity_contrast": contrast,
                "condition": condition,
                "left_label": left_identity,
                "right_label": right_identity,
            }
            direct_religion.extend(_paired_table_rows(joined, religion, registry, meta))
            direct_general.extend(_paired_table_rows(joined, general, registry, meta))

    return tuple(
        pd.DataFrame(rows)
        for rows in (within_religion, direct_religion, within_general, direct_general)
    )


def _correctness_retention(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *PAIR_KEYS,
        "text_available",
        "validated_committed",
        "analysis_is_correct",
    ]
    neutral = frame[frame["condition"].eq("neutral")][columns]
    rows: list[dict[str, object]] = []

    def append_rows(joined: pd.DataFrame, meta: dict[str, object]) -> None:
        for (model_id, subset), group in joined.groupby(["model_id", "subset"], observed=True):
            if subset not in CONSTRUCT_VALID_SUBSETS:
                continue
            text_pair = group["text_available_left"] & group["text_available_right"]
            committed_pair = group["validated_committed_left"] & group["validated_committed_right"]
            correct_pair = (
                committed_pair
                & group["analysis_is_correct_left"].fillna(False)
                & group["analysis_is_correct_right"].fillna(False)
            )
            rows.append(
                {
                    **meta,
                    "model_id": model_id,
                    "subset": subset,
                    "n_question_pairs": len(group),
                    "n_text_pairs": int(text_pair.sum()),
                    "n_both_validated_committed": int(committed_pair.sum()),
                    "n_both_correct": int(correct_pair.sum()),
                    "both_correct_retention_vs_questions": float(correct_pair.mean()),
                    "both_correct_retention_vs_text": float(correct_pair.sum() / text_pair.sum())
                    if text_pair.sum()
                    else math.nan,
                }
            )

    for identity in IDENTITIES:
        for condition in CONDITIONS:
            left = frame[
                frame["identity"].eq(identity) & frame["condition"].eq(condition)
            ][columns]
            joined = left.merge(neutral, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            append_rows(
                joined,
                {
                    "comparison_type": "persona_vs_neutral",
                    "identity": identity,
                    "identity_contrast": pd.NA,
                    "condition": condition,
                },
            )
    for condition in CONDITIONS:
        for left_identity, right_identity in DIRECT_PAIRS:
            left = frame[
                frame["identity"].eq(left_identity) & frame["condition"].eq(condition)
            ][columns]
            right = frame[
                frame["identity"].eq(right_identity) & frame["condition"].eq(condition)
            ][columns]
            joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            append_rows(
                joined,
                {
                    "comparison_type": "direct_identity",
                    "identity": pd.NA,
                    "identity_contrast": f"{left_identity}_vs_{right_identity}",
                    "condition": condition,
                },
            )
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.3,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, image_dir: Path, stem: str) -> tuple[Path, Path]:
    png = image_dir / f"{stem}.png"
    svg = image_dir / f"{stem}.svg"
    fig.savefig(png, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if png.stat().st_size < 10_000 or svg.stat().st_size < 5_000:
        raise ValueError(f"Rendered figure is unexpectedly small: {stem}")
    return png, svg


def _marker_label(marker: str) -> str:
    return RELIGION_LABELS.get(marker, GENERAL_LABELS.get(marker, marker.replace("_", " ")))


def _model_legend() -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODELS
    ]
    handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model"))
    return handles


def _plot_four_scenarios(
    data: pd.DataFrame,
    markers: list[str],
    value_column: str,
    title: str,
    xlabel: str,
    image_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    selected = data[data["subset"].eq("all") & data["marker_id"].isin(markers)].copy()
    limit = float(np.nanquantile(np.abs(selected[value_column]), 0.995)) if len(selected) else 1.0
    if value_column == "density_rank_biserial":
        limit = 1.0
    else:
        limit = max(1.0, min(100.0, 1.08 * limit))
    y_lookup = {marker: len(markers) - 1 - index for index, marker in enumerate(markers)}
    height = max(9.2, 0.47 * len(markers) + 2.7)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, height), sharex=True, sharey=True)
    for panel_index, (ax, condition) in enumerate(zip(axes.flat, CONDITIONS)):
        panel = selected[selected["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(
                point[value_column],
                y_lookup[point["marker_id"]],
                s=27,
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.88,
                edgecolors="white",
                linewidths=0.3,
            )
        medians = panel.groupby("marker_id")[value_column].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=32, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.75)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.55)
        ax.set_xlim(-limit, limit)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        if panel_index % 2 == 0:
            ax.set_yticks([y_lookup[marker] for marker in markers], [_marker_label(marker) for marker in markers])
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel(xlabel, y=0.088)
    fig.suptitle(title, y=0.995, fontweight="bold")
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3)
    fig.tight_layout(rect=(0.01, 0.14, 0.99, 0.96), h_pad=1.5, w_pad=1.0)
    return _save_figure(fig, image_dir, stem)


def _plot_status_independence(data: pd.DataFrame, image_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    offsets = {"religion_only": -0.13, "religion_plus_neutral": 0.13}
    colors = {"religion_only": "#2563EB", "religion_plus_neutral": "#D97706"}
    y_lookup = {model: len(MODELS) - 1 - index for index, model in enumerate(MODELS)}
    for scope in offsets:
        panel = data[data["scope"].eq(scope)]
        ax.scatter(
            panel["cramers_v"],
            [y_lookup[model] + offsets[scope] for model in panel["model_id"]],
            s=55,
            color=colors[scope],
            label="16 persona agama" if scope == "religion_only" else "16 persona + N",
        )
    ax.set_yticks([y_lookup[model] for model in MODELS], [MODEL_LABELS[model] for model in MODELS])
    ax.set_xlabel("Cramér's V: persona × answer status")
    ax.set_title("Ketergantungan status jawaban terhadap persona", fontweight="bold")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_figure(fig, image_dir, "01_persona_status_independence")


def _plot_conflict_sensitivity(data: pd.DataFrame, image_dir: Path) -> tuple[Path, Path]:
    summary = (
        data.groupby(["model_id", "subset"], as_index=False)["delta_excluding_minus_all_pp"]
        .median()
        .pivot(index="model_id", columns="subset", values="delta_excluding_minus_all_pp")
        .reindex(index=MODELS, columns=SUBSETS)
    )
    values = summary.to_numpy(float)
    limit = max(0.5, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(SUBSETS)), [name.title() for name in SUBSETS])
    ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[model] for model in MODELS])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8.8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Delta setelah konflik dikeluarkan (poin persentase)")
    ax.set_title("Sensitivity terhadap numeric-semantic conflict", fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, image_dir, "02_numeric_semantic_conflict_sensitivity")


def build_bundle(
    marker_dir: Path,
    canonical_csv: Path,
    audit_manifest_path: Path,
    output_dir: Path,
) -> Path:
    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    audit_manifest_path = audit_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()

    frame, registry, marker_manifest, audit_manifest = _load_sources(
        marker_dir, canonical_csv, audit_manifest_path
    )
    print("[3/8] Building conservative scoring and parser-audit tables")
    status_view = _analysis_status_view(frame)
    scoring = _scoring_summary(frame)
    semantic_inventory = _semantic_inventory(frame)
    conflicts = _conflict_sensitivity(frame)
    independence = _persona_independence(frame)

    print("[4/8] Building paired marker tables")
    within_religion, direct_religion, within_general, direct_general = _paired_marker_tables(
        frame, registry
    )
    print("[5/8] Building correctness-retention table")
    retention = _correctness_retention(frame)

    tables = {
        "analysis_status_view.csv.gz": status_view,
        "scoring_summary_validated_by_model_persona_subset.csv": scoring,
        "semantic_unaudited_inventory.csv": semantic_inventory,
        "numeric_semantic_conflict_sensitivity.csv": conflicts,
        "persona_status_independence.csv": independence,
        "religion_marker_vs_neutral_all_markers_descriptive.csv": within_religion,
        "religion_marker_direct_identity_all_markers_descriptive.csv": direct_religion,
        "general_marker_vs_neutral_all_markers_descriptive.csv": within_general,
        "general_marker_direct_identity_all_markers_descriptive.csv": direct_general,
        "correctness_pair_retention_validated.csv": retention,
        "marker_registry_snapshot.csv": registry,
    }
    for filename, table in tables.items():
        print(f"[6/8] Writing {filename}: {len(table):,} rows")
        table.to_csv(output_dir / filename, index=False, compression="infer")

    print("[7/8] Rendering thesis-readable figures")
    _style()
    figures: list[tuple[str, str]] = []

    def add(paths: tuple[Path, Path], description: str) -> None:
        png, svg = paths
        figures.append((png.name, svg.name, description))

    add(_plot_status_independence(independence, image_dir), "Persona × answer status; technical parser audit.")
    add(_plot_conflict_sensitivity(conflicts, image_dir), "Sensitivity after excluding numeric-semantic conflicts.")

    religion_markers = registry.loc[
        registry["marker_id"].str.startswith("religion_"), "marker_id"
    ].tolist()
    primary_general = registry.loc[
        ~registry["marker_id"].str.startswith("religion_")
        & registry["analysis_role"].eq("primary"),
        "marker_id",
    ].tolist()
    for identity in IDENTITIES:
        panel = within_religion[within_religion["identity"].eq(identity)]
        add(
            _plot_four_scenarios(
                panel,
                religion_markers,
                "prevalence_delta_pp",
                f"Persona {IDENTITY_LABELS[identity]} terhadap skenario netral",
                "Delta prevalensi marker agama (poin persentase; deskriptif)",
                image_dir,
                f"03_{identity}_religion_markers_vs_neutral",
            ),
            f"All 20 Religion markers for {identity} versus Neutral.",
        )
        general_panel = within_general[within_general["identity"].eq(identity)]
        add(
            _plot_four_scenarios(
                general_panel,
                primary_general,
                "density_rank_biserial",
                f"Penanda linguistik umum: {IDENTITY_LABELS[identity]} terhadap N",
                "Paired rank-biserial density per 100 kata (deskriptif)",
                image_dir,
                f"04_{identity}_general_markers_vs_neutral",
            ),
            f"Primary general markers for {identity} versus Neutral.",
        )

    for left_identity, right_identity in DIRECT_PAIRS:
        contrast = f"{left_identity}_vs_{right_identity}"
        label = f"{IDENTITY_LABELS[left_identity]} − {IDENTITY_LABELS[right_identity]}"
        panel = direct_religion[direct_religion["identity_contrast"].eq(contrast)]
        add(
            _plot_four_scenarios(
                panel,
                religion_markers,
                "prevalence_delta_pp",
                f"Perbandingan langsung {label}",
                "Delta prevalensi marker agama (poin persentase; deskriptif)",
                image_dir,
                f"05_{contrast}_religion_markers_direct",
            ),
            f"All 20 Religion markers for direct {contrast} comparison.",
        )
        general_panel = direct_general[direct_general["identity_contrast"].eq(contrast)]
        add(
            _plot_four_scenarios(
                general_panel,
                primary_general,
                "density_rank_biserial",
                f"Penanda linguistik umum: {label}",
                "Paired rank-biserial density per 100 kata (deskriptif)",
                image_dir,
                f"06_{contrast}_general_markers_direct",
            ),
            f"Primary general markers for direct {contrast} comparison.",
        )

    figure_index = pd.DataFrame(figures, columns=["png", "svg", "description"])
    figure_index["png_bytes"] = figure_index["png"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index["svg_bytes"] = figure_index["svg"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)

    expected_rows = {
        "scoring": 6 * 17 * 4,
        "conflict_sensitivity": 6 * 17 * 4,
        "independence": 6 * 2,
        "within_religion": 6 * 4 * 4 * 5 * 20,
        "direct_religion": 6 * 6 * 4 * 5 * 20,
        "within_general": 6 * 4 * 4 * 5 * 30,
        "direct_general": 6 * 6 * 4 * 5 * 30,
        "correctness_retention": 6 * (4 * 4 + 6 * 4) * 2,
    }
    actual_rows = {
        "scoring": len(scoring),
        "conflict_sensitivity": len(conflicts),
        "independence": len(independence),
        "within_religion": len(within_religion),
        "direct_religion": len(direct_religion),
        "within_general": len(within_general),
        "direct_general": len(direct_general),
        "correctness_retention": len(retention),
    }
    validation = {
        "valid": actual_rows == expected_rows,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "semantic_unaudited_demoted": int(frame["candidate_uncommitted"].sum()),
        "semantic_unaudited_remaining_committed": int(
            (frame["candidate_uncommitted"] & frame["analysis_answer_status"].eq("committed")).sum()
        ),
        "numeric_semantic_conflict_rows": int(frame["numeric_semantic_conflict"].sum()),
        "religion_markers_in_tables": int(within_religion["marker_id"].nunique()),
        "general_markers_in_tables": int(within_general["marker_id"].nunique()),
        "all_persona_status_tests_significant_holm": bool(
            independence["significant_holm_0_05"].all()
        ),
        "png_files": len(list(image_dir.glob("*.png"))),
        "svg_files": len(list(image_dir.glob("*.svg"))),
    }
    if not validation["valid"] or validation["semantic_unaudited_remaining_committed"]:
        raise ValueError(f"Bundle validation failed: {validation}")
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "canonical_csv": str(canonical_csv),
        "canonical_sha256": _sha256(canonical_csv),
        "marker_dir": str(marker_dir),
        "marker_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "audit_manifest": str(audit_manifest_path),
        "audit_manifest_sha256": _sha256(audit_manifest_path),
        "marker_pipeline_version": marker_manifest["pipeline_version"],
        "marker_registry_status": marker_manifest["registry_status"],
        "human_review_status": audit_manifest["human_review_status"],
        "rows": len(frame),
        "models": len(MODELS),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "semantic_unaudited_demoted_in_analysis_view": int(frame["candidate_uncommitted"].sum()),
        "numeric_semantic_conflict_rows": int(frame["numeric_semantic_conflict"].sum()),
        "marker_count_religion": int(within_religion["marker_id"].nunique()),
        "marker_count_general": int(within_general["marker_id"].nunique()),
        "marker_inference_run": False,
        "marker_p_values_computed": False,
        "marker_confidence_intervals_computed": False,
        "parser_status_inference_run": True,
        "generated_tables": [*tables, "figure_index.csv", "validation.json"],
        "generated_figures": figure_index.to_dict(orient="records"),
        "interpretation_guardrails": [
            "The same marker registry is applied to all models and personas.",
            "All visible responses are included in marker comparisons, including abstain responses.",
            "Missing and technical-only responses are NA, never marker zero.",
            "All 20 Religion and all 30 general markers are present in exact paired tables.",
            "Raw occurrence, density per 100 words, presence, and paired effect are stored together.",
            "semantic_unaudited responses are uncommitted candidates in the analysis view.",
            "Commonsense and Justice use committed accuracy; Deontology and Virtue use answer distributions.",
            "Marker p-values and confidence intervals remain blocked until two independent human reviews finish.",
            "The persona-status chi-square is a parser measurement audit, not marker inference.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[8/8] Done: {output_dir}")
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    build_bundle(args.marker_dir, args.canonical_csv, args.audit_manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build write-once supplemental visuals for ETHICS Religion Persona.

The source marker registry is still a candidate.  Consequently this builder
produces descriptive paired effects only: it does not calculate p-values,
confidence intervals, or significance flags for linguistic markers.  The
same visible-response surface and marker columns are used for every model and
persona.  Ground truth is read only for the explicitly labelled correctness
sensitivity analysis; it never changes marker extraction.
"""

from __future__ import annotations

import argparse
import hashlib
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


BUNDLE_VERSION = "religion-ethics-bab4-visual-supplement-v1.0.0"
EXPECTED_PIPELINE_VERSION = "ethics-religion-lm-v1.1.10-candidate"
EXPECTED_INPUT_SHA256 = "c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a"
DOMAIN_KEY = "religion"
DOMAIN_LABEL = "agama"
DOMAIN_TITLE = "Religion"
LEXICON_FILENAME = "religion_lexicon_coverage.csv"
LEXICON_CAPTURE_COLUMN = "captured_by_relevant_religion_marker"
TERM_TABLE_FILENAME = "religion_terms_explicit_response_prevalence.csv"
WITHIN_TABLE_FILENAME = "religion_marker_vs_neutral_descriptive.csv"
DIRECT_TABLE_FILENAME = "religion_marker_direct_identity_descriptive.csv"
TERM_FIGURE_FILENAME = "05_religious_terms_ranked_explicit_response_prevalence.png"
WITHIN_FIGURE_TOKEN = "religion_marker"
DIRECT_FIGURE_TOKEN = "religion"

IDENTITIES = ("islam", "christian", "jewish", "atheist")
CONDITIONS = ("neutral", "explicit", "implicit_1", "implicit_2", "implicit_3")
PAIR_CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
SUBSETS = ("commonsense", "deontology", "justice", "virtue")
CONSTRUCT_VALID_SUBSETS = ("commonsense", "justice")
DIRECT_IDENTITY_PAIRS = tuple(combinations(IDENTITIES, 2))

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
MODEL_ORDER = tuple(MODEL_LABELS)
MODEL_COLORS = dict(
    zip(MODEL_ORDER, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
)
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Christian",
    "jewish": "Jewish",
    "atheist": "Atheist",
}
IDENTITY_COLORS = {
    "islam": "#0F9D8A",
    "christian": "#7C3AED",
    "jewish": "#2563EB",
    "atheist": "#D97706",
}
CONDITION_LABELS = {
    "neutral": "N",
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
PAIR_LABELS = {
    f"{left}_vs_{right}": f"{IDENTITY_LABELS[left]} − {IDENTITY_LABELS[right]}"
    for left, right in DIRECT_IDENTITY_PAIRS
}

RELIGION_MARKERS = (
    "religion_identity_islam",
    "religion_identity_christian",
    "religion_identity_jewish",
    "religion_identity_atheist",
    "religion_generic_reference",
    "religion_deity_invocation",
    "religion_scripture_reference",
    "religion_practice_community",
    "religion_perspective_adoption",
    "religion_moral_authority_framing",
    "religion_secular_humanist_framing",
    "religion_identity_irrelevance",
)
RELIGION_MARKER_LABELS = {
    "religion_identity_islam": "Identitas Islam",
    "religion_identity_christian": "Identitas Christian",
    "religion_identity_jewish": "Identitas Jewish",
    "religion_identity_atheist": "Identitas atheist",
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Tuhan/deity",
    "religion_scripture_reference": "Kitab/teks agama",
    "religion_practice_community": "Praktik/komunitas",
    "religion_perspective_adoption": "Adopsi perspektif",
    "religion_moral_authority_framing": "Otoritas moral agama",
    "religion_secular_humanist_framing": "Kerangka sekuler",
    "religion_identity_irrelevance": "Identitas tidak relevan",
}

GENERAL_MARKERS = (
    "organization_explicit_step",
    "organization_bullet",
    "organization_section_header",
    "stance_hedge",
    "stance_booster",
    "stance_assumption",
    "stance_self_correction",
    "stance_contextual_hedge",
    "stance_distancing",
    "connective_causal",
    "connective_contrastive",
    "connective_conditional",
    "connective_additive",
    "connective_concessive",
    "self_reference_first_person",
    "self_reference_inclusive_we",
    "engagement_reader_directive",
    "affect_empathy_validation",
)
GENERAL_OUTCOMES = ("word_count", "avg_sentence_length", "ttr_300", *GENERAL_MARKERS)
GENERAL_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Panjang kalimat",
    "ttr_300": "TTR-300",
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

PAIR_KEYS = ("model_id", "subset", "question_id")
OPENING_COLORS = {
    "narrative": "#1B9E77",
    "answer_first": "#59A14F",
    "problem_restatement": "#FF9DA7",
    "refusal": "#E15759",
    "contextual": "#9C755F",
    "document_heading": "#B07AA1",
    "instructional": "#4E79A7",
    "other": "#BAB0AC",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Finish an interrupted write-once bundle without replacing existing files.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def _nullable_bool(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.map({"true": True, "false": False, "1": True, "0": False}).astype("boolean")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17.5,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titlesize": 20,
            "legend.frameon": False,
            "legend.fontsize": 17,
        }
    )


def _save(fig: plt.Figure, image_dir: Path, filename: str) -> Path:
    path = image_dir / filename
    if path.exists():
        plt.close(fig)
        return path
    fig.savefig(path, dpi=235, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _model_legend(include_median: bool = True) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    if include_median:
        handles.append(Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model"))
    return handles


def _load_sources(marker_dir: Path, canonical_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest_path = marker_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError(f"Unexpected pipeline version: {manifest.get('pipeline_version')}")
    if manifest.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"Canonical input SHA-256 does not match the validated ETHICS {DOMAIN_TITLE} source."
        )
    if manifest.get("inference_run") is not False:
        raise ValueError("The candidate marker bundle must not contain marker inference.")

    response_path = marker_dir / "response_level_markers.csv.gz"
    marker_value_columns = [f"{marker}__present" for marker in RELIGION_MARKERS]
    marker_value_columns += [f"{marker}__density" for marker in GENERAL_MARKERS]
    usecols = [
        "model_id",
        "persona",
        "identity",
        "condition",
        "question_id",
        "subset",
        "answer_status",
        "text_available",
        "word_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        *marker_value_columns,
    ]
    print(f"[1/7] Reading response-level marker surface: {response_path}")
    frame = pd.read_csv(response_path, usecols=usecols, low_memory=False)
    frame["text_available"] = _bool_mask(frame["text_available"])
    for column in marker_value_columns + ["word_count", "avg_sentence_length", "ttr_300"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    print(f"[2/7] Reading correctness keys only: {canonical_csv}")
    correctness = pd.read_csv(
        canonical_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "is_correct",
            "construct_valid",
            "extracted_stance",
        ],
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    correctness["is_correct"] = _nullable_bool(correctness["is_correct"])
    correctness["construct_valid"] = _nullable_bool(correctness["construct_valid"])

    key = ["model_id", "persona", "question_id"]
    if frame.duplicated(key).any() or correctness.duplicated(key).any():
        raise ValueError("Duplicate model/persona/question key found in a canonical source.")
    frame = frame.merge(correctness, on=key, how="left", validate="one_to_one")
    if len(frame) != manifest["rows_input"] or frame[key].isna().any().any():
        raise ValueError("Merged response-level row count or keys do not match the manifest.")
    if set(frame["model_id"].dropna().unique()) != set(MODEL_ORDER):
        raise ValueError("Unexpected model set in response-level source.")
    if set(frame["subset"].dropna().unique()) != set(SUBSETS):
        raise ValueError("Unexpected ETHICS subset set.")

    lexicon = pd.read_csv(marker_dir / LEXICON_FILENAME, low_memory=False)
    registry = pd.read_csv(marker_dir / "marker_registry.csv", low_memory=False)
    return frame, lexicon, manifest, registry


def _scope_groups(joined: pd.DataFrame):
    yield "all", joined
    for subset in SUBSETS:
        yield subset, joined[joined["subset"].eq(subset)]


def _rank_biserial(left: pd.Series, right: pd.Series) -> tuple[int, int, float]:
    valid = pd.concat([left, right], axis=1).dropna()
    if valid.empty:
        return 0, 0, math.nan
    difference = valid.iloc[:, 0].to_numpy(float) - valid.iloc[:, 1].to_numpy(float)
    nonzero = difference[~np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)]
    if not len(nonzero):
        return len(valid), 0, 0.0
    effect = (np.count_nonzero(nonzero > 0) - np.count_nonzero(nonzero < 0)) / len(nonzero)
    return len(valid), len(nonzero), float(effect)


def _matrix_statistics(
    frame: pd.DataFrame,
    left_columns: list[str],
    right_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized paired n, nonzero n, mean delta, and rank-biserial."""

    left = frame[left_columns].to_numpy(dtype=float, copy=False)
    right = frame[right_columns].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(left) & np.isfinite(right)
    difference = np.where(valid, left - right, np.nan)
    n_pairs = valid.sum(axis=0).astype(int)
    nonzero_mask = valid & ~np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)
    n_nonzero = nonzero_mask.sum(axis=0).astype(int)

    delta_sum = np.nansum(difference, axis=0)
    mean_delta = np.full(len(left_columns), np.nan, dtype=float)
    np.divide(delta_sum, n_pairs, out=mean_delta, where=n_pairs > 0)

    positive = ((difference > 0) & nonzero_mask).sum(axis=0)
    negative = ((difference < 0) & nonzero_mask).sum(axis=0)
    rank_biserial = np.full(len(left_columns), np.nan, dtype=float)
    np.divide(positive - negative, n_nonzero, out=rank_biserial, where=n_nonzero > 0)
    rank_biserial[(n_pairs > 0) & (n_nonzero == 0)] = 0.0
    return n_pairs, n_nonzero, mean_delta, rank_biserial


def _paired_effect_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = frame[frame["text_available"]].copy()
    neutral_columns = [
        *PAIR_KEYS,
        "is_correct",
        *[f"{marker}__present" for marker in RELIGION_MARKERS],
        *[f"{marker}__density" for marker in GENERAL_MARKERS],
        "word_count",
        "avg_sentence_length",
        "ttr_300",
    ]
    neutral = available[available["persona"].eq("neutral")][neutral_columns]
    religion_rows: list[dict[str, object]] = []
    general_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    religion_source_columns = [f"{marker}__present" for marker in RELIGION_MARKERS]
    general_source_columns = [
        outcome if outcome in {"word_count", "avg_sentence_length", "ttr_300"} else f"{outcome}__density"
        for outcome in GENERAL_OUTCOMES
    ]
    sensitivity_outcomes = [*GENERAL_OUTCOMES, *RELIGION_MARKERS]
    sensitivity_source_columns = [*general_source_columns, *religion_source_columns]

    for identity in IDENTITIES:
        for condition in PAIR_CONDITIONS:
            target = available[
                available["identity"].eq(identity) & available["condition"].eq(condition)
            ][neutral_columns]
            joined = target.merge(neutral, on=list(PAIR_KEYS), suffixes=("_persona", "_neutral"), validate="one_to_one")
            for model, model_rows in joined.groupby("model_id", sort=False):
                for subset_scope, scoped in _scope_groups(model_rows):
                    religion_n, _, religion_delta, _ = _matrix_statistics(
                        scoped,
                        [f"{column}_persona" for column in religion_source_columns],
                        [f"{column}_neutral" for column in religion_source_columns],
                    )
                    for index, marker in enumerate(RELIGION_MARKERS):
                        religion_rows.append(
                            {
                                "model_id": model,
                                "identity": identity,
                                "condition": condition,
                                "subset": subset_scope,
                                "outcome_id": marker,
                                "n_pairs": int(religion_n[index]),
                                "prevalence_delta_pp": 100 * religion_delta[index],
                            }
                        )
                    general_n, general_nonzero, _, general_effect = _matrix_statistics(
                        scoped,
                        [f"{column}_persona" for column in general_source_columns],
                        [f"{column}_neutral" for column in general_source_columns],
                    )
                    for index, outcome in enumerate(GENERAL_OUTCOMES):
                        general_rows.append(
                            {
                                "model_id": model,
                                "identity": identity,
                                "condition": condition,
                                "subset": subset_scope,
                                "outcome_id": outcome,
                                "n_pairs": int(general_n[index]),
                                "n_nonzero": int(general_nonzero[index]),
                                "rank_biserial": general_effect[index],
                            }
                        )

                construct_valid = model_rows[model_rows["subset"].isin(CONSTRUCT_VALID_SUBSETS)]
                both_correct = construct_valid[
                    construct_valid["is_correct_persona"].fillna(False)
                    & construct_valid["is_correct_neutral"].fillna(False)
                ]
                left_columns = [f"{column}_persona" for column in sensitivity_source_columns]
                right_columns = [f"{column}_neutral" for column in sensitivity_source_columns]
                n_all, nz_all, _, effect_all = _matrix_statistics(construct_valid, left_columns, right_columns)
                n_correct, nz_correct, _, effect_correct = _matrix_statistics(both_correct, left_columns, right_columns)
                for index, outcome in enumerate(sensitivity_outcomes):
                    sensitivity_rows.append(
                        {
                            "comparison_type": "persona_vs_neutral",
                            "model_id": model,
                            "identity": identity,
                            "condition": condition,
                            "identity_contrast": "",
                            "outcome_id": outcome,
                            "n_pairs_all": int(n_all[index]),
                            "n_nonzero_all": int(nz_all[index]),
                            "effect_all": effect_all[index],
                            "n_pairs_both_correct": int(n_correct[index]),
                            "n_nonzero_both_correct": int(nz_correct[index]),
                            "effect_both_correct": effect_correct[index],
                        }
                    )
    return pd.DataFrame(religion_rows), pd.DataFrame(general_rows), pd.DataFrame(sensitivity_rows)


def _direct_effect_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = frame[frame["text_available"]].copy()
    value_columns = [
        *PAIR_KEYS,
        "is_correct",
        *[f"{marker}__present" for marker in RELIGION_MARKERS],
        *[f"{marker}__density" for marker in GENERAL_MARKERS],
        "word_count",
        "avg_sentence_length",
        "ttr_300",
    ]
    religion_source_columns = [f"{marker}__present" for marker in RELIGION_MARKERS]
    general_source_columns = [
        outcome if outcome in {"word_count", "avg_sentence_length", "ttr_300"} else f"{outcome}__density"
        for outcome in GENERAL_OUTCOMES
    ]
    sensitivity_outcomes = [*GENERAL_OUTCOMES, *RELIGION_MARKERS]
    sensitivity_source_columns = [*general_source_columns, *religion_source_columns]
    effect_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for condition in PAIR_CONDITIONS:
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            contrast = f"{left_identity}_vs_{right_identity}"
            left = available[
                available["identity"].eq(left_identity) & available["condition"].eq(condition)
            ][value_columns]
            right = available[
                available["identity"].eq(right_identity) & available["condition"].eq(condition)
            ][value_columns]
            joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            for model, model_rows in joined.groupby("model_id", sort=False):
                for subset_scope, scoped in _scope_groups(model_rows):
                    n_pairs, _, mean_delta, _ = _matrix_statistics(
                        scoped,
                        [f"{column}_left" for column in religion_source_columns],
                        [f"{column}_right" for column in religion_source_columns],
                    )
                    for index, marker in enumerate(RELIGION_MARKERS):
                        effect_rows.append(
                            {
                                "model_id": model,
                                "identity_contrast": contrast,
                                "condition": condition,
                                "subset": subset_scope,
                                "outcome_id": marker,
                                "n_pairs": int(n_pairs[index]),
                                "prevalence_delta_pp": 100 * mean_delta[index],
                            }
                        )

                construct_valid = model_rows[model_rows["subset"].isin(CONSTRUCT_VALID_SUBSETS)]
                both_correct = construct_valid[
                    construct_valid["is_correct_left"].fillna(False)
                    & construct_valid["is_correct_right"].fillna(False)
                ]
                left_columns = [f"{column}_left" for column in sensitivity_source_columns]
                right_columns = [f"{column}_right" for column in sensitivity_source_columns]
                n_all, nz_all, _, effect_all = _matrix_statistics(construct_valid, left_columns, right_columns)
                n_correct, nz_correct, _, effect_correct = _matrix_statistics(both_correct, left_columns, right_columns)
                for index, outcome in enumerate(sensitivity_outcomes):
                    sensitivity_rows.append(
                        {
                            "comparison_type": "direct_identity",
                            "model_id": model,
                            "identity": "",
                            "condition": condition,
                            "identity_contrast": contrast,
                            "outcome_id": outcome,
                            "n_pairs_all": int(n_all[index]),
                            "n_nonzero_all": int(nz_all[index]),
                            "effect_all": effect_all[index],
                            "n_pairs_both_correct": int(n_correct[index]),
                            "n_nonzero_both_correct": int(nz_correct[index]),
                            "effect_both_correct": effect_correct[index],
                        }
                    )
    return pd.DataFrame(effect_rows), pd.DataFrame(sensitivity_rows)


def _term_table(frame: pd.DataFrame, lexicon: pd.DataFrame) -> pd.DataFrame:
    denominators = (
        frame[frame["text_available"]]
        .groupby(["model_id", "persona", "identity", "condition"], observed=True)
        .size()
        .rename("n_available")
        .reset_index()
    )
    selected = lexicon[
        _bool_mask(lexicon[LEXICON_CAPTURE_COLUMN])
        & ~_bool_mask(lexicon["same_term_in_scenario"])
    ].copy()
    selected = selected.groupby(["model_id", "persona", "term"], as_index=False)["n_responses"].sum()
    result = selected.merge(denominators, on=["model_id", "persona"], how="left", validate="many_to_one")
    result["response_prevalence_pct"] = 100 * result["n_responses"] / result["n_available"]
    return result


def _opening_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = frame[frame["text_available"] & frame["opening_category"].notna()].copy()
    neutral = available[available["condition"].eq("neutral")].copy()
    non_neutral = available[~available["condition"].eq("neutral")].copy()
    neutral = pd.concat([neutral.assign(identity=identity) for identity in IDENTITIES], ignore_index=True)
    expanded = pd.concat([non_neutral, neutral], ignore_index=True)
    counts = (
        expanded.groupby(["model_id", "identity", "condition", "opening_category"], observed=True)
        .size().rename("n").reset_index()
    )
    counts["total"] = counts.groupby(["model_id", "identity", "condition"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / counts["total"]

    direct_rows: list[dict[str, object]] = []
    categories = sorted(available["opening_category"].dropna().unique())
    for condition in PAIR_CONDITIONS:
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            contrast = f"{left_identity}_vs_{right_identity}"
            for model in MODEL_ORDER:
                left = available[
                    available["model_id"].eq(model)
                    & available["identity"].eq(left_identity)
                    & available["condition"].eq(condition)
                ]["opening_category"].value_counts(normalize=True)
                right = available[
                    available["model_id"].eq(model)
                    & available["identity"].eq(right_identity)
                    & available["condition"].eq(condition)
                ]["opening_category"].value_counts(normalize=True)
                mtv = 0.5 * sum(abs(float(left.get(category, 0)) - float(right.get(category, 0))) for category in categories)
                direct_rows.append(
                    {
                        "model_id": model,
                        "identity_contrast": contrast,
                        "condition": condition,
                        "n_left": int(left.sum() > 0 and available[
                            available["model_id"].eq(model)
                            & available["identity"].eq(left_identity)
                            & available["condition"].eq(condition)
                        ].shape[0]),
                        "n_right": int(right.sum() > 0 and available[
                            available["model_id"].eq(model)
                            & available["identity"].eq(right_identity)
                            & available["condition"].eq(condition)
                        ].shape[0]),
                        "marginal_total_variation": mtv,
                    }
                )
    return counts, pd.DataFrame(direct_rows)


def _scoring_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report answer status and committed accuracy without pooling ETHICS subsets."""

    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        committed_mask = group["answer_status"].eq("committed")
        abstain_mask = group["answer_status"].eq("abstain")
        missing_mask = group["answer_status"].eq("missing")
        committed = int(committed_mask.sum())
        total = len(group)
        n_correct = int((committed_mask & group["is_correct"].fillna(False)).sum())
        wrong = int((committed_mask & group["extracted_stance"].eq("wrong")).sum())
        acceptable = int((committed_mask & group["extracted_stance"].eq("acceptable")).sum())
        rows.append(
            {
                "model_id": model_id,
                "persona": persona,
                "subset": subset,
                "total": total,
                "committed": committed,
                "commit_rate": committed / total if total else math.nan,
                "abstain": int(abstain_mask.sum()),
                "abstain_rate": float(abstain_mask.mean()) if total else math.nan,
                "missing": int(missing_mask.sum()),
                "missing_rate": float(missing_mask.mean()) if total else math.nan,
                "n_correct": n_correct,
                "accuracy_committed": n_correct / committed if committed else math.nan,
                "pct_answer_wrong": wrong / committed if committed else math.nan,
                "pct_answer_acceptable": acceptable / committed if committed else math.nan,
                "construct_valid": bool(group["construct_valid"].fillna(False).all()),
            }
        )
    return pd.DataFrame(rows)


def _availability_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in PAIR_CONDITIONS:
        for left_identity, right_identity in DIRECT_IDENTITY_PAIRS:
            contrast = f"{left_identity}_vs_{right_identity}"
            left = frame[
                frame["identity"].eq(left_identity) & frame["condition"].eq(condition)
            ][[*PAIR_KEYS, "text_available"]]
            right = frame[
                frame["identity"].eq(right_identity) & frame["condition"].eq(condition)
            ][[*PAIR_KEYS, "text_available"]]
            joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_left", "_right"), validate="one_to_one")
            for model, model_rows in joined.groupby("model_id", sort=False):
                rows.append(
                    {
                        "model_id": model,
                        "identity_contrast": contrast,
                        "condition": condition,
                        "n_pairs": len(model_rows),
                        "left_available": int(model_rows["text_available_left"].sum()),
                        "right_available": int(model_rows["text_available_right"].sum()),
                        "availability_delta_pp": 100
                        * (model_rows["text_available_left"].astype(int) - model_rows["text_available_right"].astype(int)).mean(),
                    }
                )
    return pd.DataFrame(rows)


def _plot_term_lollipop(term_table: pd.DataFrame, image_dir: Path) -> Path:
    data = term_table[term_table["condition"].eq("explicit")].copy()
    aggregate = (
        data.groupby(["identity", "term"], as_index=False)
        .agg(prevalence_pct=("response_prevalence_pct", "mean"), raw_responses=("n_responses", "sum"))
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = aggregate[aggregate["identity"].eq(identity)].nlargest(12, "prevalence_pct").sort_values("prevalence_pct")
        ax.hlines(panel["term"], 0, panel["prevalence_pct"], color="#CBD5E1", linewidth=2.2)
        ax.scatter(panel["prevalence_pct"], panel["term"], s=85, color=IDENTITY_COLORS[identity])
        for _, point in panel.iterrows():
            ax.text(point["prevalence_pct"], point["term"], f"  {point['prevalence_pct']:.2f}%", va="center", fontsize=13)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.supxlabel("Rerata setara antar-model: respons dengan term (%)", y=0.035)
    fig.suptitle(
        f"Term {DOMAIN_LABEL} response-only yang paling sering muncul pada persona Explicit",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=2.2, w_pad=2.0)
    return _save(fig, image_dir, TERM_FIGURE_FILENAME)


def _plot_within_religion(data: pd.DataFrame, image_dir: Path, identity: str) -> Path:
    selected = data[data["identity"].eq(identity) & data["subset"].eq("all")]
    y_lookup = {marker: len(RELIGION_MARKERS) - 1 - index for index, marker in enumerate(RELIGION_MARKERS)}
    limit = max(1.0, float(np.nanquantile(np.abs(selected["prevalence_delta_pp"]), 0.99)) * 1.08)
    figure_height = max(12.5, 0.62 * len(RELIGION_MARKERS))
    fig, axes = plt.subplots(1, 4, figsize=(23, figure_height), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = selected[selected["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(point["prevalence_delta_pp"], y_lookup[point["outcome_id"]], s=70, color=MODEL_COLORS[point["model_id"]], alpha=0.9)
        medians = panel.groupby("outcome_id")["prevalence_delta_pp"].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=80, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(f"{CONDITION_LABELS[condition]} − N", fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[m] for m in RELIGION_MARKERS], [RELIGION_MARKER_LABELS[m] for m in RELIGION_MARKERS])
    fig.supxlabel("Δ prevalensi (poin persentase; deskriptif)", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle(
        f"Persona {IDENTITY_LABELS[identity]} terhadap Neutral: marker {DOMAIN_LABEL} ETHICS",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.95), w_pad=1.2)
    return _save(
        fig,
        image_dir,
        f"07_{identity}_{WITHIN_FIGURE_TOKEN}_forest_vs_neutral_descriptive.png",
    )


def _plot_general_effects(data: pd.DataFrame, image_dir: Path, identity: str) -> Path:
    selected = data[data["identity"].eq(identity) & data["subset"].eq("all")]
    y_lookup = {outcome: len(GENERAL_OUTCOMES) - 1 - index for index, outcome in enumerate(GENERAL_OUTCOMES)}
    fig, axes = plt.subplots(1, 4, figsize=(23, 14), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = selected[selected["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(
                point["rank_biserial"],
                y_lookup[point["outcome_id"]],
                s=42 + 20 * math.log10(float(point["n_nonzero"]) + 1),
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.88,
                edgecolors="white",
                linewidths=0.5,
            )
        medians = panel.groupby("outcome_id")["rank_biserial"].median()
        for outcome, value in medians.items():
            ax.scatter(value, y_lookup[outcome], marker="D", s=78, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.axvline(-0.3, color="#CBD5E1", linestyle="--", linewidth=0.7)
        ax.axvline(0.3, color="#CBD5E1", linestyle="--", linewidth=0.7)
        ax.set_xlim(-1.03, 1.03)
        ax.set_title(f"{CONDITION_LABELS[condition]} − N", fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[o] for o in GENERAL_OUTCOMES], [GENERAL_LABELS[o] for o in GENERAL_OUTCOMES])
    fig.supxlabel("Paired rank-biserial (deskriptif)", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle(f"Persona {IDENTITY_LABELS[identity]} terhadap Neutral: marker linguistik umum ETHICS", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.95), w_pad=1.2)
    return _save(fig, image_dir, f"08_{identity}_general_effects_vs_neutral_descriptive.png")


def _plot_direct_religion(data: pd.DataFrame, image_dir: Path, condition: str) -> Path:
    selected = data[data["condition"].eq(condition) & data["subset"].eq("all")]
    y_lookup = {marker: len(RELIGION_MARKERS) - 1 - index for index, marker in enumerate(RELIGION_MARKERS)}
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    limit = max(1.0, float(np.nanquantile(np.abs(selected["prevalence_delta_pp"]), 0.99)) * 1.08)
    figure_height = max(22, 0.72 * len(RELIGION_MARKERS))
    fig, axes = plt.subplots(3, 2, figsize=(18, figure_height), sharex=True, sharey=True)
    for ax, contrast in zip(axes.flat, pairs):
        panel = selected[selected["identity_contrast"].eq(contrast)]
        for _, point in panel.iterrows():
            ax.scatter(point["prevalence_delta_pp"], y_lookup[point["outcome_id"]], s=70, color=MODEL_COLORS[point["model_id"]], alpha=0.9)
        medians = panel.groupby("outcome_id")["prevalence_delta_pp"].median()
        for marker, value in medians.items():
            ax.scatter(value, y_lookup[marker], marker="D", s=80, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(PAIR_LABELS[contrast], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    for ax in axes[:, 0]:
        ax.set_yticks([y_lookup[m] for m in RELIGION_MARKERS], [RELIGION_MARKER_LABELS[m] for m in RELIGION_MARKERS])
    fig.supxlabel("Δ prevalensi (poin persentase; deskriptif)", y=0.055)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
    fig.suptitle(f"Perbandingan langsung antaridentitas pada kondisi {CONDITION_LABELS[condition]} — ETHICS", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.96), h_pad=2.0, w_pad=1.6)
    return _save(
        fig,
        image_dir,
        f"09_direct_identity_{DIRECT_FIGURE_TOKEN}_forest_{condition}_descriptive.png",
    )


def _prepare_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    result["pair_retention"] = result["n_pairs_both_correct"] / result["n_pairs_all"].replace(0, np.nan)
    result["supported_both"] = result["n_nonzero_all"].ge(20) & result["n_nonzero_both_correct"].ge(20)
    result["same_direction"] = np.sign(result["effect_all"]).eq(np.sign(result["effect_both_correct"]))
    group_columns = ["comparison_type"]
    summary = (
        result.groupby(group_columns, as_index=False)
        .agg(
            n_tests=("outcome_id", "size"),
            median_pair_retention=("pair_retention", "median"),
            supported_tests=("supported_both", "sum"),
            median_abs_effect_change=("effect_all", lambda values: math.nan),
        )
    )
    for index, row in summary.iterrows():
        subset = result[result["comparison_type"].eq(row["comparison_type"])]
        summary.loc[index, "median_abs_effect_change"] = (subset["effect_all"] - subset["effect_both_correct"]).abs().median()
        supported = subset[subset["supported_both"]]
        summary.loc[index, "same_direction_rate_supported"] = supported["same_direction"].mean() if len(supported) else math.nan
    return result, summary


def _plot_sensitivity(data: pd.DataFrame, image_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, comparison_type, title in (
        (axes[0], "persona_vs_neutral", "Persona vs Neutral"),
        (axes[1], "direct_identity", "Direct antaridentitas"),
    ):
        selected = data[data["comparison_type"].eq(comparison_type) & data["supported_both"]]
        ax.scatter(selected["effect_all"], selected["effect_both_correct"], s=48, alpha=0.45, color="#2563EB")
        ax.axline((0, 0), slope=1, color="#111827", linewidth=0.8)
        ax.axhline(0, color="#94A3B8", linewidth=0.6)
        ax.axvline(0, color="#94A3B8", linewidth=0.6)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(title, fontweight="bold")
        ax.grid(color="#E2E8F0", linewidth=0.5)
    fig.supxlabel("Rank-biserial seluruh pasangan construct-valid", y=0.035)
    fig.supylabel("Rank-biserial ketika kedua respons benar", x=0.01)
    fig.suptitle("Sensitivitas deskriptif marker terhadap correctness (Commonsense + Justice)", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.09, 0.99, 0.93), w_pad=1.8)
    return _save(fig, image_dir, "10_correctness_sensitivity_construct_valid_descriptive.png")


def _plot_opening_composition(composition: pd.DataFrame, image_dir: Path) -> Path:
    aggregate = composition.groupby(["identity", "condition", "opening_category"], as_index=False)["proportion"].mean()
    categories = [category for category in OPENING_COLORS if aggregate["opening_category"].eq(category).any()]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13.5), sharex=True)
    for ax, identity in zip(axes.flat, IDENTITIES):
        panel = aggregate[aggregate["identity"].eq(identity)]
        bottoms = np.zeros(len(CONDITIONS))
        for category in categories:
            values = [
                float(panel[panel["condition"].eq(condition) & panel["opening_category"].eq(category)]["proportion"].iloc[0])
                if len(panel[panel["condition"].eq(condition) & panel["opening_category"].eq(category)])
                else 0.0
                for condition in CONDITIONS
            ]
            ax.bar(range(len(CONDITIONS)), values, bottom=bottoms, color=OPENING_COLORS[category], width=0.72, label=category)
            bottoms += np.asarray(values)
        ax.set_title(IDENTITY_LABELS[identity], fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
        ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    fig.supxlabel("Kondisi", y=0.16)
    fig.supylabel("Proporsi respons tersedia", x=0.012)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=14)
    fig.suptitle("Komposisi opening respons ETHICS (rerata setara antar-model)", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.055, 0.22, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, "11_opening_composition_100pct.png")


def _plot_direct_opening(data: pd.DataFrame, image_dir: Path) -> Path:
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    y_lookup = {pair: len(pairs) - 1 - index for index, pair in enumerate(pairs)}
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(100 * point["marginal_total_variation"], y_lookup[point["identity_contrast"]], s=75, color=MODEL_COLORS[point["model_id"]], alpha=0.9)
        medians = panel.groupby("identity_contrast")["marginal_total_variation"].median()
        for contrast, value in medians.items():
            ax.scatter(100 * value, y_lookup[contrast], marker="D", s=82, color="#111827", zorder=5)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[p] for p in pairs], [PAIR_LABELS[p] for p in pairs])
    fig.supxlabel("Marginal total variation (%) — deskriptif", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Perbedaan komposisi opening antaridentitas pada ETHICS", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.93), w_pad=1.2)
    return _save(fig, image_dir, "12_direct_identity_opening_descriptive.png")


def _plot_availability(data: pd.DataFrame, image_dir: Path) -> Path:
    pairs = [f"{left}_vs_{right}" for left, right in DIRECT_IDENTITY_PAIRS]
    y_lookup = {pair: len(pairs) - 1 - index for index, pair in enumerate(pairs)}
    limit = max(1.0, float(np.nanmax(np.abs(data["availability_delta_pp"]))) * 1.12)
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, condition in zip(axes, PAIR_CONDITIONS):
        panel = data[data["condition"].eq(condition)]
        for _, point in panel.iterrows():
            ax.scatter(point["availability_delta_pp"], y_lookup[point["identity_contrast"]], s=75, color=MODEL_COLORS[point["model_id"]], alpha=0.9)
        medians = panel.groupby("identity_contrast")["availability_delta_pp"].median()
        for contrast, value in medians.items():
            ax.scatter(value, y_lookup[contrast], marker="D", s=82, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    axes[0].set_yticks([y_lookup[p] for p in pairs], [PAIR_LABELS[p] for p in pairs])
    fig.supxlabel("Δ teks tersedia (poin persentase; deskriptif)", y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle("Differential availability pada perbandingan antaridentitas ETHICS", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.18, 0.99, 0.93), w_pad=1.2)
    return _save(fig, image_dir, "13_direct_identity_availability_descriptive.png")


def build_bundle(marker_dir: Path, canonical_csv: Path, output_dir: Path) -> Path:
    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    _style()

    frame, lexicon, source_manifest, registry = _load_sources(marker_dir, canonical_csv)
    print("[3/7] Building paired persona-vs-Neutral descriptive tables")
    within_religion, within_general, within_sensitivity = _paired_effect_tables(frame)
    print("[4/7] Building direct-identity and correctness-sensitivity tables")
    direct_religion, direct_sensitivity = _direct_effect_tables(frame)
    sensitivity, sensitivity_summary = _prepare_sensitivity(pd.concat([within_sensitivity, direct_sensitivity], ignore_index=True))
    print("[5/7] Building term, opening, and availability tables")
    terms = _term_table(frame, lexicon)
    opening_composition, direct_opening = _opening_tables(frame)
    availability = _availability_table(frame)
    scoring_summary = _scoring_summary(frame)

    tables = {
        TERM_TABLE_FILENAME: terms,
        WITHIN_TABLE_FILENAME: within_religion,
        "general_marker_vs_neutral_rank_biserial_descriptive.csv": within_general,
        DIRECT_TABLE_FILENAME: direct_religion,
        "correctness_sensitivity_construct_valid_descriptive.csv": sensitivity,
        "correctness_sensitivity_summary.csv": sensitivity_summary,
        "opening_composition_model_identity_condition.csv": opening_composition,
        "direct_identity_opening_mtv_descriptive.csv": direct_opening,
        "direct_identity_availability_descriptive.csv": availability,
        "scoring_summary_by_model_persona_subset.csv": scoring_summary,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    print("[6/7] Rendering GSM8K-aligned supplemental figures")
    figures: list[tuple[str, str]] = []

    def add(path: Path, description: str) -> None:
        figures.append((path.name, description))

    add(
        _plot_term_lollipop(terms, image_dir),
        f"Term {DOMAIN_LABEL} response-only pada Explicit; prevalence deskriptif.",
    )
    for identity in IDENTITIES:
        add(
            _plot_within_religion(within_religion, image_dir, identity),
            f"Marker {DOMAIN_LABEL} {identity} vs Neutral; deskriptif.",
        )
    for identity in IDENTITIES:
        add(_plot_general_effects(within_general, image_dir, identity), f"Marker umum {identity} vs Neutral; paired rank-biserial deskriptif.")
    for condition in PAIR_CONDITIONS:
        add(
            _plot_direct_religion(direct_religion, image_dir, condition),
            f"Direct identity marker {DOMAIN_LABEL} {condition}; deskriptif.",
        )
    add(_plot_sensitivity(sensitivity, image_dir), "Sensitivity pada pasangan construct-valid yang keduanya benar; deskriptif.")
    add(_plot_opening_composition(opening_composition, image_dir), "Komposisi opening 100 persen dengan bobot model setara.")
    add(_plot_direct_opening(direct_opening, image_dir), "Direct identity opening marginal total variation; deskriptif.")
    add(_plot_availability(availability, image_dir), "Direct identity text availability; deskriptif.")
    figure_index = pd.DataFrame(figures, columns=["figure", "description"])
    figure_index["size_bytes"] = figure_index["figure"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)

    registry_statuses = sorted(registry["validation_status"].dropna().astype(str).unique())
    bundle_manifest = {
        "bundle_version": BUNDLE_VERSION,
        "source_marker_dir": str(marker_dir),
        "source_pipeline_version": source_manifest["pipeline_version"],
        "source_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "canonical_csv": str(canonical_csv),
        "canonical_csv_sha256": _sha256(canonical_csv),
        "rows": len(frame),
        "models": int(frame["model_id"].nunique()),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": list(SUBSETS),
        "construct_valid_correctness_subsets": list(CONSTRUCT_VALID_SUBSETS),
        "marker_registry_statuses": registry_statuses,
        "marker_inference_run": False,
        "p_values_computed": False,
        "confidence_intervals_computed": False,
        "hidden_reasoning_used": False,
        "ground_truth_use": "Correctness sensitivity filter only; never marker extraction or resolution.",
        "implicit_proxies_aggregated": False,
        f"{DOMAIN_KEY}_identities_pooled": False,
        "direct_identity_pairs": [list(pair) for pair in DIRECT_IDENTITY_PAIRS],
        "generated_figures": figure_index["figure"].tolist(),
        "generated_tables": [*tables, "figure_index.csv"],
        "interpretation_guardrails": [
            "All marker effects are descriptive while the registry remains candidate.",
            "Model points are retained; black diamonds summarize the median of six models.",
            "All responses with visible text are included, including abstain responses.",
            "Missing and technical-only surfaces are not recoded as zero marker values.",
            "Correctness sensitivity is restricted to Commonsense and Justice.",
            "Per-subset descriptive tables accompany pooled four-subset display figures.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[7/7] Done: {len(frame):,} responses, {len(figures)} figures, {len(tables)} tables")
    print(f"[DONE] Visual supplement: {output_dir}")
    return output_dir


def finalize_existing_bundle(marker_dir: Path, canonical_csv: Path, output_dir: Path) -> Path:
    """Resume figure rendering after an interrupted run without overwriting files."""

    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if manifest_path.exists():
        raise FileExistsError(f"Bundle is already finalized: {manifest_path}")
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    table_paths = {
        "terms": output_dir / TERM_TABLE_FILENAME,
        "within_religion": output_dir / WITHIN_TABLE_FILENAME,
        "within_general": output_dir / "general_marker_vs_neutral_rank_biserial_descriptive.csv",
        "direct_religion": output_dir / DIRECT_TABLE_FILENAME,
        "sensitivity": output_dir / "correctness_sensitivity_construct_valid_descriptive.csv",
        "sensitivity_summary": output_dir / "correctness_sensitivity_summary.csv",
        "opening_composition": output_dir / "opening_composition_model_identity_condition.csv",
        "direct_opening": output_dir / "direct_identity_opening_mtv_descriptive.csv",
        "availability": output_dir / "direct_identity_availability_descriptive.csv",
        "scoring_summary": output_dir / "scoring_summary_by_model_persona_subset.csv",
    }
    missing = [path for path in table_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot resume bundle; missing tables: " + ", ".join(map(str, missing)))
    tables = {name: pd.read_csv(path, low_memory=False) for name, path in table_paths.items()}
    _style()

    def existing_or_create(filename: str, creator) -> Path:
        path = image_dir / filename
        if path.is_file():
            print(f"[resume] keep existing: {filename}")
            return path
        print(f"[resume] render missing: {filename}")
        return creator()

    figures: list[tuple[str, str]] = []

    def add(path: Path, description: str) -> None:
        figures.append((path.name, description))

    add(
        existing_or_create(
            TERM_FIGURE_FILENAME,
            lambda: _plot_term_lollipop(tables["terms"], image_dir),
        ),
        f"Term {DOMAIN_LABEL} response-only pada Explicit; prevalence deskriptif.",
    )
    for identity in IDENTITIES:
        filename = f"07_{identity}_{WITHIN_FIGURE_TOKEN}_forest_vs_neutral_descriptive.png"
        add(
            existing_or_create(
                filename,
                lambda identity=identity: _plot_within_religion(tables["within_religion"], image_dir, identity),
            ),
            f"Marker {DOMAIN_LABEL} {identity} vs Neutral; deskriptif.",
        )
    for identity in IDENTITIES:
        filename = f"08_{identity}_general_effects_vs_neutral_descriptive.png"
        add(
            existing_or_create(
                filename,
                lambda identity=identity: _plot_general_effects(tables["within_general"], image_dir, identity),
            ),
            f"Marker umum {identity} vs Neutral; paired rank-biserial deskriptif.",
        )
    for condition in PAIR_CONDITIONS:
        filename = f"09_direct_identity_{DIRECT_FIGURE_TOKEN}_forest_{condition}_descriptive.png"
        add(
            existing_or_create(
                filename,
                lambda condition=condition: _plot_direct_religion(tables["direct_religion"], image_dir, condition),
            ),
            f"Direct identity marker {DOMAIN_LABEL} {condition}; deskriptif.",
        )
    add(
        existing_or_create(
            "10_correctness_sensitivity_construct_valid_descriptive.png",
            lambda: _plot_sensitivity(tables["sensitivity"], image_dir),
        ),
        "Sensitivity pada pasangan construct-valid yang keduanya benar; deskriptif.",
    )
    add(
        existing_or_create(
            "11_opening_composition_100pct.png",
            lambda: _plot_opening_composition(tables["opening_composition"], image_dir),
        ),
        "Komposisi opening 100 persen dengan bobot model setara.",
    )
    add(
        existing_or_create(
            "12_direct_identity_opening_descriptive.png",
            lambda: _plot_direct_opening(tables["direct_opening"], image_dir),
        ),
        "Direct identity opening marginal total variation; deskriptif.",
    )
    add(
        existing_or_create(
            "13_direct_identity_availability_descriptive.png",
            lambda: _plot_availability(tables["availability"], image_dir),
        ),
        "Direct identity text availability; deskriptif.",
    )

    figure_index = pd.DataFrame(figures, columns=["figure", "description"])
    figure_index["size_bytes"] = figure_index["figure"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)

    source_manifest = json.loads((marker_dir / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError("Marker pipeline changed while resuming the visual bundle.")
    registry = pd.read_csv(marker_dir / "marker_registry.csv", low_memory=False)
    bundle_manifest = {
        "bundle_version": BUNDLE_VERSION,
        "resumed_after_interrupted_render": True,
        "source_marker_dir": str(marker_dir),
        "source_pipeline_version": source_manifest["pipeline_version"],
        "source_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "canonical_csv": str(canonical_csv),
        "canonical_csv_sha256": _sha256(canonical_csv),
        "rows": source_manifest["rows_input"],
        "models": source_manifest["models"],
        "personas": source_manifest["personas"],
        "questions": source_manifest["unique_questions"],
        "subsets": list(SUBSETS),
        "construct_valid_correctness_subsets": list(CONSTRUCT_VALID_SUBSETS),
        "marker_registry_statuses": sorted(registry["validation_status"].dropna().astype(str).unique()),
        "marker_inference_run": False,
        "p_values_computed": False,
        "confidence_intervals_computed": False,
        "hidden_reasoning_used": False,
        "ground_truth_use": "Correctness sensitivity filter only; never marker extraction or resolution.",
        "implicit_proxies_aggregated": False,
        f"{DOMAIN_KEY}_identities_pooled": False,
        "direct_identity_pairs": [list(pair) for pair in DIRECT_IDENTITY_PAIRS],
        "generated_figures": figure_index["figure"].tolist(),
        "generated_tables": [path.name for path in table_paths.values()] + ["figure_index.csv"],
        "interpretation_guardrails": [
            "All marker effects are descriptive while the registry remains candidate.",
            "Model points are retained; black diamonds summarize the median of six models.",
            "All responses with visible text are included, including abstain responses.",
            "Missing and technical-only surfaces are not recoded as zero marker values.",
            "Correctness sensitivity is restricted to Commonsense and Justice.",
            "Per-subset descriptive tables accompany pooled four-subset display figures.",
        ],
    }
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Resumed visual supplement: {len(figures)} figures, {len(table_paths)} tables")
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    if args.resume_existing:
        finalize_existing_bundle(args.marker_dir, args.canonical_csv, args.output_dir)
    else:
        build_bundle(args.marker_dir, args.canonical_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

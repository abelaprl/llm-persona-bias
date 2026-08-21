"""Publication-oriented paired analysis for ETHICS Gender Persona + Neutral.

This module intentionally treats ``committed``, ``abstain``, and ``missing``
as different outcomes.  It never reparses model text and never calls an API.

Run from the repository root::

    python src/MAIN_PARSING/ETHICS/gender_publication_analysis.py

The default input is the verified parser refinement dated 2026-08-20.  All
outputs are written to a new report directory; the input and canonical reports
are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, chi2, chi2_contingency


ANALYSIS_VERSION = "ethics-gender-publication-paired-holm-v2-readable-20260821"
# Keep the inferential RNG stream frozen when only presentation changes.
STATISTICAL_SEED_VERSION = "ethics-gender-publication-paired-holm-v1-20260821"
EXPECTED_ROWS = 81_756
EXPECTED_QUESTIONS = 1_514
BOOTSTRAP_REPLICATES = 10_000

# Publication figures are commonly viewed after being scaled down in a thesis
# page or notebook.  Keep text comfortably legible at that final size instead
# of relying on the notebook renderer to enlarge a low-resolution image.
HEATMAP_ANNOT_KWS = {"fontsize": 12, "fontweight": "semibold"}
HEATMAP_COMPACT_ANNOT_KWS = {"fontsize": 11, "fontweight": "semibold"}

MODEL_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABELS = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
}

PERSONA_ORDER = [
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
    "neutral",
]
PERSONA_LABELS = {
    "man_explicit_1": "Laki-laki E",
    "man_implicit_1": "Laki-laki I1",
    "man_implicit_2": "Laki-laki I2",
    "man_implicit_3": "Laki-laki I3",
    "woman_explicit_1": "Perempuan E",
    "woman_implicit_1": "Perempuan I1",
    "woman_implicit_2": "Perempuan I2",
    "woman_implicit_3": "Perempuan I3",
    "neutral": "Netral",
}

SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
SUBSET_EXPECTED_N = {
    "commonsense": 388,
    "deontology": 359,
    "justice": 270,
    "virtue": 497,
}
CONSTRUCT_VALID = {
    "commonsense": True,
    "deontology": False,
    "justice": True,
    "virtue": False,
}

STATUS_ORDER = ["committed", "abstain", "missing"]
STATUS_COLORS = {
    "committed": "#2A9D8F",
    "abstain": "#E9C46A",
    "missing": "#E76F51",
}

USECOLS = [
    "model",
    "persona_folder",
    "question_id",
    "subset",
    "persona_group",
    "answer_status",
    "parse_source",
    "is_correct",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "truncated",
    "extracted_stance",
    "digit_convention",
    "candidate_answer",
    "candidate_source",
    "behavior_category",
    "abstain_reason",
    "extraction_confidence",
    "numeric_semantic_conflict",
    "response_flags",
    "construct_valid",
]

OUTCOME_DEFINITIONS = {
    "coverage": {
        "label": "Komitmen menjawab",
        "subsets": SUBSET_ORDER,
        "eligibility": "kedua respons tidak missing",
        "event": "answer_status == committed",
    },
    "conditional_accuracy": {
        "label": "Akurasi pada pasangan sama-sama committed",
        "subsets": ["commonsense", "justice"],
        "eligibility": "kedua respons committed",
        "event": "is_correct == True",
    },
    "effective_success": {
        "label": "Keberhasilan efektif (sensitivity)",
        "subsets": ["commonsense", "justice"],
        "eligibility": "kedua respons tidak missing; abstain dihitung tidak berhasil",
        "event": "committed dan correct",
    },
    "stance_wrong": {
        "label": "Proporsi stance wrong",
        "subsets": ["deontology", "virtue"],
        "eligibility": "kedua respons committed",
        "event": "extracted_stance == wrong",
    },
}

ABSTAIN_REASON_LABELS = {
    "both_sides": "Tidak berpihak / kedua sisi",
    "analysis_without_verdict": "Analisis tanpa verdict",
    "no_opinion_stop": "Tidak punya opini lalu berhenti",
    "explicit_binary_refusal": "Menolak jawaban biner",
    "task_substitution": "Mengganti tugas",
    "ai_refusal": "Penolakan sebagai AI",
    "insufficient_context": "Konteks dianggap kurang",
    "placeholder_confusion": "Bingung placeholder",
    "returns_question": "Mengulang pertanyaan",
    "capability_mismatch": "Mengaku tidak mampu",
    "detector_disagreement": "Detektor tidak sepakat",
    "conditional_only": "Jawaban hanya bersyarat",
}


@dataclass(frozen=True)
class PairSpec:
    identity: str
    family: str
    scope: str
    persona_a: str
    persona_b: str
    comparison_label: str
    display_order: int


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository without assuming the notebook working directory."""

    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "AGENT.md").exists() and (path / "result" / "ETHICS").exists():
            return path
    raise FileNotFoundError("Repo root tidak ditemukan dari working directory saat ini.")


def default_input_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "parser_refinement_20260820"
        / "all_models_results_full.csv"
    )


def default_output_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "persona_publication_analysis_v2_readable_paired_holm_six_models"
    )


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    unexpected = set(normalized.unique()) - {"true", "false", "1", "0", ""}
    if unexpected:
        raise ValueError(f"{column}: nilai boolean tidak dikenal: {sorted(unexpected)}")
    return normalized.map(
        {"true": True, "1": True, "false": False, "0": False, "": False}
    ).astype(bool)


def load_data(path: Path) -> pd.DataFrame:
    """Load analysis fields from all rows; no response is sampled."""

    print(f"[LOAD] Membaca seluruh baris: {path}")
    frame = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    for column in ["is_correct", "numeric_semantic_conflict", "construct_valid"]:
        frame[column] = normalize_bool(frame[column], column)
    for column in [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["persona"] = frame["persona_folder"]
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    frame["persona_label"] = frame["persona"].map(PERSONA_LABELS)
    frame["subset_label"] = frame["subset"].map(SUBSET_LABELS)
    frame["is_committed"] = frame["answer_status"].eq("committed")
    frame["is_abstain"] = frame["answer_status"].eq("abstain")
    frame["is_missing"] = frame["answer_status"].eq("missing")
    frame["effective_success"] = frame["is_committed"] & frame["is_correct"]
    frame["stance_wrong"] = frame["extracted_stance"].eq("wrong")
    print(f"[LOAD] Selesai: {len(frame):,} baris, {frame.shape[1]} kolom analisis")
    return frame


def validate_input(frame: pd.DataFrame) -> dict[str, object]:
    """Fail loudly if pairing or construct invariants are violated."""

    checks: dict[str, object] = {}
    checks["row_count"] = len(frame)
    checks["row_count_expected"] = len(frame) == EXPECTED_ROWS
    checks["models"] = sorted(frame["model"].unique())
    checks["models_expected"] = set(checks["models"]) == set(MODEL_ORDER)
    checks["personas"] = sorted(frame["persona"].unique())
    checks["personas_expected"] = set(checks["personas"]) == set(PERSONA_ORDER)
    checks["subsets"] = sorted(frame["subset"].unique())
    checks["subsets_expected"] = set(checks["subsets"]) == set(SUBSET_ORDER)
    checks["statuses"] = sorted(frame["answer_status"].unique())
    checks["statuses_expected"] = set(checks["statuses"]) == set(STATUS_ORDER)

    key_columns = ["model", "persona", "question_id"]
    duplicate_keys = int(frame.duplicated(key_columns).sum())
    checks["duplicate_response_keys"] = duplicate_keys
    checks["duplicate_response_keys_zero"] = duplicate_keys == 0

    condition_counts = frame.groupby(["model", "persona"]).size()
    checks["all_conditions_equal_1514"] = bool(
        condition_counts.eq(EXPECTED_QUESTIONS).all()
    )

    reference_ids = None
    identical_ids = True
    for _, group in frame.groupby(["model", "persona"], sort=False):
        ids = frozenset(group["question_id"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            identical_ids = False
            break
    checks["all_conditions_share_identical_question_ids"] = identical_ids

    observed_subset_n = frame.groupby(["model", "persona", "subset"]).size()
    subset_size_valid = all(
        int(count) == SUBSET_EXPECTED_N[subset]
        for (_, _, subset), count in observed_subset_n.items()
    )
    checks["subset_sizes_expected"] = subset_size_valid

    construct_observed = (
        frame.groupby("subset")["construct_valid"].agg(lambda values: set(values))
    )
    construct_valid = all(
        values == {CONSTRUCT_VALID[subset]}
        for subset, values in construct_observed.items()
    )
    checks["construct_mapping_expected"] = construct_valid

    committed_have_stance = frame.loc[frame["is_committed"], "extracted_stance"].isin(
        ["wrong", "acceptable"]
    ).all()
    noncommitted_no_correctness = frame.loc[~frame["is_committed"], "is_correct"].eq(
        False
    ).all()
    checks["committed_have_binary_stance"] = bool(committed_have_stance)
    checks["noncommitted_correctness_not_imputed"] = bool(noncommitted_no_correctness)

    boolean_checks = [
        value
        for key, value in checks.items()
        if isinstance(value, (bool, np.bool_)) and not key.endswith("_expected_n")
    ]
    checks["valid"] = bool(all(boolean_checks))
    if not checks["valid"]:
        failed = [key for key, value in checks.items() if value is False]
        raise ValueError(f"Validasi input gagal: {failed}")
    print("[VALIDATE] Pairing, status, subset, dan construct mapping valid")
    return checks


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def build_condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", "persona_group", "persona", "subset"]
    for keys, group in frame.groupby(group_columns, sort=False):
        model, persona_group, persona, subset = keys
        total = len(group)
        committed_mask = group["is_committed"]
        abstain_mask = group["is_abstain"]
        missing_mask = group["is_missing"]
        conflict_mask = group["numeric_semantic_conflict"]
        clean_committed = committed_mask & ~conflict_mask
        committed = int(committed_mask.sum())
        nonmissing = total - int(missing_mask.sum())
        n_correct = int((committed_mask & group["is_correct"]).sum())
        n_correct_clean = int((clean_committed & group["is_correct"]).sum())
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "persona_group": persona_group,
                "persona": persona,
                "persona_label": PERSONA_LABELS[persona],
                "subset": subset,
                "subset_label": SUBSET_LABELS[subset],
                "construct_valid": CONSTRUCT_VALID[subset],
                "total": total,
                "nonmissing": nonmissing,
                "committed": committed,
                "abstain": int(abstain_mask.sum()),
                "missing": int(missing_mask.sum()),
                "commit_rate": safe_rate(committed, total),
                "commit_rate_nonmissing": safe_rate(committed, nonmissing),
                "abstain_rate": safe_rate(int(abstain_mask.sum()), total),
                "missing_rate": safe_rate(int(missing_mask.sum()), total),
                "n_correct": n_correct,
                "accuracy_committed": safe_rate(n_correct, committed),
                "effective_success_rate_nonmissing": safe_rate(n_correct, nonmissing),
                "n_conflict": int((committed_mask & conflict_mask).sum()),
                "accuracy_committed_excl_conflict": safe_rate(
                    n_correct_clean, int(clean_committed.sum())
                ),
                "pct_answer_wrong": safe_rate(
                    int((committed_mask & group["stance_wrong"]).sum()), committed
                ),
                "pct_answer_acceptable": safe_rate(
                    int(
                        (
                            committed_mask
                            & group["extracted_stance"].eq("acceptable")
                        ).sum()
                    ),
                    committed,
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["conflict_sensitivity_delta_pp"] = 100 * (
        result["accuracy_committed_excl_conflict"]
        - result["accuracy_committed"]
    )
    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    persona_rank = {persona: index for index, persona in enumerate(PERSONA_ORDER)}
    subset_rank = {subset: index for index, subset in enumerate(SUBSET_ORDER)}
    result = result.assign(
        _model=result["model"].map(model_rank),
        _persona=result["persona"].map(persona_rank),
        _subset=result["subset"].map(subset_rank),
    ).sort_values(["_model", "_persona", "_subset"], kind="stable")
    return result.drop(columns=["_model", "_persona", "_subset"]).reset_index(drop=True)


def validate_against_refinement_summary(
    generated: pd.DataFrame, source_summary_path: Path
) -> dict[str, object]:
    source = pd.read_csv(source_summary_path)
    keys = ["model", "persona_group", "persona", "subset"]
    count_columns = [
        "total",
        "committed",
        "abstain",
        "missing",
        "n_correct",
        "n_conflict",
    ]
    rate_columns = [
        "commit_rate",
        "abstain_rate",
        "missing_rate",
        "accuracy_committed",
        "accuracy_committed_excl_conflict",
        "pct_answer_wrong",
        "pct_answer_acceptable",
    ]
    merged = source.merge(
        generated[keys + count_columns + rate_columns],
        on=keys,
        suffixes=("_source", "_generated"),
        validate="one_to_one",
    )
    count_differences = {
        column: int(
            (
                pd.to_numeric(merged[f"{column}_source"])
                != pd.to_numeric(merged[f"{column}_generated"])
            ).sum()
        )
        for column in count_columns
    }
    rate_max_abs_difference = {
        column: float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(merged[f"{column}_source"], errors="coerce")
                    - pd.to_numeric(merged[f"{column}_generated"], errors="coerce")
                )
            )
        )
        for column in rate_columns
    }
    valid = all(value == 0 for value in count_differences.values()) and all(
        value <= 5.1e-7 for value in rate_max_abs_difference.values()
    )
    if not valid:
        raise ValueError(
            "Ringkasan hasil hitung ulang tidak sama dengan summary refinement: "
            f"counts={count_differences}, rates={rate_max_abs_difference}"
        )
    return {
        "valid": valid,
        "rows_compared": len(merged),
        "count_differences": count_differences,
        "rate_max_abs_difference": rate_max_abs_difference,
    }


def build_model_overview(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        total = len(group)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "total": total,
                "committed": int(group["is_committed"].sum()),
                "abstain": int(group["is_abstain"].sum()),
                "missing": int(group["is_missing"].sum()),
                "commit_rate": float(group["is_committed"].mean()),
                "abstain_rate": float(group["is_abstain"].mean()),
                "missing_rate": float(group["is_missing"].mean()),
                "numeric_semantic_conflict": int(
                    group["numeric_semantic_conflict"].sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    return result.set_index("model").reindex(MODEL_ORDER).reset_index()


def build_status_independence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        "gender_only": frame["persona_group"].eq("gender"),
        "gender_plus_neutral": frame["persona"].isin(PERSONA_ORDER),
    }
    for scope, scope_mask in scopes.items():
        for model in MODEL_ORDER:
            group = frame[scope_mask & frame["model"].eq(model)]
            table = pd.crosstab(group["persona"], group["answer_status"])
            table = table.loc[:, table.sum(axis=0).gt(0)]
            statistic, p_value, degrees_of_freedom, _ = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            denominator = min(table.shape[0] - 1, table.shape[1] - 1)
            cramer_v = math.sqrt(statistic / (n * denominator)) if denominator else 0.0
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "personas": table.shape[0],
                    "status_categories": table.shape[1],
                    "n": n,
                    "chi2": float(statistic),
                    "dof": int(degrees_of_freedom),
                    "p_value": float(p_value),
                    "cramer_v": float(cramer_v),
                    "significant_0_05": bool(p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def build_behavior_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["model", "persona", "subset"]
    denominator = (
        frame.groupby(keys, as_index=False)
        .agg(
            total=("question_id", "size"),
            nonmissing=("is_missing", lambda values: int((~values).sum())),
            abstain=("is_abstain", "sum"),
        )
    )

    abstain_rows = frame[frame["is_abstain"]].copy()
    abstain_rows["abstain_reason"] = abstain_rows["abstain_reason"].replace(
        "", "unspecified"
    )
    abstain = (
        abstain_rows.groupby(keys + ["abstain_reason"], as_index=False)
        .size()
        .rename(columns={"size": "reason_count"})
        .merge(denominator, on=keys, validate="many_to_one")
    )
    abstain["pct_of_abstain"] = abstain["reason_count"] / abstain["abstain"]
    abstain["pct_of_nonmissing"] = abstain["reason_count"] / abstain["nonmissing"]

    behavior = (
        frame.groupby(keys + ["behavior_category"], as_index=False)
        .size()
        .rename(columns={"size": "behavior_count"})
        .merge(denominator, on=keys, validate="many_to_one")
    )
    behavior["pct_of_total"] = behavior["behavior_count"] / behavior["total"]
    behavior["pct_of_nonmissing"] = behavior["behavior_count"] / behavior[
        "nonmissing"
    ]

    parse_source_rows = frame.copy()
    parse_source_rows["parse_source"] = parse_source_rows["parse_source"].replace(
        "", "none"
    )
    parse_source = (
        parse_source_rows.groupby(
            keys + ["answer_status", "parse_source"], as_index=False
        )
        .size()
        .rename(columns={"size": "source_count"})
        .merge(denominator, on=keys, validate="many_to_one")
    )
    parse_source["pct_of_total"] = parse_source["source_count"] / parse_source[
        "total"
    ]

    flags = frame[keys + ["question_id", "response_flags"]].copy()
    flags["response_flag"] = flags["response_flags"].str.split(";")
    flags = flags.explode("response_flag")
    flags["response_flag"] = flags["response_flag"].str.strip()
    flags = flags[flags["response_flag"].ne("")]
    response_flags = (
        flags.groupby(keys + ["response_flag"], as_index=False)
        .size()
        .rename(columns={"size": "flag_count"})
        .merge(denominator, on=keys, validate="many_to_one")
    )
    response_flags["pct_of_total"] = response_flags["flag_count"] / response_flags[
        "total"
    ]
    response_flags["pct_of_nonmissing"] = response_flags["flag_count"] / response_flags[
        "nonmissing"
    ]
    return abstain, behavior, parse_source, response_flags


def build_pair_specs() -> list[PairSpec]:
    specs: list[PairSpec] = []
    display_order = 0

    def add(
        identity: str,
        family: str,
        scope: str,
        persona_a: str,
        persona_b: str,
    ) -> None:
        nonlocal display_order
        specs.append(
            PairSpec(
                identity=identity,
                family=family,
                scope=scope,
                persona_a=persona_a,
                persona_b=persona_b,
                comparison_label=(
                    f"{PERSONA_LABELS[persona_a]} - {PERSONA_LABELS[persona_b]}"
                ),
                display_order=display_order,
            )
        )
        display_order += 1

    for identity in ["man", "woman"]:
        explicit = f"{identity}_explicit_1"
        family = f"{identity}_primary_seven"
        add(identity, family, "primary", explicit, "neutral")
        for variant in [1, 2, 3]:
            add(
                identity,
                family,
                "primary",
                f"{identity}_implicit_{variant}",
                "neutral",
            )
        for variant in [1, 2, 3]:
            add(
                identity,
                family,
                "primary",
                f"{identity}_implicit_{variant}",
                explicit,
            )

    for suffix in ["explicit_1", "implicit_1", "implicit_2", "implicit_3"]:
        add(
            "man_vs_woman",
            "gender_cross_identity_four",
            "exploratory",
            f"man_{suffix}",
            f"woman_{suffix}",
        )
    return specs


def holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
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


def paired_bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    key: str,
    replicates: int,
) -> tuple[float, float]:
    """Deterministic paired bootstrap via the {-1, 0, +1} distribution."""

    difference = a.astype(np.int8) - b.astype(np.int8)
    counts = np.array(
        [
            int((difference == -1).sum()),
            int((difference == 0).sum()),
            int((difference == 1).sum()),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(difference), counts / len(difference), size=replicates)
    bootstrap = 100 * (draws[:, 2] - draws[:, 0]) / len(difference)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def cochran_q_test(matrix: np.ndarray) -> tuple[float, int, float]:
    values = np.asarray(matrix, dtype=np.int8)
    _, conditions = values.shape
    column_sums = values.sum(axis=0)
    row_sums = values.sum(axis=1)
    total = int(column_sums.sum())
    denominator = conditions * total - int(np.square(row_sums).sum())
    if denominator == 0:
        return 0.0, conditions - 1, 1.0
    statistic = (
        (conditions - 1)
        * (conditions * float(np.square(column_sums).sum()) - total**2)
        / denominator
    )
    return float(statistic), conditions - 1, float(chi2.sf(statistic, conditions - 1))


def _indexed_groups(frame: pd.DataFrame) -> dict[tuple[str, str, str], pd.DataFrame]:
    columns = [
        "question_id",
        "answer_status",
        "is_correct",
        "extracted_stance",
        "numeric_semantic_conflict",
    ]
    return {
        key: group[columns].set_index("question_id").sort_index()
        for key, group in frame.groupby(["model", "persona", "subset"], sort=False)
    }


def _pair_arrays(
    paired: pd.DataFrame, outcome: str
) -> tuple[pd.Series, np.ndarray, np.ndarray, int, int]:
    missing_either = paired["answer_status_a"].eq("missing") | paired[
        "answer_status_b"
    ].eq("missing")
    both_committed = paired["answer_status_a"].eq("committed") & paired[
        "answer_status_b"
    ].eq("committed")

    if outcome == "coverage":
        eligible = ~missing_either
        event_a = paired.loc[eligible, "answer_status_a"].eq("committed").to_numpy()
        event_b = paired.loc[eligible, "answer_status_b"].eq("committed").to_numpy()
        excluded_not_both_committed = 0
    elif outcome == "effective_success":
        eligible = ~missing_either
        event_a = (
            paired.loc[eligible, "answer_status_a"].eq("committed")
            & paired.loc[eligible, "is_correct_a"]
        ).to_numpy()
        event_b = (
            paired.loc[eligible, "answer_status_b"].eq("committed")
            & paired.loc[eligible, "is_correct_b"]
        ).to_numpy()
        excluded_not_both_committed = 0
    elif outcome == "conditional_accuracy":
        eligible = both_committed
        event_a = paired.loc[eligible, "is_correct_a"].to_numpy(dtype=bool)
        event_b = paired.loc[eligible, "is_correct_b"].to_numpy(dtype=bool)
        excluded_not_both_committed = int((~missing_either & ~both_committed).sum())
    elif outcome == "stance_wrong":
        eligible = both_committed
        event_a = paired.loc[eligible, "extracted_stance_a"].eq("wrong").to_numpy()
        event_b = paired.loc[eligible, "extracted_stance_b"].eq("wrong").to_numpy()
        excluded_not_both_committed = int((~missing_either & ~both_committed).sum())
    else:
        raise ValueError(f"Outcome tidak dikenal: {outcome}")
    return (
        eligible,
        event_a,
        event_b,
        int(missing_either.sum()),
        excluded_not_both_committed,
    )


def build_omnibus_results(
    frame: pd.DataFrame,
    indexed: dict[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    del frame  # Pairing uses the already validated indexed views.
    rows: list[dict[str, object]] = []
    for outcome, definition in OUTCOME_DEFINITIONS.items():
        for subset in definition["subsets"]:
            for model in MODEL_ORDER:
                for identity in ["man", "woman"]:
                    personas = [
                        "neutral",
                        f"{identity}_explicit_1",
                        f"{identity}_implicit_1",
                        f"{identity}_implicit_2",
                        f"{identity}_implicit_3",
                    ]
                    wide: pd.DataFrame | None = None
                    for persona in personas:
                        source = indexed[(model, persona, subset)].copy()
                        renamed = source.add_suffix(f"_{persona}")
                        wide = renamed if wide is None else wide.join(renamed, how="inner")
                    assert wide is not None
                    missing_any = np.zeros(len(wide), dtype=bool)
                    committed_all = np.ones(len(wide), dtype=bool)
                    for persona in personas:
                        status = wide[f"answer_status_{persona}"]
                        missing_any |= status.eq("missing").to_numpy()
                        committed_all &= status.eq("committed").to_numpy()
                    if outcome in {"coverage", "effective_success"}:
                        eligible = ~missing_any
                    else:
                        eligible = committed_all

                    event_columns = []
                    for persona in personas:
                        if outcome == "coverage":
                            event = wide.loc[eligible, f"answer_status_{persona}"].eq(
                                "committed"
                            )
                        elif outcome == "effective_success":
                            event = wide.loc[eligible, f"answer_status_{persona}"].eq(
                                "committed"
                            ) & wide.loc[eligible, f"is_correct_{persona}"]
                        elif outcome == "conditional_accuracy":
                            event = wide.loc[eligible, f"is_correct_{persona}"]
                        else:
                            event = wide.loc[
                                eligible, f"extracted_stance_{persona}"
                            ].eq("wrong")
                        event_columns.append(event.to_numpy(dtype=bool))
                    matrix = np.column_stack(event_columns)
                    statistic, degrees_of_freedom, p_value = cochran_q_test(matrix)
                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": definition["label"],
                            "subset": subset,
                            "subset_label": SUBSET_LABELS[subset],
                            "construct_valid": CONSTRUCT_VALID[subset],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "identity": identity,
                            "family": f"{identity}_primary_seven",
                            "conditions": len(personas),
                            "n_complete": int(eligible.sum()),
                            "cochran_q_statistic": statistic,
                            "degrees_of_freedom": degrees_of_freedom,
                            "cochran_q_p_value": p_value,
                            "significant_0_05": bool(p_value < 0.05),
                        }
                    )
    return pd.DataFrame(rows)


def build_paired_effects(
    frame: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_pair_specs()
    indexed = _indexed_groups(frame)
    omnibus = build_omnibus_results(frame, indexed)
    omnibus_lookup = omnibus.set_index(
        ["outcome", "subset", "model", "family"]
    )["cochran_q_p_value"].to_dict()

    rows: list[dict[str, object]] = []
    total_blocks = sum(len(value["subsets"]) for value in OUTCOME_DEFINITIONS.values())
    completed_blocks = 0
    for outcome, definition in OUTCOME_DEFINITIONS.items():
        for subset in definition["subsets"]:
            completed_blocks += 1
            print(
                f"[PAIRED] {completed_blocks}/{total_blocks}: "
                f"{definition['label']} / {SUBSET_LABELS[subset]}"
            )
            expected_n = SUBSET_EXPECTED_N[subset]
            for spec in specs:
                for model in MODEL_ORDER:
                    left = indexed[(model, spec.persona_a, subset)].add_suffix("_a")
                    right = indexed[(model, spec.persona_b, subset)].add_suffix("_b")
                    paired = left.join(right, how="inner", validate="one_to_one")
                    if len(paired) != expected_n:
                        raise ValueError(
                            f"Pair tidak lengkap: {model}/{subset}/{spec.comparison_label} "
                            f"= {len(paired)}, expected {expected_n}"
                        )
                    (
                        eligible,
                        event_a,
                        event_b,
                        excluded_missing,
                        excluded_not_both_committed,
                    ) = _pair_arrays(paired, outcome)
                    if len(event_a) == 0:
                        raise ValueError(
                            f"Tidak ada pasangan eligible: {outcome}/{model}/{subset}/"
                            f"{spec.comparison_label}"
                        )
                    n10 = int((event_a & ~event_b).sum())
                    n01 = int((~event_a & event_b).sum())
                    discordant = n10 + n01
                    p_value = (
                        float(binomtest(n10, discordant, p=0.5).pvalue)
                        if discordant
                        else 1.0
                    )
                    rate_a = float(event_a.mean())
                    rate_b = float(event_b.mean())
                    key = (
                        f"{STATISTICAL_SEED_VERSION}|{outcome}|{subset}|{model}|"
                        f"{spec.persona_a}|{spec.persona_b}"
                    )
                    ci_low, ci_high = paired_bootstrap_difference_ci(
                        event_a,
                        event_b,
                        key,
                        replicates=bootstrap_replicates,
                    )
                    if n10 == 0 and n01 == 0:
                        matched_odds_ratio = math.nan
                    elif n01 == 0:
                        matched_odds_ratio = math.inf
                    else:
                        matched_odds_ratio = n10 / n01

                    nonmissing_pair = ~(
                        paired["answer_status_a"].eq("missing")
                        | paired["answer_status_b"].eq("missing")
                    )
                    commit_rate_a = float(
                        paired.loc[nonmissing_pair, "answer_status_a"]
                        .eq("committed")
                        .mean()
                    )
                    commit_rate_b = float(
                        paired.loc[nonmissing_pair, "answer_status_b"]
                        .eq("committed")
                        .mean()
                    )
                    conflict_either_eligible = int(
                        (
                            eligible
                            & (
                                paired["numeric_semantic_conflict_a"]
                                | paired["numeric_semantic_conflict_b"]
                            )
                        ).sum()
                    )
                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": definition["label"],
                            "eligibility_definition": definition["eligibility"],
                            "event_definition": definition["event"],
                            "subset": subset,
                            "subset_label": SUBSET_LABELS[subset],
                            "construct_valid": CONSTRUCT_VALID[subset],
                            "identity": spec.identity,
                            "family": spec.family,
                            "analysis_scope": spec.scope,
                            "persona_a": spec.persona_a,
                            "persona_b": spec.persona_b,
                            "comparison_label": spec.comparison_label,
                            "display_order": spec.display_order,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "expected_questions": expected_n,
                            "n_pairs": len(event_a),
                            "pair_coverage_pct": 100 * len(event_a) / expected_n,
                            "excluded_missing_either": excluded_missing,
                            "excluded_not_both_committed": excluded_not_both_committed,
                            "eligible_conflict_either": conflict_either_eligible,
                            "event_a": int(event_a.sum()),
                            "event_b": int(event_b.sum()),
                            "rate_a_pct": 100 * rate_a,
                            "rate_b_pct": 100 * rate_b,
                            "risk_difference_pp": 100 * (rate_a - rate_b),
                            "risk_difference_ci95_low_pp": ci_low,
                            "risk_difference_ci95_high_pp": ci_high,
                            "cohen_h": (
                                2 * math.asin(math.sqrt(rate_a))
                                - 2 * math.asin(math.sqrt(rate_b))
                            ),
                            "discordant_a_event_only": n10,
                            "discordant_b_event_only": n01,
                            "matched_odds_ratio": matched_odds_ratio,
                            "mcnemar_exact_p_value": p_value,
                            "paired_nonmissing_commit_rate_a_pct": 100 * commit_rate_a,
                            "paired_nonmissing_commit_rate_b_pct": 100 * commit_rate_b,
                            "statistical_method": "two-sided Exact McNemar",
                            "multiple_comparison_method": "Holm",
                        }
                    )

    paired = pd.DataFrame(rows)
    paired["holm_adjusted_p_value"] = math.nan
    paired["family_size"] = 0
    family_columns = ["outcome", "subset", "model", "family"]
    for _, indices in paired.groupby(family_columns).groups.items():
        paired.loc[indices, "holm_adjusted_p_value"] = holm_adjust(
            paired.loc[indices, "mcnemar_exact_p_value"]
        )
        paired.loc[indices, "family_size"] = len(indices)
    paired["omnibus_cochran_q_p_value"] = paired.apply(
        lambda row: omnibus_lookup.get(
            (row["outcome"], row["subset"], row["model"], row["family"]),
            math.nan,
        ),
        axis=1,
    )
    paired["significant_holm_0_05"] = paired["holm_adjusted_p_value"].lt(0.05)
    paired["decision_significant_0_05"] = np.where(
        paired["analysis_scope"].eq("primary"),
        paired["significant_holm_0_05"]
        & paired["omnibus_cochran_q_p_value"].lt(0.05),
        paired["significant_holm_0_05"],
    )
    paired["abs_cohen_h"] = paired["cohen_h"].abs()
    paired["cohen_h_magnitude"] = pd.cut(
        paired["abs_cohen_h"],
        bins=[-math.inf, 0.2, 0.5, 0.8, math.inf],
        labels=["negligible", "small", "medium", "large"],
        right=False,
    )
    paired = paired.sort_values(
        ["outcome", "subset", "display_order", "model"], kind="stable"
    ).reset_index(drop=True)
    print(f"[PAIRED] Selesai: {len(paired):,} perbandingan")
    return paired, omnibus


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 17,
            "figure.titlesize": 20,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ["png", "svg"]:
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def style_heatmap(ax: plt.Axes, x_rotation: float = 32) -> None:
    """Apply readable tick and colorbar typography to every heatmap."""

    ax.tick_params(axis="x", labelsize=12, rotation=x_rotation, pad=6)
    ax.tick_params(axis="y", labelsize=12, rotation=0, pad=4)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    if ax.collections and ax.collections[0].colorbar is not None:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=11)
        colorbar.ax.yaxis.label.set_size(13)


def _matrix(
    frame: pd.DataFrame,
    value: str,
    index: str = "model_label",
    columns: str = "persona_label",
) -> pd.DataFrame:
    result = frame.pivot(index=index, columns=columns, values=value)
    index_order = [MODEL_LABELS[model] for model in MODEL_ORDER]
    column_order = [PERSONA_LABELS[persona] for persona in PERSONA_ORDER]
    return result.reindex(index=index_order, columns=column_order)


def plot_status_composition(
    overview: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    data = overview.set_index("model_label")
    fig, ax = plt.subplots(figsize=(16, 8.5))
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    rate_columns = {
        "committed": "commit_rate",
        "abstain": "abstain_rate",
        "missing": "missing_rate",
    }
    for status in STATUS_ORDER:
        values = 100 * data[rate_columns[status]].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=status.capitalize(),
            color=STATUS_COLORS[status],
            edgecolor="white",
        )
        for index, value in enumerate(values):
            if value >= 2.0:
                ax.text(
                    index,
                    bottom[index] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
        bottom += values
    ax.set_xticks(x, data.index, rotation=20, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Persentase seluruh respons")
    ax.set_title("ETHICS Gender + Neutral: Komposisi Committed, Abstain, dan Missing")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.text(
        0.5,
        -0.03,
        "Abstain adalah perilaku model; missing adalah kegagalan infrastruktur dan tidak digabungkan.",
        ha="center",
        fontsize=11,
    )
    return save_figure(fig, figure_dir, "01_status_composition_by_model")


def plot_condition_heatmaps(
    summary: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(28, 16), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER, strict=True):
        subset_frame = summary[summary["subset"].eq(subset)].copy()
        matrix = _matrix(subset_frame, "commit_rate") * 100
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Commit rate (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax)
    fig.suptitle(
        "Commit Rate per Model, Persona, dan Subset (Denominator = Seluruh Soal)",
        fontsize=20,
        fontweight="bold",
    )
    paths += save_figure(fig, figure_dir, "02_commit_rate_by_persona_subset_heatmap")

    valid = summary[summary["construct_valid"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(28, 9), constrained_layout=True)
    for ax, subset in zip(axes, ["commonsense", "justice"], strict=True):
        subset_frame = valid[valid["subset"].eq(subset)]
        matrix = _matrix(subset_frame, "accuracy_committed") * 100
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=40,
            vmax=90,
            center=65,
            linewidths=0.5,
            cbar_kws={"label": "Accuracy committed (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax)
    fig.suptitle(
        "Akurasi Conditional pada Subset dengan Construct Valid\n"
        "Baca bersama Figure 02 karena denominator committed berbeda",
        fontsize=20,
        fontweight="bold",
    )
    paths += save_figure(
        fig, figure_dir, "05_accuracy_committed_valid_subsets_heatmap"
    )

    fig, axes = plt.subplots(1, 2, figsize=(28, 9), constrained_layout=True)
    for ax, subset in zip(axes, ["commonsense", "justice"], strict=True):
        subset_frame = valid[valid["subset"].eq(subset)]
        matrix = _matrix(subset_frame, "conflict_sensitivity_delta_pp")
        limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt="+.1f",
            cmap="vlag",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.5,
            cbar_kws={"label": "Delta akurasi (pp)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax)
    fig.suptitle(
        "Sensitivity: Akurasi tanpa Konflik Numerik-Semantik dikurangi Akurasi Utama",
        fontsize=20,
        fontweight="bold",
    )
    paths += save_figure(
        fig, figure_dir, "06_numeric_semantic_conflict_sensitivity_heatmap"
    )

    invalid = summary[~summary["construct_valid"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(28, 9), constrained_layout=True)
    for ax, subset in zip(axes, ["deontology", "virtue"], strict=True):
        subset_frame = invalid[invalid["subset"].eq(subset)]
        matrix = _matrix(subset_frame, "pct_answer_wrong") * 100
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt=".1f",
            cmap="PuOr_r",
            vmin=0,
            vmax=100,
            center=50,
            linewidths=0.5,
            cbar_kws={"label": "Jawaban wrong (%)"},
            ax=ax,
        )
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax)
    fig.suptitle(
        "Distribusi Stance untuk Subset Construct-Invalid (Bukan Akurasi)",
        fontsize=20,
        fontweight="bold",
    )
    paths += save_figure(
        fig, figure_dir, "07_stance_wrong_construct_invalid_subsets_heatmap"
    )
    return paths


def plot_abstain_and_flags(
    frame: pd.DataFrame,
    abstain: pd.DataFrame,
    behavior: pd.DataFrame,
    response_flags: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    persona_nonmissing = frame.groupby("persona")["is_missing"].agg(
        lambda values: int((~values).sum())
    )

    reason_counts = abstain.groupby(["abstain_reason", "persona"])[
        "reason_count"
    ].sum()
    reason_matrix = reason_counts.unstack(fill_value=0).reindex(columns=PERSONA_ORDER)
    reason_matrix = 100 * reason_matrix.div(persona_nonmissing.reindex(PERSONA_ORDER), axis=1)
    reason_order = (
        reason_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
    )
    reason_matrix = reason_matrix.reindex(reason_order)
    reason_matrix.index = [
        ABSTAIN_REASON_LABELS.get(value, value.replace("_", " ").title())
        for value in reason_matrix.index
    ]
    reason_matrix.columns = [PERSONA_LABELS[value] for value in reason_matrix.columns]
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(
        reason_matrix,
        annot=True,
        annot_kws=HEATMAP_ANNOT_KWS,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"label": "% dari respons nonmissing"},
        ax=ax,
    )
    ax.set_title("Alasan Abstain per Persona — Semua Model dan Subset")
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap(ax)
    paths += save_figure(fig, figure_dir, "03_abstain_reason_by_persona_heatmap")

    behavior_counts = behavior.groupby(["behavior_category", "persona"])[
        "behavior_count"
    ].sum()
    behavior_matrix = behavior_counts.unstack(fill_value=0).reindex(columns=PERSONA_ORDER)
    persona_total = frame.groupby("persona").size().reindex(PERSONA_ORDER)
    behavior_matrix = 100 * behavior_matrix.div(persona_total, axis=1)
    behavior_matrix = behavior_matrix.reindex(
        behavior_matrix.sum(axis=1).sort_values(ascending=False).index
    )
    behavior_matrix.index = [value.replace("_", " ").title() for value in behavior_matrix.index]
    behavior_matrix.columns = [PERSONA_LABELS[value] for value in behavior_matrix.columns]
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        behavior_matrix,
        annot=True,
        annot_kws=HEATMAP_ANNOT_KWS,
        fmt=".1f",
        cmap="crest",
        linewidths=0.4,
        cbar_kws={"label": "% dari seluruh respons"},
        ax=ax,
    )
    ax.set_title("Kategori Perilaku Respons per Persona")
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap(ax)
    paths += save_figure(fig, figure_dir, "04_behavior_category_by_persona_heatmap")

    flag_counts = response_flags.groupby(["response_flag", "persona"])[
        "flag_count"
    ].sum()
    flag_matrix = flag_counts.unstack(fill_value=0).reindex(columns=PERSONA_ORDER)
    flag_matrix = 100 * flag_matrix.div(persona_total, axis=1)
    top_flags = flag_matrix.sum(axis=1).nlargest(min(14, len(flag_matrix))).index
    flag_matrix = flag_matrix.loc[top_flags]
    flag_matrix.index = [value.replace("_", " ").title() for value in flag_matrix.index]
    flag_matrix.columns = [PERSONA_LABELS[value] for value in flag_matrix.columns]
    fig, ax = plt.subplots(figsize=(20, max(8.5, 0.75 * len(flag_matrix))))
    sns.heatmap(
        flag_matrix,
        annot=True,
        annot_kws=HEATMAP_ANNOT_KWS,
        fmt=".1f",
        cmap="mako",
        linewidths=0.4,
        cbar_kws={"label": "% dari seluruh respons"},
        ax=ax,
    )
    ax.set_title("Response Flags Terbanyak per Persona (Termasuk AI Disclaimer)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap(ax)
    paths += save_figure(fig, figure_dir, "08_response_flags_by_persona_heatmap")
    return paths


def plot_status_independence(
    independence: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    matrix = independence.pivot(
        index="model_label", columns="scope", values="cramer_v"
    ).reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER])
    matrix = matrix.reindex(columns=["gender_only", "gender_plus_neutral"])
    matrix.columns = ["Gender saja", "Gender + Neutral"]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        matrix,
        annot=True,
        annot_kws=HEATMAP_ANNOT_KWS,
        fmt=".3f",
        cmap="rocket_r",
        vmin=0,
        vmax=max(0.25, float(matrix.max().max())),
        linewidths=0.5,
        cbar_kws={"label": "Cramer's V"},
        ax=ax,
    )
    ax.set_title("Ketergantungan Status Parser/Respons terhadap Persona")
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap(ax, x_rotation=0)
    fig.text(
        0.5,
        -0.02,
        "Semua model signifikan; Cramer's V dipakai untuk besar asosiasi, bukan hanya p-value.",
        ha="center",
        fontsize=11,
    )
    return save_figure(fig, figure_dir, "09_status_persona_independence_cramers_v")


def _paired_matrix(
    subset: pd.DataFrame,
    value: str,
) -> pd.DataFrame:
    order = (
        subset[["comparison_label", "display_order"]]
        .drop_duplicates()
        .sort_values("display_order")["comparison_label"]
        .tolist()
    )
    matrix = subset.pivot(
        index="model_label", columns="comparison_label", values=value
    )
    return matrix.reindex(
        index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=order
    )


def _effect_annotations(
    values: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            marker = "*" if bool(decisions.loc[row, column]) else ""
            annotations.loc[row, column] = f"{value:+.1f}{marker}"
    return annotations


def plot_paired_results(
    paired: pd.DataFrame,
    figure_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    paths: list[Path] = []
    figure_rows: list[dict[str, str]] = []
    for outcome, definition in OUTCOME_DEFINITIONS.items():
        for subset_name in definition["subsets"]:
            selected = paired[
                paired["outcome"].eq(outcome) & paired["subset"].eq(subset_name)
            ].copy()
            values = _paired_matrix(selected, "risk_difference_pp")
            decisions = _paired_matrix(selected, "decision_significant_0_05").astype(bool)
            annotations = _effect_annotations(values, decisions)
            limit = max(5.0, float(np.nanmax(np.abs(values.to_numpy()))))
            fig, ax = plt.subplots(figsize=(32, 10.5))
            sns.heatmap(
                values,
                annot=annotations,
                annot_kws=HEATMAP_ANNOT_KWS,
                fmt="",
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                linewidths=0.45,
                cbar_kws={"label": "Paired risk difference (poin persentase)"},
                ax=ax,
            )
            ax.set_title(
                f"{definition['label']} — {SUBSET_LABELS[subset_name]}\n"
                "* = Holm p < 0,05; primary comparisons juga melewati Cochran's Q"
            )
            ax.set_xlabel("Perbandingan kondisi (kondisi pertama dikurangi pembanding)")
            ax.set_ylabel("")
            style_heatmap(ax)
            stem = f"paired_{outcome}_{subset_name}_effect_holm_heatmap"
            saved = save_figure(fig, figure_dir, stem)
            paths += saved
            figure_rows.append(
                {
                    "stem": stem,
                    "outcome": outcome,
                    "subset": subset_name,
                    "description": (
                        f"Paired risk difference {definition['label']} pada "
                        f"{SUBSET_LABELS[subset_name]} dengan keputusan Holm."
                    ),
                }
            )

            if outcome == "conditional_accuracy":
                raw = _paired_matrix(selected, "mcnemar_exact_p_value")
                adjusted = _paired_matrix(selected, "holm_adjusted_p_value")
                decision = _paired_matrix(
                    selected, "decision_significant_0_05"
                ).astype(bool)
                fig, axes = plt.subplots(2, 1, figsize=(32, 17), constrained_layout=True)
                for ax, matrix, title, markers in [
                    (
                        axes[0],
                        raw,
                        "Raw two-sided Exact McNemar p-value (* raw p < 0,05)",
                        raw.lt(0.05),
                    ),
                    (
                        axes[1],
                        adjusted,
                        "Holm-adjusted p-value (* final hierarchical decision)",
                        decision,
                    ),
                ]:
                    annotations = matrix.copy().astype(object)
                    for row in matrix.index:
                        for column in matrix.columns:
                            value = float(matrix.loc[row, column])
                            marker = "*" if bool(markers.loc[row, column]) else ""
                            annotations.loc[row, column] = (
                                f"<.001{marker}" if value < 0.001 else f"{value:.3f}{marker}"
                            )
                    sns.heatmap(
                        matrix.clip(upper=0.05),
                        annot=annotations,
                        annot_kws=HEATMAP_COMPACT_ANNOT_KWS,
                        fmt="",
                        cmap="rocket_r",
                        vmin=0,
                        vmax=0.05,
                        linewidths=0.4,
                        cbar_kws={"label": "p-value (0–0,05; nilai >0,05 diclip)"},
                        ax=ax,
                    )
                    ax.set_title(title)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    style_heatmap(ax)
                fig.suptitle(
                    "ETHICS Gender: Significance Value untuk Conditional Accuracy — "
                    f"{SUBSET_LABELS[subset_name]}",
                    fontsize=20,
                    fontweight="bold",
                )
                p_stem = (
                    f"paired_conditional_accuracy_{subset_name}_exact_mcnemar_holm_pvalue"
                )
                paths += save_figure(fig, figure_dir, p_stem)
                figure_rows.append(
                    {
                        "stem": p_stem,
                        "outcome": outcome,
                        "subset": subset_name,
                        "description": (
                            "Raw dan Holm-adjusted Exact McNemar p-value; "
                            "bintang memakai keputusan hierarchical."
                        ),
                    }
                )

                n_pairs = _paired_matrix(selected, "n_pairs")
                pair_coverage = _paired_matrix(selected, "pair_coverage_pct")
                annotations = n_pairs.copy().astype(object)
                for row in n_pairs.index:
                    for column in n_pairs.columns:
                        annotations.loc[row, column] = (
                            f"n={int(n_pairs.loc[row, column])}\n"
                            f"{pair_coverage.loc[row, column]:.1f}%"
                        )
                fig, ax = plt.subplots(figsize=(32, 10.5))
                sns.heatmap(
                    pair_coverage,
                    annot=annotations,
                    annot_kws=HEATMAP_COMPACT_ANNOT_KWS,
                    fmt="",
                    cmap="YlGnBu",
                    vmin=0,
                    vmax=100,
                    linewidths=0.4,
                    cbar_kws={"label": "% soal yang sama-sama committed"},
                    ax=ax,
                )
                ax.set_title(
                    "Denominator Paired Conditional Accuracy — "
                    f"{SUBSET_LABELS[subset_name]}"
                )
                ax.set_xlabel("Perbandingan kondisi")
                ax.set_ylabel("")
                style_heatmap(ax)
                n_stem = f"paired_conditional_accuracy_{subset_name}_n_pairs_heatmap"
                paths += save_figure(fig, figure_dir, n_stem)
                figure_rows.append(
                    {
                        "stem": n_stem,
                        "outcome": outcome,
                        "subset": subset_name,
                        "description": "Jumlah dan persentase pasangan sama-sama committed.",
                    }
                )
    return paths, pd.DataFrame(figure_rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_output_index(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "output_index.csv":
            continue
        relative = path.relative_to(output_dir).as_posix()
        row: dict[str, object] = {
            "type": path.parent.name.rstrip("s") if path.parent != output_dir else "metadata",
            "file": relative,
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path, nrows=5)
            row["columns"] = len(table.columns)
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                row["rows"] = max(sum(1 for _ in handle) - 1, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Run the full analysis and return frames for notebook display."""

    repo_root = find_repo_root()
    input_path = Path(input_path) if input_path else default_input_path(repo_root)
    output_dir = Path(output_dir) if output_dir else default_output_dir(repo_root)
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in [table_dir, figure_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ETHICS GENDER PERSONA PUBLICATION ANALYSIS")
    print(f"Version : {ANALYSIS_VERSION}")
    print(f"Input   : {input_path}")
    print(f"Output  : {output_dir}")
    print(f"API call: 0")
    print("=" * 78)

    input_hash_before = sha256_file(input_path)
    frame = load_data(input_path)
    input_validation = validate_input(frame)

    print("[SUMMARY] Menghitung ulang seluruh 216 kondisi × subset")
    condition_summary = build_condition_summary(frame)
    source_summary_path = input_path.parent / "summary_by_condition_subset.csv"
    summary_validation = validate_against_refinement_summary(
        condition_summary, source_summary_path
    )
    model_overview = build_model_overview(frame)
    status_independence = build_status_independence(frame)
    abstain, behavior, parse_source, response_flags = build_behavior_tables(frame)

    paired, omnibus = build_paired_effects(frame, bootstrap_replicates)

    print("[WRITE] Menulis tabel")
    tables = {
        "condition_summary_full.csv": condition_summary,
        "model_overview.csv": model_overview,
        "persona_status_independence.csv": status_independence,
        "abstain_reason_by_condition.csv": abstain,
        "behavior_category_by_condition.csv": behavior,
        "parse_source_by_condition.csv": parse_source,
        "response_flags_by_condition.csv": response_flags,
        "cochran_q_omnibus.csv": omnibus,
        "paired_effects_all_outcomes.csv": paired,
        "paired_coverage_all_subsets.csv": paired[paired["outcome"].eq("coverage")],
        "paired_conditional_accuracy_valid_subsets.csv": paired[
            paired["outcome"].eq("conditional_accuracy")
        ],
        "paired_effective_success_valid_subsets.csv": paired[
            paired["outcome"].eq("effective_success")
        ],
        "paired_stance_construct_invalid_subsets.csv": paired[
            paired["outcome"].eq("stance_wrong")
        ],
    }
    for name, table in tables.items():
        write_csv(table, table_dir / name)

    print("[PLOT] Membuat visual publikasi")
    configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths += plot_status_composition(model_overview, figure_dir)
    figure_paths += plot_condition_heatmaps(condition_summary, figure_dir)
    figure_paths += plot_abstain_and_flags(
        frame, abstain, behavior, response_flags, figure_dir
    )
    figure_paths += plot_status_independence(status_independence, figure_dir)
    paired_paths, paired_figure_index = plot_paired_results(paired, figure_dir)
    figure_paths += paired_paths
    write_csv(paired_figure_index, table_dir / "paired_figure_index.csv")

    input_hash_after = sha256_file(input_path)
    output_validation = {
        "valid": bool(
            input_hash_before == input_hash_after
            and len(condition_summary) == 216
            and len(paired) == 1_080
            and paired["n_pairs"].gt(0).all()
            and paired["holm_adjusted_p_value"].between(0, 1).all()
            and set(paired["family_size"].unique()) == {4, 7}
            and set(paired["outcome"].unique()) == set(OUTCOME_DEFINITIONS)
        ),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "source_input_unchanged": input_hash_before == input_hash_after,
        "condition_summary_rows": len(condition_summary),
        "paired_effect_rows": len(paired),
        "paired_outcome_counts": paired["outcome"].value_counts().to_dict(),
        "holm_family_sizes": sorted(
            int(value) for value in paired["family_size"].unique()
        ),
        "minimum_n_pairs": int(paired["n_pairs"].min()),
        "maximum_n_pairs": int(paired["n_pairs"].max()),
        "figure_files": len(figure_paths),
        "bootstrap_replicates": bootstrap_replicates,
    }
    if not output_validation["valid"]:
        raise ValueError(f"Validasi output gagal: {output_validation}")

    manifest = {
        "version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "method": (
            "Full-data descriptive analysis; paired two-sided Exact McNemar; "
            "Holm correction per model-subset-outcome-family; Cochran Q gate for "
            "primary comparisons; deterministic paired bootstrap confidence intervals."
        ),
        "api_calls": 0,
        "input": str(input_path),
        "input_sha256": input_hash_before,
        "source_summary": str(source_summary_path),
        "output": str(output_dir),
        "row_count": len(frame),
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "status_counts": {
            key: int(value)
            for key, value in frame["answer_status"].value_counts().items()
        },
        "outcome_definitions": OUTCOME_DEFINITIONS,
        "multiple_comparison_family": "model × subset × outcome × prespecified family",
        "statistical_seed_version": STATISTICAL_SEED_VERSION,
        "family_definitions": {
            "man_primary_seven": 7,
            "woman_primary_seven": 7,
            "gender_cross_identity_four": 4,
        },
        "construct_valid_accuracy_subsets": ["commonsense", "justice"],
        "construct_invalid_stance_subsets": ["deontology", "virtue"],
        "input_validation": input_validation,
        "summary_validation": summary_validation,
        "output_validation": output_validation,
        "limitations": [
            "Conditional accuracy is selected on both personas being committed.",
            "Missing is infrastructure failure and is excluded pairwise, never linguistically recovered.",
            "Deontology and virtue are not interpreted as moral-judgement accuracy.",
            "Persona-status association is significant for all six models and must accompany accuracy claims.",
        ],
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(output_validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_index = build_output_index(output_dir)
    write_csv(output_index, output_dir / "output_index.csv")

    print("[DONE] Analisis selesai dan tervalidasi")
    print(
        f"       rows={len(frame):,} | summary={len(condition_summary):,} | "
        f"paired={len(paired):,} | figures={len(figure_paths):,}"
    )
    print(f"       output={output_dir}")
    return {
        "repo_root": repo_root,
        "input_path": input_path,
        "output_dir": output_dir,
        "condition_summary": condition_summary,
        "model_overview": model_overview,
        "status_independence": status_independence,
        "abstain_reason": abstain,
        "behavior_category": behavior,
        "response_flags": response_flags,
        "paired_effects": paired,
        "omnibus": omnibus,
        "figure_paths": figure_paths,
        "output_index": output_index,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build publication analysis for ETHICS Gender Persona + Neutral."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
        help="Deterministic paired-bootstrap replicates for risk-difference CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        input_path=args.input,
        output_dir=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()

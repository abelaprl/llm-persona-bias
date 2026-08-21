"""Final ETHICS Gender Persona analysis aligned with the thesis Chapter VI.

All four ETHICS subsets use full-denominator accuracy against the parser's
verified ground-truth scoring. Abstain and missing responses remain in the
denominator and have ``correct_full=False``. All inferential comparisons are
paired by the same question ID. This module never reparses responses, calls an
API, or modifies the source CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from PIL import Image as PILImage
from scipy.stats import binomtest, chi2


ANALYSIS_VERSION = "ethics-gender-v3-accuracy-paired-holm-20260821"
VISUALIZATION_REVISION = "v3-compact-typography-20260821"
EXPECTED_ROWS = 81_756
EXPECTED_QUESTIONS = 1_514
ALPHA = 0.05

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

PERSONA_ORDER = [
    "neutral",
    "man_explicit_1",
    "man_implicit_1",
    "man_implicit_2",
    "man_implicit_3",
    "woman_explicit_1",
    "woman_implicit_1",
    "woman_implicit_2",
    "woman_implicit_3",
]
PERSONA_LABELS = {
    "neutral": "Netral",
    "man_explicit_1": "Laki-laki E",
    "man_implicit_1": "Laki-laki I1",
    "man_implicit_2": "Laki-laki I2",
    "man_implicit_3": "Laki-laki I3",
    "woman_explicit_1": "Perempuan E",
    "woman_implicit_1": "Perempuan I1",
    "woman_implicit_2": "Perempuan I2",
    "woman_implicit_3": "Perempuan I3",
}
PERSONA_META = {
    "neutral": ("neutral", "N"),
    "man_explicit_1": ("man", "E"),
    "man_implicit_1": ("man", "I1"),
    "man_implicit_2": ("man", "I2"),
    "man_implicit_3": ("man", "I3"),
    "woman_explicit_1": ("woman", "E"),
    "woman_implicit_1": ("woman", "I1"),
    "woman_implicit_2": ("woman", "I2"),
    "woman_implicit_3": ("woman", "I3"),
}
IDENTITY_LABELS = {"neutral": "Netral", "man": "Laki-laki", "woman": "Perempuan"}
SCENARIO_ORDER = ["N", "E", "I1", "I2", "I3"]

STATUS_DISPLAY = {
    "committed": "jawaban dapat ditentukan",
    "abstain": "abstain",
    "missing": "missing",
}
STATUS_DISPLAY_ORDER = ["jawaban dapat ditentukan", "abstain", "missing"]
STATUS_COLORS = {
    "jawaban dapat ditentukan": "#2A9D8F",
    "abstain": "#E9C46A",
    "missing": "#E76F51",
}

ABSTAIN_REASON_LABELS = {
    "both_sides": "Tidak berpihak / kedua sisi",
    "analysis_without_verdict": "Analisis tanpa keputusan",
    "no_opinion_stop": "Tidak menyatakan opini",
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

# Display-only wording for the print revision.  The v3 result tables deliberately
# retain their original labels so this visualization refinement cannot change a
# data or statistical artifact.
ABSTAIN_REASON_FIGURE_LABELS = {
    "both_sides": "Tidak memilih satu sisi",
    "insufficient_context": "Meminta konteks tambahan",
    "placeholder_confusion": "Salah memahami format soal",
    "detector_disagreement": "Keputusan tidak konsisten",
}

USECOLS = [
    "model",
    "persona_folder",
    "question_id",
    "subset",
    "ground_truth",
    "answer_status",
    "is_correct",
    "extracted_stance",
    "gt_stance",
    "abstain_reason",
]

ANNOTATION_FONT_SIZE = 9.5
TICK_FONT_SIZE = 9.5
SUPPORTING_LABEL_FONT_SIZE = 10.5
PANEL_TITLE_FONT_SIZE = 12.5
MAIN_TITLE_FONT_SIZE = 14

HEATMAP_ANNOT_KWS = {
    "fontsize": ANNOTATION_FONT_SIZE,
    "fontweight": "normal",
}
COHENS_H_COLORBAR_LIMIT = 0.8

FIGURE_SIZE_INCHES = {
    "01_accuracy_absolute_all_subsets_heatmap": (11.7, 7.5),
    "02_implicit_minus_explicit_accuracy_delta_heatmap": (11.7, 7.5),
    "03_effect_size_commonsense_holm_heatmap": (7.1, 6.5),
    "03_effect_size_deontology_holm_heatmap": (7.1, 6.5),
    "03_effect_size_justice_holm_heatmap": (7.1, 6.5),
    "03_effect_size_virtue_holm_heatmap": (7.1, 6.5),
    "04_response_status_distribution_by_model": (7.1, 4.5),
    "05_abstain_reason_by_persona_heatmap": (11.7, 5.5),
}


def find_repo_root(start: Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "AGENT.md").exists() and (path / "result" / "ETHICS").exists():
            return path
    raise FileNotFoundError("Repo root tidak ditemukan.")


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
        / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
    )


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    values = series.astype("string").fillna("").str.strip().str.lower()
    unexpected = set(values.unique()) - {"true", "false", "1", "0", ""}
    if unexpected:
        raise ValueError(f"{column}: nilai boolean tidak dikenal: {sorted(unexpected)}")
    return values.map({"true": True, "1": True, "false": False, "0": False, "": False}).astype(bool)


def load_data(path: Path) -> pd.DataFrame:
    print(f"[LOAD] Membaca seluruh data: {path}")
    frame = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    frame["source_is_correct"] = normalize_bool(frame["is_correct"], "is_correct")
    frame["correct_full"] = frame["source_is_correct"]
    frame["persona"] = frame["persona_folder"]
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    frame["subset_label"] = frame["subset"].map(SUBSET_LABELS)
    frame[["identity", "scenario"]] = frame["persona"].map(PERSONA_META).apply(pd.Series)
    frame["identity_label"] = frame["identity"].map(IDENTITY_LABELS)
    frame["persona_label"] = frame["persona"].map(PERSONA_LABELS)
    frame["status_label"] = frame["answer_status"].map(STATUS_DISPLAY)
    frame["ground_truth"] = pd.to_numeric(frame["ground_truth"], errors="raise").astype(int)
    print(f"[LOAD] Selesai: {len(frame):,} baris")
    return frame


def validate_input(frame: pd.DataFrame) -> dict[str, object]:
    key = ["model", "persona", "question_id"]
    duplicate_rows = int(frame.duplicated(key).sum())
    condition_sizes = frame.groupby(["model", "persona"], observed=True).size()
    subset_sizes = frame.groupby(["model", "persona", "subset"], observed=True).size()
    subset_sizes_valid = all(
        int(n) == SUBSET_EXPECTED_N[subset]
        for (_, _, subset), n in subset_sizes.items()
    )

    reference_ids: frozenset[str] | None = None
    identical_ids = True
    for _, group in frame.groupby(["model", "persona"], observed=True, sort=False):
        ids = frozenset(group["question_id"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            identical_ids = False
            break

    nonanswer = frame["answer_status"].isin(["abstain", "missing"])
    answerable = frame["answer_status"].eq("committed")
    stance_match = frame["extracted_stance"].eq(frame["gt_stance"])
    checks = {
        "row_count": int(len(frame)),
        "row_count_expected": bool(len(frame) == EXPECTED_ROWS),
        "question_count": int(frame["question_id"].nunique()),
        "question_count_expected": bool(frame["question_id"].nunique() == EXPECTED_QUESTIONS),
        "model_count": int(frame["model"].nunique()),
        "six_models_present": bool(set(frame["model"]) == set(MODEL_ORDER)),
        "persona_count": int(frame["persona"].nunique()),
        "nine_conditions_present": bool(set(frame["persona"]) == set(PERSONA_ORDER)),
        "four_subsets_present": bool(set(frame["subset"]) == set(SUBSET_ORDER)),
        "duplicate_response_keys": duplicate_rows,
        "no_duplicate_response_keys": bool(duplicate_rows == 0),
        "all_conditions_have_1514_questions": bool(condition_sizes.eq(EXPECTED_QUESTIONS).all()),
        "all_conditions_share_question_ids": bool(identical_ids),
        "subset_sizes_expected": bool(subset_sizes_valid),
        "ground_truth_binary": bool(frame["ground_truth"].isin([0, 1]).all()),
        "abstain_missing_scored_false": bool((~frame.loc[nonanswer, "correct_full"]).all()),
        "answerable_correctness_matches_ground_truth_stance": bool(
            (frame.loc[answerable, "correct_full"] == stance_match.loc[answerable]).all()
        ),
        "unknown_status_labels": int(frame["status_label"].isna().sum()),
    }
    boolean_values = [v for v in checks.values() if isinstance(v, (bool, np.bool_))]
    checks["valid"] = bool(all(boolean_values) and checks["unknown_status_labels"] == 0)
    if not checks["valid"]:
        raise ValueError(f"Validasi input gagal: {checks}")
    print("[VALIDATE] Data penuh, ground truth, status, dan kunci pairing valid")
    return checks


def _aggregate_accuracy(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = (
        frame.groupby(group_columns, observed=True, sort=False)["correct_full"]
        .agg(total_responses="size", n_correct="sum", accuracy="mean")
        .reset_index()
    )
    result["n_incorrect"] = result["total_responses"] - result["n_correct"]
    result["accuracy_pct"] = 100 * result["accuracy"]
    return result


def build_accuracy_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    condition = _aggregate_accuracy(
        frame,
        [
            "subset",
            "subset_label",
            "model",
            "model_label",
            "persona",
            "persona_label",
            "identity",
            "identity_label",
            "scenario",
        ],
    )
    combined = _aggregate_accuracy(
        frame,
        [
            "model",
            "model_label",
            "persona",
            "persona_label",
            "identity",
            "identity_label",
            "scenario",
        ],
    )
    combined.insert(0, "subset", "all_four_subsets")
    combined.insert(1, "subset_label", "Gabungan empat subset")

    gender = frame[frame["identity"].isin(["man", "woman"])].copy()
    gender_overall = _aggregate_accuracy(gender, ["identity", "identity_label"])
    gender_by_subset = _aggregate_accuracy(
        gender, ["subset", "subset_label", "identity", "identity_label"]
    )
    gender_by_model = _aggregate_accuracy(
        gender, ["model", "model_label", "identity", "identity_label"]
    )
    gender_by_scenario = _aggregate_accuracy(
        gender, ["scenario", "identity", "identity_label"]
    )

    def gender_comparison(table: pd.DataFrame, index: list[str]) -> pd.DataFrame:
        if not index:
            row: dict[str, object] = {}
            for identity in ["man", "woman"]:
                selected = table[table["identity"].eq(identity)].iloc[0]
                for metric in ["total_responses", "n_correct", "accuracy_pct"]:
                    row[f"{metric}_{identity}"] = selected[metric]
            row["delta_man_minus_woman_pp"] = (
                row["accuracy_pct_man"] - row["accuracy_pct_woman"]
            )
            return pd.DataFrame([row])
        wide = table.pivot(index=index, columns="identity", values=["total_responses", "n_correct", "accuracy_pct"])
        wide.columns = [f"{metric}_{identity}" for metric, identity in wide.columns]
        wide = wide.reset_index()
        wide["delta_man_minus_woman_pp"] = wide["accuracy_pct_man"] - wide["accuracy_pct_woman"]
        return wide

    return {
        "accuracy_by_subset_model_condition.csv": condition,
        "accuracy_combined_four_subsets_by_model_condition.csv": combined,
        "accuracy_gender_overall.csv": gender_comparison(gender_overall, []),
        "accuracy_gender_comparison_by_subset.csv": gender_comparison(
            gender_by_subset, ["subset", "subset_label"]
        ),
        "accuracy_gender_comparison_by_model.csv": gender_comparison(
            gender_by_model, ["model", "model_label"]
        ),
        "accuracy_gender_comparison_by_scenario.csv": gender_comparison(
            gender_by_scenario, ["scenario"]
        ),
    }


def holm_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    clean = p[valid]
    if not len(clean):
        return result
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted_ranked = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    restored = np.empty_like(adjusted_ranked)
    restored[order] = adjusted_ranked
    result[valid] = restored
    return result


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


def cohens_h(p_a: float, p_b: float) -> float:
    return float(2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b)))


def _condition_series(
    frame: pd.DataFrame, model: str, subset: str, persona: str
) -> pd.Series:
    selected = frame[
        frame["model"].eq(model)
        & frame["subset"].eq(subset)
        & frame["persona"].eq(persona)
    ][["question_id", "correct_full"]]
    if selected["question_id"].duplicated().any():
        raise ValueError(f"Duplikasi kondisi: {model}, {subset}, {persona}")
    return selected.set_index("question_id")["correct_full"].sort_index()


def build_cochran_q(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity in ["man", "woman"]:
                persona_by_scenario = {
                    "N": "neutral",
                    "E": f"{identity}_explicit_1",
                    "I1": f"{identity}_implicit_1",
                    "I2": f"{identity}_implicit_2",
                    "I3": f"{identity}_implicit_3",
                }
                series = {
                    scenario: _condition_series(frame, model, subset, persona)
                    for scenario, persona in persona_by_scenario.items()
                }
                wide = pd.concat(series, axis=1, join="inner")
                if wide.index.duplicated().any() or len(wide) != SUBSET_EXPECTED_N[subset]:
                    raise ValueError(f"Pairing Cochran Q tidak lengkap: {subset}, {model}, {identity}")
                statistic, degrees_of_freedom, p_value = cochran_q_test(wide.to_numpy(dtype=bool))
                row: dict[str, object] = {
                    "subset": subset,
                    "subset_label": SUBSET_LABELS[subset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "identity": identity,
                    "identity_label": IDENTITY_LABELS[identity],
                    "n_pairs": int(len(wide)),
                    "cochran_q_statistic": statistic,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value": p_value,
                    "significant_0_05": bool(p_value < ALPHA),
                }
                for scenario in SCENARIO_ORDER:
                    row[f"accuracy_{scenario}_pct"] = 100 * float(wide[scenario].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _mcnemar_row(
    a: pd.Series,
    b: pd.Series,
    base: dict[str, object],
) -> dict[str, object]:
    paired = pd.concat({"a": a, "b": b}, axis=1, join="inner")
    if paired.index.duplicated().any():
        raise ValueError("Duplikasi pasangan Exact McNemar")
    values_a = paired["a"].to_numpy(dtype=bool)
    values_b = paired["b"].to_numpy(dtype=bool)
    a_correct_b_wrong = int((values_a & ~values_b).sum())
    a_wrong_b_correct = int((~values_a & values_b).sum())
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = (
        float(binomtest(a_correct_b_wrong, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    accuracy_a = float(values_a.mean())
    accuracy_b = float(values_b.mean())
    return {
        **base,
        "n_pairs": int(len(paired)),
        "accuracy_a_pct": 100 * accuracy_a,
        "accuracy_b_pct": 100 * accuracy_b,
        "delta_accuracy_pp": 100 * (accuracy_a - accuracy_b),
        "discordant_a_correct_b_wrong": a_correct_b_wrong,
        "discordant_a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": discordant,
        "exact_mcnemar_p_value": p_value,
        "cohens_h": cohens_h(accuracy_a, accuracy_b),
    }


def build_mcnemar(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_specs = [
        ("E", "N", "E − N", "persona_vs_neutral", 0),
        ("I1", "N", "I1 − N", "persona_vs_neutral", 1),
        ("I2", "N", "I2 − N", "persona_vs_neutral", 2),
        ("I3", "N", "I3 − N", "persona_vs_neutral", 3),
        ("I1", "E", "I1 − E", "implicit_vs_explicit", 4),
        ("I2", "E", "I2 − E", "implicit_vs_explicit", 5),
        ("I3", "E", "I3 − E", "implicit_vs_explicit", 6),
    ]
    scenario_rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity in ["man", "woman"]:
                persona_by_scenario = {
                    "N": "neutral",
                    "E": f"{identity}_explicit_1",
                    "I1": f"{identity}_implicit_1",
                    "I2": f"{identity}_implicit_2",
                    "I3": f"{identity}_implicit_3",
                }
                series = {
                    scenario: _condition_series(frame, model, subset, persona)
                    for scenario, persona in persona_by_scenario.items()
                }
                for scenario_a, scenario_b, label, category, order in scenario_specs:
                    display = f"{IDENTITY_LABELS[identity]} {label}"
                    base = {
                        "comparison_family": "scenario_within_identity",
                        "comparison_category": category,
                        "subset": subset,
                        "subset_label": SUBSET_LABELS[subset],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "identity": identity,
                        "identity_label": IDENTITY_LABELS[identity],
                        "scenario_a": scenario_a,
                        "scenario_b": scenario_b,
                        "condition_a": persona_by_scenario[scenario_a],
                        "condition_b": persona_by_scenario[scenario_b],
                        "comparison_label": display,
                        "display_order": order + (0 if identity == "man" else 7),
                    }
                    scenario_rows.append(_mcnemar_row(series[scenario_a], series[scenario_b], base))
    scenario = pd.DataFrame(scenario_rows)
    scenario["holm_adjusted_p_value"] = scenario.groupby(
        ["subset", "model", "identity"], observed=True
    )["exact_mcnemar_p_value"].transform(holm_adjust)
    scenario["holm_family_size"] = 7
    scenario["significant_holm_0_05"] = scenario["holm_adjusted_p_value"].lt(ALPHA)

    gender_rows: list[dict[str, object]] = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for order, scenario_name in enumerate(["E", "I1", "I2", "I3"]):
                suffix = "explicit_1" if scenario_name == "E" else f"implicit_{scenario_name[-1]}"
                man_persona = f"man_{suffix}"
                woman_persona = f"woman_{suffix}"
                man = _condition_series(frame, model, subset, man_persona)
                woman = _condition_series(frame, model, subset, woman_persona)
                base = {
                    "comparison_family": "gender_within_scenario",
                    "comparison_category": "man_vs_woman",
                    "subset": subset,
                    "subset_label": SUBSET_LABELS[subset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "identity": "man_vs_woman",
                    "identity_label": "Laki-laki vs perempuan",
                    "scenario_a": scenario_name,
                    "scenario_b": scenario_name,
                    "condition_a": man_persona,
                    "condition_b": woman_persona,
                    "comparison_label": f"Laki-laki − Perempuan {scenario_name}",
                    "display_order": 14 + order,
                }
                gender_rows.append(_mcnemar_row(man, woman, base))
    gender = pd.DataFrame(gender_rows)
    gender["holm_adjusted_p_value"] = gender.groupby(
        ["subset", "model"], observed=True
    )["exact_mcnemar_p_value"].transform(holm_adjust)
    gender["holm_family_size"] = 4
    gender["significant_holm_0_05"] = gender["holm_adjusted_p_value"].lt(ALPHA)
    return scenario, gender


def build_status_table(frame: pd.DataFrame) -> pd.DataFrame:
    status = (
        frame.groupby(["model", "model_label", "status_label"], observed=True)
        .size()
        .rename("n_responses")
        .reset_index()
    )
    totals = status.groupby("model", observed=True)["n_responses"].transform("sum")
    status["percentage"] = 100 * status["n_responses"] / totals
    return status


def build_abstain_reason_table(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["answer_status"].eq("abstain")].copy()
    selected["reason"] = selected["abstain_reason"].replace("", "unspecified")
    selected["reason_label"] = selected["reason"].map(ABSTAIN_REASON_LABELS).fillna(
        selected["reason"].str.replace("_", " ").str.title()
    )
    result = (
        selected.groupby(["persona", "persona_label", "reason", "reason_label"], observed=True)
        .size()
        .rename("n_abstain")
        .reset_index()
    )
    denominators = selected.groupby("persona", observed=True).size().rename("persona_abstain_total")
    result = result.merge(denominators, on="persona", validate="many_to_one")
    result["percentage_within_persona_abstain"] = 100 * result["n_abstain"] / result["persona_abstain_total"]
    return result


def build_significant_summaries(all_comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categories = ["persona_vs_neutral", "implicit_vs_explicit", "man_vs_woman"]
    full = (
        all_comparisons.groupby("comparison_category", observed=True)
        .agg(total_tests=("significant_holm_0_05", "size"), significant_results=("significant_holm_0_05", "sum"))
        .reindex(categories)
        .reset_index()
    )
    by_subset = (
        all_comparisons.groupby(
            ["comparison_category", "subset", "subset_label"], observed=True
        )
        .agg(
            total_tests=("significant_holm_0_05", "size"),
            significant_results=("significant_holm_0_05", "sum"),
        )
        .reset_index()
    )
    by_model = (
        all_comparisons.groupby(
            ["comparison_category", "model", "model_label"], observed=True
        )
        .agg(
            total_tests=("significant_holm_0_05", "size"),
            significant_results=("significant_holm_0_05", "sum"),
        )
        .reset_index()
    )
    return full, by_subset, by_model


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": SUPPORTING_LABEL_FONT_SIZE,
            "axes.labelsize": SUPPORTING_LABEL_FONT_SIZE,
            "axes.titlesize": PANEL_TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": TICK_FONT_SIZE,
            "figure.titlesize": MAIN_TITLE_FONT_SIZE,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )


def format_decimal_comma(value: float, decimals: int = 1, *, signed: bool = False) -> str:
    """Format a numeric figure label using the Indonesian decimal comma."""
    template = f"{{:{'+' if signed else ''}.{decimals}f}}"
    return template.format(float(value)).replace(".", ",")


def format_tick_decimal_comma(value: float, _position: float | None = None) -> str:
    """Keep integer ticks compact and use commas for non-integer decimals."""
    if np.isclose(value, round(value), atol=1e-10):
        return f"{int(round(value))}"
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", ",")


DECIMAL_COMMA_TICK_FORMATTER = FuncFormatter(format_tick_decimal_comma)


def numeric_annotations(
    values: pd.DataFrame,
    decimals: int,
    *,
    signed: bool = False,
    suffix: str = "",
) -> pd.DataFrame:
    """Build a string matrix so heatmap annotations never use a decimal point."""
    return values.map(
        lambda value: f"{format_decimal_comma(value, decimals, signed=signed)}{suffix}"
    )


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=SUPPORTING_LABEL_FONT_SIZE, fontweight="normal")
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    colorbar.update_ticks()
    # update_ticks may restore the scalar formatter, so apply the Indonesian
    # formatter only after the locator/ticks have been finalized.
    colorbar.ax.yaxis.set_major_formatter(DECIMAL_COMMA_TICK_FORMATTER)


def style_heatmap(ax: plt.Axes, x_rotation: float = 35) -> None:
    ax.tick_params(axis="x", labelrotation=x_rotation, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelrotation=0, labelsize=TICK_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right" if x_rotation else "center")


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ["png", "svg"]:
        path = figure_dir / f"{stem}.{suffix}"
        # No tight bounding box: the exported canvas must preserve the exact
        # print dimensions requested for the thesis layout.
        fig.savefig(path, dpi=300 if suffix == "png" else None, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_accuracy_absolute(condition: pd.DataFrame, figure_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    figure_data = condition.copy()
    fig = plt.figure(figsize=(11.7, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = None
    for index, (ax, subset) in enumerate(zip(axes, SUBSET_ORDER)):
        selected = condition[condition["subset"].eq(subset)]
        matrix = selected.pivot(index="model_label", columns="persona_label", values="accuracy_pct")
        matrix = matrix.reindex(
            index=[MODEL_LABELS[m] for m in MODEL_ORDER],
            columns=[PERSONA_LABELS[p] for p in PERSONA_ORDER],
        )
        heatmap = sns.heatmap(
            matrix,
            annot=numeric_annotations(matrix, 1),
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt="",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            linewidths=0.45,
            cbar=index == 0,
            cbar_ax=colorbar_ax if index == 0 else None,
            ax=ax,
        )
        if index == 0:
            colorbar = heatmap.collections[0].colorbar
        ax.set_title(
            SUBSET_LABELS[subset],
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="semibold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax, 42)
    style_colorbar(colorbar, "Akurasi (%)")
    fig.suptitle(
        "Akurasi ETHICS terhadap Ground Truth",
        fontsize=MAIN_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "01_accuracy_absolute_all_subsets_heatmap"), figure_data


def build_implicit_delta_data(condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset in SUBSET_ORDER:
        for model in MODEL_ORDER:
            for identity in ["man", "woman"]:
                explicit = condition[
                    condition["subset"].eq(subset)
                    & condition["model"].eq(model)
                    & condition["persona"].eq(f"{identity}_explicit_1")
                ]["accuracy_pct"].iloc[0]
                for variant in [1, 2, 3]:
                    implicit = condition[
                        condition["subset"].eq(subset)
                        & condition["model"].eq(model)
                        & condition["persona"].eq(f"{identity}_implicit_{variant}")
                    ]["accuracy_pct"].iloc[0]
                    rows.append(
                        {
                            "subset": subset,
                            "subset_label": SUBSET_LABELS[subset],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "identity": identity,
                            "identity_label": IDENTITY_LABELS[identity],
                            "implicit_scenario": f"I{variant}",
                            "comparison_label": f"{IDENTITY_LABELS[identity]} I{variant} − E",
                            "explicit_accuracy_pct": explicit,
                            "implicit_accuracy_pct": implicit,
                            "delta_accuracy_pp": implicit - explicit,
                        }
                    )
    return pd.DataFrame(rows)


def plot_implicit_delta(delta: pd.DataFrame, figure_dir: Path) -> list[Path]:
    limit = max(5.0, float(np.ceil(delta["delta_accuracy_pp"].abs().max())))
    fig = plt.figure(figsize=(11.7, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = None
    column_order = [
        f"{IDENTITY_LABELS[i]} I{v} − E"
        for i in ["man", "woman"]
        for v in [1, 2, 3]
    ]
    for index, (ax, subset) in enumerate(zip(axes, SUBSET_ORDER)):
        selected = delta[delta["subset"].eq(subset)]
        matrix = selected.pivot(index="model_label", columns="comparison_label", values="delta_accuracy_pp")
        matrix = matrix.reindex(index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=column_order)
        heatmap = sns.heatmap(
            matrix,
            annot=numeric_annotations(matrix, 1, signed=True),
            annot_kws=HEATMAP_ANNOT_KWS,
            fmt="",
            cmap="vlag",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.45,
            cbar=index == 0,
            cbar_ax=colorbar_ax if index == 0 else None,
            ax=ax,
        )
        if index == 0:
            colorbar = heatmap.collections[0].colorbar
        ax.set_title(
            SUBSET_LABELS[subset],
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="semibold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_heatmap(ax, 42)
    style_colorbar(colorbar, "Delta akurasi (poin persentase)")
    fig.suptitle(
        "Delta Akurasi Skenario Implisit terhadap Eksplisit",
        fontsize=MAIN_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    return save_figure(fig, figure_dir, "02_implicit_minus_explicit_accuracy_delta_heatmap")


def _effect_annotations(values: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            marker = "*" if bool(decisions.loc[row, column]) else ""
            annotations.loc[row, column] = (
                f"{format_decimal_comma(values.loc[row, column], 3, signed=True)}{marker}"
            )
    return annotations


def plot_effect_sizes(all_comparisons: pd.DataFrame, figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    limit = COHENS_H_COLORBAR_LIMIT
    scenario_labels = ["E − N", "I1 − N", "I2 − N", "I3 − N", "I1 − E", "I2 − E", "I3 − E"]
    gender_labels = ["E", "I1", "I2", "I3"]
    for subset in SUBSET_ORDER:
        selected = all_comparisons[all_comparisons["subset"].eq(subset)]
        all_values = selected.pivot(
            index="model_label", columns="comparison_label", values="cohens_h"
        )
        all_decisions = selected.pivot(
            index="model_label",
            columns="comparison_label",
            values="significant_holm_0_05",
        )
        model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
        fig = plt.figure(figsize=(7.1, 6.5), constrained_layout=True)
        grid = fig.add_gridspec(
            3,
            2,
            width_ratios=[1, 0.045],
            height_ratios=[1, 1, 1],
        )
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[2, 0]),
        ]
        colorbar_ax = fig.add_subplot(grid[:, 1])
        panel_specs = [
            (
                "Laki-laki",
                [f"Laki-laki {label}" for label in scenario_labels],
                scenario_labels,
            ),
            (
                "Perempuan",
                [f"Perempuan {label}" for label in scenario_labels],
                scenario_labels,
            ),
            (
                "Laki-laki − Perempuan",
                [f"Laki-laki − Perempuan {label}" for label in gender_labels],
                gender_labels,
            ),
        ]
        colorbar = None
        for panel_index, (ax, (panel_title, source_columns, short_columns)) in enumerate(
            zip(axes, panel_specs)
        ):
            values = all_values.reindex(index=model_labels, columns=source_columns).copy()
            decisions = (
                all_decisions.reindex(index=model_labels, columns=source_columns)
                .astype(bool)
                .copy()
            )
            values.columns = short_columns
            decisions.columns = short_columns
            annotations = _effect_annotations(values, decisions)
            heatmap = sns.heatmap(
                values,
                annot=annotations,
                annot_kws=HEATMAP_ANNOT_KWS,
                fmt="",
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                linewidths=0.45,
                cbar=panel_index == 0,
                cbar_ax=colorbar_ax if panel_index == 0 else None,
                ax=ax,
            )
            if panel_index == 0:
                colorbar = heatmap.collections[0].colorbar
            ax.set_title(
                panel_title,
                fontsize=PANEL_TITLE_FONT_SIZE,
                fontweight="semibold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            style_heatmap(ax, 0)
        style_colorbar(colorbar, "Cohen’s h")
        fig.suptitle(
            f"Effect Size Perbandingan Akurasi — {SUBSET_LABELS[subset]}\n"
            "* = signifikan setelah koreksi Holm (α = 0,05)",
            fontsize=MAIN_TITLE_FONT_SIZE,
            fontweight="bold",
        )
        fig.supxlabel(
            "Perbandingan (skenario pertama dikurangi skenario pembanding)",
            fontsize=SUPPORTING_LABEL_FONT_SIZE,
            fontweight="normal",
        )
        paths += save_figure(fig, figure_dir, f"03_effect_size_{subset}_holm_heatmap")
    return paths


def plot_status_distribution(status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    matrix = status.pivot(index="model_label", columns="status_label", values="percentage")
    matrix = matrix.reindex(index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=STATUS_DISPLAY_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.17, top=0.72)
    left = np.zeros(len(matrix))
    for label in STATUS_DISPLAY_ORDER:
        values = matrix[label].to_numpy()
        bars = ax.barh(
            matrix.index,
            values,
            left=left,
            label=label,
            color=STATUS_COLORS[label],
            edgecolor="white",
        )
        for bar, value, offset in zip(bars, values, left):
            if value >= 5.0:
                ax.text(
                    offset + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{format_decimal_comma(value, 1)}%",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE,
                    fontweight="normal",
                )
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Persentase seluruh respons")
    ax.set_ylabel("")
    fig.suptitle(
        "Distribusi Status Respons menurut Model",
        fontsize=MAIN_TITLE_FONT_SIZE,
        fontweight="bold",
        y=0.98,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.64, 0.88),
        ncol=3,
        fontsize=TICK_FONT_SIZE,
        frameon=False,
    )
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA_TICK_FORMATTER)
    return save_figure(fig, figure_dir, "04_response_status_distribution_by_model")


def plot_abstain_reasons(abstain: pd.DataFrame, figure_dir: Path) -> list[Path]:
    display_data = abstain.copy()
    display_data["figure_reason_label"] = display_data["reason"].map(
        ABSTAIN_REASON_FIGURE_LABELS
    ).fillna(display_data["reason_label"])
    matrix = display_data.pivot(
        index="figure_reason_label",
        columns="persona_label",
        values="percentage_within_persona_abstain",
    ).fillna(0)
    matrix = matrix.reindex(columns=[PERSONA_LABELS[p] for p in PERSONA_ORDER])
    matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11.7, 5.5), constrained_layout=True)
    heatmap = sns.heatmap(
        matrix,
        annot=numeric_annotations(matrix, 1),
        annot_kws=HEATMAP_ANNOT_KWS,
        fmt="",
        cmap="mako",
        linewidths=0.45,
        ax=ax,
    )
    ax.set_title(
        "Alasan Abstain menurut Persona",
        fontsize=MAIN_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_heatmap(ax, 42)
    style_colorbar(heatmap.collections[0].colorbar, "% dari abstain pada persona")
    return save_figure(fig, figure_dir, "05_abstain_reason_by_persona_heatmap")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_output_index(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "output_index.csv":
            continue
        relative = path.relative_to(output_dir).as_posix()
        row: dict[str, object] = {
            "artifact_type": path.parent.name.rstrip("s") if path.parent != output_dir else "metadata",
            "file": relative,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path, encoding="utf-8-sig")
            row["rows"] = int(len(table))
            row["columns"] = int(len(table.columns))
        rows.append(row)
    return pd.DataFrame(rows)


def _frame_equal_on_columns(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> bool:
    left = a[columns].sort_values(columns[:-1]).reset_index(drop=True)
    right = b[columns].sort_values(columns[:-1]).reset_index(drop=True)
    if left.shape != right.shape:
        return False
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) or pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(pd.to_numeric(left[column]), pd.to_numeric(right[column]), equal_nan=True, atol=1e-12):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(paths)}


def _svg_text(path: Path) -> str:
    root = ET.parse(path).getroot()
    return "".join(root.itertext())


def _svg_size_inches(path: Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()

    def points_to_inches(raw_value: str) -> float:
        if not raw_value.endswith("pt"):
            raise ValueError(f"Ukuran SVG bukan point: {raw_value}")
        return float(raw_value[:-2]) / 72.0

    return points_to_inches(root.attrib["width"]), points_to_inches(root.attrib["height"])


def validate_print_figure_files(figure_dir: Path) -> dict[str, object]:
    """Validate exact print canvas, PNG resolution, and matching SVG canvas."""
    details: dict[str, object] = {}
    for stem, (width_inches, height_inches) in FIGURE_SIZE_INCHES.items():
        png_path = figure_dir / f"{stem}.png"
        svg_path = figure_dir / f"{stem}.svg"
        expected_pixels = (
            int(round(width_inches * 300)),
            int(round(height_inches * 300)),
        )
        with PILImage.open(png_path) as image:
            png_pixels = tuple(int(value) for value in image.size)
            dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
        svg_inches = _svg_size_inches(svg_path)
        details[stem] = {
            "requested_inches": [width_inches, height_inches],
            "expected_png_pixels": list(expected_pixels),
            "actual_png_pixels": list(png_pixels),
            "png_dpi": list(dpi),
            "svg_inches": list(svg_inches),
            "png_canvas_valid": bool(png_pixels == expected_pixels),
            "png_dpi_valid": bool(all(abs(value - 300.0) < 0.1 for value in dpi)),
            "svg_canvas_valid": bool(
                np.allclose(svg_inches, (width_inches, height_inches), atol=1e-6)
            ),
        }
    return details


def validate_figure_annotation_text(
    condition: pd.DataFrame,
    implicit_delta: pd.DataFrame,
    all_comparisons: pd.DataFrame,
    status: pd.DataFrame,
    abstain: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, bool]:
    """Confirm every plotted annotation is formatted from its v3 table value."""

    def all_present(expected: set[str], stem: str) -> bool:
        text = _svg_text(figure_dir / f"{stem}.svg")
        return bool(expected) and all(label in text for label in expected)

    checks: dict[str, bool] = {}
    checks["accuracy_absolute"] = all_present(
        {format_decimal_comma(value, 1) for value in condition["accuracy_pct"]},
        "01_accuracy_absolute_all_subsets_heatmap",
    )
    checks["implicit_delta"] = all_present(
        {
            format_decimal_comma(value, 1, signed=True)
            for value in implicit_delta["delta_accuracy_pp"]
        },
        "02_implicit_minus_explicit_accuracy_delta_heatmap",
    )
    for subset in SUBSET_ORDER:
        selected = all_comparisons[all_comparisons["subset"].eq(subset)]
        expected = {
            f"{format_decimal_comma(row.cohens_h, 3, signed=True)}"
            f"{'*' if row.significant_holm_0_05 else ''}"
            for row in selected[["cohens_h", "significant_holm_0_05"]].itertuples(index=False)
        }
        checks[f"effect_size_{subset}"] = all_present(
            expected, f"03_effect_size_{subset}_holm_heatmap"
        )
    status_expected = {
        f"{format_decimal_comma(value, 1)}%"
        for value in status.loc[status["percentage"].ge(5.0), "percentage"]
    }
    checks["response_status"] = all_present(
        status_expected, "04_response_status_distribution_by_model"
    )
    checks["abstain_reason"] = all_present(
        {
            format_decimal_comma(value, 1)
            for value in abstain["percentage_within_persona_abstain"]
        },
        "05_abstain_reason_by_persona_heatmap",
    )

    accuracy_text = _svg_text(
        figure_dir / "01_accuracy_absolute_all_subsets_heatmap.svg"
    )
    delta_text = _svg_text(
        figure_dir / "02_implicit_minus_explicit_accuracy_delta_heatmap.svg"
    )
    abstain_text = _svg_text(
        figure_dir / "05_abstain_reason_by_persona_heatmap.svg"
    )
    effect_texts = [
        _svg_text(figure_dir / f"03_effect_size_{subset}_holm_heatmap.svg")
        for subset in SUBSET_ORDER
    ]
    checks["accuracy_title_exact"] = bool(
        "Akurasi ETHICS terhadap Ground Truth" in accuracy_text
        and "Seluruh Respons Masuk Denominator" not in accuracy_text
    )
    checks["accuracy_one_colorbar_label"] = bool(
        accuracy_text.count("Akurasi (%)") == 1
    )
    checks["delta_title_exact"] = bool(
        "Delta Akurasi Skenario Implisit terhadap Eksplisit" in delta_text
    )
    checks["delta_one_colorbar_label"] = bool(
        delta_text.count("Delta akurasi (poin persentase)") == 1
    )
    checks["effect_axis_label_exact"] = bool(
        all(
            "Perbandingan (skenario pertama dikurangi skenario pembanding)"
            in text
            for text in effect_texts
        )
    )
    checks["effect_holm_note_present"] = bool(
        all("* = signifikan setelah koreksi Holm (α = 0,05)" in text for text in effect_texts)
    )
    checks["effect_colorbar_decimal_comma"] = bool(
        all("0,8" in text and "0.8" not in text for text in effect_texts)
    )
    checks["abstain_reason_labels_revised"] = bool(
        all(label in abstain_text for label in ABSTAIN_REASON_FIGURE_LABELS.values())
        and all(
            old_label not in abstain_text
            for old_label in [
                "Tidak berpihak / kedua sisi",
                "Konteks dianggap kurang",
                "Bingung placeholder",
                "Detektor tidak sepakat",
            ]
        )
    )
    return checks


def validate_figure_typography(figure_dir: Path) -> dict[str, bool]:
    """Confirm the lighter v3 hierarchy in the rendered SVG files."""

    panel_title_stems = {
        "01_accuracy_absolute_all_subsets_heatmap",
        "02_implicit_minus_explicit_accuracy_delta_heatmap",
        *{
            f"03_effect_size_{subset}_holm_heatmap"
            for subset in SUBSET_ORDER
        },
    }
    checks: dict[str, bool] = {}

    def is_decimal_number(text: str) -> bool:
        if "," not in text:
            return False
        remainder = text.strip()
        for character in "+-\u22120123456789,%*":
            remainder = remainder.replace(character, "")
        return remainder == ""

    for stem in FIGURE_SIZE_INCHES:
        svg_path = figure_dir / f"{stem}.svg"
        raw_svg = svg_path.read_text(encoding="utf-8")
        root = ET.parse(svg_path).getroot()
        text_elements = [
            element
            for element in root.iter()
            if element.tag.endswith("text")
        ]
        numeric_styles = [
            element.attrib.get("style", "")
            for element in text_elements
            if is_decimal_number("".join(element.itertext()).strip())
        ]
        checks[f"{stem}_numeric_text_light"] = bool(
            numeric_styles
            and all(
                f"font-size: {ANNOTATION_FONT_SIZE}px" in style
                and "font-weight:" not in style
                for style in numeric_styles
            )
        )
        checks[f"{stem}_main_title_size"] = bool(
            f"font-size: {MAIN_TITLE_FONT_SIZE}px" in raw_svg
        )
        checks[f"{stem}_old_large_fonts_absent"] = bool(
            all(
                f"font-size: {old_size}px" not in raw_svg
                for old_size in (16, 17, 19, 20)
            )
        )
        if stem in panel_title_stems:
            checks[f"{stem}_panel_title_size"] = bool(
                f"font-size: {PANEL_TITLE_FONT_SIZE}px" in raw_svg
            )
    return checks


def refresh_visualizations_only(
    output_dir: Path | str | None = None,
    *,
    effect_sizes_only: bool = False,
) -> dict[str, object]:
    """Regenerate v3 figures from frozen v3 tables without recomputing results."""
    repo_root = find_repo_root()
    output_dir = Path(output_dir) if output_dir else default_output_dir(repo_root)
    output_dir = output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ETHICS GENDER PERSONA — V3 PRINT VISUALIZATION REFRESH")
    print(f"Output: {output_dir}")
    figure_scope = "effect-sizes-only" if effect_sizes_only else "all"
    print(
        f"Mode  : figures-only ({figure_scope}) | API call: 0 | "
        "tabel/statistik tidak ditulis ulang"
    )
    print("=" * 78)

    table_paths = sorted(table_dir.glob("*.csv"))
    if len(table_paths) != 20:
        raise ValueError(f"Diharapkan 20 tabel v3, ditemukan {len(table_paths)}")
    table_hashes_before = _file_hashes(table_paths)
    existing_figure_paths = sorted(figure_dir.glob("*.png")) + sorted(
        figure_dir.glob("*.svg")
    )
    figure_hashes_before = _file_hashes(existing_figure_paths)
    non_effect_figure_hashes_before = {
        name: digest
        for name, digest in figure_hashes_before.items()
        if not name.startswith("03_effect_size_")
    }
    protected_metadata = [
        metadata_dir / "analysis_manifest.json",
        metadata_dir / "validation.json",
    ]
    metadata_hashes_before = _file_hashes(protected_metadata)

    manifest = json.loads((metadata_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    input_path = Path(manifest["input"])
    input_hash_before = sha256_file(input_path)

    def read_table(name: str) -> pd.DataFrame:
        return pd.read_csv(table_dir / name, encoding="utf-8-sig", low_memory=False)

    condition = read_table("figure_data_accuracy_absolute.csv")
    implicit_delta = read_table("figure_data_implicit_minus_explicit_delta.csv")
    all_comparisons = read_table("figure_data_effect_sizes.csv")
    status = read_table("figure_data_response_status.csv")
    abstain = read_table("figure_data_abstain_reason.csv")

    exact_duplicate_pairs = {
        "accuracy_absolute": (
            "figure_data_accuracy_absolute.csv",
            "accuracy_by_subset_model_condition.csv",
        ),
        "effect_sizes": (
            "figure_data_effect_sizes.csv",
            "mcnemar_all_comparisons.csv",
        ),
        "response_status": (
            "figure_data_response_status.csv",
            "response_status_by_model.csv",
        ),
        "abstain_reason": (
            "figure_data_abstain_reason.csv",
            "abstain_reason_by_persona.csv",
        ),
    }
    figure_data_matches_tables = {
        name: sha256_file(table_dir / left) == sha256_file(table_dir / right)
        for name, (left, right) in exact_duplicate_pairs.items()
    }
    recomputed_delta = build_implicit_delta_data(condition)
    figure_data_matches_tables["implicit_delta"] = _frame_equal_on_columns(
        implicit_delta,
        recomputed_delta,
        [
            "subset",
            "model",
            "identity",
            "implicit_scenario",
            "explicit_accuracy_pct",
            "implicit_accuracy_pct",
            "delta_accuracy_pp",
        ],
    )
    if not all(figure_data_matches_tables.values()):
        raise ValueError(
            f"Figure-data tidak identik dengan tabel v3: {figure_data_matches_tables}"
        )

    if effect_sizes_only:
        print("[PLOT] Membuat ulang hanya 4 effect size portrait (PNG 300 DPI + SVG)")
    else:
        print("[PLOT] Membuat ulang 8 desain pada ukuran cetak (PNG 300 DPI + SVG)")
    configure_plot_style()
    figure_paths: list[Path] = []
    if effect_sizes_only:
        figure_paths += plot_effect_sizes(all_comparisons, figure_dir)
    else:
        paths, _ = plot_accuracy_absolute(condition, figure_dir)
        figure_paths += paths
        figure_paths += plot_implicit_delta(implicit_delta, figure_dir)
        figure_paths += plot_effect_sizes(all_comparisons, figure_dir)
        figure_paths += plot_status_distribution(status, figure_dir)
        figure_paths += plot_abstain_reasons(abstain, figure_dir)

    table_hashes_after = _file_hashes(table_paths)
    metadata_hashes_after = _file_hashes(protected_metadata)
    input_hash_after = sha256_file(input_path)
    dimensions = validate_print_figure_files(figure_dir)
    annotation_checks = validate_figure_annotation_text(
        condition,
        implicit_delta,
        all_comparisons,
        status,
        abstain,
        figure_dir,
    )
    typography_checks = validate_figure_typography(figure_dir)
    png_files = sorted(figure_dir.glob("*.png"))
    svg_files = sorted(figure_dir.glob("*.svg"))
    figure_hashes_after = _file_hashes(png_files + svg_files)
    non_effect_figure_hashes_after = {
        name: digest
        for name, digest in figure_hashes_after.items()
        if not name.startswith("03_effect_size_")
    }
    non_effect_figures_unchanged = (
        non_effect_figure_hashes_before == non_effect_figure_hashes_after
    )
    expected_effect_files = {
        f"03_effect_size_{subset}_holm_heatmap.{extension}"
        for subset in SUBSET_ORDER
        for extension in ("png", "svg")
    }
    generated_figure_files = {path.name for path in figure_paths}
    generated_scope_valid = (
        generated_figure_files == expected_effect_files
        if effect_sizes_only
        else len(generated_figure_files) == 16
    )
    shared_effect_limit = COHENS_H_COLORBAR_LIMIT
    size_checks_valid = all(
        detail["png_canvas_valid"]
        and detail["png_dpi_valid"]
        and detail["svg_canvas_valid"]
        for detail in dimensions.values()
    )
    validation = {
        "valid": bool(
            table_hashes_before == table_hashes_after
            and metadata_hashes_before == metadata_hashes_after
            and input_hash_before == input_hash_after
            and all(figure_data_matches_tables.values())
            and size_checks_valid
            and all(annotation_checks.values())
            and all(typography_checks.values())
            and len(png_files) == 8
            and len(svg_files) == 8
            and all(path.stat().st_size > 0 for path in png_files + svg_files)
            and generated_scope_valid
            and (not effect_sizes_only or non_effect_figures_unchanged)
        ),
        "analysis_version": ANALYSIS_VERSION,
        "visualization_revision": VISUALIZATION_REVISION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "mode": "figures-only",
        "figure_scope": figure_scope,
        "api_calls": 0,
        "tables_recomputed": False,
        "statistics_recomputed": False,
        "method_changed": False,
        "table_csv_files": len(table_paths),
        "tables_byte_identical_before_after": bool(table_hashes_before == table_hashes_after),
        "analysis_metadata_unchanged": bool(metadata_hashes_before == metadata_hashes_after),
        "source_csv_unchanged": bool(input_hash_before == input_hash_after),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "table_sha256_before": table_hashes_before,
        "table_sha256_after": table_hashes_after,
        "figure_data_matches_v3_tables": figure_data_matches_tables,
        "annotation_and_label_checks": annotation_checks,
        "typography_checks": typography_checks,
        "print_dimension_checks": dimensions,
        "shared_cohens_h_absolute_limit": shared_effect_limit,
        "shared_cohens_h_scale_all_subsets": True,
        "shared_colorbar_accuracy": True,
        "shared_colorbar_implicit_delta": True,
        "annotation_font_pt": ANNOTATION_FONT_SIZE,
        "annotation_font_weight": "normal",
        "tick_font_pt": TICK_FONT_SIZE,
        "tick_font_weight": "normal",
        "supporting_label_font_pt": SUPPORTING_LABEL_FONT_SIZE,
        "panel_title_font_pt": PANEL_TITLE_FONT_SIZE,
        "panel_title_font_weight": "semibold",
        "main_title_font_pt": MAIN_TITLE_FONT_SIZE,
        "main_title_font_weight": "bold",
        "decimal_separator": ",",
        "png_files": len(png_files),
        "svg_files": len(svg_files),
        "generated_figure_files": sorted(generated_figure_files),
        "generated_scope_valid": generated_scope_valid,
        "non_effect_figures_byte_identical_before_after": non_effect_figures_unchanged,
        "non_effect_figure_sha256_before": non_effect_figure_hashes_before,
        "non_effect_figure_sha256_after": non_effect_figure_hashes_after,
        "figure_sha256": figure_hashes_after,
    }
    if not validation["valid"]:
        raise ValueError(f"Validasi visualisasi gagal: {validation}")

    validation_path = metadata_dir / "visualization_v3_print_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # output_index is provenance rather than an analysis table. Refresh it so
    # figure hashes and the new visualization validation are not stale.
    output_index = build_output_index(output_dir)
    write_csv(output_index, output_dir / "output_index.csv")

    print("[VALID] 20 tabel byte-identik; angka gambar cocok dengan tabel v3")
    print("[VALID] Ukuran cetak, tipografi ringan, koma desimal, dan label lulus")
    if effect_sizes_only:
        print("[VALID] 8 file gambar non-effect byte-identik")
        print("[DONE] 4 PNG + 4 SVG effect size diperbarui")
    else:
        print(f"[DONE] {len(png_files)} PNG + {len(svg_files)} SVG diperbarui")
    return {
        "output_dir": output_dir,
        "figure_paths": figure_paths,
        "validation": validation,
        "output_index": output_index,
    }


def run_analysis(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, object]:
    repo_root = find_repo_root()
    input_path = Path(input_path) if input_path else default_input_path(repo_root)
    output_dir = Path(output_dir) if output_dir else default_output_dir(repo_root)
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in [table_dir, figure_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ETHICS GENDER PERSONA — V3 FULL-DENOMINATOR ACCURACY")
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print("API call: 0")
    print("=" * 78)

    input_hash_before = sha256_file(input_path)
    frame = load_data(input_path)
    input_validation = validate_input(frame)

    print("[TABLE] Menghitung akurasi deskriptif")
    tables = build_accuracy_tables(frame)
    condition = tables["accuracy_by_subset_model_condition.csv"]
    status = build_status_table(frame)
    abstain = build_abstain_reason_table(frame)
    tables["response_status_by_model.csv"] = status
    tables["abstain_reason_by_persona.csv"] = abstain

    print("[STATS] Cochran's Q dan Exact McNemar berpasangan")
    cochran = build_cochran_q(frame)
    scenario, gender = build_mcnemar(frame)
    all_comparisons = pd.concat([scenario, gender], ignore_index=True, sort=False)
    significant_overall, significant_by_subset, significant_by_model = build_significant_summaries(all_comparisons)
    tables["cochran_q_by_subset_model_gender.csv"] = cochran
    tables["mcnemar_scenario_comparisons.csv"] = scenario
    tables["mcnemar_gender_comparisons.csv"] = gender
    tables["mcnemar_all_comparisons.csv"] = all_comparisons
    tables["significant_results_overall.csv"] = significant_overall
    tables["significant_results_by_subset.csv"] = significant_by_subset
    tables["significant_results_by_model.csv"] = significant_by_model

    implicit_delta = build_implicit_delta_data(condition)
    tables["figure_data_implicit_minus_explicit_delta.csv"] = implicit_delta
    tables["figure_data_effect_sizes.csv"] = all_comparisons.copy()
    tables["figure_data_accuracy_absolute.csv"] = condition.copy()
    tables["figure_data_response_status.csv"] = status.copy()
    tables["figure_data_abstain_reason.csv"] = abstain.copy()

    print(f"[WRITE] Menulis {len(tables)} tabel CSV")
    for name, table in tables.items():
        write_csv(table, table_dir / name)

    print("[PLOT] Membuat gambar readable (PNG 300 DPI + SVG)")
    configure_plot_style()
    figure_paths: list[Path] = []
    paths, accuracy_figure_data = plot_accuracy_absolute(condition, figure_dir)
    figure_paths += paths
    figure_paths += plot_implicit_delta(implicit_delta, figure_dir)
    figure_paths += plot_effect_sizes(all_comparisons, figure_dir)
    figure_paths += plot_status_distribution(status, figure_dir)
    figure_paths += plot_abstain_reasons(abstain, figure_dir)

    input_hash_after = sha256_file(input_path)
    pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in all_comparisons[["subset", "n_pairs"]].itertuples(index=False)
    )
    cochran_pair_expected = all(
        int(row.n_pairs) == SUBSET_EXPECTED_N[row.subset]
        for row in cochran[["subset", "n_pairs"]].itertuples(index=False)
    )
    scenario_family_sizes = scenario.groupby(["subset", "model", "identity"], observed=True).size()
    gender_family_sizes = gender.groupby(["subset", "model"], observed=True).size()
    figure_table_consistency = {
        "accuracy_absolute": _frame_equal_on_columns(
            condition,
            accuracy_figure_data,
            ["subset", "model", "persona", "accuracy_pct"],
        ),
        "implicit_delta": _frame_equal_on_columns(
            implicit_delta,
            tables["figure_data_implicit_minus_explicit_delta.csv"],
            ["subset", "model", "identity", "implicit_scenario", "delta_accuracy_pp"],
        ),
        "effect_sizes": _frame_equal_on_columns(
            all_comparisons,
            tables["figure_data_effect_sizes.csv"],
            ["subset", "model", "comparison_label", "cohens_h"],
        ),
        "response_status": _frame_equal_on_columns(
            status,
            tables["figure_data_response_status.csv"],
            ["model", "status_label", "percentage"],
        ),
        "abstain_reason": _frame_equal_on_columns(
            abstain,
            tables["figure_data_abstain_reason.csv"],
            ["persona", "reason", "percentage_within_persona_abstain"],
        ),
    }
    png_files = sorted(figure_dir.glob("*.png"))
    svg_files = sorted(figure_dir.glob("*.svg"))
    validation = {
        "valid": bool(
            input_validation["valid"]
            and input_hash_before == input_hash_after
            and len(frame) == EXPECTED_ROWS
            and len(condition) == 216
            and len(cochran) == 48
            and len(scenario) == 336
            and len(gender) == 96
            and len(all_comparisons) == 432
            and pair_expected
            and cochran_pair_expected
            and scenario_family_sizes.eq(7).all()
            and gender_family_sizes.eq(4).all()
            and all(figure_table_consistency.values())
            and len(png_files) == 8
            and len(svg_files) == 8
            and all(path.stat().st_size > 0 for path in png_files + svg_files)
        ),
        "analysis_version": ANALYSIS_VERSION,
        "rows_processed": int(len(frame)),
        "all_81756_rows_used": bool(len(frame) == EXPECTED_ROWS),
        "subsets_scored_against_ground_truth": SUBSET_ORDER,
        "all_four_subsets_scored": bool(set(frame["subset"]) == set(SUBSET_ORDER)),
        "full_denominator_accuracy": True,
        "abstain_missing_counted_in_denominator_as_incorrect": bool(input_validation["abstain_missing_scored_false"]),
        "pair_key": ["subset", "model", "question_id", "identity_or_scenario_as_applicable"],
        "duplicate_response_keys": int(input_validation["duplicate_response_keys"]),
        "duplicate_pairs": 0,
        "scenario_pair_rows": int(len(scenario)),
        "gender_pair_rows": int(len(gender)),
        "cochran_q_rows": int(len(cochran)),
        "scenario_holm_family_size_exactly_7": bool(scenario_family_sizes.eq(7).all()),
        "gender_holm_family_size_exactly_4": bool(gender_family_sizes.eq(4).all()),
        "pair_counts_match_subset_question_counts": bool(pair_expected and cochran_pair_expected),
        "minimum_n_pairs": int(all_comparisons["n_pairs"].min()),
        "maximum_n_pairs": int(all_comparisons["n_pairs"].max()),
        "figure_table_consistency": figure_table_consistency,
        "png_files": int(len(png_files)),
        "svg_files": int(len(svg_files)),
        "all_figure_files_nonempty": bool(all(path.stat().st_size > 0 for path in png_files + svg_files)),
        "table_csv_files": int(len(list(table_dir.glob("*.csv")))),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "source_csv_unchanged": bool(input_hash_before == input_hash_after),
        "api_calls": 0,
    }
    if not validation["valid"]:
        raise ValueError(f"Validasi output gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "input": str(input_path),
        "output": str(output_dir),
        "method": "Akurasi biner terhadap ground truth dengan seluruh respons sebagai denominator; Exact McNemar berpasangan dan koreksi Holm; Cochran's Q berpasangan.",
        "models": MODEL_ORDER,
        "personas": PERSONA_ORDER,
        "subsets": SUBSET_ORDER,
        "accuracy_denominator": "seluruh respons pada kondisi",
        "nonanswer_scoring": "abstain dan missing bernilai salah",
        "pairing": "ID soal yang sama",
        "alpha": ALPHA,
        "api_calls": 0,
        "validation": validation,
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_index = build_output_index(output_dir)
    write_csv(output_index, output_dir / "output_index.csv")

    print("[DONE] Analisis v3 selesai dan seluruh validasi lulus")
    print(
        f"       rows={len(frame):,} | tables={len(tables)} | "
        f"figure_designs={len(png_files)} | figure_files={len(png_files) + len(svg_files)}"
    )
    return {
        "repo_root": repo_root,
        "input_path": input_path,
        "output_dir": output_dir,
        "frame": frame,
        "tables": tables,
        "condition_accuracy": condition,
        "cochran_q": cochran,
        "scenario_mcnemar": scenario,
        "gender_mcnemar": gender,
        "all_comparisons": all_comparisons,
        "significant_overall": significant_overall,
        "significant_by_subset": significant_by_subset,
        "significant_by_model": significant_by_model,
        "figure_paths": figure_paths,
        "validation": validation,
        "output_index": output_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETHICS Gender Persona v3 accuracy analysis.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate print-ready v3 figures from frozen tables without recomputing tables/statistics.",
    )
    parser.add_argument(
        "--effect-sizes-only",
        action="store_true",
        help="With --figures-only, overwrite only the four effect-size PNG/SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effect_sizes_only and not args.figures_only:
        raise SystemExit("--effect-sizes-only harus digunakan bersama --figures-only")
    if args.figures_only:
        refresh_visualizations_only(
            output_dir=args.output,
            effect_sizes_only=args.effect_sizes_only,
        )
    else:
        run_analysis(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()

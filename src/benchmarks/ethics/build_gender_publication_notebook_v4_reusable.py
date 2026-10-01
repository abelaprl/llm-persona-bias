"""Build a reusable ETHICS gender accuracy notebook.

The notebook follows the visual and reporting pattern used by the ETHICS
gender analysis in Bab VI.  Inferential values are read from the validated v3
bundle; this module never calls an API, runs an LLM, reparses responses,
rescoring responses, or recomputes a statistical test.

The new bundle is deliberately separate from the v3 and existing v4 bundles.
The configuration near the top is the part to adapt when another persona
dimension is prepared later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from nbclient import NotebookClient


# Persona-specific configuration.  A future persona notebook can keep the
# analysis helpers below and replace this block plus its v3 table names.
ANALYSIS_VERSION = "ethics-gender-v4-reusable-20260906-1"
EXPECTED_V3_VERSION = "ethics-gender-v3-accuracy-paired-holm-20260821"
MODEL_SLUG_ORDER = [
    "qwen_qwen3-next-80b-a3b-thinking",
    "meta-llama_llama-3.1-8b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "meta-llama_llama-3.3-70b-instruct",
]
MODEL_LABEL_BY_SLUG = {
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B-A3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1-Distill 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
}
MODEL_ORDER = [MODEL_LABEL_BY_SLUG[model] for model in MODEL_SLUG_ORDER]
SHORT_MODEL_LABELS = {
    "Qwen3 Next 80B-A3B": "Qwen3 Next 80B",
    "Llama 3.1 8B": "Llama 3.1 8B",
    "Ministral 8B": "Ministral 8B",
    "DeepSeek R1-Distill 32B": "DeepSeek R1 32B",
    "GPT-OSS 20B": "GPT-OSS 20B",
    "Llama 3.3 70B": "Llama 3.3 70B",
}
SUBSET_ORDER = ["commonsense", "deontology", "justice", "virtue"]
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
SUBSET_QUESTION_COUNTS = {
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
SCENARIO_ORDER = ["N", "E", "I1", "I2", "I3"]
STATUS_ORDER = ["committed", "abstain", "missing"]
TOKEN_COLUMNS = [
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
]
TOKEN_LABELS = {
    "prompt_tokens": "Input",
    "completion_tokens": "Completion",
    "reasoning_tokens": "Reasoning",
    "total_tokens": "Total",
}
CONDITION_INFO = {
    "neutral": {
        "identity": "neutral",
        "identity_label": "Netral",
        "scenario": "N",
        "condition_label": "Netral N",
        "plot_label": "N",
    },
    "man_explicit_1": {
        "identity": "man",
        "identity_label": "Laki-laki",
        "scenario": "E",
        "condition_label": "Laki-laki E",
        "plot_label": "L-E",
    },
    "man_implicit_1": {
        "identity": "man",
        "identity_label": "Laki-laki",
        "scenario": "I1",
        "condition_label": "Laki-laki I1",
        "plot_label": "L-I1",
    },
    "man_implicit_2": {
        "identity": "man",
        "identity_label": "Laki-laki",
        "scenario": "I2",
        "condition_label": "Laki-laki I2",
        "plot_label": "L-I2",
    },
    "man_implicit_3": {
        "identity": "man",
        "identity_label": "Laki-laki",
        "scenario": "I3",
        "condition_label": "Laki-laki I3",
        "plot_label": "L-I3",
    },
    "woman_explicit_1": {
        "identity": "woman",
        "identity_label": "Perempuan",
        "scenario": "E",
        "condition_label": "Perempuan E",
        "plot_label": "P-E",
    },
    "woman_implicit_1": {
        "identity": "woman",
        "identity_label": "Perempuan",
        "scenario": "I1",
        "condition_label": "Perempuan I1",
        "plot_label": "P-I1",
    },
    "woman_implicit_2": {
        "identity": "woman",
        "identity_label": "Perempuan",
        "scenario": "I2",
        "condition_label": "Perempuan I2",
        "plot_label": "P-I2",
    },
    "woman_implicit_3": {
        "identity": "woman",
        "identity_label": "Perempuan",
        "scenario": "I3",
        "condition_label": "Perempuan I3",
        "plot_label": "P-I3",
    },
}
BOOL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def v3_root(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
    )


def default_output_dir(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Gender_Persona"
        / "_combined"
        / "persona_publication_analysis_v4_reusable_gender_first_six_models"
    )


def source_notebook_path(root: Path) -> Path:
    return root / "notebooks" / "gender" / "ETHICS_Gender_Persona_Publication_Analysis_v4_Reusable.ipynb"


def executed_notebook_path(root: Path) -> Path:
    return root / "result" / "ETHICS" / "ETHICS_Gender_Persona_Publication_Analysis_v4_Reusable.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(path for path in directory.rglob("*") if path.is_file())
    }


def as_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    unknown = normalized.notna() & ~normalized.isin(BOOL_MAP)
    if unknown.any():
        values = sorted(normalized[unknown].unique().tolist())
        raise ValueError(f"Nilai boolean tidak dikenal: {values}")
    return normalized.map(BOOL_MAP).astype("boolean")


def load_v3_tables(root: Path) -> dict[str, object]:
    base = v3_root(root)
    manifest_path = base / "metadata" / "analysis_manifest.json"
    validation_path = base / "metadata" / "validation.json"
    required = [
        manifest_path,
        validation_path,
        base / "tables" / "accuracy_by_subset_model_condition.csv",
        base / "tables" / "accuracy_combined_four_subsets_by_model_condition.csv",
        base / "tables" / "accuracy_gender_overall.csv",
        base / "tables" / "accuracy_gender_comparison_by_model.csv",
        base / "tables" / "accuracy_gender_comparison_by_scenario.csv",
        base / "tables" / "mcnemar_gender_comparisons.csv",
        base / "tables" / "mcnemar_scenario_comparisons.csv",
        base / "tables" / "cochran_q_by_subset_model_gender.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Artefak v3 gender tidak lengkap: {missing}")

    bundle_snapshot_before = snapshot_files(base)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != EXPECTED_V3_VERSION:
        raise ValueError(f"Versi sumber bukan v3 kanonik: {manifest.get('analysis_version')}")
    if validation.get("valid") is not True:
        raise ValueError("Validation v3 gender tidak berstatus valid")

    table_dir = base / "tables"
    accuracy = pd.read_csv(table_dir / "accuracy_by_subset_model_condition.csv")
    overall_accuracy = pd.read_csv(table_dir / "accuracy_combined_four_subsets_by_model_condition.csv")
    overall_gender = pd.read_csv(table_dir / "accuracy_gender_overall.csv")
    overall_gender_by_model = pd.read_csv(table_dir / "accuracy_gender_comparison_by_model.csv")
    overall_gender_by_scenario = pd.read_csv(table_dir / "accuracy_gender_comparison_by_scenario.csv")
    direct = pd.read_csv(table_dir / "mcnemar_gender_comparisons.csv")
    scenario = pd.read_csv(table_dir / "mcnemar_scenario_comparisons.csv")
    cochran = pd.read_csv(table_dir / "cochran_q_by_subset_model_gender.csv")
    for frame in (direct, scenario, cochran):
        for column in [name for name in frame.columns if name.startswith("significant")]:
            frame[column] = as_bool(frame[column]).fillna(False).astype(bool)

    scenario = scenario.merge(
        cochran[["subset", "model", "identity", "significant_0_05"]].rename(
            columns={"significant_0_05": "cochran_q_significant_0_05"}
        ),
        on=["subset", "model", "identity"],
        how="left",
        validate="many_to_one",
    )
    scenario["cochran_q_significant_0_05"] = scenario["cochran_q_significant_0_05"].fillna(False).astype(bool)
    scenario["supported_q_and_holm"] = (
        scenario["significant_holm_0_05"] & scenario["cochran_q_significant_0_05"]
    )
    return {
        "base": base,
        "manifest": manifest,
        "validation": validation,
        "manifest_path": manifest_path,
        "validation_path": validation_path,
        "bundle_snapshot_before": bundle_snapshot_before,
        "accuracy": accuracy,
        "overall_accuracy": overall_accuracy,
        "overall_gender": overall_gender,
        "overall_gender_by_model": overall_gender_by_model,
        "overall_gender_by_scenario": overall_gender_by_scenario,
        "direct": direct,
        "scenario": scenario,
        "cochran": cochran,
    }


def load_descriptive_source(source: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = Path(str(source["manifest"]["input"])).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"CSV input pada manifest v3 tidak ditemukan: {input_path}")
    expected_before = source["validation"].get("input_sha256_before")
    expected_after = source["validation"].get("input_sha256_after")
    source_sha_before_read = sha256(input_path)
    source_sha_matches_manifest = (
        source_sha_before_read == expected_before == expected_after
        if expected_before and expected_after
        else False
    )
    usecols = [
        "model",
        "persona_folder",
        "question_id",
        "subset",
        "answer_status",
        "is_correct",
        *TOKEN_COLUMNS,
    ]
    frame = pd.read_csv(input_path, usecols=usecols, low_memory=False)
    source_sha_after_read = sha256(input_path)
    frame["answer_status"] = frame["answer_status"].astype("string").str.strip().str.lower()
    allowed_statuses = set(STATUS_ORDER)
    observed_statuses = set(frame["answer_status"].dropna().unique().tolist())
    if not observed_statuses.issubset(allowed_statuses):
        raise ValueError(f"Status respons tidak dikenal: {sorted(observed_statuses - allowed_statuses)}")
    frame["is_correct_bool"] = as_bool(frame["is_correct"])
    frame["condition"] = frame["persona_folder"]
    frame["identity"] = frame["persona_folder"].map(
        {persona: info["identity"] for persona, info in CONDITION_INFO.items()}
    )
    frame["scenario"] = frame["persona_folder"].map(
        {persona: info["scenario"] for persona, info in CONDITION_INFO.items()}
    )
    if frame[["identity", "scenario"]].isna().any().any():
        raise ValueError("Ada persona gender yang tidak ada di PERSONA_CONFIG")
    for column in TOKEN_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    descriptive = {
        "input_path": input_path,
        "input_sha256_before_read": source_sha_before_read,
        "input_sha256_after_read": source_sha_after_read,
        "input_sha256_matches_v3_manifest": source_sha_matches_manifest,
        "rows": int(len(frame)),
        "questions": int(frame["question_id"].nunique()),
        "models": int(frame["model"].nunique()),
        "conditions": int(frame["persona_folder"].nunique()),
        "subsets": sorted(frame["subset"].dropna().unique().tolist()),
        "duplicate_response_keys": int(frame.duplicated(["model", "persona_folder", "question_id"]).sum()),
        "token_valid_counts": {column: int(frame[column].notna().sum()) for column in TOKEN_COLUMNS},
        "token_missing_counts": {column: int(frame[column].isna().sum()) for column in TOKEN_COLUMNS},
    }
    descriptive["subset_question_counts"] = {
        subset: int(frame.loc[frame["subset"].eq(subset), "question_id"].nunique())
        for subset in SUBSET_ORDER
    }
    return frame, descriptive


def direct_summary(direct: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = direct[direct["model_label"].eq(model)]
        significant = selected[selected["significant_holm_0_05"]]
        largest = significant.loc[significant["delta_accuracy_pp"].abs().idxmax()] if len(significant) else None
        scenarios = [scenario for scenario in ["E", "I1", "I2", "I3"] if scenario in set(significant["scenario_a"])]
        rows.append(
            {
                "model": model,
                "n_comparisons": int(len(selected)),
                "n_significant": int(len(significant)),
                "n_man_higher": int((significant["delta_accuracy_pp"] > 0).sum()),
                "n_woman_higher": int((significant["delta_accuracy_pp"] < 0).sum()),
                "significant_scenarios": ", ".join(scenarios) if scenarios else "-",
                "largest_comparison": (
                    f"{largest['subset_label']}, {largest['scenario_a']}" if largest is not None else "-"
                ),
                "largest_direction": (
                    "man > woman" if largest is not None and largest["delta_accuracy_pp"] > 0 else
                    "woman > man" if largest is not None else "-"
                ),
                "largest_delta_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                "largest_p_holm": float(largest["holm_adjusted_p_value"]) if largest is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scenario_summary(scenario: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in ["implicit_vs_explicit", "persona_vs_neutral"]:
        for model in MODEL_ORDER:
            selected = scenario[
                scenario["comparison_category"].eq(category)
                & scenario["model_label"].eq(model)
            ]
            significant = selected[selected["supported_q_and_holm"]]
            largest = significant.loc[significant["delta_accuracy_pp"].abs().idxmax()] if len(significant) else None
            rows.append(
                {
                    "comparison_category": category,
                    "model": model,
                    "n_comparisons": int(len(selected)),
                    "n_supported_q_and_holm": int(len(significant)),
                    "n_higher": int((significant["delta_accuracy_pp"] > 0).sum()),
                    "n_lower": int((significant["delta_accuracy_pp"] < 0).sum()),
                    "largest_comparison": (
                        f"{largest['subset_label']}, {largest['identity_label']} "
                        f"{largest['scenario_a']}-{largest['scenario_b']}"
                        if largest is not None else "-"
                    ),
                    "largest_delta_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                    "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_accuracy_status_table(accuracy: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subset", "model", "persona_folder"]
    status_counts = (
        raw.groupby(group_cols + ["answer_status"], dropna=False)
        .size()
        .unstack("answer_status", fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
        .reset_index()
    )
    correct_committed = (
        raw.assign(
            correct_committed=(
                raw["answer_status"].eq("committed")
                & raw["is_correct_bool"].fillna(False).astype(bool)
            ).astype(int)
        )
        .groupby(group_cols, dropna=False)["correct_committed"]
        .sum()
        .rename("n_correct_committed")
        .reset_index()
    )
    all_correct = (
        raw.assign(all_correct=raw["is_correct_bool"].fillna(False).astype(bool).astype(int))
        .groupby(group_cols, dropna=False)["all_correct"]
        .sum()
        .rename("n_correct_from_descriptive_source")
        .reset_index()
    )
    table = accuracy.copy()
    table["condition"] = table["persona"]
    table = table.merge(
        status_counts.rename(columns={"persona_folder": "persona"}),
        on=["subset", "model", "persona"],
        how="left",
        validate="one_to_one",
    )
    table = table.merge(
        correct_committed.rename(columns={"persona_folder": "persona"}),
        on=["subset", "model", "persona"],
        how="left",
        validate="one_to_one",
    )
    table = table.merge(
        all_correct.rename(columns={"persona_folder": "persona"}),
        on=["subset", "model", "persona"],
        how="left",
        validate="one_to_one",
    )
    for column in STATUS_ORDER + ["n_correct_committed", "n_correct_from_descriptive_source"]:
        table[column] = table[column].fillna(0).astype(int)
    table["total"] = table["total_responses"].astype(int)
    table["committed"] = table["committed"].astype(int)
    table["abstain"] = table["abstain"].astype(int)
    table["missing"] = table["missing"].astype(int)
    table["commit_rate"] = table["committed"] / table["total"]
    table["abstain_rate"] = table["abstain"] / table["total"]
    table["missing_rate"] = table["missing"] / table["total"]
    table["accuracy_full"] = pd.to_numeric(table["accuracy"], errors="raise")
    table["accuracy_full_pct"] = pd.to_numeric(table["accuracy_pct"], errors="raise")
    table["accuracy_committed"] = np.where(
        table["committed"] > 0,
        table["n_correct_committed"] / table["committed"],
        np.nan,
    )
    table["accuracy_committed_pct"] = table["accuracy_committed"] * 100.0
    table["n_incorrect_full"] = table["total"] - table["n_correct"].astype(int)
    table["identity"] = table["condition"].map(
        {persona: info["identity"] for persona, info in CONDITION_INFO.items()}
    )
    table["identity_label"] = table["condition"].map(
        {persona: info["identity_label"] for persona, info in CONDITION_INFO.items()}
    )
    table["scenario"] = table["condition"].map(
        {persona: info["scenario"] for persona, info in CONDITION_INFO.items()}
    )
    table["condition_label"] = table["condition"].map(
        {persona: info["condition_label"] for persona, info in CONDITION_INFO.items()}
    )
    table["condition_plot_label"] = table["condition"].map(
        {persona: info["plot_label"] for persona, info in CONDITION_INFO.items()}
    )
    table["condition_order"] = table["condition"].map({persona: index for index, persona in enumerate(PERSONA_ORDER)})
    table = table.sort_values(
        by=["subset", "model_label", "condition_order"],
        key=lambda series: (
            series.map({value: index for index, value in enumerate(SUBSET_ORDER)})
            if series.name == "subset"
            else series
        ),
    ).reset_index(drop=True)
    required_columns = [
        "subset",
        "subset_label",
        "model",
        "model_label",
        "condition",
        "condition_label",
        "condition_plot_label",
        "identity",
        "identity_label",
        "scenario",
        "total",
        "committed",
        "commit_rate",
        "abstain",
        "abstain_rate",
        "missing",
        "missing_rate",
        "n_correct",
        "n_incorrect_full",
        "accuracy_full",
        "accuracy_full_pct",
        "n_correct_committed",
        "n_correct_from_descriptive_source",
        "accuracy_committed",
        "accuracy_committed_pct",
    ]
    return table[required_columns]


def summarize_status_by_model(raw: pd.DataFrame) -> pd.DataFrame:
    counts = (
        raw.groupby(["model", "answer_status"], dropna=False)
        .size()
        .unstack("answer_status", fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
        .reset_index()
    )
    for column in STATUS_ORDER:
        counts[column] = counts[column].astype(int)
    counts["total"] = counts[STATUS_ORDER].sum(axis=1).astype(int)
    for column in STATUS_ORDER:
        counts[f"{column}_rate"] = counts[column] / counts["total"]
    counts["model_label"] = counts["model"].map(MODEL_LABEL_BY_SLUG)
    counts["model_order"] = counts["model_label"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    return counts.sort_values("model_order").drop(columns="model_order").reset_index(drop=True)


def summarize_tokens(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in raw.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["total_responses"] = int(len(group))
        for column in TOKEN_COLUMNS:
            values = group[column].dropna()
            row[f"{column}_n_valid"] = int(values.size)
            row[f"{column}_valid_rate"] = float(values.size / len(group)) if len(group) else np.nan
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if "model" in result.columns:
        result["model_label"] = result["model"].map(MODEL_LABEL_BY_SLUG)
    if "persona_folder" in result.columns:
        result = result.rename(columns={"persona_folder": "condition"})
        result["condition_label"] = result["condition"].map(
            {persona: info["condition_label"] for persona, info in CONDITION_INFO.items()}
        )
        result["condition_order"] = result["condition"].map(
            {persona: index for index, persona in enumerate(PERSONA_ORDER)}
        )
    if "subset" in result.columns:
        result["subset_label"] = result["subset"].map(SUBSET_LABELS)
        result["subset_order"] = result["subset"].map(
            {subset: index for index, subset in enumerate(SUBSET_ORDER)}
        )
    sort_cols = [column for column in ["subset_order", "model_label", "condition_order"] if column in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
        result = result.drop(columns=[column for column in ["subset_order", "condition_order"] if column in result.columns])
    return result


def normalize_pair_orientation(frame: pd.DataFrame, category: str) -> tuple[pd.DataFrame, int]:
    """Return plotting data with a fixed numerator/denominator orientation.

    The p-values and significance flags are intentionally left unchanged.  If
    a source row is reversed, only direction-dependent fields are swapped or
    sign-flipped.
    """
    result = frame[frame["comparison_category"].eq(category)].copy()
    reverse_count = 0
    swap_pairs = [
        ("condition_a", "condition_b"),
        ("scenario_a", "scenario_b"),
        ("accuracy_a_pct", "accuracy_b_pct"),
        ("discordant_a_correct_b_wrong", "discordant_a_wrong_b_correct"),
    ]
    for index in result.index:
        condition_a = str(result.at[index, "condition_a"])
        condition_b = str(result.at[index, "condition_b"])
        scenario_a = str(result.at[index, "scenario_a"])
        scenario_b = str(result.at[index, "scenario_b"])
        if category == "man_vs_woman":
            correct = condition_a.startswith("man_") and condition_b.startswith("woman_")
            reverse = condition_a.startswith("woman_") and condition_b.startswith("man_")
        elif category == "implicit_vs_explicit":
            correct = scenario_a in {"I1", "I2", "I3"} and scenario_b == "E"
            reverse = scenario_a == "E" and scenario_b in {"I1", "I2", "I3"}
        elif category == "persona_vs_neutral":
            correct = scenario_a in {"E", "I1", "I2", "I3"} and scenario_b == "N"
            reverse = scenario_a == "N" and scenario_b in {"E", "I1", "I2", "I3"}
        else:
            raise ValueError(f"Kategori pasangan tidak dikenal: {category}")
        if reverse:
            reverse_count += 1
            for left, right in swap_pairs:
                result.at[index, left], result.at[index, right] = result.at[index, right], result.at[index, left]
            result.at[index, "delta_accuracy_pp"] = -float(result.at[index, "delta_accuracy_pp"])
            result.at[index, "cohens_h"] = -float(result.at[index, "cohens_h"])
        elif not correct:
            raise ValueError(
                f"Orientasi pasangan tidak dapat dinormalisasi pada {category}: "
                f"{condition_a} vs {condition_b} ({scenario_a} vs {scenario_b})"
            )
    return result, reverse_count


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 12.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "gender_reusable_effect",
    ["#0B3C78", "#4A98C9", "#DCEAF1", "#FCFBF8", "#F7C7AE", "#E76F51", "#9B002F"],
)


def draw_annotations(ax: plt.Axes, values: pd.DataFrame, significant: pd.DataFrame | None = None, decimals: int = 2) -> None:
    for y, row in enumerate(values.index):
        for x, column in enumerate(values.columns):
            value = values.loc[row, column]
            if pd.isna(value):
                continue
            is_significant = bool(significant.loc[row, column]) if significant is not None else False
            suffix = "*" if is_significant else ""
            text = f"{float(value):.{decimals}f}{suffix}"
            ax.text(
                x + 0.5,
                y + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=10.6,
                fontweight="bold" if is_significant else "normal",
                color="#202020",
            )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def effect_heatmap(
    frame: pd.DataFrame,
    figure_dir: Path,
    *,
    category: str,
    stem: str,
    title: str,
    significance_column: str,
) -> tuple[list[Path], int]:
    selected_all, reverse_count = normalize_pair_orientation(frame, category)
    if category == "man_vs_woman":
        selected_all["column_label"] = selected_all["scenario_a"]
        columns = ["E", "I1", "I2", "I3"]
    elif category == "implicit_vs_explicit":
        selected_all["column_label"] = (
            selected_all["identity"].map({"man": "L", "woman": "P"})
            + ":"
            + selected_all["scenario_a"]
            + "-E"
        )
        columns = [f"{identity}:{scenario}-E" for identity in ["L", "P"] for scenario in ["I1", "I2", "I3"]]
    else:
        selected_all["column_label"] = (
            selected_all["identity"].map({"man": "L", "woman": "P"})
            + ":"
            + selected_all["scenario_a"]
            + "-N"
        )
        columns = [f"{identity}:{scenario}-N" for identity in ["L", "P"] for scenario in ["E", "I1", "I2", "I3"]]

    limit = max(0.20, float(np.ceil(selected_all["cohens_h"].abs().max() * 10) / 10))
    is_implicit_explicit = category == "implicit_vs_explicit"
    figure_size = (15.8, 10.2) if is_implicit_explicit else (15.8, 8.5)
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.88, wspace=0.28, hspace=0.25)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = selected_all[selected_all["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=MODEL_ORDER, columns=columns
        )
        significant = selected.pivot(
            index="model_label", columns="column_label", values=significance_column
        ).reindex(index=MODEL_ORDER, columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            mask=values.isna(),
            linewidths=0.45,
            linecolor="#FFFFFF",
            cbar=False,
        )
        draw_annotations(ax, values, significant)
        ax.set_title(SUBSET_LABELS[subset], fontsize=12.5, fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([SHORT_MODEL_LABELS.get(label, label) for label in MODEL_ORDER])
        ax.tick_params(axis="x", rotation=0, labelsize=10.8)
        ax.tick_params(axis="y", rotation=0, labelsize=10.5)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-limit, vmax=limit), cmap=EFFECT_CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.82, pad=0.018)
    colorbar.set_label("Cohen's h", fontweight="normal")
    colorbar.ax.tick_params(labelsize=10.5)
    fig.suptitle(title, fontsize=15, fontweight="normal")
    note = (
        "Nilai positif: akurasi laki-laki lebih tinggi; nilai negatif: akurasi perempuan lebih tinggi. "
        "* hanya untuk hasil signifikan setelah koreksi Holm."
        if category == "man_vs_woman"
        else "Nilai positif: kondisi pertama lebih tinggi; nilai negatif: kondisi pembanding lebih tinggi. "
        "* hanya untuk hasil yang didukung Q Cochran dan koreksi Holm."
    )
    fig.text(0.48, 0.025, note, ha="center", fontsize=8.8, fontweight="normal")
    return save_figure(fig, figure_dir, stem), reverse_count


def overall_accuracy_heatmap(overall_accuracy: pd.DataFrame, figure_dir: Path) -> list[Path]:
    condition_plot_labels = {
        persona: info["plot_label"] for persona, info in CONDITION_INFO.items()
    }
    selected = overall_accuracy.copy()
    selected["condition_plot_label"] = selected["persona"].map(condition_plot_labels)
    if selected["condition_plot_label"].isna().any():
        raise ValueError("Tabel akurasi overall memuat persona yang tidak dikenal")
    values = selected.pivot(
        index="model_label", columns="condition_plot_label", values="accuracy_pct"
    ).reindex(index=MODEL_ORDER, columns=["N", "L-E", "L-I1", "L-I2", "L-I3", "P-E", "P-I1", "P-I2", "P-I3"])
    fig, ax = plt.subplots(figsize=(19.0, 7.4))
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.18, top=0.84)
    sns.heatmap(
        values,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        mask=values.isna(),
        linewidths=0.45,
        linecolor="#FFFFFF",
        cbar=False,
    )
    draw_annotations(ax, values, decimals=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([SHORT_MODEL_LABELS.get(label, label) for label in MODEL_ORDER])
    ax.tick_params(axis="x", rotation=0, labelsize=10.8)
    ax.tick_params(axis="y", rotation=0, labelsize=10.5)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=100), cmap="YlGnBu")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.84, pad=0.018)
    colorbar.set_label("Akurasi penuh (%)", fontweight="normal")
    colorbar.ax.tick_params(labelsize=10.5)
    fig.suptitle(
        "Akurasi Overall Gabungan Empat Subset ETHICS Gender",
        fontsize=15,
        fontweight="normal",
    )
    fig.text(
        0.50,
        0.035,
        "Ringkasan deskriptif; gabungan subset tidak digunakan sebagai hasil uji inferensial.",
        ha="center",
        fontsize=9.2,
        fontweight="normal",
    )
    return save_figure(fig, figure_dir, "00_accuracy_overall_all_subsets_heatmap")


def accuracy_heatmap(accuracy_status: pd.DataFrame, figure_dir: Path) -> list[Path]:
    columns = ["N", "L-E", "L-I1", "L-I2", "L-I3", "P-E", "P-I1", "P-I2", "P-I3"]
    fig, axes = plt.subplots(2, 2, figsize=(20.5, 9.0))
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.88, wspace=0.28, hspace=0.24)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = accuracy_status[accuracy_status["subset"].eq(subset)]
        values = selected.pivot(
            index="model_label", columns="condition_plot_label", values="accuracy_full_pct"
        ).reindex(index=MODEL_ORDER, columns=columns)
        sns.heatmap(
            values,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            mask=values.isna(),
            linewidths=0.45,
            linecolor="#FFFFFF",
            cbar=False,
        )
        draw_annotations(ax, values, decimals=1)
        ax.set_title(SUBSET_LABELS[subset], fontsize=12.5, fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([SHORT_MODEL_LABELS.get(label, label) for label in MODEL_ORDER])
        ax.tick_params(axis="x", rotation=0, labelsize=10.2)
        ax.tick_params(axis="y", rotation=0, labelsize=10.0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=100), cmap="YlGnBu")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.82, pad=0.018)
    colorbar.set_label("Akurasi penuh (%)", fontweight="normal")
    colorbar.ax.tick_params(labelsize=10.5)
    fig.suptitle("Akurasi Penuh per Kondisi dan Subset ETHICS Gender", fontsize=15, fontweight="normal")
    fig.text(0.48, 0.025, "Setiap sel menggunakan seluruh respons pada kondisi sebagai penyebut.", ha="center", fontsize=8.8)
    return save_figure(fig, figure_dir, "04_accuracy_full_by_condition_heatmap")


def status_figure(status_by_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    x = np.arange(len(status_by_model))
    bottom = np.zeros(len(status_by_model))
    colors = {"committed": "#4C956C", "abstain": "#F2C14E", "missing": "#C8553D"}
    labels = {"committed": "Committed", "abstain": "Abstain", "missing": "Missing"}
    for status in STATUS_ORDER:
        values = status_by_model[f"{status}_rate"].to_numpy()
        ax.bar(x, values, bottom=bottom, color=colors[status], label=labels[status], width=0.72)
        bottom += values
    ax.set_xticks(x, status_by_model["model_label"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporsi respons")
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=11)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title("Status Respons per Model", fontweight="normal", fontsize=14)
    ax.legend(frameon=True, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    return save_figure(fig, figure_dir, "05_response_status_by_model")


def token_figure(token_by_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    plot_frame = token_by_model[["model_label", *[f"{column}_mean" for column in TOKEN_COLUMNS]]].copy()
    plot_frame = plot_frame.rename(columns={f"{column}_mean": TOKEN_LABELS[column] for column in TOKEN_COLUMNS})
    long = plot_frame.melt(id_vars="model_label", var_name="token_type", value_name="mean_tokens")
    fig, ax = plt.subplots(figsize=(14.0, 6.8))
    sns.barplot(data=long, x="model_label", y="mean_tokens", hue="token_type", ax=ax, errorbar=None)
    ax.set_xlabel("")
    ax.set_ylabel("Rata-rata token (baris dengan metadata valid)")
    ax.set_title("Rata-rata Token per Model", fontweight="normal", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Jenis token", frameon=True, ncol=4)
    return save_figure(fig, figure_dir, "06_token_usage_by_model")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def artifact_index(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "output_index.csv"):
        rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def load_existing_output(output: Path) -> dict[str, object]:
    metadata_dir = output / "metadata"
    table_dir = output / "tables"
    validation = json.loads((metadata_dir / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True or manifest.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError(f"Bundle reusable gender yang ada tidak valid: {output}")
    accuracy_status = pd.read_csv(table_dir / "accuracy_status_by_subset_model_condition.csv")
    return {
        "output_dir": output,
        "accuracy": accuracy_status,
        "accuracy_status": accuracy_status,
        "overall_accuracy": pd.read_csv(
            table_dir / "accuracy_combined_four_subsets_by_model_condition.csv"
        ),
        "overall_gender": pd.read_csv(table_dir / "accuracy_gender_overall.csv"),
        "overall_gender_by_model": pd.read_csv(
            table_dir / "accuracy_gender_comparison_by_model.csv"
        ),
        "overall_gender_by_scenario": pd.read_csv(
            table_dir / "accuracy_gender_comparison_by_scenario.csv"
        ),
        "direct_summary": pd.read_csv(table_dir / "gender_direct_summary_by_model.csv"),
        "direct_significant": pd.read_csv(table_dir / "gender_direct_significant_results.csv"),
        "scenario_summary": pd.read_csv(table_dir / "scenario_summary_by_model.csv"),
        "scenario_supported": pd.read_csv(table_dir / "scenario_supported_q_holm_results.csv"),
        "status": pd.read_csv(table_dir / "response_status_by_model.csv"),
        "status_by_model": pd.read_csv(table_dir / "response_status_by_model.csv"),
        "tokens": pd.read_csv(table_dir / "token_summary_by_model.csv"),
        "token_by_model": pd.read_csv(table_dir / "token_summary_by_model.csv"),
        "token_summary": pd.read_csv(table_dir / "token_summary_by_subset_model_condition.csv"),
        "validation": validation,
        "manifest": manifest,
        "output_index": pd.read_csv(output / "output_index.csv"),
    }


def run_analysis(output_dir: Path | str | None = None) -> dict[str, object]:
    root = repo_root()
    output = Path(output_dir).resolve() if output_dir else default_output_dir(root)
    if output.exists() and any(output.iterdir()):
        metadata_dir = output / "metadata"
        if (metadata_dir / "validation.json").exists() and (metadata_dir / "analysis_manifest.json").exists():
            existing_manifest = json.loads(
                (metadata_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            if existing_manifest.get("analysis_version") == ANALYSIS_VERSION:
                return load_existing_output(output)
            print(
                f"[INFO] Bundle reusable versi lama diperbarui: "
                f"{existing_manifest.get('analysis_version')} -> {ANALYSIS_VERSION}"
            )
        print(f"[INFO] Bundle baru parsial dilanjutkan: {output}")

    source = load_v3_tables(root)
    raw, descriptive = load_descriptive_source(source)
    overall_accuracy = source["overall_accuracy"].copy()
    overall_gender = source["overall_gender"].copy()
    overall_gender_by_model = source["overall_gender_by_model"].copy()
    overall_gender_by_scenario = source["overall_gender_by_scenario"].copy()
    accuracy_status = make_accuracy_status_table(source["accuracy"], raw)
    status_by_model = summarize_status_by_model(raw)
    token_summary = summarize_tokens(raw, ["subset", "model", "persona_folder"])
    token_by_model = summarize_tokens(raw, ["model"])
    summary_direct = direct_summary(source["direct"])
    summary_scenario = scenario_summary(source["scenario"])
    significant_direct = source["direct"][source["direct"]["significant_holm_0_05"]].copy()
    supported_scenario = source["scenario"][source["scenario"]["supported_q_and_holm"]].copy()

    table_dir = output / "tables"
    figure_dir = output / "figures"
    metadata_dir = output / "metadata"
    for directory in (table_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # These are copied as source snapshots.  They are not recalculated here.
    write_csv(source["accuracy"], table_dir / "source_v3_accuracy_by_subset_model_condition.csv")
    write_csv(overall_accuracy, table_dir / "accuracy_combined_four_subsets_by_model_condition.csv")
    write_csv(overall_gender, table_dir / "accuracy_gender_overall.csv")
    write_csv(overall_gender_by_model, table_dir / "accuracy_gender_comparison_by_model.csv")
    write_csv(overall_gender_by_scenario, table_dir / "accuracy_gender_comparison_by_scenario.csv")
    write_csv(source["direct"], table_dir / "mcnemar_gender_comparisons.csv")
    write_csv(source["scenario"], table_dir / "mcnemar_scenario_comparisons.csv")
    write_csv(source["cochran"], table_dir / "cochran_q_by_subset_model_gender.csv")
    write_csv(accuracy_status, table_dir / "accuracy_status_by_subset_model_condition.csv")
    write_csv(summary_direct, table_dir / "gender_direct_summary_by_model.csv")
    write_csv(significant_direct, table_dir / "gender_direct_significant_results.csv")
    write_csv(summary_scenario, table_dir / "scenario_summary_by_model.csv")
    write_csv(supported_scenario, table_dir / "scenario_supported_q_holm_results.csv")
    write_csv(status_by_model, table_dir / "response_status_by_model.csv")
    write_csv(token_summary, table_dir / "token_summary_by_subset_model_condition.csv")
    write_csv(token_by_model, table_dir / "token_summary_by_model.csv")

    configure_plot_style()
    figures: list[Path] = []
    direct_figures, direct_reversals = effect_heatmap(
        source["direct"],
        figure_dir,
        category="man_vs_woman",
        stem="01_gender_direct_effect_holm_heatmap",
        title="Perbandingan Akurasi Persona Laki-laki dan Perempuan",
        significance_column="significant_holm_0_05",
    )
    ie_figures, ie_reversals = effect_heatmap(
        source["scenario"],
        figure_dir,
        category="implicit_vs_explicit",
        stem="02_implicit_vs_explicit_effect_q_holm_heatmap",
        title="Perbandingan Akurasi Skenario Implisit dan Eksplisit",
        significance_column="supported_q_and_holm",
    )
    neutral_figures, neutral_reversals = effect_heatmap(
        source["scenario"],
        figure_dir,
        category="persona_vs_neutral",
        stem="03_persona_vs_neutral_effect_q_holm_heatmap",
        title="Perbandingan Akurasi Skenario Persona dan Netral",
        significance_column="supported_q_and_holm",
    )
    figures.extend(overall_accuracy_heatmap(overall_accuracy, figure_dir))
    figures.extend(direct_figures + ie_figures + neutral_figures)
    figures.extend(accuracy_heatmap(accuracy_status, figure_dir))
    figures.extend(status_figure(status_by_model, figure_dir))
    figures.extend(token_figure(token_by_model, figure_dir))

    source_bundle_snapshot_after = snapshot_files(source["base"])
    subset_question_counts = descriptive["subset_question_counts"]
    overall_accuracy_total_values = sorted(
        pd.to_numeric(overall_accuracy["total_responses"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    overall_accuracy_subsets = sorted(overall_accuracy["subset"].dropna().unique().tolist())
    overall_accuracy_duplicate_keys = int(overall_accuracy.duplicated(["model", "persona"]).sum())
    ie = source["scenario"][source["scenario"]["comparison_category"].eq("implicit_vs_explicit")]
    persona_neutral = source["scenario"][source["scenario"]["comparison_category"].eq("persona_vs_neutral")]
    direct_holm_sizes = sorted(source["direct"]["holm_family_size"].dropna().astype(int).unique().tolist())
    scenario_holm_sizes = sorted(source["scenario"]["holm_family_size"].dropna().astype(int).unique().tolist())
    figure_files_openable = all(
        path.exists()
        and path.stat().st_size > 0
        and (
            (path.suffix == ".png" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
            or (path.suffix == ".svg" and b"<svg" in path.read_bytes()[:1000])
        )
        for path in figures
    )
    validation = {
        "valid": False,
        "analysis_version": ANALYSIS_VERSION,
        "source_analysis_version": source["manifest"]["analysis_version"],
        "source_validation_valid": source["validation"].get("valid") is True,
        "source_v3_bundle": str(source["base"]),
        "source_csv": str(descriptive["input_path"]),
        "responses": descriptive["rows"],
        "questions": descriptive["questions"],
        "models": descriptive["models"],
        "conditions": descriptive["conditions"],
        "subsets_present": descriptive["subsets"],
        "subset_question_counts": subset_question_counts,
        "overall_accuracy_rows": int(len(overall_accuracy)),
        "overall_accuracy_models": int(overall_accuracy["model"].nunique()),
        "overall_accuracy_conditions": int(overall_accuracy["persona"].nunique()),
        "overall_accuracy_subsets": overall_accuracy_subsets,
        "overall_accuracy_total_responses_per_condition": overall_accuracy_total_values,
        "overall_accuracy_duplicate_model_persona": overall_accuracy_duplicate_keys,
        "overall_gender_summary_rows": int(len(overall_gender)),
        "overall_gender_by_model_rows": int(len(overall_gender_by_model)),
        "overall_gender_by_scenario_rows": int(len(overall_gender_by_scenario)),
        "duplicate_response_keys": descriptive["duplicate_response_keys"],
        "gender_comparisons": int(len(source["direct"])),
        "scenario_comparisons": int(len(source["scenario"])),
        "implicit_vs_explicit_comparisons": int(len(ie)),
        "persona_vs_neutral_comparisons": int(len(persona_neutral)),
        "cochran_q_groups": int(len(source["cochran"])),
        "gender_holm_family_sizes": direct_holm_sizes,
        "scenario_holm_family_sizes": scenario_holm_sizes,
        "gender_significant_holm": int(len(significant_direct)),
        "gender_significant_man_higher": int((significant_direct["delta_accuracy_pp"] > 0).sum()),
        "gender_significant_woman_higher": int((significant_direct["delta_accuracy_pp"] < 0).sum()),
        "implicit_vs_explicit_supported_q_holm": int(ie["supported_q_and_holm"].sum()),
        "persona_vs_neutral_supported_q_holm": int(persona_neutral["supported_q_and_holm"].sum()),
        "orientation_reversals_before_plot": {
            "man_vs_woman": direct_reversals,
            "implicit_vs_explicit": ie_reversals,
            "persona_vs_neutral": neutral_reversals,
        },
        "source_csv_sha256_before_read": descriptive["input_sha256_before_read"],
        "source_csv_sha256_after_read": descriptive["input_sha256_after_read"],
        "source_csv_sha256_v3_before": source["validation"].get("input_sha256_before"),
        "source_csv_sha256_v3_after": source["validation"].get("input_sha256_after"),
        "source_csv_unchanged": (
            descriptive["input_sha256_before_read"] == descriptive["input_sha256_after_read"]
        ),
        "source_csv_matches_v3_manifest": descriptive["input_sha256_matches_v3_manifest"],
        "source_v3_bundle_unchanged": source["bundle_snapshot_before"] == source_bundle_snapshot_after,
        "v3_overwritten": False,
        "combined_four_subset_accuracy_used": True,
        "combined_four_subset_accuracy_used_descriptively": True,
        "combined_four_subset_inference_used": False,
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_api_calls": 0,
        "token_valid_counts": descriptive["token_valid_counts"],
        "token_missing_counts": descriptive["token_missing_counts"],
        "figures_png": len([path for path in figures if path.suffix == ".png"]),
        "figures_svg": len([path for path in figures if path.suffix == ".svg"]),
        "figure_files_nonempty_and_openable": figure_files_openable,
        "tables_csv": len(list(table_dir.glob("*.csv"))),
        "gate_results": {},
    }
    validation["gate_results"] = {
        "responses_81756": validation["responses"] == 81756,
        "questions_1514": validation["questions"] == 1514,
        "models_6": validation["models"] == 6,
        "conditions_9": validation["conditions"] == 9,
        "subset_question_counts": subset_question_counts == SUBSET_QUESTION_COUNTS,
        "cochran_q_groups_48": validation["cochran_q_groups"] == 48,
        "gender_comparisons_96": validation["gender_comparisons"] == 96,
        "scenario_comparisons_336": validation["scenario_comparisons"] == 336,
        "implicit_vs_explicit_144": validation["implicit_vs_explicit_comparisons"] == 144,
        "persona_vs_neutral_192": validation["persona_vs_neutral_comparisons"] == 192,
        "scenario_holm_family_size_7": scenario_holm_sizes == [7],
        "gender_holm_family_size_4": direct_holm_sizes == [4],
        "duplicate_response_keys_zero": validation["duplicate_response_keys"] == 0,
        "source_v3_validation_true": validation["source_validation_valid"],
        "source_csv_sha256_unchanged": validation["source_csv_unchanged"],
        "source_csv_sha256_matches_v3": validation["source_csv_matches_v3_manifest"],
        "source_v3_unchanged": validation["source_v3_bundle_unchanged"],
        "no_v3_overwrite": validation["v3_overwritten"] is False,
        "no_statistics_recompute": validation["statistics_recomputed"] is False,
        "no_rescoring": validation["response_rescoring"] is False,
        "no_llm_api_call": validation["llm_api_calls"] == 0,
        "all_four_subsets_present": validation["subsets_present"] == SUBSET_ORDER,
        "overall_accuracy_rows_54": validation["overall_accuracy_rows"] == 54,
        "overall_accuracy_models_6": validation["overall_accuracy_models"] == 6,
        "overall_accuracy_conditions_9": validation["overall_accuracy_conditions"] == 9,
        "overall_accuracy_is_all_four_subsets": validation["overall_accuracy_subsets"] == ["all_four_subsets"],
        "overall_accuracy_total_per_condition_1514": validation["overall_accuracy_total_responses_per_condition"] == [1514],
        "overall_accuracy_unique_model_persona": validation["overall_accuracy_duplicate_model_persona"] == 0,
        "overall_gender_summary_one_row": validation["overall_gender_summary_rows"] == 1,
        "overall_gender_by_model_rows_6": validation["overall_gender_by_model_rows"] == 6,
        "overall_gender_by_scenario_rows_4": validation["overall_gender_by_scenario_rows"] == 4,
        "overall_accuracy_descriptive_only": (
            validation["combined_four_subset_accuracy_used_descriptively"]
            and not validation["combined_four_subset_inference_used"]
        ),
        "no_combined_four_subset_inference": validation["combined_four_subset_inference_used"] is False,
        "figure_files_openable": validation["figure_files_nonempty_and_openable"],
        "accuracy_status_rows_216": len(accuracy_status) == 216,
        "accuracy_full_denominator": bool(
            (accuracy_status["total"] == accuracy_status["committed"] + accuracy_status["abstain"] + accuracy_status["missing"]).all()
        ),
        "v3_accuracy_matches_descriptive_correct_count": bool(
            (accuracy_status["n_correct"] == accuracy_status["n_correct_from_descriptive_source"]).all()
        ),
    }
    validation["valid"] = all(validation["gate_results"].values())

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "source_bundle": str(source["base"]),
        "source_manifest": str(source["manifest_path"]),
        "source_validation": str(source["validation_path"]),
        "descriptive_source_csv": str(descriptive["input_path"]),
        "output": str(output),
        "purpose": "Template reusable presentasi akurasi ETHICS; overall deskriptif, lalu hasil inferensial per subset.",
        "inferential_source": "Bundle v3 gender tervalidasi; tabel inferensial disalin tanpa perhitungan ulang.",
        "descriptive_source": "Tabel overall v3 dipakai sebagai ringkasan gabungan; CSV pada analysis_manifest.json dipakai untuk status dan metadata token.",
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "api_calls": 0,
        "combined_four_subset_accuracy_used": True,
        "combined_four_subset_accuracy_used_descriptively": True,
        "combined_four_subset_inference_used": False,
        "subsets_retained_separately": SUBSET_ORDER,
        "persona_order": PERSONA_ORDER,
        "status_definition": "abstain dan missing dihitung sebagai tidak benar pada accuracy_full; accuracy_committed hanya metrik tambahan.",
        "interpretation_limits": {
            "commonsense": "akurasi kecocokan label biner ETHICS",
            "justice": "akurasi kecocokan label biner ETHICS",
            "deontology": "kecocokan label per butir, bukan kemampuan moral absolut",
            "virtue": "kecocokan label per butir, bukan kemampuan moral absolut",
        },
        "validation": validation,
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index = artifact_index(output)
    write_csv(index, output / "output_index.csv")
    if not validation["valid"]:
        raise ValueError(f"Validasi reusable gender gagal: {validation}")
    return {
        "output_dir": output,
        "accuracy": accuracy_status,
        "accuracy_status": accuracy_status,
        "overall_accuracy": overall_accuracy,
        "overall_gender": overall_gender,
        "overall_gender_by_model": overall_gender_by_model,
        "overall_gender_by_scenario": overall_gender_by_scenario,
        "direct_summary": summary_direct,
        "direct_significant": significant_direct,
        "scenario_summary": summary_scenario,
        "scenario_supported": supported_scenario,
        "status": status_by_model,
        "status_by_model": status_by_model,
        "tokens": token_by_model,
        "token_by_model": token_by_model,
        "token_summary": token_summary,
        "validation": validation,
        "manifest": manifest,
        "output_index": index,
    }


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(path: Path, *, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Notebook sumber sudah ada; gunakan tanpa --rebuild-source: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
# ETHICS Gender Persona - Template Reusable Akurasi

Notebook ini mempertahankan gaya analisis ETHICS gender yang dipakai untuk
visual Bab VI: heatmap efek berpasangan dengan empat panel subset. Template
ini juga menyediakan tabel akurasi penuh, status respons, metadata token, dan
indeks artefak agar dapat diadaptasi untuk persona lain.

Angka inferensial diambil dari bundle v3 tervalidasi. Notebook tidak melakukan
API call, inference, parsing ulang, rescoring, atau perhitungan ulang uji
statistik.
"""
        ),
        md("## 0. Provenance dan validation gate"),
        code(
            """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display

start = Path.cwd().resolve()
REPO_ROOT = next(
    path for path in [start, *start.parents]
    if (path / "AGENTS.md").exists() and (path / "src").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.ethics.build_gender_publication_notebook_v4_reusable import run_analysis

result = run_analysis()
OUTPUT_DIR = Path(result["output_dir"])
FIGURE_DIR = OUTPUT_DIR / "figures"
display(pd.Series(result["validation"], name="value").to_frame())
assert result["validation"]["valid"] is True
"""
        ),
        md(
            """
## 1. Akurasi penuh untuk setiap kondisi dan subset

Akurasi penuh menggunakan seluruh respons pada kondisi sebagai penyebut.
Kolom `accuracy_committed` disajikan sebagai metrik tambahan dan harus dibaca
bersama `commit_rate`; `abstain` dan `missing` tidak dihapus dari denominator.
"""
        ),
        code(
            """
accuracy_columns = [
    "subset_label", "model_label", "condition_label", "scenario", "total",
    "committed", "commit_rate", "abstain", "abstain_rate", "missing",
    "missing_rate", "n_correct", "accuracy_full_pct", "n_correct_committed",
    "accuracy_committed_pct",
]
display(
    result["accuracy_status"][accuracy_columns]
    .style.format({
        "commit_rate": "{:.2%}", "abstain_rate": "{:.2%}",
        "missing_rate": "{:.2%}", "accuracy_full_pct": "{:.2f}",
        "accuracy_committed_pct": "{:.2f}",
    })
)
"""
        ),
        md("## 2. Perbandingan langsung laki-laki dan perempuan - hasil utama"),
        code(
            """
display(
    result["direct_summary"].style.format({
        "largest_delta_pp": "{:+.2f}",
        "largest_cohens_h": "{:+.2f}",
        "largest_p_holm": "{:.4g}",
    })
)
display(Image(filename=str(FIGURE_DIR / "01_gender_direct_effect_holm_heatmap.png"), width=1100))
"""
        ),
        md("## 3. Perbandingan I1-I3 terhadap E"),
        code(
            """
display(
    result["scenario_summary"].query("comparison_category == 'implicit_vs_explicit'")
    .style.format({"largest_delta_pp": "{:+.2f}", "largest_cohens_h": "{:+.2f}"})
)
display(Image(filename=str(FIGURE_DIR / "02_implicit_vs_explicit_effect_q_holm_heatmap.png"), width=1100))
"""
        ),
        md("## 4. Perbandingan E, I1, I2, dan I3 terhadap N"),
        code(
            """
display(
    result["scenario_summary"].query("comparison_category == 'persona_vs_neutral'")
    .style.format({"largest_delta_pp": "{:+.2f}", "largest_cohens_h": "{:+.2f}"})
)
display(Image(filename=str(FIGURE_DIR / "03_persona_vs_neutral_effect_q_holm_heatmap.png"), width=1100))
"""
        ),
        md("## 5. Visualisasi akurasi penuh, status respons, dan token"),
        code(
            """
display(Image(filename=str(FIGURE_DIR / "04_accuracy_full_by_condition_heatmap.png"), width=1200))
display(result["status_by_model"].style.format({
    "committed_rate": "{:.2%}", "abstain_rate": "{:.2%}", "missing_rate": "{:.2%}"
}))
display(Image(filename=str(FIGURE_DIR / "05_response_status_by_model.png"), width=1000))
display(result["token_by_model"].style.format({
    "prompt_tokens_mean": "{:.1f}", "completion_tokens_mean": "{:.1f}",
    "reasoning_tokens_mean": "{:.1f}", "total_tokens_mean": "{:.1f}",
}))
display(Image(filename=str(FIGURE_DIR / "06_token_usage_by_model.png"), width=1050))
"""
        ),
        md(
            """
## 6. Batas interpretasi dan provenance

Analisis dipertahankan terpisah untuk Commonsense, Deontology, Justice, dan
Virtue. Untuk Deontology dan Virtue, metrik yang ditampilkan adalah kecocokan
label per butir, bukan ukuran kemampuan moral absolut. Tidak ada satu angka
akurasi gabungan empat subset sebagai hasil inferensial.

Status dan token bersifat deskriptif. Metadata token yang hilang tidak
diimputasi; jumlah baris validnya tersedia pada tabel token.
"""
        ),
        code(
            """
display(pd.DataFrame({
    "subset": ["Commonsense", "Deontology", "Justice", "Virtue"],
    "interpretation_limit": [
        "Kecocokan label biner ETHICS",
        "Kecocokan label per butir; bukan kemampuan moral absolut",
        "Kecocokan label biner ETHICS",
        "Kecocokan label per butir; bukan kemampuan moral absolut",
    ],
}))
display(result["output_index"])
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, path)
    return path


def execute_notebook(source: Path, output: Path, *, overwrite: bool = False) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Notebook tereksekusi sudah ada; gunakan --overwrite-executed bila memang diminta: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root())}},
    )
    client.execute()
    nbf.write(notebook, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally execute the reusable ETHICS gender notebook")
    parser.add_argument("--execute", action="store_true", help="execute the source notebook after it is available")
    parser.add_argument("--rebuild-source", action="store_true", help="rebuild this generated source notebook")
    parser.add_argument("--overwrite-executed", action="store_true", help="allow replacing this new executed notebook")
    args = parser.parse_args()
    root = repo_root()
    source = source_notebook_path(root)
    if args.rebuild_source or not source.exists():
        build_notebook(source, overwrite=args.rebuild_source)
        print(f"[NOTEBOOK] Source dibuat: {source}")
    else:
        print(f"[NOTEBOOK] Source dipakai tanpa ditimpa: {source}")
    if args.execute:
        executed = execute_notebook(
            source,
            executed_notebook_path(root),
            overwrite=args.overwrite_executed,
        )
        print(f"[NOTEBOOK] Executed: {executed}")


if __name__ == "__main__":
    main()

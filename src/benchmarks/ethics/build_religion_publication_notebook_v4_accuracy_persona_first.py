"""Build the thesis-facing ETHICS Religion accuracy v4 notebook.

The builder is a presentation layer over the validated religion v3 bundle. It
does not call an API, rerun extraction, rescore responses, or recompute an
inferential test. Full-denominator accuracy and all inferential values are read
from v3 tables. Response status and token summaries are descriptive aggregates
of the source CSV referenced by the v3 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient
from matplotlib.colors import LinearSegmentedColormap


ANALYSIS_VERSION = "ethics-religion-v4-accuracy-persona-first-20260905"
EXPECTED_V3_VERSION = "ethics-religion-v3-accuracy-paired-holm-20260822"

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
EXPECTED_SUBSET_QUESTION_COUNTS = {
    "commonsense": 388,
    "deontology": 359,
    "justice": 270,
    "virtue": 497,
}

IDENTITY_ORDER = ["islam", "christian", "jewish", "atheist"]
IDENTITY_LABELS = {
    "islam": "Islam",
    "christian": "Kristen",
    "jewish": "Yahudi",
    "atheist": "Ateis",
}
SCENARIO_ORDER = ["N", "E", "I1", "I2", "I3"]
PERSONA_ORDER = [
    "neutral",
    *[f"{identity}_explicit" for identity in IDENTITY_ORDER],
    *[f"{identity}_implicit_{number}" for identity in IDENTITY_ORDER for number in (1, 2, 3)],
]
# The display order is intentionally interleaved by identity, not by source CSV order.
CONDITION_ORDER = ["neutral"] + [
    f"{identity}_{suffix}"
    for identity in IDENTITY_ORDER
    for suffix in ("explicit", "implicit_1", "implicit_2", "implicit_3")
]
PAIR_ORDER = [
    ("islam", "christian"),
    ("islam", "jewish"),
    ("islam", "atheist"),
    ("christian", "jewish"),
    ("christian", "atheist"),
    ("jewish", "atheist"),
]
IDENTITY_RANK = {identity: index for index, identity in enumerate(IDENTITY_ORDER)}
SOURCE_USECOLS = [
    "model",
    "persona_folder",
    "question_id",
    "subset",
    "answer_status",
    "is_correct",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
]
TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]
STATUS_COLUMNS = ["committed", "abstain", "missing"]

EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "religion_effect",
    ["#2166AC", "#D7E5F0", "#FCFBF8", "#F2D2CC", "#B2182B"],
)
ACCURACY_CMAP = "YlGnBu"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def v3_root(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Religion_Persona"
        / "_combined"
        / "persona_publication_analysis_v3_accuracy_paired_holm_six_models"
    )


def default_output_dir(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "reports"
        / "Religion_Persona"
        / "_combined"
        / "persona_publication_analysis_v4_accuracy_persona_first_six_models"
    )


def source_notebook_path(root: Path) -> Path:
    return (
        root
        / "notebooks"
        / "religion"
        / "ETHICS_Religion_Persona_Publication_Analysis_v4_Accuracy_Persona_First.ipynb"
    )


def executed_notebook_path(root: Path) -> Path:
    return (
        root
        / "result"
        / "ETHICS"
        / "ETHICS_Religion_Persona_Publication_Analysis_v4_Accuracy_Persona_First.ipynb"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_snapshot(path: Path) -> str:
    """Hash file names, sizes, and contents without modifying the bundle."""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(f"{relative}\t{file_path.stat().st_size}\t".encode("utf-8"))
        digest.update(bytes.fromhex(sha256(file_path)))
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(value: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_bool_series(series: pd.Series, name: str, *, allow_na: bool = False) -> pd.Series:
    """Parse boolean columns without silently treating unknown values as false."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    result = normalized.map({"true": True, "false": False})
    if result.isna().any():
        unknown = sorted(normalized[result.isna()].dropna().unique().tolist())
        if unknown or not allow_na:
            raise ValueError(f"Nilai boolean tidak dikenal pada {name}: {unknown}")
        return result.fillna(False).astype(bool)
    return result.astype(bool)


def condition_parts(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "N"
    pieces = persona.split("_")
    if len(pieces) != 2 and len(pieces) != 3:
        raise ValueError(f"Format persona tidak dikenal: {persona}")
    identity = pieces[0]
    if pieces[1] == "explicit":
        scenario = "E"
    elif pieces[1] == "implicit" and len(pieces) == 3 and pieces[2] in {"1", "2", "3"}:
        scenario = f"I{pieces[2]}"
    else:
        raise ValueError(f"Format persona tidak dikenal: {persona}")
    if identity not in IDENTITY_ORDER:
        raise ValueError(f"Identitas agama tidak dikenal: {persona}")
    return identity, scenario


def add_condition_parts(frame: pd.DataFrame, persona_column: str = "persona") -> pd.DataFrame:
    result = frame.copy()
    parts = result[persona_column].map(condition_parts)
    result["identity"] = parts.map(lambda value: value[0])
    result["scenario"] = parts.map(lambda value: value[1])
    result["identity_label"] = result["identity"].map({"neutral": "Netral", **IDENTITY_LABELS})
    result["persona_label"] = result["persona"].map(
        lambda value: "Netral N"
        if value == "neutral"
        else f"{IDENTITY_LABELS[condition_parts(value)[0]]} {condition_parts(value)[1]}"
    )
    return result


def load_v3_tables(root: Path) -> dict[str, object]:
    base = v3_root(root)
    manifest_path = base / "metadata" / "analysis_manifest.json"
    validation_path = base / "metadata" / "validation.json"
    if not manifest_path.exists() or not validation_path.exists():
        raise FileNotFoundError(f"Metadata v3 tidak lengkap: {base}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != EXPECTED_V3_VERSION:
        raise ValueError(f"Versi sumber bukan v3 kanonik: {manifest.get('analysis_version')}")
    if validation.get("valid") is not True:
        raise ValueError("Validation v3 harus bernilai true")

    source_csv = Path(manifest["input"]).resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"CSV sumber manifest tidak ditemukan: {source_csv}")
    source_sha_before = sha256(source_csv)
    expected_sha = validation.get("input_sha256_before")
    if expected_sha and source_sha_before != expected_sha:
        raise ValueError(
            "SHA-256 CSV sumber berbeda dari manifest v3: "
            f"expected={expected_sha}, actual={source_sha_before}"
        )

    table_dir = base / "tables"
    table_names = {
        "accuracy": "accuracy_by_subset_model_condition.csv",
        "pairwise": "mcnemar_religion_pairwise_comparisons.csv",
        "scenario": "mcnemar_scenario_comparisons.csv",
        "cochran": "cochran_q_by_subset_model_religion.csv",
    }
    tables = {
        key: pd.read_csv(table_dir / filename) for key, filename in table_names.items()
    }
    for frame in tables.values():
        for column in [name for name in frame.columns if name.startswith("significant")]:
            frame[column] = normalize_bool_series(frame[column], column)
    return {
        "base": base,
        "manifest": manifest,
        "validation": validation,
        "manifest_path": manifest_path,
        "validation_path": validation_path,
        "source_csv": source_csv,
        "source_sha_before": source_sha_before,
        **tables,
    }


def aggregate_source_csv(source_csv: Path) -> dict[str, object]:
    """Aggregate only pre-existing status/token fields from the manifest CSV."""
    parts: list[pd.DataFrame] = []
    question_ids: defaultdict[str, set[str]] = defaultdict(set)
    response_keys: set[tuple[str, str, str, str]] = set()
    status_values: set[str] = set()
    row_count = 0

    for chunk in pd.read_csv(
        source_csv,
        usecols=SOURCE_USECOLS,
        chunksize=100_000,
        low_memory=False,
    ):
        row_count += len(chunk)
        if chunk["question_id"].isna().any():
            raise ValueError("question_id kosong pada CSV sumber")
        if chunk["subset"].isna().any() or chunk["model"].isna().any():
            raise ValueError("subset/model kosong pada CSV sumber")

        chunk = chunk.rename(columns={"persona_folder": "persona"})
        chunk = add_condition_parts(chunk)
        status_values.update(chunk["answer_status"].fillna("<NA>").astype(str).unique().tolist())
        # Empty source scores occur on nonanswer rows; v3 defines those rows as incorrect.
        chunk["is_correct_bool"] = normalize_bool_series(chunk["is_correct"], "is_correct", allow_na=True)
        chunk["is_committed"] = chunk["answer_status"].eq("committed")
        chunk["is_abstain"] = chunk["answer_status"].eq("abstain")
        chunk["is_missing"] = chunk["answer_status"].eq("missing")
        if not chunk["answer_status"].isin(STATUS_COLUMNS).all():
            unexpected = sorted(set(chunk.loc[~chunk["answer_status"].isin(STATUS_COLUMNS), "answer_status"].astype(str)))
            raise ValueError(f"Status respons tidak dikenal: {unexpected}")
        chunk["correct_committed"] = chunk["is_correct_bool"] & chunk["is_committed"]

        for subset, values in chunk.groupby("subset", sort=False)["question_id"]:
            question_ids[str(subset)].update(values.astype(str).tolist())
        response_keys.update(
            zip(
                chunk["model"].astype(str),
                chunk["persona"].astype(str),
                chunk["subset"].astype(str),
                chunk["question_id"].astype(str),
            )
        )

        for column in TOKEN_COLUMNS:
            chunk[f"{column}_numeric"] = pd.to_numeric(chunk[column], errors="coerce")

        group_columns = ["model", "persona", "identity", "scenario", "subset"]
        aggregation: dict[str, tuple[str, str]] = {
            "total": ("answer_status", "size"),
            "committed": ("is_committed", "sum"),
            "abstain": ("is_abstain", "sum"),
            "missing": ("is_missing", "sum"),
            "n_correct_full_source": ("is_correct_bool", "sum"),
            "n_correct_committed": ("correct_committed", "sum"),
        }
        for column in TOKEN_COLUMNS:
            aggregation[f"sum_{column}"] = (f"{column}_numeric", "sum")
            aggregation[f"n_valid_{column}"] = (f"{column}_numeric", "count")
        grouped = chunk.groupby(group_columns, observed=True, sort=False).agg(**aggregation).reset_index()
        parts.append(grouped)

    if not parts:
        raise ValueError("CSV sumber kosong")
    group_columns = ["model", "persona", "identity", "scenario", "subset"]
    numeric_columns = [
        "total",
        "committed",
        "abstain",
        "missing",
        "n_correct_full_source",
        "n_correct_committed",
        *[f"sum_{column}" for column in TOKEN_COLUMNS],
        *[f"n_valid_{column}" for column in TOKEN_COLUMNS],
    ]
    descriptive = pd.concat(parts, ignore_index=True).groupby(group_columns, as_index=False).agg(
        {column: "sum" for column in numeric_columns}
    )
    for column in TOKEN_COLUMNS:
        descriptive.loc[descriptive[f"n_valid_{column}"].eq(0), f"sum_{column}"] = np.nan
        descriptive[f"avg_{column}"] = descriptive[f"sum_{column}"] / descriptive[f"n_valid_{column}"].replace(0, np.nan)
    descriptive = descriptive.sort_values(
        by=["model", "subset", "persona"],
        key=lambda series: series.map(
            {value: index for index, value in enumerate(MODEL_ORDER if series.name == "model" else SUBSET_ORDER if series.name == "subset" else CONDITION_ORDER)}
        ).fillna(999),
    ).reset_index(drop=True)
    descriptive["model_label"] = descriptive["model"].map(MODEL_LABELS)
    descriptive["subset_label"] = descriptive["subset"].map(SUBSET_LABELS)
    descriptive["identity_label"] = descriptive["identity"].map({"neutral": "Netral", **IDENTITY_LABELS})
    descriptive["persona_label"] = descriptive["persona"].map(
        lambda value: "Netral N"
        if value == "neutral"
        else f"{IDENTITY_LABELS[condition_parts(value)[0]]} {condition_parts(value)[1]}"
    )
    return {
        "data": descriptive,
        "row_count": row_count,
        "question_ids": question_ids,
        "duplicate_response_keys": row_count - len(response_keys),
        "status_values": sorted(status_values),
    }


def add_accuracy_status_metrics(accuracy_v3: pd.DataFrame, descriptive: pd.DataFrame) -> pd.DataFrame:
    accuracy = accuracy_v3.copy().rename(columns={"persona": "persona"})
    accuracy = accuracy.rename(
        columns={
            "total_responses": "total",
            "n_correct": "n_correct_full",
            "accuracy": "accuracy_full",
            "accuracy_pct": "accuracy_full_pct",
            "n_incorrect": "n_incorrect_full",
        }
    )
    key_columns = ["model", "persona", "identity", "scenario", "subset"]
    source_columns = key_columns + [
        "total",
        "committed",
        "abstain",
        "missing",
        "n_correct_full_source",
        "n_correct_committed",
    ]
    merged = accuracy.merge(
        descriptive[source_columns],
        on=key_columns,
        how="left",
        suffixes=("", "_source"),
        validate="one_to_one",
    )
    if merged["total_source"].isna().any():
        raise ValueError("Kunci akurasi v3 tidak cocok dengan sumber CSV")
    if not (merged["total"].eq(merged["total_source"])).all():
        raise ValueError("Denominator akurasi v3 berbeda dari CSV sumber")
    if not (merged["n_correct_full"].eq(merged["n_correct_full_source"])).all():
        raise ValueError("n_correct v3 berbeda dari is_correct pada CSV sumber")
    merged = merged.drop(columns=["total_source", "n_correct_full_source"])
    merged["n_incorrect_full"] = merged["total"] - merged["n_correct_full"]
    merged["commit_rate"] = merged["committed"] / merged["total"]
    merged["abstain_rate"] = merged["abstain"] / merged["total"]
    merged["missing_rate"] = merged["missing"] / merged["total"]
    merged["accuracy_committed"] = merged["n_correct_committed"] / merged["committed"].replace(0, np.nan)
    for column in ["commit_rate", "abstain_rate", "missing_rate", "accuracy_committed"]:
        merged[f"{column}_pct"] = merged[column] * 100
    merged["construct_valid"] = merged["subset"].isin(["commonsense", "justice"])
    merged["interpretation_note"] = merged["subset"].map(
        {
            "commonsense": "Ketepatan keputusan biner; label 1 berarti tindakan salah.",
            "justice": "Ketepatan keputusan biner; label 1 berarti pernyataan wajar.",
            "deontology": "Kecocokan label per butir; bukan kemampuan moral absolut.",
            "virtue": "Kecocokan label per butir; bukan kemampuan moral absolut.",
        }
    )
    order = {value: index for index, value in enumerate(CONDITION_ORDER)}
    merged["_condition_order"] = merged["persona"].map(order)
    merged["_model_order"] = merged["model"].map({value: index for index, value in enumerate(MODEL_ORDER)})
    merged["_subset_order"] = merged["subset"].map({value: index for index, value in enumerate(SUBSET_ORDER)})
    merged = merged.sort_values(["_subset_order", "_model_order", "_condition_order"]).drop(
        columns=["_condition_order", "_model_order", "_subset_order"]
    )
    return merged.reset_index(drop=True)


def aggregate_status(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    aggregations = {
        "total": ("total", "sum"),
        "committed": ("committed", "sum"),
        "abstain": ("abstain", "sum"),
        "missing": ("missing", "sum"),
        "n_correct_full": ("n_correct_full", "sum"),
        "n_correct_committed": ("n_correct_committed", "sum"),
    }
    result = frame.groupby(group_columns, as_index=False).agg(**aggregations)
    result["n_incorrect_full"] = result["total"] - result["n_correct_full"]
    result["accuracy_full"] = result["n_correct_full"] / result["total"]
    result["accuracy_full_pct"] = result["accuracy_full"] * 100
    result["commit_rate"] = result["committed"] / result["total"]
    result["abstain_rate"] = result["abstain"] / result["total"]
    result["missing_rate"] = result["missing"] / result["total"]
    result["accuracy_committed"] = result["n_correct_committed"] / result["committed"].replace(0, np.nan)
    for column in ["commit_rate", "abstain_rate", "missing_rate", "accuracy_committed"]:
        result[f"{column}_pct"] = result[column] * 100
    return result


def add_token_labels(token_frame: pd.DataFrame) -> pd.DataFrame:
    result = token_frame.copy()
    if "model" in result.columns and "model_label" not in result.columns:
        result["model_label"] = result["model"].map(MODEL_LABELS)
    if "subset" in result.columns and "subset_label" not in result.columns:
        result["subset_label"] = result["subset"].map(SUBSET_LABELS)
    if "identity" in result.columns and "identity_label" not in result.columns:
        result["identity_label"] = result["identity"].map({"neutral": "Netral", **IDENTITY_LABELS})
    if "persona" in result.columns and "persona_label" not in result.columns:
        result["persona_label"] = result["persona"].map(
            lambda value: "Netral N"
            if value == "neutral"
            else f"{IDENTITY_LABELS[condition_parts(value)[0]]} {condition_parts(value)[1]}"
        )
    sort_columns = []
    if "subset" in result.columns:
        result["_subset_order"] = result["subset"].map({value: index for index, value in enumerate(SUBSET_ORDER)})
        sort_columns.append("_subset_order")
    if "model" in result.columns:
        result["_model_order"] = result["model"].map({value: index for index, value in enumerate(MODEL_ORDER)})
        sort_columns.append("_model_order")
    if "persona" in result.columns:
        result["_condition_order"] = result["persona"].map({value: index for index, value in enumerate(CONDITION_ORDER)})
        sort_columns.append("_condition_order")
    if sort_columns:
        result = result.sort_values(sort_columns).drop(columns=sort_columns)
    return result.reset_index(drop=True)


def aggregate_tokens(token_condition: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for column in TOKEN_COLUMNS:
        aggregations[f"sum_{column}"] = (f"sum_{column}", "sum")
        aggregations[f"n_valid_{column}"] = (f"n_valid_{column}", "sum")
    result = token_condition.groupby(group_columns, as_index=False).agg(**aggregations)
    for column in TOKEN_COLUMNS:
        result.loc[result[f"n_valid_{column}"].eq(0), f"sum_{column}"] = np.nan
        result[f"avg_{column}"] = result[f"sum_{column}"] / result[f"n_valid_{column}"].replace(0, np.nan)
    return add_token_labels(result)


def canonical_scenario_comparisons(scenario: pd.DataFrame) -> pd.DataFrame:
    result = scenario.copy()
    result["source_comparison_label"] = result["comparison_label"]
    result["orientation_flipped"] = False
    for index, row in result.iterrows():
        category = row["comparison_category"]
        if category == "implicit_vs_explicit":
            should_flip = row["scenario_a"] == "E" and row["scenario_b"] in {"I1", "I2", "I3"}
        elif category == "persona_vs_neutral":
            should_flip = row["scenario_a"] == "N" and row["scenario_b"] in {"E", "I1", "I2", "I3"}
        else:
            should_flip = False
        if should_flip:
            _swap_pairwise_values(result, index)
            result.at[index, "orientation_flipped"] = True
        result.at[index, "comparison_label"] = (
            f"{row['identity_label']} {result.at[index, 'scenario_a']} − {result.at[index, 'scenario_b']}"
        )
    return result


def _swap_pairwise_values(frame: pd.DataFrame, index: int) -> None:
    for first, second in [
        ("scenario_a", "scenario_b"),
        ("condition_a", "condition_b"),
        ("accuracy_a_pct", "accuracy_b_pct"),
        ("discordant_a_correct_b_wrong", "discordant_a_wrong_b_correct"),
    ]:
        first_value = frame.at[index, first]
        frame.at[index, first] = frame.at[index, second]
        frame.at[index, second] = first_value
    frame.at[index, "delta_accuracy_pp"] = -frame.at[index, "delta_accuracy_pp"]
    frame.at[index, "cohens_h"] = -frame.at[index, "cohens_h"]


def canonical_pairwise_comparisons(pairwise: pd.DataFrame) -> pd.DataFrame:
    result = pairwise.copy()
    result["source_identity"] = result["identity"]
    result["source_identity_label"] = result["identity_label"]
    result["source_comparison_label"] = result["comparison_label"]
    result["source_delta_accuracy_pp"] = result["delta_accuracy_pp"]
    result["source_cohens_h"] = result["cohens_h"]
    result["orientation_flipped"] = False
    result["persona_a"] = result["condition_a"].map(lambda value: condition_parts(value)[0])
    result["persona_b"] = result["condition_b"].map(lambda value: condition_parts(value)[0])
    for index, row in result.iterrows():
        persona_a = row["persona_a"]
        persona_b = row["persona_b"]
        if IDENTITY_RANK[persona_a] > IDENTITY_RANK[persona_b]:
            _swap_pairwise_values(result, index)
            result.at[index, "persona_a"] = persona_b
            result.at[index, "persona_b"] = persona_a
            result.at[index, "orientation_flipped"] = True
        first = result.at[index, "persona_a"]
        second = result.at[index, "persona_b"]
        result.at[index, "identity"] = f"{first}_vs_{second}"
        result.at[index, "identity_label"] = f"{IDENTITY_LABELS[first]} vs {IDENTITY_LABELS[second]}"
        result.at[index, "comparison_label"] = (
            f"{IDENTITY_LABELS[first]} − {IDENTITY_LABELS[second]} {result.at[index, 'scenario_a']}"
        )
        result.at[index, "canonical_display_order"] = (
            PAIR_ORDER.index((first, second)) * 4 + ["E", "I1", "I2", "I3"].index(result.at[index, "scenario_a"])
        )
    if not result["holm_adjusted_p_value"].equals(pairwise["holm_adjusted_p_value"]):
        raise ValueError("Normalisasi orientasi mengubah p-value Holm")
    if not result["exact_mcnemar_p_value"].equals(pairwise["exact_mcnemar_p_value"]):
        raise ValueError("Normalisasi orientasi mengubah p-value McNemar")
    return result


def largest_significant_row(frame: pd.DataFrame, significance_column: str) -> pd.Series | None:
    selected = frame[frame[significance_column].astype(bool)]
    if selected.empty:
        return None
    return selected.loc[selected["delta_accuracy_pp"].abs().idxmax()]


def persona_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", "model_label", "subset", "subset_label", "scenario_a"]
    for keys, selected in pairwise.groupby(group_columns, sort=False):
        model, model_label, subset, subset_label, scenario = keys
        significant = selected[selected["significant_holm_0_05"]]
        largest = largest_significant_row(selected, "significant_holm_0_05")
        winners = pd.Series(dtype="string")
        if not significant.empty:
            winners = significant.apply(
                lambda row: row["persona_a"] if row["delta_accuracy_pp"] > 0 else row["persona_b"] if row["delta_accuracy_pp"] < 0 else "tie",
                axis=1,
            )
        row: dict[str, object] = {
            "model": model,
            "model_label": model_label,
            "subset": subset,
            "subset_label": subset_label,
            "scenario": scenario,
            "n_comparisons": int(len(selected)),
            "n_significant_holm": int(len(significant)),
            "n_ties_significant": int((winners == "tie").sum()),
            "largest_comparison": (
                str(largest["comparison_label"]) if largest is not None else "—"
            ),
            "largest_delta_accuracy_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
            "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
            "largest_holm_adjusted_p_value": float(largest["holm_adjusted_p_value"]) if largest is not None else np.nan,
        }
        for identity in IDENTITY_ORDER:
            row[f"n_{identity}_higher"] = int((winners == identity).sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_model_order"] = result["model"].map({value: index for index, value in enumerate(MODEL_ORDER)})
    result["_subset_order"] = result["subset"].map({value: index for index, value in enumerate(SUBSET_ORDER)})
    result["_scenario_order"] = result["scenario"].map({value: index for index, value in enumerate(["E", "I1", "I2", "I3"])})
    return result.sort_values(["_model_order", "_subset_order", "_scenario_order"]).drop(
        columns=["_model_order", "_subset_order", "_scenario_order"]
    ).reset_index(drop=True)


def persona_summary_by_model(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        selected = pairwise[pairwise["model"].eq(model)]
        significant = selected[selected["significant_holm_0_05"]]
        winners = significant.apply(
            lambda row: row["persona_a"] if row["delta_accuracy_pp"] > 0 else row["persona_b"] if row["delta_accuracy_pp"] < 0 else "tie",
            axis=1,
        )
        largest = largest_significant_row(selected, "significant_holm_0_05")
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_comparisons": int(len(selected)),
                "n_significant_holm": int(len(significant)),
                "n_ties_significant": int((winners == "tie").sum()),
                **{f"n_{identity}_higher": int((winners == identity).sum()) for identity in IDENTITY_ORDER},
                "significant_scenarios": ", ".join(sorted(significant["scenario_a"].unique().tolist(), key=["E", "I1", "I2", "I3"].index)) if not significant.empty else "—",
                "largest_comparison": str(largest["comparison_label"]) if largest is not None else "—",
                "largest_subset": str(largest["subset_label"]) if largest is not None else "—",
                "largest_delta_accuracy_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                "largest_holm_adjusted_p_value": float(largest["holm_adjusted_p_value"]) if largest is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scenario_summary(scenario: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["comparison_category", "model", "model_label", "subset", "subset_label"]
    for keys, selected in scenario.groupby(group_columns, sort=False):
        category, model, model_label, subset, subset_label = keys
        supported = selected[selected["supported_q_and_holm"]]
        largest = largest_significant_row(selected, "supported_q_and_holm")
        rows.append(
            {
                "comparison_category": category,
                "model": model,
                "model_label": model_label,
                "subset": subset,
                "subset_label": subset_label,
                "n_comparisons": int(len(selected)),
                "n_supported_q_and_holm": int(len(supported)),
                "n_higher": int((supported["delta_accuracy_pp"] > 0).sum()),
                "n_lower": int((supported["delta_accuracy_pp"] < 0).sum()),
                "n_cochran_q_significant": int(selected["cochran_q_significant_0_05"].iloc[0]),
                "largest_comparison": str(largest["comparison_label"]) if largest is not None else "—",
                "largest_delta_accuracy_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                "largest_holm_adjusted_p_value": float(largest["holm_adjusted_p_value"]) if largest is not None else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["_category_order"] = result["comparison_category"].map({"implicit_vs_explicit": 0, "persona_vs_neutral": 1})
    result["_model_order"] = result["model"].map({value: index for index, value in enumerate(MODEL_ORDER)})
    result["_subset_order"] = result["subset"].map({value: index for index, value in enumerate(SUBSET_ORDER)})
    return result.sort_values(["_category_order", "_model_order", "_subset_order"]).drop(
        columns=["_category_order", "_model_order", "_subset_order"]
    ).reset_index(drop=True)


def scenario_summary_by_model(scenario: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in ["implicit_vs_explicit", "persona_vs_neutral"]:
        for model in MODEL_ORDER:
            selected = scenario[
                scenario["comparison_category"].eq(category) & scenario["model"].eq(model)
            ]
            supported = selected[selected["supported_q_and_holm"]]
            largest = largest_significant_row(selected, "supported_q_and_holm")
            rows.append(
                {
                    "comparison_category": category,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n_comparisons": int(len(selected)),
                    "n_supported_q_and_holm": int(len(supported)),
                    "n_higher": int((supported["delta_accuracy_pp"] > 0).sum()),
                    "n_lower": int((supported["delta_accuracy_pp"] < 0).sum()),
                    "largest_comparison": str(largest["comparison_label"]) if largest is not None else "—",
                    "largest_subset": str(largest["subset_label"]) if largest is not None else "—",
                    "largest_delta_accuracy_pp": float(largest["delta_accuracy_pp"]) if largest is not None else np.nan,
                    "largest_cohens_h": float(largest["cohens_h"]) if largest is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "normal",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )


def annotation_matrix(values: pd.DataFrame, significant: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = ""
            else:
                star = "*" if bool(significant.loc[row, column]) else ""
                annotations.loc[row, column] = f"{value:+.{decimals}f}{star}"
    return annotations


def emphasize_significant_annotations(ax: plt.Axes) -> None:
    """Use bold type only for heatmap annotations carrying the significance star."""
    for annotation in ax.texts:
        annotation.set_fontweight("bold" if annotation.get_text().endswith("*") else "normal")


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def condition_label(persona: str) -> str:
    if persona == "neutral":
        return "N"
    identity, scenario = condition_parts(persona)
    return f"{IDENTITY_LABELS[identity]} {scenario}"


def accuracy_heatmap(accuracy: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = accuracy.copy()
    frame["condition_label"] = frame["persona"].map(condition_label)
    columns = [condition_label(persona) for persona in CONDITION_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(19, 9), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = frame[frame["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="condition_label", values="accuracy_full_pct").reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns
        )
        sns.heatmap(
            values,
            ax=ax,
            cmap=ACCURACY_CMAP,
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            linewidths=0.35,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": 8.2},
        )
        emphasize_significant_annotations(ax)
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=100), cmap=ACCURACY_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
    colorbar.set_label("Akurasi penuh (%)")
    fig.suptitle("Akurasi Penuh ETHICS Agama pada Seluruh Kondisi", fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "N = netral; E = eksplisit; I1–I3 = tiga skenario implisit. Setiap panel adalah satu subset.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, "01_religion_accuracy_full_all_conditions")


def scenario_effect_heatmap(scenario: pd.DataFrame, figure_dir: Path, category: str, stem: str, title: str) -> list[Path]:
    selected_all = scenario[scenario["comparison_category"].eq(category)].copy()
    if category == "implicit_vs_explicit":
        scenario_pairs = [(scenario_name, "E") for scenario_name in ["I1", "I2", "I3"]]
    else:
        scenario_pairs = [(scenario_name, "N") for scenario_name in ["E", "I1", "I2", "I3"]]
    columns = [
        f"{IDENTITY_LABELS[identity]} {first}–{second}"
        for identity in IDENTITY_ORDER
        for first, second in scenario_pairs
    ]
    selected_all["column_label"] = selected_all.apply(
        lambda row: f"{row['identity_label']} {row['scenario_a']}–{row['scenario_b']}", axis=1
    )
    max_effect = float(selected_all["cohens_h"].abs().max())
    limit = max(0.2, float(np.ceil(max_effect * 10) / 10))
    fig, axes = plt.subplots(2, 2, figsize=(19, 10), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = selected_all[selected_all["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns
        )
        significant = selected.pivot(
            index="model_label", columns="column_label", values="supported_q_and_holm"
        ).reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=annotation_matrix(values, significant),
            fmt="",
            linewidths=0.35,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": 7.8},
        )
        emphasize_significant_annotations(ax)
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-limit, vmax=limit), cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
    colorbar.set_label("Cohen's h")
    fig.suptitle(title, fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "Nilai positif menunjukkan kondisi pertama lebih tinggi; * didukung Q Cochran dan tetap signifikan setelah Holm.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, stem)


def pairwise_effect_heatmap(pairwise: pd.DataFrame, figure_dir: Path, scenario: str) -> list[Path]:
    selected_all = pairwise[pairwise["scenario_a"].eq(scenario)].copy()
    columns = [f"{IDENTITY_LABELS[first]}–{IDENTITY_LABELS[second]}" for first, second in PAIR_ORDER]
    selected_all["column_label"] = selected_all.apply(
        lambda row: f"{IDENTITY_LABELS[row['persona_a']]}–{IDENTITY_LABELS[row['persona_b']]}", axis=1
    )
    max_effect = float(selected_all["cohens_h"].abs().max())
    limit = max(0.2, float(np.ceil(max_effect * 10) / 10))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    for ax, subset in zip(axes.flat, SUBSET_ORDER):
        selected = selected_all[selected_all["subset"].eq(subset)]
        values = selected.pivot(index="model_label", columns="column_label", values="cohens_h").reindex(
            index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns
        )
        significant = selected.pivot(
            index="model_label", columns="column_label", values="significant_holm_0_05"
        ).reindex(index=[MODEL_LABELS[model] for model in MODEL_ORDER], columns=columns).fillna(False).astype(bool)
        sns.heatmap(
            values,
            ax=ax,
            cmap=EFFECT_CMAP,
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=annotation_matrix(values, significant),
            fmt="",
            linewidths=0.4,
            linecolor="#FFFFFF",
            cbar=False,
            annot_kws={"fontsize": 8.2},
        )
        emphasize_significant_annotations(ax)
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0)
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-limit, vmax=limit), cmap=EFFECT_CMAP)
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.8, pad=0.015)
    colorbar.set_label("Cohen's h")
    fig.suptitle(f"Perbandingan Antarpersona Agama: {scenario}", fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "Orientasi: Islam, Kristen, Yahudi, lalu Ateis. Nilai positif berarti persona pertama lebih tinggi; * signifikan setelah Holm.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, f"04_religion_pairwise_{scenario.lower()}_effect_holm")


def status_figure(status_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = status_model.set_index("model_label").reindex([MODEL_LABELS[model] for model in MODEL_ORDER])
    percentages = frame[["commit_rate_pct", "abstain_rate_pct", "missing_rate_pct"]]
    percentages.columns = ["Committed", "Abstain", "Missing"]
    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    bottom = np.zeros(len(percentages))
    colors = {"Committed": "#4C956C", "Abstain": "#E9C46A", "Missing": "#D1495B"}
    for column in percentages.columns:
        values = percentages[column].to_numpy(dtype=float)
        ax.bar(percentages.index, values, bottom=bottom, label=column, color=colors[column])
        bottom += values
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Proporsi respons")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Status Respons ETHICS Agama Menurut Model", fontsize=15, fontweight="normal")
    ax.legend(title="Status", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return save_figure(fig, figure_dir, "05_religion_response_status_by_model")


def token_figure(token_model_condition: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = token_model_condition.copy()
    frame["condition_label"] = frame["persona"].map(condition_label)
    condition_labels = [condition_label(persona) for persona in CONDITION_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        selected = frame[frame["model"].eq(model)].set_index("condition_label").reindex(condition_labels)
        ax.plot(condition_labels, selected["avg_total_tokens"], marker="o", color="#2A6F97", linewidth=1.8)
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylabel("Rata-rata total token")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=50, labelsize=7)
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle("Rata-rata Total Token Menurut Kondisi", fontsize=15, fontweight="normal")
    fig.text(
        0.5,
        -0.015,
        "Rata-rata dihitung hanya dari metadata total_tokens yang valid; metadata hilang tidak diimputasi.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, figure_dir, "06_religion_total_tokens_by_condition")


def validate_figure_files(figures: list[Path]) -> dict[str, object]:
    png_files = [path for path in figures if path.suffix.lower() == ".png"]
    svg_files = [path for path in figures if path.suffix.lower() == ".svg"]
    png_valid = all(path.exists() and path.stat().st_size > 100 and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in png_files)
    svg_valid = all(path.exists() and path.stat().st_size > 100 and "<svg" in path.read_text(encoding="utf-8")[:1000] and "</svg>" in path.read_text(encoding="utf-8") for path in svg_files)
    return {
        "png_files": len(png_files),
        "svg_files": len(svg_files),
        "all_figure_files_nonempty": bool(png_valid and svg_valid and len(png_files) == len(svg_files)),
    }


def artifact_index(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "output_index.csv"):
        relative = path.relative_to(output_dir).as_posix()
        if path.suffix.lower() == ".csv":
            shape = pd.read_csv(path, nrows=0).columns.size
            rows_count = max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)
            artifact_type = "table"
        elif path.suffix.lower() in {".png", ".svg"}:
            shape = ""
            rows_count = ""
            artifact_type = "figure"
        else:
            shape = ""
            rows_count = ""
            artifact_type = "metadata"
        rows.append(
            {
                "artifact_type": artifact_type,
                "file": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": rows_count,
                "columns": shape,
            }
        )
    return pd.DataFrame(rows)


def load_existing_output(output: Path, root: Path) -> dict[str, object]:
    metadata_dir = output / "metadata"
    table_dir = output / "tables"
    validation = json.loads((metadata_dir / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((metadata_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("analysis_version") != ANALYSIS_VERSION or validation.get("valid") is not True:
        raise ValueError(f"Bundle v4 yang ada tidak valid: {output}")
    # Recheck the immutable v3 provenance even when the v4 bundle is cached.
    source = load_v3_tables(root)
    current_source_sha = sha256(source["source_csv"])
    current_bundle_snapshot = bundle_snapshot(source["base"])
    if current_source_sha != validation.get("source_sha256_after"):
        raise ValueError("SHA-256 CSV sumber berubah sejak bundle v4 dibuat")
    if current_bundle_snapshot != validation.get("source_bundle_snapshot_after"):
        raise ValueError("Bundle v3 berubah sejak bundle v4 dibuat")
    output_index = pd.read_csv(output / "output_index.csv")
    return {
        "output_dir": output,
        "accuracy": pd.read_csv(table_dir / "accuracy_all_conditions.csv"),
        "accuracy_all_conditions": pd.read_csv(table_dir / "accuracy_all_conditions.csv"),
        "persona_summary": pd.read_csv(table_dir / "religion_persona_summary_by_model_subset_scenario.csv"),
        "persona_summary_by_model": pd.read_csv(table_dir / "religion_persona_summary_by_model.csv"),
        "scenario_summary": pd.read_csv(table_dir / "scenario_summary_by_model_subset.csv"),
        "scenario_summary_by_model": pd.read_csv(table_dir / "scenario_summary_by_model.csv"),
        "status": pd.read_csv(table_dir / "response_status_by_condition.csv"),
        "status_by_model": pd.read_csv(table_dir / "response_status_by_model.csv"),
        "status_by_subset": pd.read_csv(table_dir / "response_status_by_subset.csv"),
        "tokens": pd.read_csv(table_dir / "token_summary_by_condition.csv"),
        "token_summary": pd.read_csv(table_dir / "token_summary_by_condition.csv"),
        "token_summary_by_model_condition": pd.read_csv(table_dir / "token_summary_by_model_condition.csv"),
        "validation": validation,
        "manifest": manifest,
        "output_index": output_index,
    }


def run_analysis(output_dir: Path | str | None = None) -> dict[str, object]:
    root = repo_root()
    output = Path(output_dir).resolve() if output_dir else default_output_dir(root)
    source_bundle = v3_root(root).resolve()
    if output == source_bundle or source_bundle in output.parents:
        raise ValueError("Output v4 tidak boleh berada di dalam atau menimpa bundle v3")
    if output.exists() and any(output.iterdir()):
        return load_existing_output(output, root)

    source = load_v3_tables(root)
    source_snapshot_before = bundle_snapshot(source["base"])
    descriptive_result = aggregate_source_csv(source["source_csv"])
    descriptive = descriptive_result["data"]

    accuracy = add_accuracy_status_metrics(source["accuracy"], descriptive)
    expected_keys = {"model", "persona", "identity", "scenario", "subset"}
    if set(accuracy["model"].unique()) != set(MODEL_ORDER):
        raise ValueError("Model pada tabel akurasi tidak sesuai enam model v3")
    if set(accuracy["persona"].unique()) != set(PERSONA_ORDER):
        raise ValueError("Kondisi persona pada tabel akurasi tidak sesuai 17 kondisi")
    if set(accuracy["subset"].unique()) != set(SUBSET_ORDER):
        raise ValueError("Subset pada tabel akurasi tidak lengkap")
    if not expected_keys.issubset(accuracy.columns):
        raise ValueError("Kolom kunci akurasi tidak lengkap")

    status_condition = accuracy[
        [
            "model",
            "model_label",
            "subset",
            "subset_label",
            "identity",
            "identity_label",
            "persona",
            "persona_label",
            "scenario",
            "total",
            "n_correct_full",
            "n_incorrect_full",
            "accuracy_full",
            "accuracy_full_pct",
            "committed",
            "commit_rate",
            "commit_rate_pct",
            "abstain",
            "abstain_rate",
            "abstain_rate_pct",
            "missing",
            "missing_rate",
            "missing_rate_pct",
            "n_correct_committed",
            "accuracy_committed",
            "accuracy_committed_pct",
            "construct_valid",
            "interpretation_note",
        ]
    ].copy()
    status_model = aggregate_status(status_condition, ["model", "model_label"])
    status_subset = aggregate_status(status_condition, ["subset", "subset_label"])

    tokens = add_token_labels(
        descriptive[
            [
                "model",
                "subset",
                "identity",
                "persona",
                "scenario",
                "total",
                *[f"sum_{column}" for column in TOKEN_COLUMNS],
                *[f"n_valid_{column}" for column in TOKEN_COLUMNS],
                *[f"avg_{column}" for column in TOKEN_COLUMNS],
            ]
        ]
    )
    tokens = tokens[
        [
            "model",
            "model_label",
            "subset",
            "subset_label",
            "identity",
            "identity_label",
            "persona",
            "persona_label",
            "scenario",
            "total",
            *[f"n_valid_{column}" for column in TOKEN_COLUMNS],
            *[f"sum_{column}" for column in TOKEN_COLUMNS],
            *[f"avg_{column}" for column in TOKEN_COLUMNS],
        ]
    ]
    token_model_condition = aggregate_tokens(
        tokens,
        ["model", "model_label", "identity", "identity_label", "persona", "persona_label", "scenario"],
    )
    token_subset = aggregate_tokens(tokens, ["subset", "subset_label"])

    cochran_key = source["cochran"][
        ["model", "subset", "identity", "significant_0_05"]
    ].rename(columns={"significant_0_05": "cochran_q_significant_0_05"})
    scenario_with_q = source["scenario"].merge(
        cochran_key,
        on=["model", "subset", "identity"],
        how="left",
        validate="many_to_one",
    )
    if scenario_with_q["cochran_q_significant_0_05"].isna().any():
        raise ValueError("Sebagian perbandingan skenario tidak memiliki hasil Q Cochran")
    scenario_with_q["supported_q_and_holm"] = (
        scenario_with_q["cochran_q_significant_0_05"] & scenario_with_q["significant_holm_0_05"]
    )
    scenario = canonical_scenario_comparisons(scenario_with_q)
    pairwise = canonical_pairwise_comparisons(source["pairwise"])
    persona_detail = persona_summary(pairwise)
    persona_model = persona_summary_by_model(pairwise)
    scenario_detail = scenario_summary(scenario)
    scenario_model = scenario_summary_by_model(scenario)

    table_dir = output / "tables"
    figure_dir = output / "figures"
    metadata_dir = output / "metadata"
    for directory in (table_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Keep raw v3 inferential tables alongside presentation derivatives.
    write_csv(source["accuracy"], table_dir / "accuracy_by_subset_model_condition.csv")
    write_csv(source["cochran"], table_dir / "cochran_q_by_subset_model_religion.csv")
    write_csv(source["scenario"], table_dir / "mcnemar_scenario_comparisons.csv")
    write_csv(source["pairwise"], table_dir / "mcnemar_religion_pairwise_comparisons.csv")
    write_csv(accuracy, table_dir / "accuracy_all_conditions.csv")
    write_csv(status_condition, table_dir / "response_status_by_condition.csv")
    write_csv(status_model, table_dir / "response_status_by_model.csv")
    write_csv(status_subset, table_dir / "response_status_by_subset.csv")
    write_csv(tokens, table_dir / "token_summary_by_condition.csv")
    write_csv(token_model_condition, table_dir / "token_summary_by_model_condition.csv")
    write_csv(token_subset, table_dir / "token_summary_by_subset.csv")
    write_csv(scenario, table_dir / "mcnemar_scenario_comparisons_with_q_gate.csv")
    write_csv(scenario, table_dir / "mcnemar_scenario_comparisons_normalized.csv")
    write_csv(pairwise, table_dir / "mcnemar_religion_pairwise_comparisons_normalized.csv")
    write_csv(persona_detail, table_dir / "religion_persona_summary_by_model_subset_scenario.csv")
    write_csv(persona_model, table_dir / "religion_persona_summary_by_model.csv")
    write_csv(
        pairwise[pairwise["significant_holm_0_05"]],
        table_dir / "religion_persona_significant_results.csv",
    )
    write_csv(scenario_detail, table_dir / "scenario_summary_by_model_subset.csv")
    write_csv(scenario_model, table_dir / "scenario_summary_by_model.csv")

    configure_plot_style()
    figures: list[Path] = []
    figures += accuracy_heatmap(accuracy, figure_dir)
    figures += scenario_effect_heatmap(
        scenario,
        figure_dir,
        "implicit_vs_explicit",
        "02_religion_implicit_vs_explicit_effect_q_holm",
        "Perbandingan Skenario Implisit dan Eksplisit pada ETHICS Agama",
    )
    figures += scenario_effect_heatmap(
        scenario,
        figure_dir,
        "persona_vs_neutral",
        "03_religion_persona_vs_neutral_effect_q_holm",
        "Perbandingan Persona Agama dan Netral pada ETHICS",
    )
    for scenario_name in ["E", "I1", "I2", "I3"]:
        figures += pairwise_effect_heatmap(pairwise, figure_dir, scenario_name)
    figures += status_figure(status_model, figure_dir)
    figures += token_figure(token_model_condition, figure_dir)
    figure_validation = validate_figure_files(figures)

    source_sha_after = sha256(source["source_csv"])
    source_snapshot_after = bundle_snapshot(source["base"])
    subset_question_counts = {
        subset: len(descriptive_result["question_ids"].get(subset, set())) for subset in SUBSET_ORDER
    }
    scenario_family_sizes = sorted(source["scenario"]["holm_family_size"].astype(int).unique().tolist())
    pairwise_family_sizes = sorted(source["pairwise"]["holm_family_size"].astype(int).unique().tolist())
    checks = {
        "rows_processed_exact": descriptive_result["row_count"] == 154_428,
        "question_count_exact": len(set().union(*descriptive_result["question_ids"].values())) == 1_514,
        "model_count_exact": accuracy["model"].nunique() == 6,
        "condition_count_exact": accuracy["persona"].nunique() == 17,
        "subset_question_counts_exact": subset_question_counts == EXPECTED_SUBSET_QUESTION_COUNTS,
        "cochran_q_rows_exact": len(source["cochran"]) == 96,
        "scenario_pair_rows_exact": len(source["scenario"]) == 672,
        "religion_pairwise_rows_exact": len(source["pairwise"]) == 576,
        "scenario_holm_family_size_exactly_7": scenario_family_sizes == [7],
        "religion_pairwise_holm_family_size_exactly_24": pairwise_family_sizes == [24],
        "duplicate_response_keys_zero": descriptive_result["duplicate_response_keys"] == 0,
        "source_v3_validation_valid": source["validation"].get("valid") is True,
        "source_sha256_unchanged": source_sha_before_equals_expected(
            source["source_sha_before"], source_sha_after, source["validation"].get("input_sha256_before")
        ),
        "source_v3_bundle_unchanged": source_snapshot_before == source_snapshot_after,
        "source_v3_not_overwritten": output != source["base"],
        "figure_files_valid": bool(figure_validation["all_figure_files_nonempty"]),
    }
    validation = {
        "valid": bool(all(checks.values())),
        "analysis_version": ANALYSIS_VERSION,
        "source_analysis_version": source["manifest"]["analysis_version"],
        "rows_processed": descriptive_result["row_count"],
        "question_count": len(set().union(*descriptive_result["question_ids"].values())),
        "model_count": int(accuracy["model"].nunique()),
        "condition_count": int(accuracy["persona"].nunique()),
        "models": MODEL_ORDER,
        "conditions": CONDITION_ORDER,
        "subset_question_counts": subset_question_counts,
        "cochran_q_rows": int(len(source["cochran"])),
        "scenario_pair_rows": int(len(source["scenario"])),
        "religion_pairwise_rows": int(len(source["pairwise"])),
        "scenario_holm_family_sizes": scenario_family_sizes,
        "religion_pairwise_holm_family_sizes": pairwise_family_sizes,
        "duplicate_response_keys": int(descriptive_result["duplicate_response_keys"]),
        "status_values": descriptive_result["status_values"],
        "source_v3_validation_valid": bool(source["validation"].get("valid")),
        "source_sha256_expected_v3": source["validation"].get("input_sha256_before"),
        "source_sha256_before": source["source_sha_before"],
        "source_sha256_after": source_sha_after,
        "source_bundle_snapshot_before": source_snapshot_before,
        "source_bundle_snapshot_after": source_snapshot_after,
        "source_csv_unchanged": source_sha_before_equals_expected(
            source["source_sha_before"], source_sha_after, source["validation"].get("input_sha256_before")
        ),
        "source_v3_bundle_unchanged": source_snapshot_before == source_snapshot_after,
        "source_v3_not_overwritten": output != source["base"],
        "full_denominator_accuracy_from_v3": True,
        "abstain_missing_counted_in_denominator_as_incorrect": bool(
            source["manifest"].get("nonanswer_scoring") == "abstain dan missing bernilai salah"
        ),
        "committed_accuracy_descriptive_only": True,
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "api_calls": 0,
        "combined_four_subset_accuracy_used": False,
        "token_valid_counts": {
            column: int(descriptive[f"n_valid_{column}"].sum()) for column in TOKEN_COLUMNS
        },
        **figure_validation,
        "checks": checks,
    }
    if not validation["valid"]:
        write_json(validation, metadata_dir / "validation.json")
        raise ValueError(f"Validation v4 gagal: {validation}")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "source_bundle": str(source["base"]),
        "source_manifest": str(source["manifest_path"]),
        "source_validation": str(source["validation_path"]),
        "source_csv": str(source["source_csv"]),
        "source_analysis_version": source["manifest"]["analysis_version"],
        "output": str(output),
        "purpose": "Penyajian akurasi ETHICS agama dengan urutan hasil utama antarpersona, lalu I/E/N.",
        "inference_source_tables": [
            "mcnemar_religion_pairwise_comparisons.csv",
            "mcnemar_scenario_comparisons.csv",
            "cochran_q_by_subset_model_religion.csv",
            "accuracy_by_subset_model_condition.csv",
        ],
        "descriptive_source_csv": str(source["source_csv"]),
        "statistics_recomputed": False,
        "response_rescoring": False,
        "llm_inference_rerun": False,
        "api_calls": 0,
        "combined_four_subset_accuracy_used": False,
        "subsets_retained_separately": SUBSET_ORDER,
        "identity_order_for_figures": IDENTITY_ORDER,
        "condition_order_for_figures": CONDITION_ORDER,
        "validation": validation,
    }
    write_json(manifest, metadata_dir / "analysis_manifest.json")
    write_json(validation, metadata_dir / "validation.json")
    output_index = artifact_index(output)
    write_csv(output_index, output / "output_index.csv")
    return {
        "output_dir": output,
        "accuracy": accuracy,
        "accuracy_all_conditions": accuracy,
        "persona_summary": persona_detail,
        "persona_summary_by_model": persona_model,
        "scenario_summary": scenario_detail,
        "scenario_summary_by_model": scenario_model,
        "status": status_condition,
        "status_by_model": status_model,
        "status_by_subset": status_subset,
        "tokens": tokens,
        "token_summary": tokens,
        "token_summary_by_model_condition": token_model_condition,
        "validation": validation,
        "manifest": manifest,
        "output_index": output_index,
    }


def source_sha_before_equals_expected(before: str, after: str, expected: str | None) -> bool:
    return before == after and (expected is None or before == expected)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(path: Path) -> Path:
    """Write the requested source notebook, including the refreshed run gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
# ETHICS Religion Persona — Analisis Akurasi v4 (Persona-First)

Notebook ini adalah lapisan penyajian di atas bundle v3 agama yang tervalidasi.
Inferensi tidak dihitung ulang. Akurasi penuh dan seluruh nilai uji statistik
dibaca dari tabel v3; status respons dan token dihitung sebagai ringkasan
deskriptif dari CSV sumber yang dirujuk manifest v3. Empat subset—Commonsense,
Deontology, Justice, dan Virtue—selalu dipertahankan terpisah.
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
REPO_ROOT = next(path for path in [start, *start.parents] if (path / "AGENTS.md").exists() and (path / "src").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.ethics.build_religion_publication_notebook_v4_accuracy_persona_first import run_analysis

result = run_analysis()
OUTPUT_DIR = Path(result["output_dir"])
FIGURE_DIR = OUTPUT_DIR / "figures"
display(pd.DataFrame([result["validation"]]).T.rename(columns={0: "value"}))
assert result["validation"]["valid"]
assert result["validation"]["api_calls"] == 0
assert result["validation"]["statistics_recomputed"] is False
"""
        ),
        md(
            """
## 1. Akurasi N, E, I1, I2, dan I3 untuk setiap subset

Tabel berikut memuat seluruh 408 kombinasi model × kondisi × subset. `accuracy_full`
berasal dari tabel v3 dengan seluruh respons sebagai penyebut. `accuracy_committed`
ditampilkan sebagai metrik tambahan dan tidak menggantikan akurasi penuh.
"""
        ),
        code(
            """
pd.set_option("display.max_rows", 1000)
display(result["accuracy_all_conditions"])
"""
        ),
        md("## 2. Perbandingan antarpersona agama — hasil utama"),
        code(
            """
display(result["persona_summary_by_model"])
display(result["persona_summary"])
display(Image(filename=str(FIGURE_DIR / "04_religion_pairwise_e_effect_holm.png"), width=1500))
"""
        ),
        md(
            """
Urutan orientasi pasangan pada tabel dan grafik adalah Islam, Kristen, Yahudi,
lalu Ateis. Jika orientasi sumber v3 dibalik, builder hanya membalik delta,
Cohen's h, dan arah discordant pair; nilai-p tidak diubah.
"""
        ),
        md("## 3. Perbandingan I1–E, I2–E, dan I3–E"),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'implicit_vs_explicit'"))
display(Image(filename=str(FIGURE_DIR / "02_religion_implicit_vs_explicit_effect_q_holm.png"), width=1500))
"""
        ),
        md("## 4. Perbandingan E–N, I1–N, I2–N, dan I3–N"),
        code(
            """
display(result["scenario_summary"].query("comparison_category == 'persona_vs_neutral'"))
display(Image(filename=str(FIGURE_DIR / "03_religion_persona_vs_neutral_effect_q_holm.png"), width=1500))
"""
        ),
        md(
            """
## 5. Akurasi penuh, committed, commit rate, abstain, dan missing

Status dihitung per model × kondisi × subset. `abstain` dan `missing` tetap
berada di dalam denominator akurasi penuh dan bernilai tidak benar sesuai
manifest v3. Perbedaan tingkat komitmen tidak boleh dibaca sebagai perbedaan
akurasi tanpa melihat kolom status pada tabel yang sama.
"""
        ),
        code(
            """
display(result["status"])
display(result["status_by_model"])
display(result["status_by_subset"])
display(Image(filename=str(FIGURE_DIR / "05_religion_response_status_by_model.png"), width=1200))
"""
        ),
        md(
            """
## 6. Input token, completion token, reasoning token, dan total token

Jumlah valid dan rata-rata token ditampilkan tanpa imputasi. `n_valid_*` dapat
lebih kecil daripada `total`; metadata token yang hilang tetap hilang.
"""
        ),
        code(
            """
display(result["tokens"])
display(result["token_summary_by_model_condition"])
display(Image(filename=str(FIGURE_DIR / "06_religion_total_tokens_by_condition.png"), width=1500))
"""
        ),
        md(
            """
## 7. Batas interpretasi subset

`Commonsense` memakai label 1 untuk tindakan salah, sedangkan `Justice` memakai
label 1 untuk pernyataan wajar. Pada `Deontology`, angka yang tampil adalah
kecocokan label per butir karena informasi penentu *excuse* tidak dikirim kepada
model. Pada `Virtue`, angka juga hanya kecocokan label per butir karena bentuk
pertanyaan tidak sepenuhnya cocok dengan konstruk label. Karena itu tidak ada
satu angka akurasi gabungan empat subset sebagai hasil inferensial.
"""
        ),
        md("## 8. Indeks artefak dan kontrak reproduksi"),
        code(
            """
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


def execute_notebook(source: Path, output: Path) -> Path:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute the refreshed notebook")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = repo_root()
    result = run_analysis(args.output_dir)
    print(f"[ANALYSIS] Output: {result['output_dir']}")
    print(f"[VALIDATION] valid={result['validation']['valid']}")
    source = build_notebook(source_notebook_path(root))
    print(f"[NOTEBOOK] Source: {source}")
    if args.execute:
        executed = execute_notebook(source, executed_notebook_path(root))
        print(f"[NOTEBOOK] Executed: {executed}")


if __name__ == "__main__":
    main()

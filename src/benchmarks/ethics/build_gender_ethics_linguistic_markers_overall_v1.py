"""Build an additive overall-ETHICS linguistic-marker summary.

This module deliberately aggregates only visible linguistic behaviour across
the four ETHICS subsets.  It does not create a cross-subset accuracy score.
The subset identity is retained in the source data and the existing subset
bundle remains untouched for appendix-level reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.linguistic_markers.config import (
    IDENTITY_CONDITIONS,
    MAIN_MODELS,
    PLANNED_CONTRASTS,
)
from src.core.linguistic_markers.statistics import (
    _bootstrap_ci,
    _paired_numeric_test,
    _stable_seed,
    holm_adjust,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MARKER_ROOT = (
    REPO_ROOT
    / "result/ETHICS/reports/Gender_Persona"
    / "linguistic_markers_gender_candidate_20260821_v1_6_0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "result/ETHICS/reports/Gender_Persona/linguistic_markers/analysis"
    / "gender_ethics_linguistic_markers_overall_candidate_r2_20260905"
)

SUBSETS = ("commonsense", "deontology", "justice", "virtue")
EXPECTED_ROWS = 81_756
EXPECTED_VISIBLE = 80_840
BOOTSTRAP_RESAMPLES = 2_000

SURFACE_OUTCOMES = (
    "word_count",
    "avg_sentence_length",
    "ttr_300",
)
PRIMARY_MARKERS = (
    "organization_explicit_step",
    "organization_bullet",
    "organization_section_header",
    "stance_hedge",
    "stance_booster",
    "stance_assumption",
    "stance_self_correction",
    "connective_causal",
    "connective_contrastive",
    "connective_conditional",
    "self_reference_first_person",
    "self_reference_inclusive_we",
    "engagement_reader_directive",
    "stance_contextual_hedge",
    "stance_distancing",
    "connective_additive",
    "connective_concessive",
    "affect_empathy_validation",
)

MODEL_LABELS = {
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-root", type=Path, default=MARKER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    return parser.parse_args()


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.casefold().eq("true")


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _load_inputs(marker_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    marker_root = marker_root.resolve()
    response_path = marker_root / "response_level_markers.csv.gz"
    registry_path = marker_root / "marker_registry.csv"
    manifest_path = marker_root / "manifest.json"
    frame = pd.read_csv(response_path, compression="gzip", low_memory=False)
    registry = pd.read_csv(registry_path)
    marker_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {
        "model_id",
        "persona",
        "question_id",
        "subset",
        "answer_status",
        "text_available",
        "text_status",
        *SURFACE_OUTCOMES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required response columns: {missing}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, got {len(frame):,}")
    if set(frame["subset"].dropna().unique()) != set(SUBSETS):
        raise ValueError("The four ETHICS subsets are not all present.")
    if int(_as_bool(frame["text_available"]).sum()) != EXPECTED_VISIBLE:
        raise ValueError("Visible-text count does not match the active v1.6 bundle.")
    if frame.duplicated(["model_id", "persona", "subset", "question_id"]).any():
        raise ValueError("Response-level input has duplicate model/persona/question keys.")

    frame = frame.copy()
    frame["text_available"] = _as_bool(frame["text_available"])
    frame["question_id"] = frame["question_id"].astype(str)
    for column in SURFACE_OUTCOMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    marker_ids = registry["marker_id"].astype(str).tolist()
    if len(marker_ids) != 41 or len(set(marker_ids)) != 41:
        raise ValueError("The active registry must contain exactly 41 unique markers.")
    for marker_id in marker_ids:
        for suffix in ("__count", "__density", "__present"):
            column = f"{marker_id}{suffix}"
            if column not in frame.columns:
                raise ValueError(f"Missing marker column: {column}")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not set(PRIMARY_MARKERS).issubset(marker_ids):
        raise ValueError("Primary marker list is not contained in the active registry.")
    return frame, registry, marker_manifest


def _build_status(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "persona"]
    total = frame.groupby(keys, as_index=False).size().rename(columns={"size": "total"})
    counts = (
        frame.groupby(keys + ["answer_status"], as_index=False)
        .size()
        .pivot(index=keys, columns="answer_status", values="size")
        .fillna(0)
        .reset_index()
    )
    for status in ("committed", "abstain", "missing"):
        if status not in counts:
            counts[status] = 0
    available = (
        frame.groupby(keys, as_index=False)["text_available"]
        .sum()
        .rename(columns={"text_available": "text_available"})
    )
    technical = (
        frame.assign(technical_only=frame["text_status"].eq("technical_only"))
        .groupby(keys, as_index=False)["technical_only"]
        .sum()
    )
    result = total.merge(counts, on=keys).merge(available, on=keys).merge(technical, on=keys)
    result["commit_rate"] = _safe_rate(result["committed"], result["total"])
    result["abstain_rate"] = _safe_rate(result["abstain"], result["total"])
    result["missing_rate"] = _safe_rate(result["missing"], result["total"])
    result["text_available_rate"] = _safe_rate(result["text_available"], result["total"])
    result["model"] = result["model_id"].map(MODEL_LABELS)
    columns = [
        "model_id", "model", "persona", "total", "committed", "commit_rate",
        "abstain", "abstain_rate", "missing", "missing_rate", "text_available",
        "text_available_rate", "technical_only",
    ]
    return result[columns].sort_values(["model_id", "persona"]).reset_index(drop=True)


def _build_marker_profile(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    group_keys: list[str],
) -> pd.DataFrame:
    available = frame[frame["text_available"]].copy()
    metadata = registry.set_index("marker_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for group_values, group in available.groupby(group_keys, sort=True, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_data = dict(zip(group_keys, group_values))
        n_available = len(group)
        for marker_id, marker_meta in metadata.items():
            counts = pd.to_numeric(group[f"{marker_id}__count"], errors="coerce").fillna(0.0)
            densities = pd.to_numeric(group[f"{marker_id}__density"], errors="coerce")
            present = pd.to_numeric(group[f"{marker_id}__present"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    **group_data,
                    "marker_id": marker_id,
                    "marker_family": marker_meta["family"],
                    "analysis_role": marker_meta["analysis_role"],
                    "measurement": marker_meta["measurement"],
                    "validation_status": marker_meta["validation_status"],
                    "n_available": n_available,
                    "responses_present": int((present > 0).sum()),
                    "response_prevalence_pct": float((present > 0).mean() * 100)
                    if n_available
                    else math.nan,
                    "total_occurrences": int(counts.sum()),
                    "mean_density_per_100_words": float(densities.mean())
                    if n_available
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_surface_profile(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"]].copy()
    rows: list[dict[str, object]] = []
    for (model_id, persona), group in available.groupby(["model_id", "persona"], sort=True):
        for outcome_id in SURFACE_OUTCOMES:
            values = pd.to_numeric(group[outcome_id], errors="coerce").dropna()
            if len(values):
                q1, median, q3 = np.quantile(values.to_numpy(float), [0.25, 0.5, 0.75])
                rows.append(
                    {
                        "model_id": model_id,
                        "model": MODEL_LABELS.get(model_id, model_id),
                        "persona": persona,
                        "outcome_id": outcome_id,
                        "n_available": len(values),
                        "mean": float(values.mean()),
                        "median": float(median),
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(q3 - q1),
                    }
                )
    return pd.DataFrame(rows)


def _outcome_specs() -> list[dict[str, str]]:
    specs = [
        {"outcome_id": outcome, "value_col": outcome, "raw_col": ""}
        for outcome in SURFACE_OUTCOMES
    ]
    specs.extend(
        {
            "outcome_id": marker,
            "value_col": f"{marker}__density",
            "raw_col": f"{marker}__count",
        }
        for marker in PRIMARY_MARKERS
    )
    return specs


def _paired_row(
    *,
    group: pd.DataFrame,
    left_persona: str,
    right_persona: str,
    outcome: dict[str, str],
    model_id: str,
    pair_filter: str,
    analysis_family: str,
    identity: str,
    contrast_id: str,
    left_condition: str,
    right_condition: str,
    bootstrap_resamples: int,
) -> dict[str, object]:
    key = ["subset", "question_id"]
    value_col = outcome["value_col"]
    left_columns = key + [value_col]
    right_columns = key + [value_col]
    left = group.loc[group["persona"].eq(left_persona), left_columns].copy()
    right = group.loc[group["persona"].eq(right_persona), right_columns].copy()
    left[value_col] = pd.to_numeric(left[value_col], errors="coerce")
    right[value_col] = pd.to_numeric(right[value_col], errors="coerce")
    paired = left.merge(right, on=key, how="inner", suffixes=("_left", "_right"))
    paired = paired.dropna(subset=[f"{value_col}_left", f"{value_col}_right"])
    left_values = paired[f"{value_col}_left"].to_numpy(float)
    right_values = paired[f"{value_col}_right"].to_numpy(float)
    deltas = left_values - right_values
    test = _paired_numeric_test(deltas)
    ci_low, ci_high = _bootstrap_ci(
        deltas,
        "median",
        bootstrap_resamples,
        _stable_seed(
            "ethics-overall",
            model_id,
            pair_filter,
            analysis_family,
            identity,
            outcome["outcome_id"],
            contrast_id,
        ),
    )
    row = {
        "result_status": "candidate",
        "analysis_scope": "overall_ethics",
        "analysis_family": analysis_family,
        "pair_filter": pair_filter,
        "model_id": model_id,
        "model": MODEL_LABELS.get(model_id, model_id),
        "benchmark": "ethics",
        "subset": "overall_ethics",
        "source_subsets": ";".join(SUBSETS),
        "identity": identity,
        "outcome_id": outcome["outcome_id"],
        "contrast_id": contrast_id,
        "left_condition": left_condition,
        "right_condition": right_condition,
        "left_persona": left_persona,
        "right_persona": right_persona,
        "median_left": float(np.median(left_values)) if len(left_values) else math.nan,
        "median_right": float(np.median(right_values)) if len(right_values) else math.nan,
        "median_paired_delta": float(np.median(deltas)) if len(deltas) else math.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        **test,
    }
    if outcome["raw_col"]:
        raw_col = outcome["raw_col"]
        raw_left = group.loc[group["persona"].eq(left_persona), key + [raw_col]].copy()
        raw_right = group.loc[group["persona"].eq(right_persona), key + [raw_col]].copy()
        raw_left[raw_col] = pd.to_numeric(raw_left[raw_col], errors="coerce")
        raw_right[raw_col] = pd.to_numeric(raw_right[raw_col], errors="coerce")
        raw_pair = raw_left.merge(raw_right, on=key, how="inner", suffixes=("_left", "_right"))
        raw_pair = raw_pair.dropna(subset=[f"{raw_col}_left", f"{raw_col}_right"])
        raw_deltas = raw_pair[f"{raw_col}_left"].to_numpy(float) - raw_pair[f"{raw_col}_right"].to_numpy(float)
        row.update(
            {
                "n_raw_count_pairs": len(raw_pair),
                "raw_count_sum_left": float(raw_pair[f"{raw_col}_left"].sum()),
                "raw_count_sum_right": float(raw_pair[f"{raw_col}_right"].sum()),
                "median_raw_count_delta": float(np.median(raw_deltas)) if len(raw_deltas) else math.nan,
            }
        )
    else:
        row.update(
            {
                "n_raw_count_pairs": math.nan,
                "raw_count_sum_left": math.nan,
                "raw_count_sum_right": math.nan,
                "median_raw_count_delta": math.nan,
            }
        )
    return row


def _add_holm(result: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    if result.empty:
        result["p_holm"] = pd.Series(dtype=float)
        return result
    result = result.copy()
    result["p_holm"] = result.groupby(family_columns, dropna=False)["p_raw"].transform(
        lambda values: holm_adjust(values.to_numpy(float))
    )
    return result


def _build_paired(
    frame: pd.DataFrame,
    *,
    pair_filter: str,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    if pair_filter == "all_visible_text":
        selected = frame[frame["text_available"]].copy()
    elif pair_filter == "both_committed":
        selected = frame[frame["text_available"] & frame["answer_status"].eq("committed")].copy()
    else:
        raise ValueError(f"Unknown pair filter: {pair_filter}")

    rows: list[dict[str, object]] = []
    for model_id, model_group in selected.groupby("model_id", sort=True):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for outcome in _outcome_specs():
                for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                    rows.append(
                        _paired_row(
                            group=model_group,
                            left_persona=conditions[left_condition],
                            right_persona=conditions[right_condition],
                            outcome=outcome,
                            model_id=model_id,
                            pair_filter=pair_filter,
                            analysis_family="within_identity",
                            identity=identity,
                            contrast_id=contrast_id,
                            left_condition=left_condition,
                            right_condition=right_condition,
                            bootstrap_resamples=bootstrap_resamples,
                        )
                    )
        for outcome in _outcome_specs():
            for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3"):
                rows.append(
                    _paired_row(
                        group=model_group,
                        left_persona=IDENTITY_CONDITIONS["man"][condition],
                        right_persona=IDENTITY_CONDITIONS["woman"][condition],
                        outcome=outcome,
                        model_id=model_id,
                        pair_filter=pair_filter,
                        analysis_family="direct_gender",
                        identity="direct_gender",
                        contrast_id=f"man_minus_woman_{condition}",
                        left_condition=f"man_{condition}",
                        right_condition=f"woman_{condition}",
                        bootstrap_resamples=bootstrap_resamples,
                    )
                )
    result = pd.DataFrame(rows)
    expected_rows = (
        selected["model_id"].nunique()
        * (2 * len(PLANNED_CONTRASTS) + 4)
        * len(_outcome_specs())
    )
    if len(result) != expected_rows:
        raise AssertionError(
            f"Unexpected overall paired row count: {len(result)} != {expected_rows}"
        )
    result = _add_holm(
        result,
        ["analysis_family", "pair_filter", "model_id", "identity", "outcome_id"],
    )
    return result


def _write_readme(output_dir: Path, manifest: dict) -> None:
    text = f"""# Overall ETHICS linguistic markers — candidate

Bundle ini adalah tambahan additive untuk tampilan **overall ETHICS**.
Hasil lama per-subset tidak dihapus dan tetap menjadi bahan lampiran.

## Scope

- Sumber: `{manifest['source_response_level_markers']}`
- Subset yang diringkas: `{', '.join(SUBSETS)}`
- Total response records: `{manifest['total_responses']:,}`
- Visible-text records: `{manifest['visible_text_records']:,}`
- Primary outcomes: 3 surface outcomes + 18 primary markers
- Registered marker profile: 41 markers
- Status: `candidate`

Overall aggregation hanya berlaku untuk linguistic marker dan response behavior.
Tidak ada satu angka accuracy ETHICS gabungan karena polaritas dan konstruk
empat subset berbeda. Accuracy tetap harus dilaporkan per subset.

Analisis utama menggunakan seluruh visible text yang tersedia, termasuk
`abstain`. Sensitivity menggunakan pasangan yang kedua responsnya committed.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build_bundle(
    marker_root: Path,
    output_dir: Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading response-level marker input...")
    frame, registry, marker_manifest = _load_inputs(marker_root)
    print(f"Loaded {len(frame):,} response records; {int(frame['text_available'].sum()):,} visible-text records.")

    status = _build_status(frame)
    status.to_csv(output_dir / "overall_response_status.csv", index=False, encoding="utf-8")

    print("Building overall marker profiles...")
    marker_profile = _build_marker_profile(frame, registry, group_keys=["model_id", "persona"])
    marker_profile.insert(0, "result_status", "candidate")
    marker_profile.to_csv(output_dir / "overall_marker_profile_41.csv", index=False, encoding="utf-8")
    marker_profile[
        marker_profile["marker_id"].isin(PRIMARY_MARKERS)
    ].to_csv(output_dir / "overall_primary_marker_profile_18.csv", index=False, encoding="utf-8")
    surface_profile = _build_surface_profile(frame)
    surface_profile.insert(0, "result_status", "candidate")
    surface_profile.to_csv(output_dir / "overall_surface_profile_3.csv", index=False, encoding="utf-8")

    print("Building overall paired effects...")
    paired_frames = []
    for pair_filter in ("all_visible_text", "both_committed"):
        print(f"  Pair filter: {pair_filter}")
        paired_frames.append(
            _build_paired(
                frame,
                pair_filter=pair_filter,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output_dir / "overall_paired_numeric_21.csv", index=False, encoding="utf-8")

    source_response_path = marker_root.resolve() / "response_level_markers.csv.gz"
    source_sha256 = hashlib.sha256(source_response_path.read_bytes()).hexdigest()
    manifest = {
        "bundle_version": "gender-ethics-linguistic-markers-overall-v1.0.0-candidate",
        "result_status": "candidate",
        "analysis_scope": "overall_ethics",
        "source_marker_manifest": str((marker_root / "manifest.json").resolve()),
        "source_response_level_markers": str(source_response_path),
        "source_response_level_markers_sha256": source_sha256,
        "source_input_sha256": marker_manifest.get(
            "input_sha256", marker_manifest.get("source_input_sha256")
        ),
        "total_responses": int(len(frame)),
        "visible_text_records": int(frame["text_available"].sum()),
        "subsets": list(SUBSETS),
        "questions": int(frame["question_id"].nunique()),
        "models": int(frame["model_id"].nunique()),
        "personas": int(frame["persona"].nunique()),
        "primary_surface_outcomes": list(SURFACE_OUTCOMES),
        "primary_marker_outcomes": list(PRIMARY_MARKERS),
        "registered_marker_count": int(len(registry)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "hidden_reasoning_used": False,
        "ground_truth_used_for_extraction": False,
        "accuracy_aggregated_across_subsets": False,
        "generated_files": [
            "README.md",
            "manifest.json",
            "overall_response_status.csv",
            "overall_marker_profile_41.csv",
            "overall_primary_marker_profile_18.csv",
            "overall_surface_profile_3.csv",
            "overall_paired_numeric_21.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_readme(output_dir, manifest)
    print(f"Overall bundle written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    args = parse_args()
    build_bundle(args.marker_root, args.output_dir, args.bootstrap_resamples)

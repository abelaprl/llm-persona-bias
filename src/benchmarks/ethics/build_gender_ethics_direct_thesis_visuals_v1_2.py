"""Build thesis-facing direct Man-Woman visuals for ETHICS Gender Persona.

The builder is intentionally write-once.  It reuses the validated visible-text
surface from the linguistic-marker pipeline, never reads hidden reasoning, and
keeps E/I1/I2/I3 and the four ETHICS subsets separate.  Marker inference is not
performed while the active registry remains a candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.linguistic_markers.gender_registry import (
    GENDER_TERM_CATEGORIES,
    compile_gender_term_pattern,
)
from src.core.linguistic_markers.surface import prepare_text_surface


BUNDLE_VERSION = "gender-ethics-direct-thesis-visuals-v1.2.0"
EXPECTED_PIPELINE_VERSION = "ethics-gender-lm-v1.6.0-candidate"
EXPECTED_INPUT_SHA256 = "9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4"

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
MODEL_SHORT = {
    "meta-llama_llama-3.1-8b-instruct": "Llama\n3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama\n3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral\n8B",
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek\nR1 32B",
    "openai_gpt-oss-20b": "GPT-OSS\n20B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next\n80B",
}
MODEL_COLORS = dict(
    zip(MODELS, ("#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"))
)
CONDITIONS = ("explicit", "implicit_1", "implicit_2", "implicit_3")
CONDITION_LABELS = {
    "explicit": "E",
    "implicit_1": "I1",
    "implicit_2": "I2",
    "implicit_3": "I3",
}
SUBSETS = ("commonsense", "deontology", "justice", "virtue")
SUBSET_LABELS = {
    "commonsense": "Commonsense",
    "deontology": "Deontology",
    "justice": "Justice",
    "virtue": "Virtue",
}
PAIR_KEYS = ("model_id", "subset", "question_id")

SURFACE_OUTCOMES = ("word_count", "avg_sentence_length", "ttr_300")
SURFACE_LABELS = {
    "word_count": "Jumlah kata",
    "avg_sentence_length": "Rerata kata per kalimat",
    "ttr_300": "TTR-300",
}
SURFACE_UNITS = {
    "word_count": "median selisih kata",
    "avg_sentence_length": "median selisih kata per kalimat",
    "ttr_300": "median selisih rasio",
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

GENDER_MARKERS = (
    "gender_persona_direct_address",
    "gender_persona_acknowledgment",
    "gender_second_person_identity_reference",
    "gender_hypothetical_identity_reference",
    "gender_first_person_identity_reference",
    "gender_perspective_adoption",
    "gender_identity_relevance_denial",
    "gender_identity_neutralization",
    "gender_identity_discourse",
    "gender_role_framing",
    "gendered_normative_evaluation",
)
GENDER_MARKER_LABELS = {
    "gender_persona_direct_address": "Sapaan persona langsung",
    "gender_persona_acknowledgment": "Acknowledgment persona",
    "gender_second_person_identity_reference": "Identitas orang kedua",
    "gender_hypothetical_identity_reference": "Identitas hipotetis",
    "gender_first_person_identity_reference": "Identitas orang pertama",
    "gender_perspective_adoption": "Adopsi perspektif",
    "gender_identity_relevance_denial": "Gender dinyatakan tak relevan",
    "gender_identity_neutralization": "Netralisasi identitas",
    "gender_identity_discourse": "Diskursus identitas",
    "gender_role_framing": "Framing peran gender",
    "gendered_normative_evaluation": "Evaluasi normatif gender",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate and add a manifest to an interrupted bundle without replacing files.",
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
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.titlesize": 16,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, image_dir: Path, filename: str) -> Path:
    png = image_dir / filename
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if png.stat().st_size < 10_000 or svg.stat().st_size < 2_000:
        raise ValueError(f"Rendered figure is unexpectedly small: {png}")
    return png


def _model_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODELS
    ] + [Line2D([0], [0], marker="D", linestyle="", color="#111827", label="Median 6 model")]


def _scope_groups(frame: pd.DataFrame):
    yield "all", frame
    for subset in SUBSETS:
        yield subset, frame[frame["subset"].eq(subset)]


def _rank_biserial(difference: np.ndarray) -> tuple[int, int, float]:
    finite = difference[np.isfinite(difference)]
    if not len(finite):
        return 0, 0, math.nan
    nonzero = finite[~np.isclose(finite, 0.0, rtol=0.0, atol=1e-12)]
    if not len(nonzero):
        return len(finite), 0, 0.0
    effect = (np.count_nonzero(nonzero > 0) - np.count_nonzero(nonzero < 0)) / len(nonzero)
    return len(finite), len(nonzero), float(effect)


def _load_sources(marker_dir: Path, canonical_csv: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((marker_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError(f"Unexpected marker pipeline: {manifest.get('pipeline_version')}")
    if manifest.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("Marker manifest does not point to the validated Gender source.")
    if _sha256(canonical_csv) != EXPECTED_INPUT_SHA256:
        raise ValueError("Canonical CSV SHA-256 changed.")
    if manifest.get("inference_run") is not False:
        raise ValueError("Marker inference must remain disabled for the candidate registry.")

    marker_columns = [f"{marker}__density" for marker in GENERAL_MARKERS]
    marker_columns += [f"{marker}__present" for marker in GENDER_MARKERS]
    usecols = [
        "model_id",
        "persona",
        "condition",
        "question_id",
        "subset",
        "answer_status",
        "text_status",
        "text_available",
        "word_count",
        "avg_sentence_length",
        "ttr_300",
        "opening_category",
        *marker_columns,
    ]
    frame = pd.read_csv(marker_dir / "response_level_markers.csv.gz", usecols=usecols, low_memory=False)
    frame["identity"] = frame["persona"].str.extract(r"^(man|woman)_", expand=False)
    frame.loc[frame["persona"].eq("neutral"), "identity"] = "neutral"
    frame["text_available"] = _bool_mask(frame["text_available"])
    for column in [*SURFACE_OUTCOMES, *marker_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    canonical = pd.read_csv(
        canonical_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "response_content_full",
            "is_correct",
            "construct_valid",
            "extracted_stance",
        ],
        keep_default_na=False,
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    canonical["is_correct"] = _nullable_bool(canonical["is_correct"])
    canonical["construct_valid"] = _nullable_bool(canonical["construct_valid"])
    key = ["model_id", "persona", "question_id"]
    if frame.duplicated(key).any() or canonical.duplicated(key).any():
        raise ValueError("Duplicate model/persona/question key found.")
    frame = frame.merge(canonical, on=key, how="left", validate="one_to_one")
    if len(frame) != manifest["rows_input"] or frame["response_content_full"].isna().any():
        raise ValueError("Merged response rows do not match the marker manifest.")
    if set(frame["model_id"].unique()) != set(MODELS):
        raise ValueError("Unexpected model set.")
    if set(frame["subset"].unique()) != set(SUBSETS):
        raise ValueError("Unexpected ETHICS subset set.")
    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int32)

    scenario_lookup = pd.read_csv(marker_dir / "scenario_lookup.csv", dtype={"question_id": str})
    lexicon_coverage = pd.read_csv(marker_dir / "gender_lexicon_coverage.csv")
    return frame, manifest, scenario_lookup, lexicon_coverage


def _build_term_hits(
    frame: pd.DataFrame,
    scenario_lookup: pd.DataFrame,
    lexicon_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    term_pattern = compile_gender_term_pattern()
    terms = tuple(GENDER_TERM_CATEGORIES)
    term_index = {term: index for index, term in enumerate(terms)}
    scenario_map = scenario_lookup.set_index("question_id")["scenario"].fillna("").astype(str).to_dict()
    scenario_terms = {
        question_id: {match.group(0).casefold() for match in term_pattern.finditer(text)}
        for question_id, text in scenario_map.items()
    }

    response_only_presence = np.zeros((len(frame), len(terms)), dtype=np.uint8)
    response_only_occurrences = np.zeros((len(frame), len(terms)), dtype=np.uint16)
    rows: list[dict[str, object]] = []
    availability_mismatches = 0
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        surface = prepare_text_surface(row.response_content_full, benchmark="ethics")
        if bool(surface.text_available) != bool(row.text_available):
            availability_mismatches += 1
        if not surface.text_available:
            continue
        counts = Counter(match.group(0).casefold() for match in term_pattern.finditer(surface.visible_text))
        in_scenario = scenario_terms.get(str(row.question_id), set())
        for term, count in counts.items():
            same_term_in_scenario = term in in_scenario
            index = term_index[term]
            if not same_term_in_scenario:
                response_only_presence[row.row_id, index] = 1
                response_only_occurrences[row.row_id, index] = count
            rows.append(
                {
                    "row_id": row.row_id,
                    "model_id": row.model_id,
                    "persona": row.persona,
                    "identity": row.identity,
                    "condition": row.condition,
                    "question_id": row.question_id,
                    "subset": row.subset,
                    "answer_status": row.answer_status,
                    "term": term,
                    "term_category": GENDER_TERM_CATEGORIES[term],
                    "raw_occurrences": count,
                    "same_term_in_scenario": same_term_in_scenario,
                }
            )
        if position % 10_000 == 0 or position == len(frame):
            print(f"[term-surface] {position:,}/{len(frame):,}")
    if availability_mismatches:
        raise ValueError(f"Visible-surface availability mismatch: {availability_mismatches}")

    hits = pd.DataFrame(rows)
    observed = (
        hits.groupby("term", as_index=False)
        .agg(response_occurrences=("raw_occurrences", "sum"), responses_with_term=("row_id", "nunique"))
    )
    expected = lexicon_coverage[["term", "response_occurrences", "responses_with_term"]].copy()
    validation = expected.merge(observed, on="term", how="left", suffixes=("_expected", "_observed"))
    validation[["response_occurrences_observed", "responses_with_term_observed"]] = validation[
        ["response_occurrences_observed", "responses_with_term_observed"]
    ].fillna(0).astype(int)
    validation["occurrences_match"] = validation["response_occurrences_expected"].eq(validation["response_occurrences_observed"])
    validation["responses_match"] = validation["responses_with_term_expected"].eq(validation["responses_with_term_observed"])
    if not validation[["occurrences_match", "responses_match"]].all().all():
        mismatches = validation[~validation["occurrences_match"] | ~validation["responses_match"]]
        raise ValueError(f"Term reconstruction does not match v1.6 coverage:\n{mismatches.to_string(index=False)}")
    return hits, response_only_presence, response_only_occurrences, validation


def _term_prevalence(frame: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"]].copy()
    available_all = pd.concat(
        [available.assign(subset_scope=available["subset"]), available.assign(subset_scope="all")],
        ignore_index=True,
    )
    denominators = (
        available_all.groupby(
            ["model_id", "persona", "identity", "condition", "subset_scope"], observed=True
        )
        .agg(n_available=("row_id", "size"), total_words=("word_count", "sum"))
        .reset_index()
    )

    hits_all = pd.concat(
        [hits.assign(subset_scope=hits["subset"]), hits.assign(subset_scope="all")],
        ignore_index=True,
    )
    hits_all["response_only"] = ~hits_all["same_term_in_scenario"]
    hits_all["response_only_occurrences"] = hits_all["raw_occurrences"].where(hits_all["response_only"], 0)
    grouped = (
        hits_all.groupby(
            ["model_id", "persona", "identity", "condition", "subset_scope", "term", "term_category"],
            observed=True,
        )
        .agg(
            responses_with_term=("row_id", "nunique"),
            raw_occurrences=("raw_occurrences", "sum"),
            responses_same_term_in_scenario=("same_term_in_scenario", "sum"),
            response_only_responses=("response_only", "sum"),
            response_only_occurrences=("response_only_occurrences", "sum"),
        )
        .reset_index()
    )

    lexicon = pd.DataFrame(
        {"term": list(GENDER_TERM_CATEGORIES), "term_category": list(GENDER_TERM_CATEGORIES.values())}
    )
    denominators["_join"] = 1
    lexicon["_join"] = 1
    complete = denominators.merge(lexicon, on="_join", how="inner").drop(columns="_join")
    result = complete.merge(
        grouped,
        on=["model_id", "persona", "identity", "condition", "subset_scope", "term", "term_category"],
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "responses_with_term",
        "raw_occurrences",
        "responses_same_term_in_scenario",
        "response_only_responses",
        "response_only_occurrences",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["response_prevalence_pct"] = 100 * result["responses_with_term"] / result["n_available"]
    result["response_only_prevalence_pct"] = 100 * result["response_only_responses"] / result["n_available"]
    result["occurrences_per_100_words"] = 100 * result["raw_occurrences"] / result["total_words"].replace(0, np.nan)
    result["response_only_occurrences_per_100_words"] = (
        100 * result["response_only_occurrences"] / result["total_words"].replace(0, np.nan)
    )
    return result.rename(columns={"subset_scope": "subset"})


def _direct_term_table(
    frame: pd.DataFrame,
    presence: np.ndarray,
    occurrences: np.ndarray,
) -> pd.DataFrame:
    terms = tuple(GENDER_TERM_CATEGORIES)
    available = frame[frame["text_available"]]
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        left = available[available["identity"].eq("man") & available["condition"].eq(condition)][
            [*PAIR_KEYS, "row_id", "word_count"]
        ]
        right = available[available["identity"].eq("woman") & available["condition"].eq(condition)][
            [*PAIR_KEYS, "row_id", "word_count"]
        ]
        joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_man", "_woman"), validate="one_to_one")
        for model_id, model_rows in joined.groupby("model_id", sort=False):
            for subset_scope, scoped in _scope_groups(model_rows):
                man_index = scoped["row_id_man"].to_numpy(dtype=int)
                woman_index = scoped["row_id_woman"].to_numpy(dtype=int)
                man_presence = presence[man_index]
                woman_presence = presence[woman_index]
                man_occurrences = occurrences[man_index]
                woman_occurrences = occurrences[woman_index]
                n_pairs = len(scoped)
                man_words = float(scoped["word_count_man"].sum())
                woman_words = float(scoped["word_count_woman"].sum())
                for index, term in enumerate(terms):
                    man_n = int(man_presence[:, index].sum())
                    woman_n = int(woman_presence[:, index].sum())
                    man_raw = int(man_occurrences[:, index].sum())
                    woman_raw = int(woman_occurrences[:, index].sum())
                    rows.append(
                        {
                            "model_id": model_id,
                            "condition": condition,
                            "subset": subset_scope,
                            "term": term,
                            "term_category": GENDER_TERM_CATEGORIES[term],
                            "n_pairs": n_pairs,
                            "man_responses_with_response_only_term": man_n,
                            "woman_responses_with_response_only_term": woman_n,
                            "man_prevalence_pct": 100 * man_n / n_pairs if n_pairs else math.nan,
                            "woman_prevalence_pct": 100 * woman_n / n_pairs if n_pairs else math.nan,
                            "prevalence_delta_pp": 100 * (man_n - woman_n) / n_pairs if n_pairs else math.nan,
                            "man_raw_occurrences": man_raw,
                            "woman_raw_occurrences": woman_raw,
                            "man_occurrences_per_100_words": 100 * man_raw / man_words if man_words else math.nan,
                            "woman_occurrences_per_100_words": 100 * woman_raw / woman_words if woman_words else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _direct_marker_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = frame[frame["text_available"]]
    general_source = {**{outcome: outcome for outcome in SURFACE_OUTCOMES}}
    general_source.update({marker: f"{marker}__density" for marker in GENERAL_MARKERS})
    gender_source = {marker: f"{marker}__present" for marker in GENDER_MARKERS}
    value_columns = [*PAIR_KEYS, *general_source.values(), *gender_source.values()]
    general_rows: list[dict[str, object]] = []
    gender_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        left = available[available["identity"].eq("man") & available["condition"].eq(condition)][value_columns]
        right = available[available["identity"].eq("woman") & available["condition"].eq(condition)][value_columns]
        joined = left.merge(right, on=list(PAIR_KEYS), suffixes=("_man", "_woman"), validate="one_to_one")
        for model_id, model_rows in joined.groupby("model_id", sort=False):
            for subset_scope, scoped in _scope_groups(model_rows):
                for outcome, column in general_source.items():
                    man = pd.to_numeric(scoped[f"{column}_man"], errors="coerce").to_numpy(float)
                    woman = pd.to_numeric(scoped[f"{column}_woman"], errors="coerce").to_numpy(float)
                    difference = man - woman
                    n_pairs, n_nonzero, effect = _rank_biserial(difference)
                    finite = difference[np.isfinite(difference)]
                    general_rows.append(
                        {
                            "model_id": model_id,
                            "condition": condition,
                            "subset": subset_scope,
                            "outcome_id": outcome,
                            "outcome_family": "surface" if outcome in SURFACE_OUTCOMES else "general_marker",
                            "n_pairs": n_pairs,
                            "n_nonzero": n_nonzero,
                            "mean_delta": float(np.mean(finite)) if len(finite) else math.nan,
                            "median_delta": float(np.median(finite)) if len(finite) else math.nan,
                            "rank_biserial": effect,
                        }
                    )
                for marker, column in gender_source.items():
                    man = pd.to_numeric(scoped[f"{column}_man"], errors="coerce").to_numpy(float)
                    woman = pd.to_numeric(scoped[f"{column}_woman"], errors="coerce").to_numpy(float)
                    valid = np.isfinite(man) & np.isfinite(woman)
                    man = man[valid]
                    woman = woman[valid]
                    n_pairs = len(man)
                    gender_rows.append(
                        {
                            "model_id": model_id,
                            "condition": condition,
                            "subset": subset_scope,
                            "outcome_id": marker,
                            "n_pairs": n_pairs,
                            "man_prevalence_pct": 100 * float(man.mean()) if n_pairs else math.nan,
                            "woman_prevalence_pct": 100 * float(woman.mean()) if n_pairs else math.nan,
                            "prevalence_delta_pp": 100 * float((man - woman).mean()) if n_pairs else math.nan,
                            "man_raw_positive_responses": int(man.sum()) if n_pairs else 0,
                            "woman_raw_positive_responses": int(woman.sum()) if n_pairs else 0,
                        }
                    )
    return pd.DataFrame(general_rows), pd.DataFrame(gender_rows)


def _response_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        left = frame[frame["identity"].eq("man") & frame["condition"].eq(condition)]
        right = frame[frame["identity"].eq("woman") & frame["condition"].eq(condition)]
        value_columns = [*PAIR_KEYS, "answer_status", "is_correct", "extracted_stance", "text_available"]
        joined = left[value_columns].merge(
            right[value_columns], on=list(PAIR_KEYS), suffixes=("_man", "_woman"), validate="one_to_one"
        )
        for (model_id, subset), group in joined.groupby(["model_id", "subset"], sort=False):
            total = len(group)
            man_committed = group["answer_status_man"].eq("committed")
            woman_committed = group["answer_status_woman"].eq("committed")
            both_committed = man_committed & woman_committed
            man_abstain = group["answer_status_man"].eq("abstain")
            woman_abstain = group["answer_status_woman"].eq("abstain")
            man_missing = group["answer_status_man"].eq("missing")
            woman_missing = group["answer_status_woman"].eq("missing")
            row: dict[str, object] = {
                "model_id": model_id,
                "condition": condition,
                "subset": subset,
                "total_pairs": total,
                "man_committed": int(man_committed.sum()),
                "woman_committed": int(woman_committed.sum()),
                "man_commit_rate": float(man_committed.mean()),
                "woman_commit_rate": float(woman_committed.mean()),
                "commit_rate_delta_pp": 100 * float((man_committed.astype(int) - woman_committed.astype(int)).mean()),
                "man_abstain": int(man_abstain.sum()),
                "woman_abstain": int(woman_abstain.sum()),
                "man_abstain_rate": float(man_abstain.mean()),
                "woman_abstain_rate": float(woman_abstain.mean()),
                "abstain_rate_delta_pp": 100 * float((man_abstain.astype(int) - woman_abstain.astype(int)).mean()),
                "man_missing": int(man_missing.sum()),
                "woman_missing": int(woman_missing.sum()),
                "man_missing_rate": float(man_missing.mean()),
                "woman_missing_rate": float(woman_missing.mean()),
                "missing_rate_delta_pp": 100 * float((man_missing.astype(int) - woman_missing.astype(int)).mean()),
                "both_committed": int(both_committed.sum()),
                "both_committed_rate": float(both_committed.mean()),
                "man_text_available": int(group["text_available_man"].sum()),
                "woman_text_available": int(group["text_available_woman"].sum()),
                "text_available_delta_pp": 100 * float(
                    (group["text_available_man"].astype(int) - group["text_available_woman"].astype(int)).mean()
                ),
            }
            if subset in {"commonsense", "justice"}:
                man_correct = group["is_correct_man"].fillna(False).astype(bool)
                woman_correct = group["is_correct_woman"].fillna(False).astype(bool)
                row.update(
                    {
                        "man_accuracy_committed": float(man_correct[man_committed].mean()) if man_committed.any() else math.nan,
                        "woman_accuracy_committed": float(woman_correct[woman_committed].mean()) if woman_committed.any() else math.nan,
                        "man_accuracy_both_committed": float(man_correct[both_committed].mean()) if both_committed.any() else math.nan,
                        "woman_accuracy_both_committed": float(woman_correct[both_committed].mean()) if both_committed.any() else math.nan,
                        "paired_accuracy_delta_pp": 100 * float(
                            (man_correct[both_committed].astype(int) - woman_correct[both_committed].astype(int)).mean()
                        ) if both_committed.any() else math.nan,
                    }
                )
            else:
                man_wrong = group["extracted_stance_man"].eq("wrong")
                woman_wrong = group["extracted_stance_woman"].eq("wrong")
                row.update(
                    {
                        "man_pct_answer_wrong": float(man_wrong[man_committed].mean()) if man_committed.any() else math.nan,
                        "woman_pct_answer_wrong": float(woman_wrong[woman_committed].mean()) if woman_committed.any() else math.nan,
                        "answer_wrong_delta_pp": 100 * (
                            float(man_wrong[man_committed].mean()) - float(woman_wrong[woman_committed].mean())
                        ) if man_committed.any() and woman_committed.any() else math.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _opening_direct(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"] & frame["opening_category"].notna()]
    categories = sorted(available["opening_category"].unique())
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for model_id in MODELS:
            man = available[
                available["model_id"].eq(model_id)
                & available["identity"].eq("man")
                & available["condition"].eq(condition)
            ]["opening_category"]
            woman = available[
                available["model_id"].eq(model_id)
                & available["identity"].eq("woman")
                & available["condition"].eq(condition)
            ]["opening_category"]
            man_prop = man.value_counts(normalize=True)
            woman_prop = woman.value_counts(normalize=True)
            mtv = 0.5 * sum(abs(float(man_prop.get(item, 0)) - float(woman_prop.get(item, 0))) for item in categories)
            rows.append(
                {
                    "model_id": model_id,
                    "condition": condition,
                    "n_man": len(man),
                    "n_woman": len(woman),
                    "marginal_total_variation": mtv,
                }
            )
    return pd.DataFrame(rows)


def _heatmap_axes(
    axes: np.ndarray,
    data: pd.DataFrame,
    *,
    value: str,
    columns: tuple[str, ...],
    column_labels: dict[str, str],
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str,
    cbar_label: str,
) -> None:
    cbar_ax = axes.flat[-1].figure.add_axes([0.92, 0.18, 0.016, 0.64])
    for index, (ax, condition) in enumerate(zip(axes.flat, CONDITIONS)):
        panel = data[data["condition"].eq(condition)].pivot(index="model_id", columns="subset", values=value)
        panel = panel.reindex(index=MODELS, columns=columns)
        panel.index = [MODEL_LABELS[item] for item in panel.index]
        panel.columns = [column_labels[item] for item in panel.columns]
        sns.heatmap(
            panel,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0 if vmin < 0 < vmax else None,
            annot=True,
            fmt=fmt,
            annot_kws={"fontsize": 8.5},
            linewidths=0.6,
            linecolor="white",
            cbar=index == 3,
            cbar_ax=cbar_ax if index == 3 else None,
            cbar_kws={"label": cbar_label} if index == 3 else None,
        )
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.set_xlabel("Subset ETHICS")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)


def _plot_surface_heatmap(data: pd.DataFrame, image_dir: Path, outcome: str, number: int) -> Path:
    selected = data[data["outcome_id"].eq(outcome) & ~data["subset"].eq("all")]
    limit = float(np.nanmax(np.abs(selected["median_delta"])))
    limit = max(limit, 0.01 if outcome == "ttr_300" else 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    _heatmap_axes(
        axes,
        selected,
        value="median_delta",
        columns=SUBSETS,
        column_labels=SUBSET_LABELS,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
        fmt="+.3f" if outcome == "ttr_300" else "+.1f",
        cbar_label=f"Man − Woman: {SURFACE_UNITS[outcome]}",
    )
    fig.suptitle(f"{SURFACE_LABELS[outcome]}: perbandingan langsung Man dan Woman", fontweight="bold", y=0.985)
    fig.text(0.5, 0.025, "Nilai positif = lebih tinggi pada Man; nilai negatif = lebih tinggi pada Woman", ha="center")
    fig.tight_layout(rect=(0.02, 0.05, 0.90, 0.95), h_pad=2.0, w_pad=2.0)
    return _save(fig, image_dir, f"{number:02d}_direct_verbosity_{outcome}.png")


def _selected_terms(term_prevalence: pd.DataFrame, n: int = 12) -> list[str]:
    selected = term_prevalence[
        term_prevalence["identity"].isin(["man", "woman"])
        & term_prevalence["condition"].isin(CONDITIONS)
        & term_prevalence["subset"].eq("all")
    ]
    ranking = selected.groupby("term")["response_only_prevalence_pct"].mean().sort_values(ascending=False)
    return ranking.head(n).index.tolist()


def _plot_term_absolute(term_prevalence: pd.DataFrame, selected_terms: list[str], image_dir: Path) -> Path:
    data = term_prevalence[
        term_prevalence["identity"].isin(["man", "woman"])
        & term_prevalence["condition"].isin(CONDITIONS)
        & term_prevalence["subset"].eq("all")
        & term_prevalence["term"].isin(selected_terms)
    ]
    aggregate = data.groupby(["identity", "condition", "term"], as_index=False)["response_only_prevalence_pct"].mean()
    term_order = list(reversed(selected_terms))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        panel = aggregate[aggregate["condition"].eq(condition)]
        for identity, color, label in (("man", "#2563EB", "Man"), ("woman", "#D946EF", "Woman")):
            values = panel[panel["identity"].eq(identity)].set_index("term")["response_only_prevalence_pct"].reindex(term_order)
            offset = -0.10 if identity == "man" else 0.10
            ax.scatter(values, np.arange(len(term_order)) + offset, s=52, color=color, label=label, zorder=3)
        ax.set_yticks(np.arange(len(term_order)), term_order)
        ax.set_title(CONDITION_LABELS[condition], fontweight="bold")
        ax.set_xlabel("Respons yang menambahkan term (%)")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Prevalensi kata gender pada jawaban Man dan Woman", fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.95), h_pad=1.8, w_pad=1.5)
    return _save(fig, image_dir, "04_gender_words_absolute_man_woman.png")


def _plot_term_direct_heatmap(
    direct_terms: pd.DataFrame,
    selected_terms: list[str],
    image_dir: Path,
    condition: str,
    number: int,
    shared_limit: float,
) -> Path:
    data = direct_terms[
        direct_terms["condition"].eq(condition)
        & direct_terms["term"].isin(selected_terms)
        & ~direct_terms["subset"].eq("all")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.19, 0.016, 0.62])
    term_order = list(reversed(selected_terms))
    for index, (ax, subset) in enumerate(zip(axes.flat, SUBSETS)):
        panel = data[data["subset"].eq(subset)].pivot(index="term", columns="model_id", values="prevalence_delta_pp")
        panel = panel.reindex(index=term_order, columns=MODELS)
        panel.columns = [MODEL_SHORT[item] for item in panel.columns]
        sns.heatmap(
            panel,
            ax=ax,
            cmap="RdBu_r",
            vmin=-shared_limit,
            vmax=shared_limit,
            center=0,
            annot=True,
            fmt="+.1f",
            annot_kws={"fontsize": 7.5},
            linewidths=0.5,
            linecolor="white",
            cbar=index == 3,
            cbar_ax=cbar_ax if index == 3 else None,
            cbar_kws={"label": "Man − Woman (poin persentase)"} if index == 3 else None,
        )
        ax.set_title(SUBSET_LABELS[subset], fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Kata gender")
        ax.tick_params(axis="x", rotation=0, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.suptitle(f"Kata gender response-only: Man − Woman pada {CONDITION_LABELS[condition]}", fontweight="bold", y=0.985)
    fig.text(0.5, 0.025, "Positif = lebih sering pada Man; negatif = lebih sering pada Woman", ha="center")
    fig.tight_layout(rect=(0.02, 0.05, 0.90, 0.95), h_pad=2.0, w_pad=1.8)
    return _save(fig, image_dir, f"{number:02d}_direct_gender_words_{condition}.png")


def _plot_marker_forest(
    data: pd.DataFrame,
    image_dir: Path,
    *,
    condition: str,
    outcomes: tuple[str, ...],
    labels: dict[str, str],
    value: str,
    xlabel: str,
    title: str,
    filename: str,
    xlimit: float,
) -> Path:
    selected = data[data["condition"].eq(condition) & ~data["subset"].eq("all")]
    y_lookup = {outcome: len(outcomes) - 1 - index for index, outcome in enumerate(outcomes)}
    height = 14.5 if len(outcomes) > 14 else 11.5
    fig, axes = plt.subplots(2, 2, figsize=(18.5, height), sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, SUBSETS):
        panel = selected[selected["subset"].eq(subset)]
        for _, point in panel.iterrows():
            ax.scatter(
                point[value],
                y_lookup[point["outcome_id"]],
                s=46,
                color=MODEL_COLORS[point["model_id"]],
                alpha=0.88,
                edgecolors="white",
                linewidths=0.4,
            )
        medians = panel.groupby("outcome_id")[value].median()
        for outcome, point in medians.items():
            ax.scatter(point, y_lookup[outcome], marker="D", s=62, color="#111827", zorder=5)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(-xlimit, xlimit)
        ax.set_title(SUBSET_LABELS[subset], fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
        ax.set_yticks([y_lookup[item] for item in outcomes], [labels[item] for item in outcomes])
        ax.tick_params(axis="y", labelsize=9.3)
    fig.supxlabel(xlabel, y=0.105)
    fig.legend(handles=_model_legend(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4)
    fig.suptitle(title, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.03, 0.17, 0.99, 0.95), h_pad=2.0, w_pad=1.5)
    return _save(fig, image_dir, filename)


def _plot_commit_delta(outcomes: pd.DataFrame, image_dir: Path) -> Path:
    limit = max(1.0, float(outcomes["commit_rate_delta_pp"].abs().max()))
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    _heatmap_axes(
        axes,
        outcomes,
        value="commit_rate_delta_pp",
        columns=SUBSETS,
        column_labels=SUBSET_LABELS,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
        fmt="+.1f",
        cbar_label="Man − Woman: commit rate (poin persentase)",
    )
    fig.suptitle("Perbedaan commit rate antara Man dan Woman", fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.02, 0.05, 0.90, 0.95), h_pad=2.0, w_pad=2.0)
    return _save(fig, image_dir, "17_direct_commit_rate_man_woman.png")


def _plot_accuracy_coverage(outcomes: pd.DataFrame, image_dir: Path) -> Path:
    data = outcomes[outcomes["subset"].isin(["commonsense", "justice"])]
    accuracy_limit = max(1.0, float(data["paired_accuracy_delta_pp"].abs().max()))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    for row_index, subset in enumerate(("commonsense", "justice")):
        panel = data[data["subset"].eq(subset)]
        accuracy = panel.pivot(index="model_id", columns="condition", values="paired_accuracy_delta_pp").reindex(index=MODELS, columns=CONDITIONS)
        retention = (100 * panel.pivot(index="model_id", columns="condition", values="both_committed_rate")).reindex(index=MODELS, columns=CONDITIONS)
        for matrix in (accuracy, retention):
            matrix.index = [MODEL_LABELS[item] for item in matrix.index]
            matrix.columns = [CONDITION_LABELS[item] for item in matrix.columns]
        sns.heatmap(
            accuracy,
            ax=axes[row_index, 0],
            cmap="RdBu_r",
            vmin=-accuracy_limit,
            vmax=accuracy_limit,
            center=0,
            annot=True,
            fmt="+.1f",
            annot_kws={"fontsize": 8.5},
            linewidths=0.6,
            cbar_kws={"label": "Δ akurasi (pp)"},
        )
        sns.heatmap(
            retention,
            ax=axes[row_index, 1],
            cmap="Blues",
            vmin=max(0.0, float(retention.min().min()) - 5),
            vmax=100,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 8.5},
            linewidths=0.6,
            cbar_kws={"label": "Kedua respons committed (%)"},
        )
        axes[row_index, 0].set_title(f"{SUBSET_LABELS[subset]} — Man − Woman", fontweight="bold")
        axes[row_index, 1].set_title(f"{SUBSET_LABELS[subset]} — paired coverage", fontweight="bold")
        for ax in axes[row_index]:
            ax.set_xlabel("Kondisi")
            ax.set_ylabel("Model")
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Akurasi pada pasangan committed dan coverage-nya", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.95), h_pad=2.0, w_pad=2.0)
    return _save(fig, image_dir, "18_direct_accuracy_committed_with_coverage.png")


def _plot_distribution_coverage(outcomes: pd.DataFrame, image_dir: Path) -> Path:
    data = outcomes[outcomes["subset"].isin(["deontology", "virtue"])]
    delta_limit = max(1.0, float(data["answer_wrong_delta_pp"].abs().max()))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    for row_index, subset in enumerate(("deontology", "virtue")):
        panel = data[data["subset"].eq(subset)]
        delta = panel.pivot(index="model_id", columns="condition", values="answer_wrong_delta_pp").reindex(index=MODELS, columns=CONDITIONS)
        minimum_commit = (
            100
            * panel.assign(min_commit=lambda d: d[["man_commit_rate", "woman_commit_rate"]].min(axis=1))
            .pivot(index="model_id", columns="condition", values="min_commit")
            .reindex(index=MODELS, columns=CONDITIONS)
        )
        for matrix in (delta, minimum_commit):
            matrix.index = [MODEL_LABELS[item] for item in matrix.index]
            matrix.columns = [CONDITION_LABELS[item] for item in matrix.columns]
        sns.heatmap(
            delta,
            ax=axes[row_index, 0],
            cmap="RdBu_r",
            vmin=-delta_limit,
            vmax=delta_limit,
            center=0,
            annot=True,
            fmt="+.1f",
            annot_kws={"fontsize": 8.5},
            linewidths=0.6,
            cbar_kws={"label": "Δ jawaban wrong (pp)"},
        )
        sns.heatmap(
            minimum_commit,
            ax=axes[row_index, 1],
            cmap="Blues",
            vmin=max(0.0, float(minimum_commit.min().min()) - 5),
            vmax=100,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 8.5},
            linewidths=0.6,
            cbar_kws={"label": "Commit rate minimum (%)"},
        )
        axes[row_index, 0].set_title(f"{SUBSET_LABELS[subset]} — Man − Woman", fontweight="bold")
        axes[row_index, 1].set_title(f"{SUBSET_LABELS[subset]} — coverage", fontweight="bold")
        for ax in axes[row_index]:
            ax.set_xlabel("Kondisi")
            ax.set_ylabel("Model")
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Distribusi jawaban Deontology/Virtue dan commit rate", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.95), h_pad=2.0, w_pad=2.0)
    return _save(fig, image_dir, "19_direct_answer_distribution_with_coverage.png")


def _plot_opening(opening: pd.DataFrame, image_dir: Path) -> Path:
    panel = (100 * opening.pivot(index="model_id", columns="condition", values="marginal_total_variation")).reindex(index=MODELS, columns=CONDITIONS)
    panel.index = [MODEL_LABELS[item] for item in panel.index]
    panel.columns = [CONDITION_LABELS[item] for item in panel.columns]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sns.heatmap(
        panel,
        ax=ax,
        cmap="YlOrBr",
        vmin=0,
        vmax=max(5.0, float(panel.max().max())),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 9},
        linewidths=0.6,
        cbar_kws={"label": "Marginal total variation (%)"},
    )
    ax.set_title("Perbedaan komposisi opening: Man dan Woman", fontweight="bold")
    ax.set_xlabel("Kondisi")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, image_dir, "20_direct_opening_composition_man_woman.png")


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

    print("[1/8] Loading canonical marker and response sources")
    frame, source_manifest, scenario_lookup, lexicon_coverage = _load_sources(marker_dir, canonical_csv)
    print("[2/8] Reconstructing closed gender lexicon on the visible surface")
    hits, response_only_presence, response_only_occurrences, term_validation = _build_term_hits(
        frame, scenario_lookup, lexicon_coverage
    )
    print("[3/8] Building term tables by model, identity, condition, and subset")
    term_prevalence = _term_prevalence(frame, hits)
    direct_terms = _direct_term_table(frame, response_only_presence, response_only_occurrences)
    print("[4/8] Building direct marker and response-outcome tables")
    direct_general, direct_gender = _direct_marker_tables(frame)
    response_outcomes = _response_outcomes(frame)
    opening = _opening_direct(frame)

    tables = {
        "gender_term_response_hits.csv.gz": hits,
        "gender_term_surface_reconstruction_validation.csv": term_validation,
        "gender_term_prevalence_by_model_identity_condition_subset.csv": term_prevalence,
        "gender_term_direct_man_woman_response_only.csv": direct_terms,
        "general_marker_direct_man_woman_descriptive.csv": direct_general,
        "gender_marker_direct_man_woman_by_subset_descriptive.csv": direct_gender,
        "response_outcomes_direct_man_woman_by_subset.csv": response_outcomes,
        "opening_direct_man_woman_descriptive.csv": opening,
    }
    print("[5/8] Writing exact tables")
    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=False,
            compression="gzip" if filename.endswith(".gz") else None,
            encoding="utf-8",
        )

    selected_terms = _selected_terms(term_prevalence)
    direct_term_selected = direct_terms[
        direct_terms["term"].isin(selected_terms) & ~direct_terms["subset"].eq("all")
    ]
    term_limit = max(1.0, float(direct_term_selected["prevalence_delta_pp"].abs().max()))
    figures: list[tuple[str, str]] = []

    def add(path: Path, description: str) -> None:
        figures.append((path.name, description))

    print("[6/8] Rendering thesis-facing figures")
    for number, outcome in enumerate(SURFACE_OUTCOMES, start=1):
        add(
            _plot_surface_heatmap(direct_general, image_dir, outcome, number),
            f"Direct Man-Woman {outcome} by model, subset, and condition.",
        )
    add(_plot_term_absolute(term_prevalence, selected_terms, image_dir), "Absolute response-only gender-term prevalence for Man and Woman.")
    for number, condition in enumerate(CONDITIONS, start=5):
        add(
            _plot_term_direct_heatmap(direct_terms, selected_terms, image_dir, condition, number, term_limit),
            f"Direct Man-Woman response-only gender terms for {condition}.",
        )
    for number, condition in enumerate(CONDITIONS, start=9):
        add(
            _plot_marker_forest(
                direct_general[direct_general["outcome_family"].eq("general_marker")],
                image_dir,
                condition=condition,
                outcomes=GENERAL_MARKERS,
                labels=GENERAL_LABELS,
                value="rank_biserial",
                xlabel="Paired rank-biserial Man − Woman (deskriptif)",
                title=f"Marker linguistik umum: Man − Woman pada {CONDITION_LABELS[condition]}",
                filename=f"{number:02d}_direct_general_markers_{condition}.png",
                xlimit=1.03,
            ),
            f"Direct general-marker paired effect for {condition} by subset.",
        )
    gender_limit = max(1.0, float(direct_gender["prevalence_delta_pp"].abs().max()) * 1.08)
    for number, condition in enumerate(CONDITIONS, start=13):
        add(
            _plot_marker_forest(
                direct_gender,
                image_dir,
                condition=condition,
                outcomes=GENDER_MARKERS,
                labels=GENDER_MARKER_LABELS,
                value="prevalence_delta_pp",
                xlabel="Man − Woman: Δ prevalensi (poin persentase; deskriptif)",
                title=f"Marker gender: Man − Woman pada {CONDITION_LABELS[condition]}",
                filename=f"{number:02d}_direct_gender_markers_{condition}.png",
                xlimit=gender_limit,
            ),
            f"Direct gender-marker prevalence for {condition} by subset.",
        )
    add(_plot_commit_delta(response_outcomes, image_dir), "Direct Man-Woman commit-rate difference by subset.")
    add(_plot_accuracy_coverage(response_outcomes, image_dir), "Committed-pair accuracy for construct-valid subsets with paired coverage.")
    add(_plot_distribution_coverage(response_outcomes, image_dir), "Deontology/Virtue answer distribution with commit-rate coverage.")
    add(_plot_opening(opening, image_dir), "Direct opening-composition difference.")

    figure_index = pd.DataFrame(figures, columns=["figure", "description"])
    figure_index["png_size_bytes"] = figure_index["figure"].map(lambda name: (image_dir / name).stat().st_size)
    figure_index["svg_size_bytes"] = figure_index["figure"].map(lambda name: (image_dir / name).with_suffix(".svg").stat().st_size)
    figure_index.to_csv(output_dir / "figure_index.csv", index=False)

    print("[7/8] Validating bundle invariants")
    expected_general_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * (len(SURFACE_OUTCOMES) + len(GENERAL_MARKERS))
    expected_gender_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * len(GENDER_MARKERS)
    expected_term_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * len(GENDER_TERM_CATEGORIES)
    validations = {
        "rows_match_source": len(frame) == source_manifest["rows_input"] == 81_756,
        "visible_rows_match_source": int(frame["text_available"].sum()) == source_manifest["responses_text_available"],
        "term_reconstruction_all_match": bool(term_validation[["occurrences_match", "responses_match"]].all().all()),
        "direct_general_row_count": len(direct_general) == expected_general_rows,
        "direct_gender_row_count": len(direct_gender) == expected_gender_rows,
        "direct_term_row_count": len(direct_terms) == expected_term_rows,
        "response_outcome_row_count": len(response_outcomes) == len(MODELS) * len(CONDITIONS) * len(SUBSETS),
        "all_figures_exist": all((image_dir / name).is_file() for name in figure_index["figure"]),
        "all_svg_figures_exist": all((image_dir / name).with_suffix(".svg").is_file() for name in figure_index["figure"]),
        "marker_inference_disabled": source_manifest["inference_run"] is False,
        "hidden_reasoning_used": False,
        "implicit_proxies_aggregated": False,
    }
    required_true = (
        "rows_match_source",
        "visible_rows_match_source",
        "term_reconstruction_all_match",
        "direct_general_row_count",
        "direct_gender_row_count",
        "direct_term_row_count",
        "response_outcome_row_count",
        "all_figures_exist",
        "all_svg_figures_exist",
        "marker_inference_disabled",
    )
    if (
        not all(validations[key] is True for key in required_true)
        or validations["hidden_reasoning_used"] is not False
        or validations["implicit_proxies_aggregated"] is not False
    ):
        raise ValueError(f"Bundle validation failed: {validations}")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "source_marker_dir": str(marker_dir),
        "source_pipeline_version": source_manifest["pipeline_version"],
        "source_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "canonical_csv": str(canonical_csv),
        "canonical_csv_sha256": _sha256(canonical_csv),
        "rows": len(frame),
        "visible_rows": int(frame["text_available"].sum()),
        "models": len(MODELS),
        "personas": int(frame["persona"].nunique()),
        "questions": int(frame["question_id"].nunique()),
        "subsets": list(SUBSETS),
        "conditions": list(CONDITIONS),
        "pair_direction": "man_minus_woman",
        "pair_key": list(PAIR_KEYS),
        "selected_gender_terms": selected_terms,
        "gender_term_definition": "response-only means the exact closed-lexicon term occurs in visible response but not in the paired question scenario",
        "marker_inference_run": False,
        "p_values_computed_for_markers": False,
        "confidence_intervals_computed_for_markers": False,
        "hidden_reasoning_used": False,
        "ground_truth_use": "Only response-outcome summaries for Commonsense and Justice; never surface or marker extraction.",
        "accuracy_reporting": "Paired accuracy uses both-committed responses and is displayed beside both-committed coverage.",
        "deontology_virtue_reporting": "Answer distribution and commit rate; no cross-persona accuracy claim.",
        "generated_tables": [*tables, "figure_index.csv"],
        "generated_figures": figure_index["figure"].tolist(),
        "generated_svg_figures": [Path(name).with_suffix(".svg").name for name in figure_index["figure"]],
        "validations": validations,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[8/8] Done")
    print(f"[DONE] {output_dir}")
    return output_dir


def finalize_existing_bundle(marker_dir: Path, canonical_csv: Path, output_dir: Path) -> Path:
    """Finalize a fully rendered interrupted run without replacing artifacts."""

    marker_dir = marker_dir.resolve()
    canonical_csv = canonical_csv.resolve()
    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"Interrupted bundle is missing: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")

    source_manifest = json.loads((marker_dir / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("pipeline_version") != EXPECTED_PIPELINE_VERSION:
        raise ValueError("Unexpected source marker pipeline.")
    if source_manifest.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("Unexpected source marker SHA-256.")
    if _sha256(canonical_csv) != EXPECTED_INPUT_SHA256:
        raise ValueError("Canonical CSV SHA-256 changed.")

    table_filenames = [
        "gender_term_response_hits.csv.gz",
        "gender_term_surface_reconstruction_validation.csv",
        "gender_term_prevalence_by_model_identity_condition_subset.csv",
        "gender_term_direct_man_woman_response_only.csv",
        "general_marker_direct_man_woman_descriptive.csv",
        "gender_marker_direct_man_woman_by_subset_descriptive.csv",
        "response_outcomes_direct_man_woman_by_subset.csv",
        "opening_direct_man_woman_descriptive.csv",
    ]
    missing = [name for name in [*table_filenames, "figure_index.csv"] if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Interrupted bundle is incomplete: {missing}")

    term_validation = pd.read_csv(output_dir / "gender_term_surface_reconstruction_validation.csv")
    term_prevalence = pd.read_csv(output_dir / "gender_term_prevalence_by_model_identity_condition_subset.csv")
    direct_terms = pd.read_csv(output_dir / "gender_term_direct_man_woman_response_only.csv")
    direct_general = pd.read_csv(output_dir / "general_marker_direct_man_woman_descriptive.csv")
    direct_gender = pd.read_csv(output_dir / "gender_marker_direct_man_woman_by_subset_descriptive.csv")
    response_outcomes = pd.read_csv(output_dir / "response_outcomes_direct_man_woman_by_subset.csv")
    figure_index = pd.read_csv(output_dir / "figure_index.csv")

    expected_general_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * (len(SURFACE_OUTCOMES) + len(GENERAL_MARKERS))
    expected_gender_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * len(GENDER_MARKERS)
    expected_term_rows = len(MODELS) * len(CONDITIONS) * (len(SUBSETS) + 1) * len(GENDER_TERM_CATEGORIES)
    validations = {
        "rows_match_source": source_manifest["rows_input"] == 81_756,
        "visible_rows_match_source": source_manifest["responses_text_available"] == 80_840,
        "term_reconstruction_all_match": bool(term_validation[["occurrences_match", "responses_match"]].all().all()),
        "direct_general_row_count": len(direct_general) == expected_general_rows,
        "direct_gender_row_count": len(direct_gender) == expected_gender_rows,
        "direct_term_row_count": len(direct_terms) == expected_term_rows,
        "response_outcome_row_count": len(response_outcomes) == len(MODELS) * len(CONDITIONS) * len(SUBSETS),
        "all_figures_exist": all((image_dir / name).is_file() for name in figure_index["figure"]),
        "all_svg_figures_exist": all((image_dir / name).with_suffix(".svg").is_file() for name in figure_index["figure"]),
        "marker_inference_disabled": source_manifest["inference_run"] is False,
        "hidden_reasoning_used": False,
        "implicit_proxies_aggregated": False,
    }
    required_true = (
        "rows_match_source",
        "visible_rows_match_source",
        "term_reconstruction_all_match",
        "direct_general_row_count",
        "direct_gender_row_count",
        "direct_term_row_count",
        "response_outcome_row_count",
        "all_figures_exist",
        "all_svg_figures_exist",
        "marker_inference_disabled",
    )
    if (
        not all(validations[key] is True for key in required_true)
        or validations["hidden_reasoning_used"] is not False
        or validations["implicit_proxies_aggregated"] is not False
    ):
        raise ValueError(f"Interrupted bundle validation failed: {validations}")

    selected_terms = _selected_terms(term_prevalence)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "finalized_from_interrupted_run": True,
        "source_marker_dir": str(marker_dir),
        "source_pipeline_version": source_manifest["pipeline_version"],
        "source_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": _sha256(marker_dir / "manifest.json"),
        "canonical_csv": str(canonical_csv),
        "canonical_csv_sha256": _sha256(canonical_csv),
        "rows": source_manifest["rows_input"],
        "visible_rows": source_manifest["responses_text_available"],
        "models": len(MODELS),
        "personas": 9,
        "questions": source_manifest["unique_questions"],
        "subsets": list(SUBSETS),
        "conditions": list(CONDITIONS),
        "pair_direction": "man_minus_woman",
        "pair_key": list(PAIR_KEYS),
        "selected_gender_terms": selected_terms,
        "gender_term_definition": "response-only means the exact closed-lexicon term occurs in visible response but not in the paired question scenario",
        "marker_inference_run": False,
        "p_values_computed_for_markers": False,
        "confidence_intervals_computed_for_markers": False,
        "hidden_reasoning_used": False,
        "ground_truth_use": "Only response-outcome summaries for Commonsense and Justice; never surface or marker extraction.",
        "accuracy_reporting": "Paired accuracy uses both-committed responses and is displayed beside both-committed coverage.",
        "deontology_virtue_reporting": "Answer distribution and commit rate; no cross-persona accuracy claim.",
        "generated_tables": [*table_filenames, "figure_index.csv"],
        "generated_figures": figure_index["figure"].tolist(),
        "generated_svg_figures": [Path(name).with_suffix(".svg").name for name in figure_index["figure"]],
        "validations": validations,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Finalized existing bundle: {output_dir}")
    return output_dir


def main() -> int:
    args = _parser().parse_args()
    if args.finalize_existing:
        finalize_existing_bundle(args.marker_dir, args.canonical_csv, args.output_dir)
    else:
        build_bundle(args.marker_dir, args.canonical_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

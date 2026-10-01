"""Full-corpus audit runner for ETHICS regional-cultural linguistic markers.

The runner is intentionally descriptive by default.  It extracts the same
registered rules for every model and persona, retains visible abstentions,
maps missing response text to NA, and keeps inferential tests disabled until
the candidate registry has independent human precision and recall review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .audit_sampler import build_recall_audit_sample
from .config import PLANNED_CONTRASTS, RANDOM_SEED
from .ethics_registry import moral_polarity_sequence
from .ethics_regional_registry import (
    ETHICS_REGIONAL_CULTURAL_MARKERS,
    REGIONAL_AUDIT_LEXICON,
    REGIONAL_CULTURAL_LEXICON_VERSION,
    REGIONAL_CULTURAL_MARKERS,
    REGIONAL_CULTURAL_REGISTRY_VERSION,
    compile_regional_audit_lexicon_pattern,
    regional_cultural_lexicon_rows,
    regional_cultural_registry_rows,
)
from .extractor import extract_markers, find_marker_matches, marker_text
from .regional_config import IDENTITY_CONDITIONS
from .regional_csv_loaders import regional_persona_metadata
from .registry import MarkerSpec
from .statistics import (
    _bootstrap_ci,
    _mcnemar_exact,
    _paired_numeric_test,
    _stable_seed,
    holm_adjust,
)
from .surface import prepare_text_surface


REQUIRED_COLUMNS = (
    "model",
    "persona_folder",
    "file",
    "question_id",
    "subset",
    "response_content_full",
    "answer_status",
    "abstain_reason",
    "parse_source",
    "reasoning_tokens",
    "truncated",
    "construct_valid",
)

_TOKEN_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)?", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_OPEN_ANCHORS: dict[str, re.Pattern[str]] = {
    "regional_geocultural": re.compile(
        r"\b(?:north[- ]america(?:n)?s?|east[- ]asia(?:n)?s?|south[- ]asia(?:n)?s?|"
        r"sub[- ]saharan[- ]africa(?:n)?s?|african|asian|american|western|chinese|"
        r"japanese|korean|indian|pakistani|bangladeshi|nigerian|ghanaian|kenyan)\b",
        re.IGNORECASE,
    ),
    "cultural_meta": re.compile(
        r"\b(?:cultures?|cultural(?:ly)?|heritage|traditions?|traditional(?:ly)?|"
        r"norms?|customs?)\b",
        re.IGNORECASE,
    ),
    "value_orientation": re.compile(
        r"\b(?:collectiv\w*|communal\w*|individualis\w*|autonomy|self[- ]reliance|"
        r"famil(?:y|ies|ial)|communit(?:y|ies)|elders?|harmony|hierarch\w*|"
        r"seniority|ubuntu|face[- ]saving|loss\s+of\s+face|reciprocity|"
        r"interdepend\w*|interconnect\w*|mutual\s+aid|social\s+cohesion|"
        r"hospitality|generosity|group\s+consensus|consensus[- ]building|"
        r"kinship\s+ties?)\b",
        re.IGNORECASE,
    ),
    "persona_context": re.compile(
        r"\b(?:thanksgiving|ch[uū]nji[eé]|hong\s?bao|red\s+envelopes?|diwali|"
        r"deepavali|diyas?|mehendi|mehndi|henna\s+ceremony|harvest\s+"
        r"(?:celebration|festival)|matching\s+fabric|tea\s+ceremony|james\s+mitchell|"
        r"wei\s+chen|priya\s+kapoor|kwame\s+asante)\b",
        re.IGNORECASE,
    ),
}

_ANCHOR_CAPTURE_MARKERS: dict[str, frozenset[str]] = {
    "regional_geocultural": frozenset(
        {
            "culture_region_north_american",
            "culture_region_east_asian",
            "culture_region_south_asian",
            "culture_region_sub_saharan_african",
            "culture_geocultural_broader_reference",
            "regional_persona_direct_address",
            "regional_perspective_adoption",
        }
    ),
    "cultural_meta": frozenset(
        {
            "culture_general_reference",
            "culture_meta_framing",
            "culture_relativist_framing",
            "culture_identity_irrelevance",
            "regional_persona_direct_address",
            "regional_perspective_adoption",
        }
    ),
    "value_orientation": frozenset(
        {
            "culture_collectivist_communal_orientation",
            "culture_individual_autonomy_orientation",
            "culture_family_obligation_orientation",
            "culture_community_obligation_orientation",
            "culture_elder_respect_orientation",
            "culture_harmony_face_orientation",
            "culture_hierarchy_authority_orientation",
            "culture_reciprocity_interdependence_orientation",
            "culture_hospitality_generosity_orientation",
            "culture_named_concept_reference",
        }
    ),
    "persona_context": frozenset(
        {
            "culture_named_practice_reference",
            "culture_warmup_name_echo",
        }
    ),
}


@dataclass(frozen=True)
class RegionalCulturalAuditConfig:
    repo_root: Path
    input_csv: Path
    raw_ethics_root: Path
    output_dir: Path
    random_seed: int = RANDOM_SEED
    precision_per_model_marker: int = 10
    recall_sample_per_condition: int = 3
    unmatched_per_model_category: int = 5
    save_response_level: bool = True
    run_inference: bool = False
    bootstrap_resamples: int = 2_000
    extraction_workers: int = 1


def default_audit_config(repo_root: Path) -> RegionalCulturalAuditConfig:
    root = Path(repo_root).resolve()
    return RegionalCulturalAuditConfig(
        repo_root=root,
        input_csv=(
            root
            / "result"
            / "ETHICS"
            / "reports"
            / "Regional_Cultural_Persona"
            / "_combined"
            / "all_models_results_full.csv"
        ),
        raw_ethics_root=root / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]",
        output_dir=(
            root
            / "result"
            / "ETHICS"
            / "reports"
            / "Regional_Cultural_Persona"
            / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_10"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def _as_number(value: object) -> int | float | None:
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _condition_bucket(persona: str) -> str:
    if persona == "neutral":
        return "neutral"
    if persona.endswith("_explicit"):
        return "explicit"
    if "_implicit_" in persona:
        return "implicit"
    raise ValueError(f"Unexpected regional persona: {persona}")


def load_canonical_ethics(path: Path) -> pd.DataFrame:
    """Load every source row without filtering answer status or validity."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(REQUIRED_COLUMNS).difference(header)
    if missing:
        raise ValueError(f"Canonical ETHICS CSV is missing columns: {sorted(missing)}")
    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    duplicate = frame.duplicated(["model", "persona_folder", "question_id"], keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, ["model", "persona_folder", "question_id"]].iloc[0]
        raise ValueError(f"Duplicate response key: {example.to_dict()}")
    observed = set(frame["persona_folder"])
    expected = {persona for conditions in IDENTITY_CONDITIONS.values() for persona in conditions.values()}
    unknown = observed.difference(expected)
    if unknown:
        raise ValueError(f"Unexpected persona folders: {sorted(unknown)}")
    frame["condition_bucket"] = frame["persona_folder"].map(_condition_bucket)
    return frame


def load_scenario_lookup(raw_root: Path, question_ids: Iterable[str]) -> pd.DataFrame:
    """Load one visible scenario JSON per question from a Neutral model directory."""

    neutral_root = raw_root / "Neutral Persona"
    directories = sorted(path for path in neutral_root.iterdir() if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No Neutral Persona model directories under {neutral_root}")
    rows: list[dict[str, object]] = []
    unique_ids = sorted(set(str(value) for value in question_ids))
    for position, question_id in enumerate(unique_ids, start=1):
        scenario = ""
        source_path = ""
        for directory in directories:
            candidate = directory / f"{question_id}.json"
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            scenario = str(payload.get("scenario") or "").strip()
            source_path = str(candidate)
            if scenario:
                break
        rows.append(
            {
                "question_id": question_id,
                "scenario": scenario,
                "scenario_source_path": source_path,
                "scenario_available": bool(scenario),
            }
        )
        if position % 250 == 0 or position == len(unique_ids):
            print(f"[scenario] {position:,}/{len(unique_ids):,} questions")
    return pd.DataFrame(rows)


def _extract_response_record(record: dict[str, object]) -> dict[str, object]:
    extracted = extract_markers(
        record,
        markers=ETHICS_REGIONAL_CULTURAL_MARKERS,
        pipeline_version=REGIONAL_CULTURAL_REGISTRY_VERSION,
    )
    if extracted["text_available"]:
        extracted.update(moral_polarity_sequence(str(extracted["explanation_text"])))
    else:
        extracted.update(
            {
                "moral_polarity_initial": None,
                "moral_polarity_final": None,
                "moral_polarity_shift": None,
                "moral_polarity_direction": None,
            }
        )
    extracted.pop("response_content", None)
    extracted.pop("explanation_text", None)
    return extracted


def _extract_response_batch(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_extract_response_record(record) for record in records]


def _iter_extraction_batches(
    frame: pd.DataFrame, batch_size: int = 100
) -> Iterable[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for record_index, row in enumerate(frame.itertuples(index=False), start=1):
        identity, condition = regional_persona_metadata(row.persona_folder)
        reasoning_tokens = _as_number(row.reasoning_tokens)
        response_content = str(row.response_content_full or "")
        batch.append(
            {
                "model_id": row.model,
                "benchmark": "ethics",
                "subset": row.subset,
                "identity": identity,
                "condition": condition,
                "condition_bucket": row.condition_bucket,
                "persona": row.persona_folder,
                "question_id": row.question_id,
                "raw_source_file": row.file,
                "source_path": str(row.file),
                "source_row": record_index + 1,
                "response_content": response_content,
                "response_success": bool(response_content.strip()),
                "reasoning_present": bool(reasoning_tokens and reasoning_tokens > 0),
                "response_truncated": _as_bool(row.truncated),
                "answer_status": row.answer_status or None,
                "abstain_reason": row.abstain_reason or None,
                "parse_source": row.parse_source or None,
                "construct_valid": _as_bool(row.construct_valid),
                "reasoning_tokens": reasoning_tokens,
            }
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


_NUMERIC_BASE_COLUMNS = {
    "source_row",
    "reasoning_tokens",
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "ttr_300",
    "technical_lines_stripped",
    "moral_polarity_shift",
}


def _compact_extraction_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column in _NUMERIC_BASE_COLUMNS or column.endswith(
            ("__count", "__density", "__present")
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce", downcast="float")
    return frame


def extract_response_markers(frame: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    """Extract all registered features in bounded, memory-compacted chunks."""

    chunk_size = 2_500
    rows: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    total = len(frame)
    processed = 0
    batches = _iter_extraction_batches(frame)
    if workers > 1:
        print(f"[extract] using {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            extracted_batches = executor.map(
                _extract_response_batch,
                batches,
                chunksize=1,
                buffersize=workers * 2,
            )
            for extracted_batch in extracted_batches:
                rows.extend(extracted_batch)
                processed += len(extracted_batch)
                if len(rows) >= chunk_size:
                    chunks.append(_compact_extraction_chunk(pd.DataFrame.from_records(rows)))
                    rows.clear()
                    gc.collect()
                if processed % 5_000 == 0 or processed == total:
                    print(f"[extract] {processed:,}/{total:,} responses")
    else:
        for batch in batches:
            for record in batch:
                rows.append(_extract_response_record(record))
                processed += 1
            if len(rows) >= chunk_size:
                chunks.append(_compact_extraction_chunk(pd.DataFrame.from_records(rows)))
                rows.clear()
                gc.collect()
            if processed % 5_000 == 0 or processed == total:
                print(f"[extract] {processed:,}/{total:,} responses")
    if rows:
        chunks.append(_compact_extraction_chunk(pd.DataFrame.from_records(rows)))
    if not chunks:
        return pd.DataFrame()
    response = pd.concat(chunks, ignore_index=True, sort=False)
    del chunks
    gc.collect()
    return response


def build_status_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["model_id", "persona", "subset", "answer_status", "text_status"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("n_responses")
        .reset_index()
    )


def build_surface_summary(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"].fillna(False)].copy()
    return (
        available.groupby(["model_id", "persona", "subset"], sort=True)
        .agg(
            n_available=("question_id", "size"),
            median_word_count=("word_count", "median"),
            mean_word_count=("word_count", "mean"),
            p95_word_count=("word_count", lambda values: values.quantile(0.95)),
            median_sentence_length=("avg_sentence_length", "median"),
            mean_ttr_300=("ttr_300", "mean"),
        )
        .reset_index()
    )


def build_opening_summary(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"].fillna(False)].copy()
    counts = (
        available.groupby(
            ["model_id", "persona", "subset", "opening_category"], sort=True
        )
        .size()
        .rename("n_responses")
        .reset_index()
    )
    totals = counts.groupby(["model_id", "persona", "subset"])["n_responses"].transform("sum")
    counts["prevalence_pct"] = 100 * counts["n_responses"] / totals
    return counts


def build_polarity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame[frame["text_available"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    columns = ["model_id", "persona", "subset", "answer_status"]
    for keys, group in available.groupby(columns, dropna=False, sort=True):
        shift = pd.to_numeric(group["moral_polarity_shift"], errors="coerce")
        rows.append(
            {
                **dict(zip(columns, keys)),
                "n_available": len(group),
                "n_positive_initial": int(group["moral_polarity_initial"].eq("positive").sum()),
                "n_negative_initial": int(group["moral_polarity_initial"].eq("negative").sum()),
                "n_positive_final": int(group["moral_polarity_final"].eq("positive").sum()),
                "n_negative_final": int(group["moral_polarity_final"].eq("negative").sum()),
                "n_polarity_shift": int(shift.fillna(0).sum()),
                "polarity_shift_rate_pct": 100 * float(shift.mean()),
                "n_positive_to_negative": int(
                    group["moral_polarity_direction"].eq("positive_to_negative").sum()
                ),
                "n_negative_to_positive": int(
                    group["moral_polarity_direction"].eq("negative_to_positive").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_marker_prevalence(
    frame: pd.DataFrame,
    markers: Iterable[MarkerSpec] = ETHICS_REGIONAL_CULTURAL_MARKERS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model_id", "persona", "subset", "answer_status"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        metadata = dict(zip(group_columns, keys))
        for marker in markers:
            counts = pd.to_numeric(group[f"{marker.marker_id}__count"], errors="coerce")
            densities = pd.to_numeric(group[f"{marker.marker_id}__density"], errors="coerce")
            present = pd.to_numeric(group[f"{marker.marker_id}__present"], errors="coerce")
            valid = present.notna()
            rows.append(
                {
                    **metadata,
                    "marker_id": marker.marker_id,
                    "marker_family": marker.family,
                    "analysis_role": marker.analysis_role,
                    "registered_measurement": marker.measurement,
                    "n_available": int(valid.sum()),
                    "responses_present": int(present[valid].sum()),
                    "response_prevalence_pct": (
                        100 * float(present[valid].mean()) if valid.any() else math.nan
                    ),
                    "total_occurrences": int(counts.fillna(0).sum()),
                    "mean_density_per_100_words": float(densities.mean()),
                }
            )
    return pd.DataFrame(rows)


def _context_excerpt(text: str, start: int, end: int, radius: int = 180) -> str:
    excerpt = text[max(0, start - radius) : min(len(text), end + radius)]
    return _WHITESPACE_RE.sub(" ", excerpt).strip()


def _normalize_for_overlap(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip().casefold()


def _balanced_indices(
    group: pd.DataFrame,
    take: int,
    rng: np.random.Generator,
) -> list[object]:
    if len(group) <= take:
        return group.index.tolist()
    selected: list[object] = []
    strata = list(group.groupby(["persona", "subset"], sort=True))
    order = rng.permutation(len(strata))
    for position in order:
        _, stratum = strata[int(position)]
        selected.append(rng.choice(stratum.index.to_numpy()).item())
        if len(selected) == take:
            return selected
    remaining = group.drop(index=selected)
    extra = rng.choice(remaining.index.to_numpy(), size=take - len(selected), replace=False)
    selected.extend(extra.tolist())
    return selected


def build_regional_precision_audit(
    frame: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    per_model_marker: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample complete marker-positive responses for lexical and attribution audit."""

    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    lexicon_pattern = compile_regional_audit_lexicon_pattern()
    rows: list[dict[str, object]] = []
    for marker in REGIONAL_CULTURAL_MARKERS:
        present_col = f"{marker.marker_id}__present"
        candidates = frame[pd.to_numeric(frame[present_col], errors="coerce").eq(1)]
        for model, group in candidates.groupby("model_id", sort=True):
            rng = np.random.default_rng(
                _stable_seed(
                    REGIONAL_CULTURAL_REGISTRY_VERSION,
                    marker.marker_id,
                    model,
                    random_seed,
                )
            )
            selected = _balanced_indices(group, min(per_model_marker, len(group)), rng)
            for source_index in selected:
                source = candidates.loc[source_index]
                surface = prepare_text_surface(source["visible_text"], benchmark="ethics")
                text = marker_text(surface, marker)
                matches = find_marker_matches(text, marker)
                scenario = str(scenario_map.get(source["question_id"], ""))
                scenario_normalized = _normalize_for_overlap(scenario)
                matched_texts = [match.text for match in matches]
                response_terms = {
                    match.group(0).casefold() for match in lexicon_pattern.finditer(text)
                }
                scenario_terms = {
                    match.group(0).casefold() for match in lexicon_pattern.finditer(scenario)
                }
                exact_overlap = any(
                    _normalize_for_overlap(value) in scenario_normalized
                    for value in matched_texts
                    if _normalize_for_overlap(value)
                )
                rows.append(
                    {
                        "response_key": (
                            f"{source['model_id']}|{source['persona']}|{source['question_id']}"
                        ),
                        "marker_id": marker.marker_id,
                        "marker_family": marker.family,
                        "model_id": source["model_id"],
                        "persona": source["persona"],
                        "condition": source["condition"],
                        "subset": source["subset"],
                        "question_id": source["question_id"],
                        "answer_status": source["answer_status"],
                        "matched_text": " || ".join(matched_texts),
                        "match_context": " || ".join(
                            _context_excerpt(text, match.start, match.end) for match in matches
                        ),
                        "response_lexicon_terms": ";".join(sorted(response_terms)),
                        "scenario_lexicon_terms": ";".join(sorted(scenario_terms)),
                        "same_lexicon_term_in_scenario": bool(response_terms & scenario_terms),
                        "exact_matched_text_in_scenario": exact_overlap,
                        "persona_attribution_status": (
                            "scenario_overlap_requires_caution"
                            if exact_overlap
                            else "response_only_candidate"
                        ),
                        "full_response": source["visible_text"],
                        "full_scenario": scenario,
                        "codex_review_label": "",
                        "codex_review_note": "",
                        "human_reviewer_1": "",
                        "human_review_1_label": "",
                        "human_review_1_note": "",
                        "human_reviewer_2": "",
                        "human_review_2_label": "",
                        "human_review_2_note": "",
                    }
                )
    return pd.DataFrame(rows)


def build_candidate_anchor_audit(
    frame: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    per_model_category: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure broad anchor capture and sample unmatched contexts for recall review."""

    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    marker_cols = [f"{marker.marker_id}__present" for marker in REGIONAL_CULTURAL_MARKERS]
    counters: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    unmatched: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    form_counter: Counter[tuple[str, str, bool]] = Counter()
    for row in frame.itertuples(index=False):
        if not bool(row.text_available):
            continue
        text = str(row.visible_text)
        scenario = str(scenario_map.get(row.question_id, ""))
        scenario_normalized = _normalize_for_overlap(scenario)
        present_ids = {
            marker.marker_id
            for marker, column in zip(REGIONAL_CULTURAL_MARKERS, marker_cols)
            if getattr(row, column) == 1
        }
        for category, pattern in _OPEN_ANCHORS.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            captured_ids = sorted(present_ids & _ANCHOR_CAPTURE_MARKERS[category])
            captured = bool(captured_ids)
            anchor_forms = sorted({match.group(0).casefold() for match in matches})
            scenario_overlap = any(
                _normalize_for_overlap(value) in scenario_normalized for value in anchor_forms
            )
            counters[
                (
                    row.model_id,
                    row.persona,
                    row.subset,
                    category,
                    captured,
                    scenario_overlap,
                )
            ] += 1
            for form in anchor_forms:
                form_counter[(category, form, captured)] += 1
            if captured:
                continue
            first = matches[0]
            key = (category, row.model_id)
            unmatched[key].append(
                {
                    "anchor_category": category,
                    "anchor_forms": ";".join(anchor_forms),
                    "model_id": row.model_id,
                    "persona": row.persona,
                    "condition": row.condition,
                    "subset": row.subset,
                    "question_id": row.question_id,
                    "answer_status": row.answer_status,
                    "scenario_anchor_overlap": scenario_overlap,
                    "anchor_context": _context_excerpt(
                        text, first.start(), first.end(), radius=240
                    ),
                    "full_response": text,
                    "full_scenario": scenario,
                    "codex_recall_label": "",
                    "proposed_marker_or_exclusion": "",
                    "codex_recall_note": "",
                    "human_reviewer_1": "",
                    "human_review_1_note": "",
                    "human_reviewer_2": "",
                    "human_review_2_note": "",
                }
            )

    summary_rows = [
        {
            "model_id": model,
            "persona": persona,
            "subset": subset,
            "anchor_category": category,
            "captured_by_relevant_regional_marker": captured,
            "scenario_anchor_overlap": overlap,
            "n_responses": count,
        }
        for (model, persona, subset, category, captured, overlap), count in counters.items()
    ]
    summary = pd.DataFrame(summary_rows)
    form_rows = [
        {
            "anchor_category": category,
            "anchor_form": form,
            "captured_by_relevant_regional_marker": captured,
            "n_responses": count,
        }
        for (category, form, captured), count in form_counter.most_common()
    ]
    forms = pd.DataFrame(form_rows)
    sampled_rows: list[dict[str, object]] = []
    for (category, model), records in sorted(unmatched.items()):
        rng = np.random.default_rng(
            _stable_seed(
                REGIONAL_CULTURAL_REGISTRY_VERSION,
                "unmatched",
                category,
                model,
                random_seed,
            )
        )
        take = min(per_model_category, len(records))
        chosen = rng.choice(len(records), size=take, replace=False)
        sampled_rows.extend(records[int(position)] for position in chosen)
    return summary, forms, pd.DataFrame(sampled_rows)


def build_lexicon_coverage(
    frame: pd.DataFrame, scenarios: pd.DataFrame
) -> pd.DataFrame:
    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    pattern = compile_regional_audit_lexicon_pattern()
    counters: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    for row in frame.itertuples(index=False):
        if not bool(row.text_available):
            continue
        text = str(row.visible_text)
        terms = {match.group(0).casefold() for match in pattern.finditer(text)}
        if not terms:
            continue
        scenario_terms = {
            match.group(0).casefold()
            for match in pattern.finditer(str(scenario_map.get(row.question_id, "")))
        }
        present_ids = {
            marker.marker_id
            for marker in REGIONAL_CULTURAL_MARKERS
            if getattr(row, f"{marker.marker_id}__present") == 1
        }
        for term in terms:
            category = REGIONAL_AUDIT_LEXICON.get(term, "surface_variant")
            if category in {"tested_region_place", "tested_region_identity", "broader_geocultural_phrase"}:
                anchor_category = "regional_geocultural"
            elif category in {"value_orientation_anchor", "named_cultural_concept_anchor"}:
                anchor_category = "value_orientation"
            elif category in {"persona_practice_anchor", "persona_name_anchor"}:
                anchor_category = "persona_context"
            else:
                anchor_category = "cultural_meta"
            captured = bool(present_ids & _ANCHOR_CAPTURE_MARKERS[anchor_category])
            counters[
                (row.model_id, row.persona, term, category, captured, term in scenario_terms)
            ] += 1
    return pd.DataFrame(
        [
            {
                "model_id": model,
                "persona": persona,
                "term": term,
                "term_category": category,
                "captured_by_relevant_regional_marker": captured,
                "same_term_in_scenario": overlap,
                "n_responses": count,
            }
            for (model, persona, term, category, captured, overlap), count in counters.items()
        ]
    )


def build_paired_marker_tests(
    frame: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    require_both_committed: bool,
) -> pd.DataFrame:
    """Optional paired inference; disabled by default until human audit promotion."""

    outcomes: list[dict[str, str]] = [
        {"outcome_id": "word_count", "value_col": "word_count", "scale": "numeric"},
        {
            "outcome_id": "avg_sentence_length",
            "value_col": "avg_sentence_length",
            "scale": "numeric",
        },
        {"outcome_id": "ttr_300", "value_col": "ttr_300", "scale": "numeric"},
    ]
    for marker in ETHICS_REGIONAL_CULTURAL_MARKERS:
        outcomes.append(
            {
                "outcome_id": marker.marker_id,
                "value_col": (
                    f"{marker.marker_id}__present"
                    if marker.measurement == "binary"
                    else f"{marker.marker_id}__density"
                ),
                "scale": marker.measurement,
            }
        )
    rows: list[dict[str, object]] = []
    for (model, subset), group in frame.groupby(["model_id", "subset"], sort=True):
        for identity, conditions in IDENTITY_CONDITIONS.items():
            for contrast_id, left_condition, right_condition in PLANNED_CONTRASTS:
                left_persona = conditions[left_condition]
                right_persona = conditions[right_condition]
                for outcome in outcomes:
                    left = group[group["persona"].eq(left_persona)][
                        ["question_id", "answer_status", outcome["value_col"]]
                    ]
                    right = group[group["persona"].eq(right_persona)][
                        ["question_id", "answer_status", outcome["value_col"]]
                    ]
                    paired = left.merge(
                        right,
                        on="question_id",
                        how="inner",
                        validate="one_to_one",
                        suffixes=("_left", "_right"),
                    )
                    if require_both_committed:
                        paired = paired[
                            paired["answer_status_left"].eq("committed")
                            & paired["answer_status_right"].eq("committed")
                        ]
                    left_values = pd.to_numeric(
                        paired[f"{outcome['value_col']}_left"], errors="coerce"
                    )
                    right_values = pd.to_numeric(
                        paired[f"{outcome['value_col']}_right"], errors="coerce"
                    )
                    valid = left_values.notna() & right_values.notna()
                    left_array = left_values[valid].to_numpy(float)
                    right_array = right_values[valid].to_numpy(float)
                    deltas = left_array - right_array
                    if outcome["scale"] == "binary":
                        test = _mcnemar_exact(left_array, right_array)
                        effect = float(deltas.mean()) if len(deltas) else math.nan
                        effect_name = "prevalence_delta"
                    else:
                        test = _paired_numeric_test(deltas)
                        effect = float(np.median(deltas)) if len(deltas) else math.nan
                        effect_name = "median_delta"
                    ci_low, ci_high = _bootstrap_ci(
                        deltas,
                        "mean" if outcome["scale"] == "binary" else "median",
                        bootstrap_resamples,
                        _stable_seed(
                            REGIONAL_CULTURAL_REGISTRY_VERSION,
                            model,
                            subset,
                            identity,
                            contrast_id,
                            outcome["outcome_id"],
                            require_both_committed,
                        ),
                    )
                    rows.append(
                        {
                            "model_id": model,
                            "subset": subset,
                            "identity": identity,
                            "contrast_id": contrast_id,
                            "left_persona": left_persona,
                            "right_persona": right_persona,
                            "outcome_id": outcome["outcome_id"],
                            "outcome_scale": outcome["scale"],
                            "pair_filter": (
                                "both_committed"
                                if require_both_committed
                                else "all_visible_text"
                            ),
                            "n_pairs": len(deltas),
                            effect_name: effect,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **test,
                        }
                    )
    result = pd.DataFrame(rows)
    if not result.empty:
        family = ["model_id", "subset", "identity", "outcome_id", "pair_filter"]
        result["p_holm"] = result.groupby(family, dropna=False)["p_raw"].transform(
            lambda values: holm_adjust(values.to_numpy(float))
        )
    return result


def _write_audit_workbook(
    output_path: Path,
    *,
    manifest: dict[str, object],
    precision: pd.DataFrame,
    recall: pd.DataFrame,
    unmatched: pd.DataFrame,
    registry: pd.DataFrame,
    prevalence: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    instructions = pd.DataFrame(
        [
            {
                "section": "purpose",
                "instruction": (
                    "Candidate precision/recall audit for visible ETHICS regional-cultural "
                    "language. This is not a cultural score and not proof that a persona caused "
                    "a phrase."
                ),
            },
            {
                "section": "precision",
                "instruction": (
                    "Read full_response and full_scenario. Label TP only when matched_text fits "
                    "the registered operational definition. Separately note whether attribution "
                    "is scenario-ambiguous."
                ),
            },
            {
                "section": "recall",
                "instruction": (
                    "Read complete responses without relying on ground truth. Record any missed "
                    "regional-cultural construct and its exact visible span."
                ),
            },
            {
                "section": "promotion",
                "instruction": (
                    "The registry remains candidate until two independent human reviewers agree; "
                    "Codex review does not satisfy that human-review requirement."
                ),
            },
            {"section": "input_sha256", "instruction": str(manifest["input_sha256"])},
        ]
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="README", index=False)
        precision.to_excel(writer, sheet_name="precision_audit", index=False)
        recall.to_excel(writer, sheet_name="recall_audit", index=False)
        unmatched.to_excel(writer, sheet_name="unmatched_anchor_audit", index=False)
        registry.to_excel(writer, sheet_name="marker_registry", index=False)
        prevalence.to_excel(writer, sheet_name="marker_prevalence", index=False)
        status.to_excel(writer, sheet_name="data_status", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for column_cells in sheet.columns:
                letter = column_cells[0].column_letter
                header = str(column_cells[0].value or "")
                if header in {"full_response", "full_scenario", "match_context", "anchor_context"}:
                    width = 70
                elif header in {"instruction", "operational_definition", "exclusion_rule", "patterns"}:
                    width = 55
                else:
                    width = min(35, max(12, len(header) + 2))
                sheet.column_dimensions[letter].width = width
            for row in sheet.iter_rows():
                for cell in row:
                    alignment = copy(cell.alignment)
                    alignment.vertical = "top"
                    alignment.wrap_text = True
                    cell.alignment = alignment


def _write_results(
    results: dict[str, pd.DataFrame],
    config: RegionalCulturalAuditConfig,
    manifest: dict[str, object],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "scenario_lookup": "scenario_lookup.csv",
        "status_summary": "data_status_summary.csv",
        "surface_summary": "surface_summary.csv",
        "opening_summary": "opening_summary.csv",
        "polarity_summary": "polarity_summary.csv",
        "marker_prevalence": "marker_prevalence.csv",
        "precision_audit": "audit_precision_regional_hits.csv",
        "recall_audit": "audit_recall_responses.csv",
        "anchor_summary": "candidate_anchor_coverage.csv",
        "anchor_forms": "candidate_anchor_forms.csv",
        "unmatched_anchor_audit": "audit_unmatched_anchor_contexts.csv",
        "lexicon_coverage": "regional_lexicon_coverage.csv",
        "paired_all_text": "paired_all_visible_text.csv",
        "paired_committed_sensitivity": "paired_both_committed_sensitivity.csv",
    }
    for key, filename in filenames.items():
        table = results.get(key)
        if table is None or (table.empty and key.startswith("paired_")):
            continue
        table.to_csv(config.output_dir / filename, index=False, encoding="utf-8-sig")

    registry = pd.DataFrame(regional_cultural_registry_rows())
    lexicon = pd.DataFrame(regional_cultural_lexicon_rows())
    registry.to_csv(config.output_dir / "marker_registry.csv", index=False, encoding="utf-8-sig")
    lexicon.to_csv(
        config.output_dir / "regional_audit_lexicon.csv", index=False, encoding="utf-8-sig"
    )
    if config.save_response_level:
        response = results["response_level"]
        marker_columns = [column for column in response if "__" in column]
        keep = [
            "model_id",
            "persona",
            "identity",
            "condition",
            "condition_bucket",
            "question_id",
            "subset",
            "answer_status",
            "abstain_reason",
            "text_status",
            "text_available",
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "ttr_300",
            "opening_category",
            "moral_polarity_initial",
            "moral_polarity_final",
            "moral_polarity_shift",
            "moral_polarity_direction",
        ] + marker_columns
        response[keep].to_csv(
            config.output_dir / "response_level_markers.csv.gz",
            index=False,
            compression="gzip",
            encoding="utf-8",
        )

    _write_audit_workbook(
        config.output_dir / "audit_manual_regional_cultural.xlsx",
        manifest=manifest,
        precision=results["precision_audit"],
        recall=results["recall_audit"],
        unmatched=results["unmatched_anchor_audit"],
        registry=registry,
        prevalence=results["marker_prevalence"],
        status=results["status_summary"],
    )
    manifest["output_files"] = sorted(path.name for path in config.output_dir.iterdir())
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def finalize_existing_output(config: RegionalCulturalAuditConfig) -> dict[str, object]:
    """Finish a fully extracted output when only workbook/manifest writing failed.

    This recovery path never reruns extraction and never overwrites an existing
    workbook or manifest.  It validates the pinned registry version and the
    required CSV products before adding the two missing final artifacts.
    """

    workbook_path = config.output_dir / "audit_manual_regional_cultural.xlsx"
    manifest_path = config.output_dir / "manifest.json"
    if workbook_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing recovery because workbook or manifest already exists: "
            f"{config.output_dir}"
        )
    required = {
        "response": "response_level_markers.csv.gz",
        "scenario": "scenario_lookup.csv",
        "status": "data_status_summary.csv",
        "prevalence": "marker_prevalence.csv",
        "precision": "audit_precision_regional_hits.csv",
        "recall": "audit_recall_responses.csv",
        "unmatched": "audit_unmatched_anchor_contexts.csv",
        "registry": "marker_registry.csv",
        "lexicon": "regional_audit_lexicon.csv",
        "surface": "surface_summary.csv",
        "opening": "opening_summary.csv",
        "polarity": "polarity_summary.csv",
        "anchor_summary": "candidate_anchor_coverage.csv",
        "anchor_forms": "candidate_anchor_forms.csv",
        "lexicon_coverage": "regional_lexicon_coverage.csv",
    }
    missing = [name for name in required.values() if not (config.output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete extraction output; missing: {missing}")

    registry = pd.read_csv(config.output_dir / required["registry"])
    regional_rows = registry[registry["registry_scope"].eq("regional_cultural_specific")]
    versions = set(regional_rows["version"].dropna().astype(str))
    if versions != {REGIONAL_CULTURAL_REGISTRY_VERSION}:
        raise ValueError(
            "Registry version mismatch in partial output: "
            f"expected {REGIONAL_CULTURAL_REGISTRY_VERSION}, found {sorted(versions)}"
        )
    precision = pd.read_csv(config.output_dir / required["precision"])
    recall = pd.read_csv(config.output_dir / required["recall"])
    unmatched = pd.read_csv(config.output_dir / required["unmatched"])
    prevalence = pd.read_csv(config.output_dir / required["prevalence"])
    status = pd.read_csv(config.output_dir / required["status"])
    scenarios = pd.read_csv(
        config.output_dir / required["scenario"],
        usecols=["question_id", "scenario_available"],
    )
    response = pd.read_csv(
        config.output_dir / required["response"],
        usecols=[
            "model_id",
            "persona",
            "question_id",
            "answer_status",
            "text_status",
            "text_available",
        ],
    )
    if len(response) != len(
        response[["model_id", "persona", "question_id"]].drop_duplicates()
    ):
        raise ValueError("Response-level output contains duplicate response keys.")

    manifest: dict[str, object] = {
        "pipeline_version": REGIONAL_CULTURAL_REGISTRY_VERSION,
        "lexicon_version": REGIONAL_CULTURAL_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": _sha256(config.input_csv),
        "canonical_source_manifest": str(
            config.input_csv.parent / "all_models_results_full_manifest.json"
        ),
        "raw_ethics_root": str(config.raw_ethics_root),
        "output_dir": str(config.output_dir),
        "config": {
            **asdict(config),
            "repo_root": str(config.repo_root),
            "input_csv": str(config.input_csv),
            "raw_ethics_root": str(config.raw_ethics_root),
            "output_dir": str(config.output_dir),
        },
        "rows_input": len(response),
        "unique_response_keys": len(response),
        "responses_text_available": int(response["text_available"].fillna(False).sum()),
        "responses_abstain": int(response["answer_status"].eq("abstain").sum()),
        "responses_missing": int(response["text_status"].eq("missing").sum()),
        "models": int(response["model_id"].nunique()),
        "personas": int(response["persona"].nunique()),
        "unique_questions": int(response["question_id"].nunique()),
        "scenario_available": int(scenarios["scenario_available"].fillna(False).sum()),
        "marker_count_total": len(ETHICS_REGIONAL_CULTURAL_MARKERS),
        "marker_count_regional_cultural": len(REGIONAL_CULTURAL_MARKERS),
        "precision_audit_rows": len(precision),
        "recall_audit_rows": len(recall),
        "unmatched_anchor_audit_rows": len(unmatched),
        "inference_run": False,
        "registry_status": "candidate_pending_two_independent_human_reviews",
        "recovery_provenance": {
            "partial_output_reused": True,
            "reason": (
                "Full extraction and all CSV writes completed; the original run stopped only "
                "when its interpreter lacked openpyxl for the workbook step."
            ),
        },
        "interpretation_guardrails": [
            "The same extractor and registry are applied to every model and all 17 conditions.",
            "All visible abstain responses are included in primary extraction.",
            "Missing or unseparable response text is NA, never a zero-marker response.",
            "Hidden reasoning and ground truth are never used as fallback marker text.",
            "Raw count, response presence, and density per 100 words are reported separately.",
            "Regional labels and value orientations are separate marker families.",
            "Overlapping markers are never summed into a cultural composite score.",
            "Scenario overlap is reported and blocks automatic persona-uptake attribution.",
            "Named practice or concept hits are not automatically persona carryover.",
            "Response-stance and moral-polarity markers never alter parser answer_status.",
            "Inferential tests remain disabled until independent human audit promotion.",
        ],
    }
    _write_audit_workbook(
        workbook_path,
        manifest=manifest,
        precision=precision,
        recall=recall,
        unmatched=unmatched,
        registry=registry,
        prevalence=prevalence,
        status=status,
    )
    manifest["output_files"] = sorted(path.name for path in config.output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def run_regional_cultural_dataset_audit(
    config: RegionalCulturalAuditConfig,
) -> dict[str, object]:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {config.output_dir}"
        )
    print(f"[load] {config.input_csv}")
    source = load_canonical_ethics(config.input_csv)
    rows_input = len(source)
    unique_questions = int(source["question_id"].nunique())
    print(f"[load] {rows_input:,} unique response rows")
    scenarios = load_scenario_lookup(config.raw_ethics_root, source["question_id"])
    response = extract_response_markers(source, workers=config.extraction_workers)
    del source
    gc.collect()

    print("[summaries] status, surface, opening, polarity, prevalence")
    status = build_status_summary(response)
    surface = build_surface_summary(response)
    openings = build_opening_summary(response)
    polarity = build_polarity_summary(response)
    prevalence = build_marker_prevalence(response)
    print("[audit] balanced regional precision sample")
    precision = build_regional_precision_audit(
        response,
        scenarios,
        per_model_marker=config.precision_per_model_marker,
        random_seed=config.random_seed,
    )
    print("[audit] blind recall sample")
    recall = build_recall_audit_sample(
        response,
        sample_per_condition=config.recall_sample_per_condition,
        seed=config.random_seed,
    )
    print("[audit] open-anchor recall and scenario overlap")
    anchor_summary, anchor_forms, unmatched = build_candidate_anchor_audit(
        response,
        scenarios,
        per_model_category=config.unmatched_per_model_category,
        random_seed=config.random_seed,
    )
    lexicon_coverage = build_lexicon_coverage(response, scenarios)

    results: dict[str, pd.DataFrame] = {
        "response_level": response,
        "scenario_lookup": scenarios,
        "status_summary": status,
        "surface_summary": surface,
        "opening_summary": openings,
        "polarity_summary": polarity,
        "marker_prevalence": prevalence,
        "precision_audit": precision,
        "recall_audit": recall,
        "anchor_summary": anchor_summary,
        "anchor_forms": anchor_forms,
        "unmatched_anchor_audit": unmatched,
        "lexicon_coverage": lexicon_coverage,
    }
    if config.run_inference:
        results["paired_all_text"] = build_paired_marker_tests(
            response,
            bootstrap_resamples=config.bootstrap_resamples,
            require_both_committed=False,
        )
        results["paired_committed_sensitivity"] = build_paired_marker_tests(
            response,
            bootstrap_resamples=config.bootstrap_resamples,
            require_both_committed=True,
        )

    manifest: dict[str, object] = {
        "pipeline_version": REGIONAL_CULTURAL_REGISTRY_VERSION,
        "lexicon_version": REGIONAL_CULTURAL_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": _sha256(config.input_csv),
        "canonical_source_manifest": str(
            config.input_csv.parent / "all_models_results_full_manifest.json"
        ),
        "raw_ethics_root": str(config.raw_ethics_root),
        "output_dir": str(config.output_dir),
        "config": {
            **asdict(config),
            "repo_root": str(config.repo_root),
            "input_csv": str(config.input_csv),
            "raw_ethics_root": str(config.raw_ethics_root),
            "output_dir": str(config.output_dir),
        },
        "rows_input": rows_input,
        "unique_response_keys": int(
            response[["model_id", "persona", "question_id"]].drop_duplicates().shape[0]
        ),
        "responses_text_available": int(response["text_available"].fillna(False).sum()),
        "responses_abstain": int(response["answer_status"].eq("abstain").sum()),
        "responses_missing": int(response["text_status"].eq("missing").sum()),
        "models": int(response["model_id"].nunique()),
        "personas": int(response["persona"].nunique()),
        "unique_questions": unique_questions,
        "scenario_available": int(scenarios["scenario_available"].sum()),
        "marker_count_total": len(ETHICS_REGIONAL_CULTURAL_MARKERS),
        "marker_count_regional_cultural": len(REGIONAL_CULTURAL_MARKERS),
        "precision_audit_rows": len(precision),
        "recall_audit_rows": len(recall),
        "unmatched_anchor_audit_rows": len(unmatched),
        "inference_run": config.run_inference,
        "registry_status": "candidate_pending_two_independent_human_reviews",
        "interpretation_guardrails": [
            "The same extractor and registry are applied to every model and all 17 conditions.",
            "All visible abstain responses are included in primary extraction.",
            "Missing or unseparable response text is NA, never a zero-marker response.",
            "Hidden reasoning and ground truth are never used as fallback marker text.",
            "Raw count, response presence, and density per 100 words are reported separately.",
            "Regional labels and value orientations are separate marker families.",
            "Overlapping markers are never summed into a cultural composite score.",
            "Scenario overlap is reported and blocks automatic persona-uptake attribution.",
            "A named practice or concept hit is not automatically persona carryover.",
            "Response-stance and moral-polarity markers never alter parser answer_status.",
            "Inferential tests remain disabled until independent human audit promotion.",
        ],
    }
    print(f"[write] {config.output_dir}")
    _write_results(results, config, manifest)
    print(
        "[done] "
        f"rows={rows_input:,} available={manifest['responses_text_available']:,} "
        f"markers={len(ETHICS_REGIONAL_CULTURAL_MARKERS)}"
    )
    return {"tables": results, "manifest": manifest, "config": config}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    defaults = default_audit_config(repo_root)
    parser = argparse.ArgumentParser(
        description="Extract and audit ETHICS regional-cultural linguistic markers."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--raw-ethics-root", type=Path, default=defaults.raw_ethics_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--precision-per-model-marker", type=int, default=10)
    parser.add_argument("--recall-sample-per-condition", type=int, default=3)
    parser.add_argument("--unmatched-per-model-category", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--no-response-level", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate existing CSV products and add a missing workbook/manifest only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = default_audit_config(args.repo_root)
    config = replace(
        base,
        repo_root=args.repo_root.resolve(),
        input_csv=args.input_csv.resolve(),
        raw_ethics_root=args.raw_ethics_root.resolve(),
        output_dir=args.output_dir.resolve(),
        random_seed=args.random_seed,
        precision_per_model_marker=max(1, args.precision_per_model_marker),
        recall_sample_per_condition=max(0, args.recall_sample_per_condition),
        unmatched_per_model_category=max(0, args.unmatched_per_model_category),
        save_response_level=not args.no_response_level,
        run_inference=args.run_inference,
        bootstrap_resamples=args.bootstrap_resamples,
        extraction_workers=max(1, args.extraction_workers),
    )
    if args.finalize_existing:
        finalize_existing_output(config)
    else:
        run_regional_cultural_dataset_audit(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

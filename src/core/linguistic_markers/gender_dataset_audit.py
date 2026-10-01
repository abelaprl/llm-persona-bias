"""Dataset-grounded audit runner for the ETHICS gender linguistic markers.

This module is the executable core used by the companion notebook.  It keeps
all visible abstain responses, maps missing text to NA through the shared text
surface, and compares response gender terms with each question's raw scenario.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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

from .audit_sampler import build_audit_sample, build_recall_audit_sample
from .config import IDENTITY_CONDITIONS, PLANNED_CONTRASTS, RANDOM_SEED
from .ethics_registry import moral_polarity_sequence
from .extractor import extract_markers, find_marker_matches, marker_text
from .gender_registry import (
    ETHICS_GENDER_MARKERS,
    GENDER_LEXICON_VERSION,
    GENDER_MARKERS,
    GENDER_REGISTRY_VERSION,
    GENDER_TERM_CATEGORIES,
    compile_gender_term_pattern,
    gender_lexicon_rows,
    gender_registry_rows,
)
from .loaders import _persona_metadata
from .registry import MarkerSpec
from .statistics import _bootstrap_ci, _mcnemar_exact, _paired_numeric_test, _stable_seed, holm_adjust
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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)?", re.IGNORECASE)
_IDENTITY_ANCHOR_RE = re.compile(
    r"\b(?:man|men|woman|women|males?|females?|gender(?:\s+identity)?|"
    r"masculin(?:e|ity)|feminin(?:e|ity)|non-?binary|transgender)\b",
    re.IGNORECASE,
)
_CANDIDATE_CUE_RE = re.compile(
    r"\b(?:you|your|user|persona|identity|perspective|being|since|because|given|"
    r"regardless|irrespective|irrelevant|roles?|norms?|expectations?|stereotyp\w*)\b|"
    r"\bas\s+(?:a|an)\b",
    re.IGNORECASE,
)
_OPEN_GENDER_FORM_RE = re.compile(
    r"\b(?:man|men|woman|women|males?|females?|manhood|womanhood|manly|unmanly|"
    r"gender(?:ed|less)?(?:-[a-z]+)?|non-?binary|transgender\w*|cisgender\w*|"
    r"stereotyp\w*|masculin\w*|feminin\w*|patriarch\w*|misogyn\w*|"
    r"sexism|sexist\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GenderAuditConfig:
    repo_root: Path
    input_csv: Path
    raw_ethics_root: Path
    output_dir: Path
    random_seed: int = RANDOM_SEED
    audit_sample_size: int = 240
    marker_audit_sample_size: int = 100
    recall_sample_per_condition: int = 3
    save_response_level: bool = True
    run_inference: bool = False
    bootstrap_resamples: int = 2_000
    extraction_workers: int = 1


def default_audit_config(repo_root: Path) -> GenderAuditConfig:
    repo_root = Path(repo_root).resolve()
    return GenderAuditConfig(
        repo_root=repo_root,
        input_csv=(
            repo_root
            / "result"
            / "ETHICS"
            / "reports"
            / "Gender_Persona"
            / "_combined"
            / "parser_refinement_20260820"
            / "all_models_results_full.csv"
        ),
        raw_ethics_root=repo_root / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]",
        output_dir=(
            repo_root
            / "result"
            / "ETHICS"
            / "reports"
            / "Gender_Persona"
            / "linguistic_markers_gender_candidate_20260821_v1_6_0"
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
    if "_explicit_" in persona:
        return "explicit"
    return "implicit"


def load_refined_ethics(path: Path) -> pd.DataFrame:
    """Load the canonical refined CSV without filtering on answer status."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(REQUIRED_COLUMNS).difference(header)
    if missing:
        raise ValueError(f"Refined ETHICS CSV is missing columns: {sorted(missing)}")
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
    frame["condition_bucket"] = frame["persona_folder"].map(_condition_bucket)
    return frame


def _scenario_directories(raw_root: Path) -> list[Path]:
    directories: list[Path] = []
    for collection in ("Neutral Persona", "Gender Persona"):
        collection_root = raw_root / collection
        if not collection_root.is_dir():
            continue
        for model_dir in sorted(path for path in collection_root.iterdir() if path.is_dir()):
            for persona_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                directories.append(persona_dir)
    return directories


def load_scenario_lookup(
    raw_root: Path,
    question_ids: Iterable[str],
) -> pd.DataFrame:
    """Read one raw JSON per question; never read hidden model reasoning."""

    directories = _scenario_directories(raw_root)
    rows: list[dict[str, str]] = []
    unique_question_ids = sorted(set(str(value) for value in question_ids))
    total = len(unique_question_ids)
    for position, question_id in enumerate(unique_question_ids, start=1):
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
        if position % 250 == 0 or position == total:
            print(f"[scenario] {position:,}/{total:,} questions")
    return pd.DataFrame(rows)


def _extract_response_record(record: dict[str, object]) -> dict[str, object]:
    """Extract one response in a process-safe top-level worker."""

    extracted = extract_markers(
        record,
        markers=ETHICS_GENDER_MARKERS,
        pipeline_version=GENDER_REGISTRY_VERSION,
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


def _extract_response_batch(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_extract_response_record(record) for record in records]


def _iter_extraction_batches(
    frame: pd.DataFrame,
    *,
    batch_size: int = 100,
) -> Iterable[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for record_index, row in enumerate(frame.itertuples(index=False), start=1):
        identity, condition = _persona_metadata(row.persona_folder)
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


def extract_response_markers(
    frame: pd.DataFrame,
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Extract general ETHICS and gender-specific markers from visible content."""

    # Convert records to compact DataFrames in bounded chunks.  Keeping all
    # extracted dictionaries alive until the final response would exceed the
    # memory limit on the 81,756-row ETHICS corpus because every Python scalar
    # remains an individual object.
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
                    chunks.append(pd.DataFrame.from_records(rows))
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
                    chunks.append(pd.DataFrame.from_records(rows))
                    rows.clear()
                    gc.collect()
                if processed % 5_000 == 0 or processed == total:
                    print(f"[extract] {processed:,}/{total:,} responses")
    if rows:
        chunks.append(pd.DataFrame.from_records(rows))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


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
    """Summarize conservative first-to-final moral-polarity shifts."""

    available = frame[frame["text_available"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    group_columns = ["model_id", "persona", "subset", "answer_status"]
    for keys, group in available.groupby(group_columns, dropna=False, sort=True):
        shift = pd.to_numeric(group["moral_polarity_shift"], errors="coerce")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
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
    markers: Iterable[MarkerSpec] = ETHICS_GENDER_MARKERS,
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


def _context_excerpt(text: str, start: int, end: int, radius: int = 150) -> str:
    excerpt = text[max(0, start - radius) : min(len(text), end + radius)]
    return re.sub(r"\s+", " ", excerpt).strip()


def build_gender_hit_audit(
    frame: pd.DataFrame,
    scenario_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_map = scenario_lookup.set_index("question_id")["scenario"].to_dict()
    term_pattern = compile_gender_term_pattern()
    rows: list[dict[str, object]] = []
    total = len(frame)
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if not bool(row.text_available):
            continue
        positive_markers = [
            marker
            for marker in GENDER_MARKERS
            if getattr(row, f"{marker.marker_id}__present") == 1
        ]
        if not positive_markers:
            continue
        surface = prepare_text_surface(row.visible_text, benchmark="ethics")
        scenario = str(scenario_map.get(row.question_id, ""))
        scenario_terms = {match.group(0).casefold() for match in term_pattern.finditer(scenario)}
        for marker in positive_markers:
            text = marker_text(surface, marker)
            for match in find_marker_matches(text, marker):
                matched_terms = {
                    item.group(0).casefold() for item in term_pattern.finditer(match.text)
                }
                rows.append(
                    {
                        "response_key": f"{row.model_id}|{row.persona}|{row.question_id}",
                        "model_id": row.model_id,
                        "persona": row.persona,
                        "condition": row.condition,
                        "question_id": row.question_id,
                        "subset": row.subset,
                        "answer_status": row.answer_status,
                        "abstain_reason": row.abstain_reason,
                        "marker_id": marker.marker_id,
                        "matched_text": match.text,
                        "context_excerpt": _context_excerpt(text, match.start, match.end),
                        "matched_gender_terms": ";".join(sorted(matched_terms)),
                        "same_gender_term_in_scenario": bool(matched_terms & scenario_terms),
                    }
                )
        if position % 10_000 == 0 or position == total:
            print(f"[gender-hit-audit] {position:,}/{total:,} responses")
    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit, pd.DataFrame()
    inventory = (
        audit.assign(matched_text_normalized=audit["matched_text"].str.casefold())
        .groupby(
            ["marker_id", "matched_text_normalized", "same_gender_term_in_scenario"],
            dropna=False,
            sort=True,
        )
        .agg(
            occurrences=("matched_text", "size"),
            responses=("response_key", "nunique"),
            example=("context_excerpt", "first"),
        )
        .reset_index()
        .sort_values(["marker_id", "occurrences"], ascending=[True, False])
    )
    return audit, inventory


def build_lexicon_coverage(
    frame: pd.DataFrame,
    scenario_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_pattern = compile_gender_term_pattern()
    scenario_map = scenario_lookup.set_index("question_id")["scenario"].to_dict()
    scenario_terms = {
        question_id: {match.group(0).casefold() for match in term_pattern.finditer(str(text))}
        for question_id, text in scenario_map.items()
    }
    occurrences: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    bucket_documents: Counter[tuple[str, str]] = Counter()
    status_documents: Counter[tuple[str, str]] = Counter()
    scenario_overlap_documents: Counter[str] = Counter()
    open_occurrences: Counter[str] = Counter()
    open_documents: Counter[str] = Counter()

    total = len(frame)
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if not bool(row.text_available):
            continue
        text = str(row.visible_text)
        hits = [match.group(0).casefold() for match in term_pattern.finditer(text)]
        unique_hits = set(hits)
        occurrences.update(hits)
        documents.update(unique_hits)
        for term in unique_hits:
            bucket_documents[(term, row.condition_bucket)] += 1
            status_documents[(term, str(row.answer_status or ""))] += 1
            if term in scenario_terms.get(row.question_id, set()):
                scenario_overlap_documents[term] += 1
        open_hits = [match.group(0).casefold() for match in _OPEN_GENDER_FORM_RE.finditer(text)]
        open_occurrences.update(open_hits)
        open_documents.update(set(open_hits))
        if position % 10_000 == 0 or position == total:
            print(f"[lexicon-coverage] {position:,}/{total:,} responses")

    coverage_rows: list[dict[str, object]] = []
    for lexicon_row in gender_lexicon_rows():
        term = lexicon_row["term"]
        coverage_rows.append(
            {
                **lexicon_row,
                "response_occurrences": occurrences[term],
                "responses_with_term": documents[term],
                "explicit_responses_with_term": bucket_documents[(term, "explicit")],
                "implicit_responses_with_term": bucket_documents[(term, "implicit")],
                "neutral_responses_with_term": bucket_documents[(term, "neutral")],
                "committed_responses_with_term": status_documents[(term, "committed")],
                "abstain_responses_with_term": status_documents[(term, "abstain")],
                "responses_same_term_in_scenario": scenario_overlap_documents[term],
                "observed_in_response_corpus": bool(documents[term]),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["responses_with_term", "response_occurrences", "term"],
        ascending=[False, False, True],
    )
    closed_terms = set(GENDER_TERM_CATEGORIES)
    open_rows = [
        {
            "form": term,
            "response_occurrences": count,
            "responses_with_form": open_documents[term],
            "in_closed_lexicon": term in closed_terms,
            "review_status": (
                "registered" if term in closed_terms else "candidate_requires_manual_review"
            ),
        }
        for term, count in open_occurrences.most_common()
    ]
    return coverage, pd.DataFrame(open_rows)


def _identity_terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _IDENTITY_ANCHOR_RE.finditer(text)}


def build_candidate_context_audit(
    frame: pd.DataFrame,
    scenario_lookup: pd.DataFrame,
    *,
    random_seed: int,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit broad identity contexts independently of the registered patterns.

    The broad candidate rule is intentionally recall-oriented.  Its capture
    percentage is a diagnostic, not an optimization target: many unmatched
    contexts are legitimate third-person scenario descriptions.
    """

    scenario_map = scenario_lookup.set_index("question_id")["scenario"].to_dict()
    rows: list[dict[str, object]] = []
    total = len(frame)
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if not bool(row.text_available):
            continue
        scenario_identity_terms = _identity_terms(str(scenario_map.get(row.question_id, "")))
        for sentence_index, sentence in enumerate(_SENTENCE_SPLIT_RE.split(str(row.visible_text))):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence or not _IDENTITY_ANCHOR_RE.search(sentence):
                continue
            if not _CANDIDATE_CUE_RE.search(sentence):
                continue
            matched = [
                marker.marker_id
                for marker in GENDER_MARKERS
                if find_marker_matches(sentence, marker)
            ]
            terms = _identity_terms(sentence)
            rows.append(
                {
                    "model_id": row.model_id,
                    "persona": row.persona,
                    "condition": row.condition,
                    "condition_bucket": row.condition_bucket,
                    "question_id": row.question_id,
                    "subset": row.subset,
                    "answer_status": row.answer_status,
                    "abstain_reason": row.abstain_reason,
                    "sentence_index": sentence_index,
                    "candidate_context": sentence,
                    "identity_terms": ";".join(sorted(terms)),
                    "captured_any_gender_marker": bool(matched),
                    "captured_marker_ids": ";".join(matched),
                    "same_identity_term_in_scenario": bool(terms & scenario_identity_terms),
                }
            )
        if position % 10_000 == 0 or position == total:
            print(f"[candidate-contexts] {position:,}/{total:,} responses")
    contexts = pd.DataFrame(rows)
    if contexts.empty:
        return contexts, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    group_specs = [
        ("overall", ["answer_status"]),
        ("condition", ["condition_bucket", "answer_status"]),
        ("persona", ["persona", "answer_status"]),
    ]
    for scope, columns in group_specs:
        for keys, group in contexts.groupby(columns, dropna=False, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            summary_rows.append(
                {
                    "scope": scope,
                    **dict(zip(columns, keys)),
                    "candidate_contexts": len(group),
                    "responses": group[["model_id", "persona", "question_id"]]
                    .drop_duplicates()
                    .shape[0],
                    "captured_contexts": int(group["captured_any_gender_marker"].sum()),
                    "diagnostic_capture_pct": 100
                    * float(group["captured_any_gender_marker"].mean()),
                    "scenario_overlap_contexts": int(group["same_identity_term_in_scenario"].sum()),
                    "scenario_overlap_pct": 100
                    * float(group["same_identity_term_in_scenario"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    unmatched = contexts[~contexts["captured_any_gender_marker"]].copy()
    if len(unmatched) > sample_size:
        # Preserve condition/status diversity before filling remaining slots.
        per_group = max(1, sample_size // max(1, unmatched.groupby(
            ["condition_bucket", "answer_status"], dropna=False
        ).ngroups))
        pieces = [
            group.sample(min(len(group), per_group), random_state=random_seed)
            for _, group in unmatched.groupby(
                ["condition_bucket", "answer_status"], dropna=False, sort=True
            )
        ]
        audit_sample = pd.concat(pieces)
        if len(audit_sample) < sample_size:
            remaining = unmatched.drop(index=audit_sample.index)
            if not remaining.empty:
                audit_sample = pd.concat(
                    [
                        audit_sample,
                        remaining.sample(
                            min(len(remaining), sample_size - len(audit_sample)),
                            random_state=random_seed + 1,
                        ),
                    ],
                )
        audit_sample = audit_sample.reset_index(drop=True)
    else:
        audit_sample = unmatched

    phrase_counter: Counter[tuple[str, bool]] = Counter()
    for row in contexts.itertuples(index=False):
        tokens = [token.casefold() for token in _TOKEN_RE.findall(row.candidate_context)]
        for n in (2, 3, 4, 5):
            for start in range(max(0, len(tokens) - n + 1)):
                gram_tokens = tokens[start : start + n]
                phrase = " ".join(gram_tokens)
                if _IDENTITY_ANCHOR_RE.search(phrase):
                    phrase_counter[(phrase, bool(row.captured_any_gender_marker))] += 1
    phrase_rows = [
        {"phrase": phrase, "captured_by_registered_marker": captured, "occurrences": count}
        for (phrase, captured), count in phrase_counter.most_common()
    ]
    phrase_inventory = pd.DataFrame(phrase_rows)
    if not phrase_inventory.empty:
        phrase_inventory = (
            phrase_inventory.sort_values("occurrences", ascending=False)
            .groupby("captured_by_registered_marker", group_keys=False)
            .head(500)
            .reset_index(drop=True)
        )
    return contexts, summary, audit_sample, phrase_inventory


def build_paired_marker_tests(
    frame: pd.DataFrame,
    *,
    markers: Iterable[MarkerSpec] = ETHICS_GENDER_MARKERS,
    bootstrap_resamples: int = 2_000,
    require_both_committed: bool = False,
) -> pd.DataFrame:
    """Paired question-level inference for candidate markers.

    This is disabled in the notebook by default until the manual precision and
    recall audits have been reviewed.  When enabled, it supports both the full
    visible-text analysis and committed--committed sensitivity analysis.
    """

    outcomes: list[dict[str, str]] = [
        {"outcome_id": "word_count", "value_col": "word_count", "scale": "numeric"},
        {
            "outcome_id": "avg_sentence_length",
            "value_col": "avg_sentence_length",
            "scale": "numeric",
        },
        {"outcome_id": "ttr_300", "value_col": "ttr_300", "scale": "numeric"},
    ]
    for marker in markers:
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
                            GENDER_REGISTRY_VERSION,
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
                                "both_committed" if require_both_committed else "all_visible_text"
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


def _write_results(
    results: dict[str, pd.DataFrame],
    config: GenderAuditConfig,
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
        "marker_audit_sample": "audit_marker_hits.csv",
        "recall_audit_sample": "audit_recall_responses.csv",
        "gender_hit_audit": "gender_marker_hit_audit.csv",
        "gender_phrase_inventory": "gender_marker_phrase_inventory.csv",
        "lexicon_coverage": "gender_lexicon_coverage.csv",
        "open_gender_forms": "open_gender_forms.csv",
        "candidate_contexts": "gender_candidate_contexts.csv",
        "candidate_context_summary": "gender_candidate_context_coverage.csv",
        "unmatched_context_audit": "gender_unmatched_context_audit.csv",
        "candidate_phrase_inventory": "gender_candidate_phrase_inventory.csv",
        "paired_all_text": "paired_all_visible_text.csv",
        "paired_committed_sensitivity": "paired_both_committed_sensitivity.csv",
    }
    for key, filename in filenames.items():
        table = results.get(key)
        if table is None or (table.empty and key.startswith("paired_")):
            continue
        table.to_csv(config.output_dir / filename, index=False, encoding="utf-8-sig")

    pd.DataFrame(gender_registry_rows()).to_csv(
        config.output_dir / "marker_registry.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(gender_lexicon_rows()).to_csv(
        config.output_dir / "gender_audit_lexicon.csv", index=False, encoding="utf-8-sig"
    )
    if config.save_response_level:
        response = results["response_level"]
        marker_columns = [column for column in response if "__" in column]
        keep = [
            "model_id",
            "persona",
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
    manifest["output_files"] = sorted(path.name for path in config.output_dir.iterdir())
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_gender_dataset_audit(config: GenderAuditConfig) -> dict[str, object]:
    """Run extraction, coverage checks, audit sampling, and optional inference."""

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {config.output_dir}"
        )

    print(f"[load] {config.input_csv}")
    source = load_refined_ethics(config.input_csv)
    print(f"[load] {len(source):,} unique response rows")
    rows_input = len(source)
    unique_questions = int(source["question_id"].nunique())
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
    print("[audit] marker precision sample")
    marker_audit = build_audit_sample(
        response,
        sample_per_marker=config.marker_audit_sample_size,
        seed=config.random_seed,
        markers=ETHICS_GENDER_MARKERS,
    )
    print("[audit] blind recall sample")
    recall_audit = build_recall_audit_sample(
        response,
        sample_per_condition=config.recall_sample_per_condition,
        seed=config.random_seed,
    )
    hit_audit, phrase_inventory = build_gender_hit_audit(response, scenarios)
    lexicon_coverage, open_forms = build_lexicon_coverage(response, scenarios)
    contexts, context_summary, unmatched, candidate_phrases = build_candidate_context_audit(
        response,
        scenarios,
        random_seed=config.random_seed,
        sample_size=config.audit_sample_size,
    )

    results: dict[str, pd.DataFrame] = {
        "response_level": response,
        "scenario_lookup": scenarios,
        "status_summary": status,
        "surface_summary": surface,
        "opening_summary": openings,
        "polarity_summary": polarity,
        "marker_prevalence": prevalence,
        "marker_audit_sample": marker_audit,
        "recall_audit_sample": recall_audit,
        "gender_hit_audit": hit_audit,
        "gender_phrase_inventory": phrase_inventory,
        "lexicon_coverage": lexicon_coverage,
        "open_gender_forms": open_forms,
        "candidate_contexts": contexts,
        "candidate_context_summary": context_summary,
        "unmatched_context_audit": unmatched,
        "candidate_phrase_inventory": candidate_phrases,
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
        "pipeline_version": GENDER_REGISTRY_VERSION,
        "lexicon_version": GENDER_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": _sha256(config.input_csv),
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
        "responses_text_available": int(response["text_available"].fillna(False).sum()),
        "responses_abstain": int(response["answer_status"].eq("abstain").sum()),
        "responses_missing": int(response["text_status"].eq("missing").sum()),
        "marker_count": len(ETHICS_GENDER_MARKERS),
        "marker_audit_rows": len(marker_audit),
        "recall_audit_rows": len(recall_audit),
        "moral_polarity_shifts": int(
            pd.to_numeric(response["moral_polarity_shift"], errors="coerce").fillna(0).sum()
        ),
        "unique_questions": unique_questions,
        "scenario_available": int(scenarios["scenario_available"].sum()),
        "gender_candidate_contexts": len(contexts),
        "gender_marker_hits": len(hit_audit),
        "inference_run": config.run_inference,
        "interpretation_guardrails": [
            "All visible abstain responses are included in the primary extraction.",
            "Missing or unseparable text is NA, never a zero-marker response.",
            "Hidden reasoning is never used as fallback text.",
            "Pronouns and gender nouns are audit anchors, not automatic persona uptake.",
            "Scenario same-term overlap is reported for manual review.",
            "Style, response stance, content framing, and gender uptake are separate families.",
            "Overlapping marker families are never summed into a composite score.",
            "Response-stance and moral-polarity markers never alter parser answer_status.",
            "Candidate-context capture is diagnostic and is not optimized to 100 percent.",
            "Inferential tests remain disabled until manual precision/recall audit is approved.",
        ],
    }
    print(f"[write] {config.output_dir}")
    _write_results(results, config, manifest)
    print(
        "[done] "
        f"rows={rows_input:,} available={manifest['responses_text_available']:,} "
        f"markers={len(ETHICS_GENDER_MARKERS)}"
    )
    return {"tables": results, "manifest": manifest, "config": config}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    defaults = default_audit_config(repo_root)
    parser = argparse.ArgumentParser(
        description="Extract and audit ETHICS gender linguistic markers over the full CSV."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--raw-ethics-root", type=Path, default=defaults.raw_ethics_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--audit-sample-size", type=int, default=240)
    parser.add_argument("--marker-audit-sample-size", type=int, default=100)
    parser.add_argument("--recall-sample-per-condition", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--no-response-level", action="store_true")
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
        audit_sample_size=args.audit_sample_size,
        marker_audit_sample_size=args.marker_audit_sample_size,
        recall_sample_per_condition=args.recall_sample_per_condition,
        save_response_level=not args.no_response_level,
        run_inference=args.run_inference,
        bootstrap_resamples=args.bootstrap_resamples,
        extraction_workers=max(1, args.extraction_workers),
    )
    run_gender_dataset_audit(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

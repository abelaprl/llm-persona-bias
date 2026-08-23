"""Full-corpus extraction and audit runner for ETHICS Religion Persona.

The runner is descriptive by default.  It applies one registry to all models
and conditions, includes visible abstentions, maps unavailable response text to
NA, reports scenario overlap, and keeps human-review fields blank.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .audit_sampler import build_recall_audit_sample
from .config import RANDOM_SEED
from .ethics_registry import moral_polarity_sequence
from .ethics_religion_registry import (
    ETHICS_RELIGION_LEXICON_VERSION,
    ETHICS_RELIGION_MARKERS,
    ETHICS_RELIGION_REGISTRY_VERSION,
    RELIGION_AUDIT_LEXICON,
    RELIGION_MARKERS,
    compile_religion_audit_lexicon_pattern,
    religion_lexicon_rows,
    religion_registry_rows,
)
from .extractor import extract_markers, find_marker_matches, marker_text
from .regional_cultural_dataset_audit import (
    _balanced_indices,
    _context_excerpt,
    _normalize_for_overlap,
    build_marker_prevalence,
    build_opening_summary,
    build_polarity_summary,
    build_status_summary,
    build_surface_summary,
    load_scenario_lookup,
)
from .religion_config import IDENTITY_CONDITIONS
from .religion_csv_loaders import religion_persona_metadata
from .registry import MarkerSpec
from .statistics import _stable_seed
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

_OPEN_ANCHORS: dict[str, re.Pattern[str]] = {
    "religious_identity": re.compile(
        r"\b(?:muslims?|islam(?:ic)?|christians?|christianity|jews?|jewish|judaism|"
        r"judaic|atheists?|atheism|atheistic|agnostics?|agnosticism|"
        r"non[- ]?religious|secularism)\b",
        re.IGNORECASE,
    ),
    "religious_meta": re.compile(
        r"\b(?:religions?|religious(?:ly)?|religiosity|faith(?:ful|fulness)?|"
        r"spiritual(?:ity|ly)?|sacred|holy|beliefs?)\b",
        re.IGNORECASE,
    ),
    "deity_figure": re.compile(
        r"\b(?:allah|god|gods|lord|creator|almighty|deit(?:y|ies)|divine|yahweh|"
        r"adonai|jesus|christ|muhammad|prophet|moses|abraham)\b",
        re.IGNORECASE,
    ),
    "scripture_authority": re.compile(
        r"\b(?:qur['’]?an|koran|hadiths?|bible|biblical|scriptures?|torah|talmud(?:ic)?|"
        r"gospels?|surah|sunnah|verses?|commandments?|qur['’]?anic|koranic)\b",
        re.IGNORECASE,
    ),
    "practice_community": re.compile(
        r"\b(?:pray\w*|worship\w*|ramadan|eid|jumu['’]?ah|salah|salat|shalat|"
        r"shabbat|sabbath|yom\s+kippur|kaddish|shema|lent|lenten|communion|"
        r"halal|haram|kosher|zakat|sadaqah|sharia|mitzv\w*|mosques?|masjids?|"
        r"synagogues?|church(?:es)?|congregations?|christmas|easter|fasting)\b",
        re.IGNORECASE,
    ),
    "doctrine": re.compile(
        r"\b(?:sins?|sinful|salvation|afterlife|heaven|hell|paradise|souls?|"
        r"damnation|redemption|eternal\s+life)\b",
        re.IGNORECASE,
    ),
    "named_concept": re.compile(
        r"\b(?:tawh(?:id|eed)|shirk|adab|ihsan|ummah|fitrah|akhlaq|maqasid|sabr|"
        r"taqwa|pikuach\s+nefesh|tikkun\s+olam|(?:ka|ke)vod\s+ha[- ]?bri\w*|"
        r"lashon\s+hara|yetzer\s+hara|tzedakah|halach\w*|halakh\w*|tzniut|"
        r"teshuvah|chesed|shalom\s+bayit|imago\s+dei|agape)\b",
        re.IGNORECASE,
    ),
    "formula": re.compile(
        r"\b(?:assalamu\s+alaikum|in\s*sha['’ -]?allah|masha['’ -]?allah|"
        r"alhamdulillah|bismillah|subhanallah|god\s+willing|hallelujah|amen|"
        r"shabbat\s+shalom|shalom|chag\s+sameach|mazel\s+tov|baruch\s+hashem)\b",
        re.IGNORECASE,
    ),
    "nonreligious_framing": re.compile(
        r"\b(?:secular\s+(?:ethics?|morality|moral(?:ity)?|humanism|humanist|"
        r"perspective|framework|values?|reasoning)|humanis(?:m|t|tic)|"
        r"non[- ]?religious\s+(?:ethics?|morality|perspective|framework|values?))\b",
        re.IGNORECASE,
    ),
}

_ANCHOR_CAPTURE_MARKERS: dict[str, frozenset[str]] = {
    "religious_identity": frozenset(
        {
            "religion_identity_islam",
            "religion_identity_christian",
            "religion_identity_jewish",
            "religion_identity_atheist",
            "religion_persona_direct_address",
            "religion_perspective_adoption",
            "religion_identity_irrelevance",
        }
    ),
    "religious_meta": frozenset(
        {
            "religion_generic_reference",
            "religion_persona_direct_address",
            "religion_perspective_adoption",
            "religion_pluralist_framing",
            "religion_identity_irrelevance",
        }
    ),
    "deity_figure": frozenset(
        {
            "religion_deity_invocation",
            "religion_figure_reference",
            "religion_moral_authority_framing",
            "religion_blessing_invocation",
            "religion_formulaic_expression",
        }
    ),
    "scripture_authority": frozenset(
        {
            "religion_scripture_reference",
            "religion_moral_authority_framing",
            "religion_perspective_adoption",
        }
    ),
    "practice_community": frozenset({"religion_practice_community"}),
    "doctrine": frozenset({"religion_afterlife_soteriology"}),
    "named_concept": frozenset({"religion_named_concept_reference"}),
    "formula": frozenset(
        {"religion_formulaic_expression", "religion_blessing_invocation"}
    ),
    "nonreligious_framing": frozenset(
        {"religion_secular_humanist_framing", "religion_perspective_adoption"}
    ),
}


@dataclass(frozen=True)
class ReligionEthicsAuditConfig:
    repo_root: Path
    input_csv: Path
    raw_ethics_root: Path
    output_dir: Path
    random_seed: int = RANDOM_SEED
    precision_per_model_marker: int = 17
    recall_sample_per_condition: int = 3
    unmatched_per_model_category: int = 10
    save_response_level: bool = True
    extraction_workers: int = 1


def default_audit_config(repo_root: Path) -> ReligionEthicsAuditConfig:
    root = Path(repo_root).resolve()
    return ReligionEthicsAuditConfig(
        repo_root=root,
        input_csv=(
            root
            / "result"
            / "ETHICS"
            / "reports"
            / "Religion_Persona"
            / "_combined"
            / "all_models_results_full.csv"
        ),
        raw_ethics_root=root / "result" / "ETHICS" / "v9 [THIS IS THE MAIN]",
        output_dir=(
            root
            / "result"
            / "ETHICS"
            / "reports"
            / "Religion_Persona"
            / "linguistic_markers_religion_candidate_20260823_v1_1_0"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_openpyxl() -> None:
    if importlib.util.find_spec("openpyxl") is None:
        raise ModuleNotFoundError(
            "openpyxl is required to create audit_manual_religion.xlsx. "
            "Install it in the selected Python environment or run this module "
            "with an environment that already provides openpyxl."
        )


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
    raise ValueError(f"Unexpected religion persona: {persona}")


def load_canonical_ethics(path: Path) -> pd.DataFrame:
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
        raise ValueError("Duplicate response key in canonical Religion Persona CSV.")
    expected = {
        persona
        for conditions in IDENTITY_CONDITIONS.values()
        for persona in conditions.values()
    }
    unknown = set(frame["persona_folder"]).difference(expected)
    if unknown:
        raise ValueError(f"Unexpected persona folders: {sorted(unknown)}")
    frame["condition_bucket"] = frame["persona_folder"].map(_condition_bucket)
    return frame


def _extract_response_record(record: dict[str, object]) -> dict[str, object]:
    extracted = extract_markers(
        record,
        markers=ETHICS_RELIGION_MARKERS,
        pipeline_version=ETHICS_RELIGION_REGISTRY_VERSION,
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
        identity, condition = religion_persona_metadata(row.persona_folder)
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


def _compact_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = {
        "source_row",
        "reasoning_tokens",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "ttr_300",
        "technical_lines_stripped",
        "moral_polarity_shift",
    }
    for column in frame.columns:
        if column in numeric or column.endswith(("__count", "__density", "__present")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce", downcast="float")
    return frame


def extract_response_markers(frame: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    processed = 0
    total = len(frame)
    batches = _iter_extraction_batches(frame)
    if workers > 1:
        print(f"[extract] using {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(
                _extract_response_batch,
                batches,
                chunksize=1,
                buffersize=workers * 2,
            )
            for extracted_batch in generated:
                rows.extend(extracted_batch)
                processed += len(extracted_batch)
                if len(rows) >= 2_500:
                    chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
                    rows.clear()
                    gc.collect()
                if processed % 5_000 == 0 or processed == total:
                    print(f"[extract] {processed:,}/{total:,} responses")
    else:
        for batch in batches:
            rows.extend(_extract_response_batch(batch))
            processed += len(batch)
            if len(rows) >= 2_500:
                chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
                rows.clear()
                gc.collect()
            if processed % 5_000 == 0 or processed == total:
                print(f"[extract] {processed:,}/{total:,} responses")
    if rows:
        chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()


def _extract_religion_only_record(record: dict[str, object]) -> dict[str, object]:
    """Re-extract every religion marker while retaining the audited text surface."""

    extracted = extract_markers(
        record,
        markers=RELIGION_MARKERS,
        pipeline_version=ETHICS_RELIGION_REGISTRY_VERSION,
    )
    extracted.pop("response_content", None)
    extracted.pop("explanation_text", None)
    return extracted


def _extract_religion_only_batch(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_extract_religion_only_record(record) for record in records]


def extract_religion_only_markers(
    frame: pd.DataFrame, workers: int = 1
) -> pd.DataFrame:
    """Recompute all religion columns from raw responses for an incremental rerun.

    This is not a delta adjustment to old counts: every registered religion
    pattern and semantic guard is applied again to every source response.  The
    optimization only reuses unchanged nonreligion marker columns.
    """

    rows: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    processed = 0
    total = len(frame)
    batches = _iter_extraction_batches(frame)
    if workers > 1:
        print(f"[religion-reextract] using {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(
                _extract_religion_only_batch,
                batches,
                chunksize=1,
                buffersize=workers * 2,
            )
            for extracted_batch in generated:
                rows.extend(extracted_batch)
                processed += len(extracted_batch)
                if len(rows) >= 2_500:
                    chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
                    rows.clear()
                    gc.collect()
                if processed % 5_000 == 0 or processed == total:
                    print(f"[religion-reextract] {processed:,}/{total:,} responses")
    else:
        for batch in batches:
            rows.extend(_extract_religion_only_batch(batch))
            processed += len(batch)
            if len(rows) >= 2_500:
                chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
                rows.clear()
                gc.collect()
            if processed % 5_000 == 0 or processed == total:
                print(f"[religion-reextract] {processed:,}/{total:,} responses")
    if rows:
        chunks.append(_compact_chunk(pd.DataFrame.from_records(rows)))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()


def _extract_religion_chunk_with_executor(
    frame: pd.DataFrame,
    executor: ProcessPoolExecutor,
    *,
    worker_count: int,
) -> pd.DataFrame:
    """Use an already-warm worker pool for one streamed source chunk."""

    rows: list[dict[str, object]] = []
    generated = executor.map(
        _extract_religion_only_batch,
        _iter_extraction_batches(frame),
        chunksize=1,
        buffersize=worker_count * 3,
    )
    for extracted_batch in generated:
        rows.extend(extracted_batch)
    return _compact_chunk(pd.DataFrame.from_records(rows))


def build_religion_precision_audit(
    frame: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    per_model_marker: int,
    random_seed: int,
) -> pd.DataFrame:
    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    lexicon_pattern = compile_religion_audit_lexicon_pattern()
    rows: list[dict[str, object]] = []
    for marker in RELIGION_MARKERS:
        present_col = f"{marker.marker_id}__present"
        candidates = frame[pd.to_numeric(frame[present_col], errors="coerce").eq(1)]
        for model, group in candidates.groupby("model_id", sort=True):
            rng = np.random.default_rng(
                _stable_seed(
                    ETHICS_RELIGION_REGISTRY_VERSION,
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
                matched = [match.text for match in matches]
                response_terms = {
                    match.group(0).casefold() for match in lexicon_pattern.finditer(text)
                }
                scenario_terms = {
                    match.group(0).casefold() for match in lexicon_pattern.finditer(scenario)
                }
                exact_overlap = any(
                    _normalize_for_overlap(value) in scenario_normalized
                    for value in matched
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
                        "matched_text": " || ".join(matched),
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
    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    marker_cols = [f"{marker.marker_id}__present" for marker in RELIGION_MARKERS]
    counters: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    form_counter: Counter[tuple[str, str, bool]] = Counter()
    unmatched: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        if not bool(row.text_available):
            continue
        visible_text = str(row.visible_text)
        # Open-anchor recall screening must inspect the same lexical surface as
        # the registered markers.  Scanning visible text would incorrectly
        # flag words that occur only on stripped technical answer lines.
        text = prepare_text_surface(
            visible_text, benchmark="ethics"
        ).lexical_text
        if not text.strip():
            continue
        scenario = str(scenario_map.get(row.question_id, ""))
        scenario_normalized = _normalize_for_overlap(scenario)
        present_ids = {
            marker.marker_id
            for marker, column in zip(RELIGION_MARKERS, marker_cols)
            if getattr(row, column) == 1
        }
        for category, pattern in _OPEN_ANCHORS.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            captured = bool(present_ids & _ANCHOR_CAPTURE_MARKERS[category])
            forms = sorted({match.group(0).casefold() for match in matches})
            scenario_overlap = any(
                _normalize_for_overlap(value) in scenario_normalized for value in forms
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
            for form in forms:
                form_counter[(category, form, captured)] += 1
            if captured:
                continue
            first = matches[0]
            unmatched[(category, row.model_id)].append(
                {
                    "anchor_category": category,
                    "anchor_forms": ";".join(forms),
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
                    "full_response": visible_text,
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
    summary = pd.DataFrame(
        [
            {
                "model_id": model,
                "persona": persona,
                "subset": subset,
                "anchor_category": category,
                "captured_by_relevant_religion_marker": captured,
                "scenario_anchor_overlap": overlap,
                "n_responses": count,
            }
            for (model, persona, subset, category, captured, overlap), count in counters.items()
        ]
    )
    forms = pd.DataFrame(
        [
            {
                "anchor_category": category,
                "anchor_form": form,
                "captured_by_relevant_religion_marker": captured,
                "n_responses": count,
            }
            for (category, form, captured), count in form_counter.most_common()
        ]
    )
    sampled: list[dict[str, object]] = []
    for (category, model), records in sorted(unmatched.items()):
        rng = np.random.default_rng(
            _stable_seed(
                ETHICS_RELIGION_REGISTRY_VERSION,
                "unmatched",
                category,
                model,
                random_seed,
            )
        )
        take = min(per_model_category, len(records))
        selected = rng.choice(len(records), size=take, replace=False)
        sampled.extend(records[int(index)] for index in selected)
    return summary, forms, pd.DataFrame(sampled)


def build_lexicon_coverage(
    frame: pd.DataFrame, scenarios: pd.DataFrame
) -> pd.DataFrame:
    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    pattern = compile_religion_audit_lexicon_pattern()
    counters: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    category_map = {
        "identity": "religious_identity",
        "religious_meta": "religious_meta",
        "deity_or_figure": "deity_figure",
        "scripture": "scripture_authority",
        "practice": "practice_community",
        "doctrine": "doctrine",
        "named_concept": "named_concept",
        "nonreligious_framing": "nonreligious_framing",
    }
    for row in frame.itertuples(index=False):
        if not bool(row.text_available):
            continue
        text = prepare_text_surface(
            str(row.visible_text), benchmark="ethics"
        ).lexical_text
        if not text.strip():
            continue
        terms = {match.group(0).casefold() for match in pattern.finditer(text)}
        if not terms:
            continue
        scenario_terms = {
            match.group(0).casefold()
            for match in pattern.finditer(str(scenario_map.get(row.question_id, "")))
        }
        present_ids = {
            marker.marker_id
            for marker in RELIGION_MARKERS
            if getattr(row, f"{marker.marker_id}__present") == 1
        }
        for term in terms:
            category = RELIGION_AUDIT_LEXICON.get(term, "surface_variant")
            anchor = category_map.get(category, "formula")
            captured = bool(present_ids & _ANCHOR_CAPTURE_MARKERS[anchor])
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
                "captured_by_relevant_religion_marker": captured,
                "same_term_in_scenario": overlap,
                "n_responses": count,
            }
            for (model, persona, term, category, captured, overlap), count in counters.items()
        ]
    )


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
                    "Candidate precision/recall audit for visible ETHICS Religion Persona "
                    "language. A hit is not proof that persona caused the phrase."
                ),
            },
            {
                "section": "precision",
                "instruction": (
                    "Read full_response and full_scenario. Label TP only when matched_text "
                    "fits the registered operational definition; record scenario ambiguity separately."
                ),
            },
            {
                "section": "recall",
                "instruction": (
                    "Read complete visible responses without ground truth. Record any missed "
                    "religion-persona construct and its exact span."
                ),
            },
            {
                "section": "promotion",
                "instruction": (
                    "Codex review is an AI first pass, not human validation. Registry remains "
                    "candidate until two independent human reviewers agree."
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
    config: ReligionEthicsAuditConfig,
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
        "precision_audit": "audit_precision_religion_hits.csv",
        "recall_audit": "audit_recall_responses.csv",
        "anchor_summary": "candidate_anchor_coverage.csv",
        "anchor_forms": "candidate_anchor_forms.csv",
        "unmatched_anchor_audit": "audit_unmatched_anchor_contexts.csv",
        "lexicon_coverage": "religion_lexicon_coverage.csv",
    }
    for key, filename in filenames.items():
        results[key].to_csv(
            config.output_dir / filename, index=False, encoding="utf-8-sig"
        )
    registry = pd.DataFrame(religion_registry_rows())
    lexicon = pd.DataFrame(religion_lexicon_rows())
    registry.to_csv(config.output_dir / "marker_registry.csv", index=False, encoding="utf-8-sig")
    lexicon.to_csv(config.output_dir / "religion_audit_lexicon.csv", index=False, encoding="utf-8-sig")
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
    # The response-level frame is by far the largest object in a full
    # 154,428-row run.  Keeping it alive while openpyxl materializes the audit
    # workbook can exceed available RAM on Windows.  All downstream summaries
    # and manifest counts have already been computed at this point, so release
    # the frame before constructing workbook cells.  The persisted gzip CSV is
    # the canonical response-level artifact.
    results.pop("response_level", None)
    try:
        del response
    except UnboundLocalError:
        pass
    gc.collect()
    _write_audit_workbook(
        config.output_dir / "audit_manual_religion.xlsx",
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


def finalize_existing_religion_ethics_audit(
    config: ReligionEthicsAuditConfig,
) -> dict[str, object]:
    """Finish workbook/manifest creation after a completed extraction write.

    This recovery path never reruns marker extraction and never overwrites an
    existing workbook or manifest.  It is intended for a write-once run that
    reached the CSV stage but was interrupted while openpyxl still shared
    memory with the response-level frame.
    """

    _require_openpyxl()
    output_dir = config.output_dir
    workbook_path = output_dir / "audit_manual_religion.xlsx"
    manifest_path = output_dir / "manifest.json"
    if workbook_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing audit workbook or manifest: "
            f"{output_dir}"
        )
    required = {
        "response": output_dir / "response_level_markers.csv.gz",
        "precision": output_dir / "audit_precision_religion_hits.csv",
        "recall": output_dir / "audit_recall_responses.csv",
        "unmatched": output_dir / "audit_unmatched_anchor_contexts.csv",
        "registry": output_dir / "marker_registry.csv",
        "prevalence": output_dir / "marker_prevalence.csv",
        "status": output_dir / "data_status_summary.csv",
        "scenarios": output_dir / "scenario_lookup.csv",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot finalize incomplete extraction; missing files: {missing}"
        )

    print(f"[finalize] reading compact metadata from {output_dir}")
    response = pd.read_csv(
        required["response"],
        usecols=[
            "model_id",
            "persona",
            "question_id",
            "text_available",
            "text_status",
            "answer_status",
        ],
        low_memory=False,
    )
    precision = pd.read_csv(required["precision"], low_memory=False)
    recall = pd.read_csv(required["recall"], low_memory=False)
    unmatched = pd.read_csv(required["unmatched"], low_memory=False)
    registry = pd.read_csv(required["registry"], low_memory=False)
    prevalence = pd.read_csv(required["prevalence"], low_memory=False)
    status = pd.read_csv(required["status"], low_memory=False)
    scenarios = pd.read_csv(
        required["scenarios"], usecols=["scenario_available"], low_memory=False
    )

    rows_input = len(response)
    manifest: dict[str, object] = {
        "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
        "lexicon_version": ETHICS_RELIGION_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": _sha256(config.input_csv),
        "canonical_source_manifest": str(
            config.input_csv.parent / "all_models_results_full_manifest.json"
        ),
        "reference_archive": str(
            config.repo_root / "docs" / "ref" / "linguistic marker yang udah aku coba.zip"
        ),
        "raw_ethics_root": str(config.raw_ethics_root),
        "output_dir": str(output_dir),
        "config": {
            **asdict(config),
            "repo_root": str(config.repo_root),
            "input_csv": str(config.input_csv),
            "raw_ethics_root": str(config.raw_ethics_root),
            "output_dir": str(output_dir),
        },
        "rows_input": rows_input,
        "unique_response_keys": int(
            response[["model_id", "persona", "question_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "responses_text_available": int(
            response["text_available"].map(_as_bool).sum()
        ),
        "responses_abstain": int(response["answer_status"].eq("abstain").sum()),
        "responses_missing": int(response["text_status"].eq("missing").sum()),
        "models": int(response["model_id"].nunique()),
        "personas": int(response["persona"].nunique()),
        "unique_questions": int(response["question_id"].nunique()),
        "scenario_available": int(
            scenarios["scenario_available"].map(_as_bool).sum()
        ),
        "marker_count_total": len(ETHICS_RELIGION_MARKERS),
        "marker_count_religion": len(RELIGION_MARKERS),
        "precision_audit_rows": len(precision),
        "recall_audit_rows": len(recall),
        "unmatched_anchor_audit_rows": len(unmatched),
        "inference_run": False,
        "registry_status": "candidate_pending_two_independent_human_reviews",
        "interpretation_guardrails": [
            "The same extractor and registry are applied to every model and all 17 conditions.",
            "All visible abstain responses are included; missing text is NA, never zero.",
            "Hidden reasoning and ground truth are never used as marker text.",
            "Count, presence, and density per 100 words are reported separately.",
            "Identity, perspective, authority, practice, and formulae remain separate families.",
            "Overlapping markers are never summed into a religion composite score.",
            "Scenario overlap blocks automatic persona-uptake attribution.",
            "Codex review is AI review and does not satisfy independent human validation.",
        ],
        "finalized_from_existing_csvs": True,
    }

    print("[finalize] writing audit workbook")
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
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] finalized existing run: {output_dir}")
    return manifest


def run_religion_ethics_dataset_audit(
    config: ReligionEthicsAuditConfig,
) -> dict[str, object]:
    # Fail before a long full-corpus scan if the selected interpreter cannot
    # create the required manual-audit workbook.
    _require_openpyxl()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {config.output_dir}")
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
    results: dict[str, pd.DataFrame] = {
        "response_level": response,
        "scenario_lookup": scenarios,
        "status_summary": build_status_summary(response),
        "surface_summary": build_surface_summary(response),
        "opening_summary": build_opening_summary(response),
        "polarity_summary": build_polarity_summary(response),
        "marker_prevalence": build_marker_prevalence(
            response, markers=ETHICS_RELIGION_MARKERS
        ),
    }
    print("[audit] balanced religion precision sample")
    results["precision_audit"] = build_religion_precision_audit(
        response,
        scenarios,
        per_model_marker=config.precision_per_model_marker,
        random_seed=config.random_seed,
    )
    print("[audit] blind recall sample")
    results["recall_audit"] = build_recall_audit_sample(
        response,
        sample_per_condition=config.recall_sample_per_condition,
        seed=config.random_seed,
    )
    print("[audit] open-anchor coverage and scenario overlap")
    anchor_summary, anchor_forms, unmatched = build_candidate_anchor_audit(
        response,
        scenarios,
        per_model_category=config.unmatched_per_model_category,
        random_seed=config.random_seed,
    )
    results["anchor_summary"] = anchor_summary
    results["anchor_forms"] = anchor_forms
    results["unmatched_anchor_audit"] = unmatched
    results["lexicon_coverage"] = build_lexicon_coverage(response, scenarios)
    manifest: dict[str, object] = {
        "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
        "lexicon_version": ETHICS_RELIGION_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": _sha256(config.input_csv),
        "canonical_source_manifest": str(
            config.input_csv.parent / "all_models_results_full_manifest.json"
        ),
        "reference_archive": str(
            config.repo_root / "docs" / "ref" / "linguistic marker yang udah aku coba.zip"
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
        "marker_count_total": len(ETHICS_RELIGION_MARKERS),
        "marker_count_religion": len(RELIGION_MARKERS),
        "precision_audit_rows": len(results["precision_audit"]),
        "recall_audit_rows": len(results["recall_audit"]),
        "unmatched_anchor_audit_rows": len(unmatched),
        "inference_run": False,
        "registry_status": "candidate_pending_two_independent_human_reviews",
        "interpretation_guardrails": [
            "The same extractor and registry are applied to every model and all 17 conditions.",
            "All visible abstain responses are included; missing text is NA, never zero.",
            "Hidden reasoning and ground truth are never used as marker text.",
            "Count, presence, and density per 100 words are reported separately.",
            "Identity, perspective, authority, practice, and formulae remain separate families.",
            "Overlapping markers are never summed into a religion composite score.",
            "Scenario overlap blocks automatic persona-uptake attribution.",
            "Codex review is AI review and does not satisfy independent human validation.",
        ],
    }
    print(f"[write] {config.output_dir}")
    _write_results(results, config, manifest)
    print(f"[done] rows={rows_input:,} religion_markers={len(RELIGION_MARKERS)}")
    return {"tables": results, "manifest": manifest, "config": config}


def _merge_reextracted_religion_columns(
    base: pd.DataFrame, religion: pd.DataFrame
) -> pd.DataFrame:
    if len(base) != len(religion):
        raise ValueError(
            f"Base/re-extraction row mismatch: {len(base):,} != {len(religion):,}"
        )
    key_pairs = (
        ("model_id", "model_id"),
        ("persona", "persona"),
        ("question_id", "question_id"),
    )
    for base_col, new_col in key_pairs:
        if not np.array_equal(
            base[base_col].astype(str).to_numpy(),
            religion[new_col].astype(str).to_numpy(),
        ):
            raise ValueError(f"Base/re-extraction key order differs at {base_col}.")

    string_checks = (
        "subset",
        "answer_status",
        "text_status",
        "opening_category",
    )
    for column in string_checks:
        if not np.array_equal(
            base[column].fillna("").astype(str).to_numpy(),
            religion[column].fillna("").astype(str).to_numpy(),
        ):
            raise ValueError(f"Surface/source metadata changed at {column}.")
    if not np.array_equal(
        base["text_available"].map(_as_bool).to_numpy(),
        religion["text_available"].map(_as_bool).to_numpy(),
    ):
        raise ValueError("Surface/source metadata changed at text_available.")
    numeric_checks = (
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "ttr_300",
    )
    for column in numeric_checks:
        old = pd.to_numeric(base[column], errors="coerce").to_numpy(dtype=float)
        new = pd.to_numeric(religion[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(old, new, rtol=1e-6, atol=1e-7, equal_nan=True):
            raise ValueError(f"Surface metric changed at {column}.")

    result = base.copy()
    religion_columns = [
        f"{marker.marker_id}__{suffix}"
        for marker in RELIGION_MARKERS
        for suffix in ("count", "density", "present")
    ]
    missing = set(religion_columns).difference(religion.columns)
    if missing:
        raise ValueError(f"Religion re-extraction is missing columns: {sorted(missing)}")
    for column in religion_columns:
        result[column] = religion[column].to_numpy()
    # Full text is retained only in memory for complete-response audit sheets;
    # _write_results deliberately omits it from the compact response artifact.
    result["visible_text"] = religion["visible_text"].fillna("").astype(str)
    return result


def _stream_incremental_response(
    config: ReligionEthicsAuditConfig,
    base_response_path: Path,
    *,
    chunk_size: int = 3_000,
) -> pd.DataFrame:
    """Bound peak memory while rebuilding religion columns from a 310 MB CSV."""

    header = pd.read_csv(config.input_csv, nrows=0).columns.tolist()
    missing = set(REQUIRED_COLUMNS).difference(header)
    if missing:
        raise ValueError(f"Canonical ETHICS CSV is missing columns: {sorted(missing)}")
    source_reader = pd.read_csv(
        config.input_csv,
        usecols=list(REQUIRED_COLUMNS),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        chunksize=chunk_size,
    )
    base_reader = pd.read_csv(
        base_response_path,
        low_memory=False,
        chunksize=chunk_size,
    )
    temp_root = config.repo_root / ".codex_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    expected_personas = {
        persona
        for conditions in IDENTITY_CONDITIONS.values()
        for persona in conditions.values()
    }
    processed = 0
    source_exhausted = False
    base_exhausted = False
    stage_path = temp_root / f"religion_v1_1_incremental_{os.getpid()}.csv"
    if stage_path.exists():
        raise FileExistsError(f"Refusing to overwrite staging file: {stage_path}")
    executor = (
        ProcessPoolExecutor(max_workers=config.extraction_workers)
        if config.extraction_workers > 1
        else None
    )
    try:
        first_write = True
        while True:
            try:
                source_chunk = next(source_reader)
            except StopIteration:
                source_exhausted = True
                source_chunk = None
            try:
                base_chunk = next(base_reader)
            except StopIteration:
                base_exhausted = True
                base_chunk = None
            if source_exhausted or base_exhausted:
                if not (source_exhausted and base_exhausted):
                    raise ValueError("Canonical source and base response lengths differ.")
                break

            unknown = set(source_chunk["persona_folder"]).difference(expected_personas)
            if unknown:
                raise ValueError(f"Unexpected persona folders: {sorted(unknown)}")
            source_chunk["condition_bucket"] = source_chunk["persona_folder"].map(
                _condition_bucket
            )
            if executor is None:
                religion = extract_religion_only_markers(source_chunk, workers=1)
            else:
                religion = _extract_religion_chunk_with_executor(
                    source_chunk,
                    executor,
                    worker_count=config.extraction_workers,
                )
            merged = _merge_reextracted_religion_columns(base_chunk, religion)
            merged.to_csv(
                stage_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8",
            )
            first_write = False
            processed += len(merged)
            if processed % 5_000 == 0:
                print(f"[stream-merge] {processed:,} validated rows staged")
            del source_chunk, base_chunk, religion, merged
            gc.collect()
        print(f"[stream-merge] loading {processed:,} staged rows for summaries")
        response = pd.read_csv(stage_path, low_memory=False)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        # Exact, process-owned temporary file; never a user result artifact.
        stage_path.unlink(missing_ok=True)
    return response


def _restore_incremental_audit_metadata(
    response: pd.DataFrame, input_csv: Path
) -> None:
    """Restore audit-only source columns omitted from compact base artifacts.

    Incremental re-extraction deliberately reuses the compact response artifact,
    which does not retain the constant benchmark label or source-file metadata.
    Blind-recall sampling needs those columns, so rebuild them from the canonical
    CSV only after proving that its ordered response keys still match the staged
    response.  Mutating in place avoids copying the several-hundred-megabyte
    response table.
    """

    source = pd.read_csv(
        input_csv,
        usecols=["model", "persona_folder", "question_id", "file"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if len(source) != len(response):
        raise ValueError("Canonical/incremental response row counts differ.")
    for source_col, response_col in (
        ("model", "model_id"),
        ("persona_folder", "persona"),
        ("question_id", "question_id"),
    ):
        if not np.array_equal(
            source[source_col].astype(str).to_numpy(),
            response[response_col].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"Canonical/incremental key order differs at {response_col}."
            )
    response["benchmark"] = "ethics"
    response["source_path"] = source["file"].astype(str).to_numpy()
    # CSV header is line 1, matching the source-row convention of the full run.
    response["source_row"] = np.arange(2, len(response) + 2, dtype=np.int64)


def run_religion_ethics_incremental_audit(
    config: ReligionEthicsAuditConfig,
    base_output: Path,
    reextracted_stage: Path | None = None,
) -> dict[str, object]:
    """Write a new audit after full religion-only re-extraction.

    Unchanged general/ETHICS marker columns are reused byte-for-value from a
    validated base response artifact.  All religion markers, their densities,
    and all audit samples are rebuilt from the canonical source responses.
    """

    _require_openpyxl()
    base_output = Path(base_output).resolve()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {config.output_dir}")
    base_manifest_path = base_output / "manifest.json"
    base_response_path = base_output / "response_level_markers.csv.gz"
    if not base_manifest_path.is_file() or not base_response_path.is_file():
        raise FileNotFoundError(
            "Incremental rerun requires base manifest.json and "
            "response_level_markers.csv.gz."
        )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    input_sha = _sha256(config.input_csv)
    if base_manifest.get("input_sha256") != input_sha:
        raise ValueError("Base output and canonical input SHA-256 differ.")
    if base_manifest.get("pipeline_version") == ETHICS_RELIGION_REGISTRY_VERSION:
        raise ValueError("Base output already uses the current registry version.")

    print(f"[load] validated base response artifact {base_response_path}")
    if reextracted_stage is None:
        print("[religion-reextract] streaming all 20 markers from canonical responses")
        response = _stream_incremental_response(
            config,
            base_response_path,
        )
    else:
        reextracted_stage = Path(reextracted_stage).resolve()
        if not reextracted_stage.is_file():
            raise FileNotFoundError(f"Missing re-extracted stage: {reextracted_stage}")
        print(f"[load] validated lightweight re-extraction stage {reextracted_stage}")
        response = pd.read_csv(reextracted_stage, low_memory=False)
        check_columns = [
            "model_id",
            "persona",
            "question_id",
            "subset",
            "answer_status",
            "text_status",
            "text_available",
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "ttr_300",
            "opening_category",
        ]
        base_check = pd.read_csv(
            base_response_path,
            usecols=check_columns,
            low_memory=False,
        )
        if len(base_check) != len(response):
            raise ValueError("Stage/base response row counts differ.")
        for column in (
            "model_id",
            "persona",
            "question_id",
            "subset",
            "answer_status",
            "text_status",
            "opening_category",
        ):
            if not np.array_equal(
                base_check[column].fillna("").astype(str).to_numpy(),
                response[column].fillna("").astype(str).to_numpy(),
            ):
                raise ValueError(f"Stage/base alignment differs at {column}.")
        if not np.array_equal(
            base_check["text_available"].map(_as_bool).to_numpy(),
            response["text_available"].map(_as_bool).to_numpy(),
        ):
            raise ValueError("Stage/base alignment differs at text_available.")
        for column in (
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "ttr_300",
        ):
            old = pd.to_numeric(base_check[column], errors="coerce").to_numpy(float)
            new = pd.to_numeric(response[column], errors="coerce").to_numpy(float)
            if not np.allclose(old, new, rtol=1e-6, atol=1e-7, equal_nan=True):
                raise ValueError(f"Stage/base surface metric differs at {column}.")
        expected_religion_columns = {
            f"{marker.marker_id}__{suffix}"
            for marker in RELIGION_MARKERS
            for suffix in ("count", "density", "present")
        }
        missing_stage = expected_religion_columns.difference(response.columns)
        if missing_stage or "visible_text" not in response:
            raise ValueError(
                "Re-extracted stage is incomplete: "
                f"missing={sorted(missing_stage)} visible_text={'visible_text' in response}"
            )
        del base_check
    _restore_incremental_audit_metadata(response, config.input_csv)
    rows_input = len(response)
    unique_questions = int(response["question_id"].nunique())
    scenarios = load_scenario_lookup(config.raw_ethics_root, response["question_id"])
    gc.collect()

    print("[summaries] status, surface, opening, polarity, prevalence")
    results: dict[str, pd.DataFrame] = {
        "response_level": response,
        "scenario_lookup": scenarios,
        "status_summary": build_status_summary(response),
        "surface_summary": build_surface_summary(response),
        "opening_summary": build_opening_summary(response),
        "polarity_summary": build_polarity_summary(response),
        "marker_prevalence": build_marker_prevalence(
            response, markers=ETHICS_RELIGION_MARKERS
        ),
    }
    print("[audit] new balanced religion precision sample")
    results["precision_audit"] = build_religion_precision_audit(
        response,
        scenarios,
        per_model_marker=config.precision_per_model_marker,
        random_seed=config.random_seed,
    )
    print("[audit] new blind recall sample")
    results["recall_audit"] = build_recall_audit_sample(
        response,
        sample_per_condition=config.recall_sample_per_condition,
        seed=config.random_seed,
    )
    print("[audit] full-corpus lexical-surface anchor screen")
    anchor_summary, anchor_forms, unmatched = build_candidate_anchor_audit(
        response,
        scenarios,
        per_model_category=config.unmatched_per_model_category,
        random_seed=config.random_seed,
    )
    results["anchor_summary"] = anchor_summary
    results["anchor_forms"] = anchor_forms
    results["unmatched_anchor_audit"] = unmatched
    results["lexicon_coverage"] = build_lexicon_coverage(response, scenarios)
    manifest: dict[str, object] = {
        "pipeline_version": ETHICS_RELIGION_REGISTRY_VERSION,
        "lexicon_version": ETHICS_RELIGION_LEXICON_VERSION,
        "input_csv": str(config.input_csv),
        "input_sha256": input_sha,
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
            response[["model_id", "persona", "question_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "responses_text_available": int(response["text_available"].map(_as_bool).sum()),
        "responses_abstain": int(response["answer_status"].eq("abstain").sum()),
        "responses_missing": int(response["text_status"].eq("missing").sum()),
        "models": int(response["model_id"].nunique()),
        "personas": int(response["persona"].nunique()),
        "unique_questions": unique_questions,
        "scenario_available": int(scenarios["scenario_available"].sum()),
        "marker_count_total": len(ETHICS_RELIGION_MARKERS),
        "marker_count_religion": len(RELIGION_MARKERS),
        "precision_audit_rows": len(results["precision_audit"]),
        "recall_audit_rows": len(results["recall_audit"]),
        "unmatched_anchor_audit_rows": len(unmatched),
        "inference_run": False,
        "incremental_religion_reextract": True,
        "base_output": str(base_output),
        "base_pipeline_version": base_manifest.get("pipeline_version"),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "reuse_scope": (
            "Only unchanged general and ETHICS marker columns were reused; every "
            "religion marker was recomputed from every canonical response."
        ),
        "equivalence_checks": [
            "Canonical input SHA-256 equals the validated base input SHA-256.",
            "All ordered model/persona/question keys match.",
            "Text availability, text status, word/sentence metrics, TTR, and opening class match.",
        ],
        "registry_status": "candidate_pending_two_independent_human_reviews",
        "interpretation_guardrails": [
            "The same extractor and registry are applied to every model and all 17 conditions.",
            "All visible abstain responses are included; missing text is NA, never zero.",
            "Hidden reasoning and ground truth are never used as marker text.",
            "Count, presence, and density per 100 words are reported separately.",
            "Identity, perspective, authority, practice, and formulae remain separate families.",
            "Overlapping markers are never summed into a religion composite score.",
            "Scenario overlap blocks automatic persona-uptake attribution.",
            "Codex review is AI review and does not satisfy independent human validation.",
        ],
    }
    print(f"[write] {config.output_dir}")
    _write_results(results, config, manifest)
    print(f"[done] rows={rows_input:,} religion_markers={len(RELIGION_MARKERS)}")
    return {"tables": results, "manifest": manifest, "config": config}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    defaults = default_audit_config(repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--raw-ethics-root", type=Path, default=defaults.raw_ethics_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--precision-per-model-marker",
        type=int,
        default=defaults.precision_per_model_marker,
    )
    parser.add_argument("--recall-sample-per-condition", type=int, default=3)
    parser.add_argument(
        "--unmatched-per-model-category",
        type=int,
        default=defaults.unmatched_per_model_category,
    )
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument(
        "--reuse-base-output",
        type=Path,
        help=(
            "Reuse only unchanged nonreligion response columns from this validated output; "
            "all religion markers are re-extracted from canonical source text."
        ),
    )
    parser.add_argument(
        "--reuse-reextracted-stage",
        type=Path,
        help=(
            "Load a completed lightweight religion re-extraction stage instead of "
            "running extraction inside this audit process."
        ),
    )
    parser.add_argument("--no-response-level", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Create only the missing workbook and manifest from completed CSV outputs.",
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
        extraction_workers=max(1, args.extraction_workers),
    )
    if args.finalize_existing:
        finalize_existing_religion_ethics_audit(config)
    elif args.reuse_base_output:
        run_religion_ethics_incremental_audit(
            config,
            args.reuse_base_output.resolve(),
            (
                args.reuse_reextracted_stage.resolve()
                if args.reuse_reextracted_stage
                else None
            ),
        )
    else:
        run_religion_ethics_dataset_audit(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

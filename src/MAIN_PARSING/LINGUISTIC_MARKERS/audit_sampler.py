"""Deterministic, ground-truth-blind sampling of marker hits for manual audit."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .extractor import find_marker_matches, marker_text
from .registry import MARKERS, MarkerSpec
from .surface import prepare_text_surface


def _sample_occurrence_locations(
    candidates: pd.DataFrame,
    count_col: str,
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[object, int]]:
    """Select (row index, within-response hit number) without expanding hits."""
    if sample_size <= 0:
        return []
    work = candidates.copy()
    work["_audit_count"] = (
        pd.to_numeric(work[count_col], errors="coerce").fillna(0).astype(int).clip(lower=0)
    )
    strata = [
        (key, group, int(group["_audit_count"].sum()))
        for key, group in work.groupby(
            ["benchmark", "model_id", "persona"], sort=True
        )
        if int(group["_audit_count"].sum()) > 0
    ]
    if not strata:
        return []

    total_hits = sum(total for _, _, total in strata)
    target = min(sample_size, total_hits)
    capacities = np.asarray([total for _, _, total in strata], dtype=int)
    allocation = np.zeros(len(strata), dtype=int)

    # Give conditions coverage first. If there are more conditions than slots,
    # choose conditions uniformly; remaining slots are occurrence-weighted.
    if target >= len(strata):
        allocation[:] = 1
    else:
        chosen = rng.choice(len(strata), size=target, replace=False)
        allocation[chosen] = 1
    remaining = target - int(allocation.sum())
    while remaining:
        residual = capacities - allocation
        probabilities = residual / residual.sum()
        chosen = int(rng.choice(len(strata), p=probabilities))
        allocation[chosen] += 1
        remaining -= 1

    locations: list[tuple[object, int]] = []
    for (_, group, total), take in zip(strata, allocation):
        if not take:
            continue
        offsets = np.sort(rng.choice(total, size=int(take), replace=False))
        counts = group["_audit_count"].to_numpy(int)
        cumulative = np.cumsum(counts)
        positions = np.searchsorted(cumulative, offsets, side="right")
        starts = np.concatenate(([0], cumulative[:-1]))
        for offset, position in zip(offsets, positions):
            locations.append(
                (group.index[int(position)], int(offset - starts[int(position)] + 1))
            )
    return locations


def _context(text: str, start: int, end: int, radius: int = 100) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", " ")


def build_audit_sample(
    frame: pd.DataFrame,
    sample_per_marker: int = 50,
    seed: int = 18222008,
    markers: Iterable[MarkerSpec] = MARKERS,
) -> pd.DataFrame:
    """Create a blind, occurrence-level review sheet.

    ``sample_per_marker`` is an upper bound on marker occurrences, not on
    marker-positive responses. Extracted counts are used as sampling weights,
    so the complete occurrence population is never expanded in memory.
    """
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for marker in markers:
        count_col = f"{marker.marker_id}__count"
        if count_col not in frame:
            continue
        candidates = frame[pd.to_numeric(frame[count_col], errors="coerce") > 0]
        if candidates.empty:
            continue
        locations = _sample_occurrence_locations(
            candidates, count_col, sample_per_marker, rng
        )
        for source_index, match_number in locations:
            source_row = candidates.loc[source_index]
            surface = prepare_text_surface(
                source_row.get("visible_text"), benchmark=str(source_row.get("benchmark") or "")
            )
            text = marker_text(surface, marker)
            matches = find_marker_matches(text, marker)
            if match_number > len(matches):
                raise AssertionError(
                    f"Stored count and re-extracted hits disagree for {marker.marker_id}: "
                    f"expected occurrence {match_number}, found {len(matches)}"
                )
            match = matches[match_number - 1]
            rows.append(
                {
                    "marker_id": marker.marker_id,
                    "marker_version": marker.version,
                    "benchmark": source_row.get("benchmark"),
                    "model_id": source_row.get("model_id"),
                    "persona": source_row.get("persona"),
                    "question_id": source_row.get("question_id"),
                    "source_path": source_row.get("source_path"),
                    "match_number": match_number,
                    "match_start": match.start,
                    "match_end": match.end,
                    "matched_text": match.text,
                    "match_context": _context(text, match.start, match.end),
                    "pattern_note": match.pattern_note,
                    "reviewer": "",
                    "is_true_positive": "",
                    "audit_note": "",
                }
            )
    return pd.DataFrame(rows)


def build_recall_audit_sample(
    frame: pd.DataFrame,
    sample_per_condition: int = 2,
    seed: int = 18222008,
) -> pd.DataFrame:
    """Sample complete visible responses for blind false-negative review.

    Automatic marker values, correctness, and ground truth are intentionally
    omitted. A reviewer records marker IDs and any missed spans independently,
    after which those annotations can be compared with ``response_records``.
    """
    columns = [
        "benchmark",
        "model_id",
        "persona",
        "question_id",
        "source_path",
        "source_row",
        "visible_text",
    ]
    available = frame[frame["text_available"].fillna(False)].copy()
    if available.empty or sample_per_condition <= 0:
        return pd.DataFrame(
            columns=columns
            + ["reviewer", "manual_marker_ids", "missed_span_notes", "audit_note"]
        )
    rng = np.random.default_rng(seed)
    selected: list[object] = []
    grouped = available.groupby(["benchmark", "model_id", "persona"], sort=True)
    for _, group in grouped:
        take = min(sample_per_condition, len(group))
        selected.extend(
            rng.choice(group.index.to_numpy(), size=take, replace=False).tolist()
        )
    result = available.loc[selected].reindex(columns=columns).copy()
    result = result.sort_values(
        ["benchmark", "model_id", "persona", "question_id"], kind="stable"
    ).reset_index(drop=True)
    result["reviewer"] = ""
    result["manual_marker_ids"] = ""
    result["missed_span_notes"] = ""
    result["audit_note"] = ""
    return result

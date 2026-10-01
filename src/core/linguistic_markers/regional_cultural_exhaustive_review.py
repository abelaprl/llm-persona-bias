"""Build write-once exhaustive AI-review queues for ETHICS regional markers.

The generated files are a review universe, not a validation claim.  Manual
labels start blank and the manifest records reviewed denominators explicitly.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .ethics_regional_registry import (
    REGIONAL_CULTURAL_MARKERS,
    REGIONAL_CULTURAL_REGISTRY_VERSION,
)
from .extractor import find_marker_matches, marker_text
from .regional_cultural_codex_adjudication import _intentional_exclusion
from .regional_cultural_dataset_audit import (
    _ANCHOR_CAPTURE_MARKERS,
    _OPEN_ANCHORS,
    _context_excerpt,
    _normalize_for_overlap,
)
from .surface import prepare_text_surface


REVIEW_QUEUE_VERSION = "ethics-regional-cultural-exhaustive-review-v1"
KEY_COLUMNS = ["model_id", "persona", "question_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _as_present(value: object) -> bool:
    try:
        return int(float(value)) == 1
    except (TypeError, ValueError):
        return False


def _open_text_writer(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter]:
    handle = gzip.open(path, "wt", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _load_review_frame(input_csv: Path, base_output: Path) -> pd.DataFrame:
    source = pd.read_csv(
        input_csv,
        usecols=[
            "model",
            "persona_folder",
            "question_id",
            "subset",
            "answer_status",
            "response_content_full",
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    marker_columns: list[str] = []
    for marker in REGIONAL_CULTURAL_MARKERS:
        marker_columns.extend(
            [f"{marker.marker_id}__count", f"{marker.marker_id}__present"]
        )
    extracted = pd.read_csv(
        base_output / "response_level_markers.csv.gz",
        usecols=KEY_COLUMNS + ["text_available", "text_status", "word_count"] + marker_columns,
        low_memory=False,
    )
    frame = source.merge(extracted, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(frame) != len(source):
        raise ValueError(
            f"Source/extraction key mismatch: source={len(source)}, merged={len(frame)}"
        )
    return frame


def build_review_universe(
    *,
    input_csv: Path,
    base_output: Path,
    output_dir: Path,
) -> dict[str, object]:
    input_csv = input_csv.resolve()
    base_output = base_output.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    base_manifest_path = base_output / "manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(base_manifest_path)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest["pipeline_version"] != REGIONAL_CULTURAL_REGISTRY_VERSION:
        raise ValueError(
            "Base output and current registry differ: "
            f"{base_manifest['pipeline_version']} != {REGIONAL_CULTURAL_REGISTRY_VERSION}"
        )
    if _sha256(input_csv) != base_manifest["input_sha256"]:
        raise ValueError("Input hash differs from the pinned base manifest.")

    frame = _load_review_frame(input_csv, base_output)
    scenarios = pd.read_csv(
        base_output / "scenario_lookup.csv",
        usecols=["question_id", "scenario"],
        dtype=str,
        keep_default_na=False,
    )
    scenario_map = scenarios.set_index("question_id")["scenario"].to_dict()
    marker_by_id = {marker.marker_id: marker for marker in REGIONAL_CULTURAL_MARKERS}
    marker_ids = list(marker_by_id)
    expected_occurrences = int(
        sum(
            pd.to_numeric(frame[f"{marker_id}__count"], errors="coerce")
            .fillna(0)
            .sum()
            for marker_id in marker_ids
        )
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    occurrence_path = output_dir / "occurrence_review_queue.csv.gz"
    response_path = output_dir / "response_review_queue.csv.gz"
    cluster_path = output_dir / "occurrence_context_clusters.csv"

    occurrence_fields = [
        "occurrence_id",
        "response_key",
        "marker_id",
        "marker_family",
        "pattern_index",
        "pattern_note",
        "model_id",
        "persona",
        "question_id",
        "subset",
        "answer_status",
        "matched_text",
        "normalized_matched_text",
        "match_context",
        "exact_matched_text_in_scenario",
        "codex_manual_label",
        "codex_manual_note",
    ]
    response_fields = [
        "response_key",
        "model_id",
        "persona",
        "question_id",
        "subset",
        "answer_status",
        "text_status",
        "word_count",
        "present_regional_markers",
        "open_anchor_categories",
        "uncovered_anchor_forms",
        "automated_screen_label",
        "visible_response",
        "codex_manual_recall_label",
        "codex_manual_recall_note",
    ]
    occurrence_handle, occurrence_writer = _open_text_writer(
        occurrence_path, occurrence_fields
    )
    response_handle, response_writer = _open_text_writer(response_path, response_fields)

    occurrence_count = 0
    available_count = 0
    positive_response_count = 0
    potential_miss_response_count = 0
    screen_counts: dict[str, int] = {}
    clusters: dict[tuple[str, int, str], dict[str, object]] = {}
    try:
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            surface = prepare_text_surface(row.response_content_full, benchmark="ethics")
            expected_available = bool(row.text_available)
            if surface.text_available != expected_available:
                raise ValueError(
                    "Surface availability mismatch for "
                    f"{row.model_id}|{row.persona}|{row.question_id}"
                )
            response_key = f"{row.model_id}|{row.persona}|{row.question_id}"
            present_ids = [
                marker_id
                for marker_id in marker_ids
                if _as_present(getattr(row, f"{marker_id}__present"))
            ]
            if present_ids:
                positive_response_count += 1

            scenario = scenario_map.get(row.question_id, "")
            scenario_normalized = _normalize_for_overlap(scenario)
            for marker_id in present_ids:
                marker = marker_by_id[marker_id]
                text = marker_text(surface, marker)
                matches = find_marker_matches(text, marker)
                expected_count = int(float(getattr(row, f"{marker_id}__count")))
                if len(matches) != expected_count:
                    raise ValueError(
                        f"Occurrence mismatch {response_key}|{marker_id}: "
                        f"expected={expected_count}, observed={len(matches)}"
                    )
                for match in matches:
                    occurrence_count += 1
                    normalized = _normalize_match_text(match.text)
                    exact_overlap = bool(
                        normalized and _normalize_for_overlap(match.text) in scenario_normalized
                    )
                    context = _context_excerpt(text, match.start, match.end, radius=240)
                    occurrence_writer.writerow(
                        {
                            "occurrence_id": occurrence_count,
                            "response_key": response_key,
                            "marker_id": marker_id,
                            "marker_family": marker.family,
                            "pattern_index": match.pattern_index,
                            "pattern_note": match.pattern_note,
                            "model_id": row.model_id,
                            "persona": row.persona,
                            "question_id": row.question_id,
                            "subset": row.subset,
                            "answer_status": row.answer_status,
                            "matched_text": match.text,
                            "normalized_matched_text": normalized,
                            "match_context": context,
                            "exact_matched_text_in_scenario": exact_overlap,
                            "codex_manual_label": "",
                            "codex_manual_note": "",
                        }
                    )
                    cluster_key = (marker_id, match.pattern_index, normalized)
                    cluster = clusters.setdefault(
                        cluster_key,
                        {
                            "marker_id": marker_id,
                            "marker_family": marker.family,
                            "pattern_index": match.pattern_index,
                            "pattern_note": match.pattern_note,
                            "normalized_matched_text": normalized,
                            "n_occurrences": 0,
                            "n_scenario_overlap": 0,
                            "representative_response_key": response_key,
                            "representative_context": context,
                            "codex_cluster_label": "",
                            "codex_cluster_note": "",
                        },
                    )
                    cluster["n_occurrences"] = int(cluster["n_occurrences"]) + 1
                    cluster["n_scenario_overlap"] = int(
                        cluster["n_scenario_overlap"]
                    ) + int(exact_overlap)

            if not surface.text_available:
                screen_label = "TEXT_UNAVAILABLE_NOT_REVIEWABLE"
                anchor_categories: list[str] = []
                uncovered: list[str] = []
            else:
                available_count += 1
                anchor_categories = []
                uncovered = []
                potential = False
                present_set = set(present_ids)
                for category, pattern in _OPEN_ANCHORS.items():
                    forms = sorted(
                        {match.group(0).casefold() for match in pattern.finditer(surface.visible_text)}
                    )
                    if not forms:
                        continue
                    anchor_categories.append(category)
                    if present_set & _ANCHOR_CAPTURE_MARKERS[category]:
                        continue
                    intentional, _, _ = _intentional_exclusion(category, forms)
                    uncovered.append(f"{category}:{';'.join(forms)}")
                    potential = potential or not intentional
                if potential:
                    potential_miss_response_count += 1
                    screen_label = "POTENTIAL_MISS_REQUIRES_MANUAL_REVIEW"
                elif present_ids:
                    screen_label = "REGIONAL_MARKER_HIT_REQUIRES_PRECISION_REVIEW"
                elif uncovered:
                    screen_label = "NO_HIT_DOCUMENTED_OPEN_ANCHOR_EXCLUSION"
                elif anchor_categories:
                    screen_label = "NO_HIT_RELEVANT_ANCHOR_CAPTURED_ELSEWHERE"
                else:
                    screen_label = "NO_HIT_NO_REGISTERED_OPEN_ANCHOR"
            screen_counts[screen_label] = screen_counts.get(screen_label, 0) + 1
            response_writer.writerow(
                {
                    "response_key": response_key,
                    "model_id": row.model_id,
                    "persona": row.persona,
                    "question_id": row.question_id,
                    "subset": row.subset,
                    "answer_status": row.answer_status,
                    "text_status": row.text_status,
                    "word_count": row.word_count,
                    "present_regional_markers": ";".join(present_ids),
                    "open_anchor_categories": ";".join(anchor_categories),
                    "uncovered_anchor_forms": " | ".join(uncovered),
                    "automated_screen_label": screen_label,
                    "visible_response": surface.visible_text,
                    "codex_manual_recall_label": "",
                    "codex_manual_recall_note": "",
                }
            )
            if index % 5_000 == 0 or index == len(frame):
                print(f"[review-universe] {index:,}/{len(frame):,} responses")
    finally:
        occurrence_handle.close()
        response_handle.close()

    cluster_frame = pd.DataFrame(clusters.values()).sort_values(
        ["marker_id", "pattern_index", "n_occurrences", "normalized_matched_text"],
        ascending=[True, True, False, True],
    )
    cluster_frame.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    if occurrence_count != expected_occurrences:
        raise ValueError(
            "Occurrence denominator differs from response-level counts: "
            f"expected={expected_occurrences}, observed={occurrence_count}"
        )

    outputs = [occurrence_path, response_path, cluster_path]
    manifest: dict[str, object] = {
        "review_queue_version": REVIEW_QUEUE_VERSION,
        "pipeline_version": REGIONAL_CULTURAL_REGISTRY_VERSION,
        "review_status": "queue_built_manual_labels_blank_not_validated",
        "reviewer_type": "codex_ai_not_human_reviewer",
        "input_csv": str(input_csv),
        "input_sha256": _sha256(input_csv),
        "base_output": str(base_output),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "responses_total": len(frame),
        "responses_text_available": available_count,
        "responses_with_any_regional_marker": positive_response_count,
        "regional_occurrences": occurrence_count,
        "occurrence_context_clusters": len(cluster_frame),
        "potential_miss_responses_by_open_anchor_screen": potential_miss_response_count,
        "automated_screen_counts": screen_counts,
        "manual_precision_reviewed": 0,
        "manual_recall_responses_reviewed": 0,
        "manual_review_complete": False,
        "output_files": {path.name: _sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_review_universe(
        input_csv=args.input_csv,
        base_output=args.base_output,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

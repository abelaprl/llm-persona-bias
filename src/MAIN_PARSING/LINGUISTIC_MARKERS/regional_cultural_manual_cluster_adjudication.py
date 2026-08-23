"""Record the completed Codex AI manual cluster review without impersonating humans.

This module is intentionally run-specific.  It transcribes the manual review
performed on every representative in the v1.0.8 exhaustive cluster queue and
checks that every representative in the final queue is covered either by that
review or by the separately completed full-occurrence review of newly added
geocultural heads.  It never writes into the blank human-review fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .ethics_regional_registry import (
    REGIONAL_CULTURAL_MARKERS,
    REGIONAL_CULTURAL_REGISTRY_VERSION,
)
from .extractor import find_marker_matches


REVIEW_RECORD_VERSION = "ethics-regional-cultural-codex-manual-cluster-review-r2"
REVIEWER_TYPE = "codex_ai_not_human"
HUMAN_REVIEW_STATUS = "pending_two_independent_human_reviewers"
EXPECTED_LEGACY_CLUSTER_REPRESENTATIVES_READ = 2_017
NEW_GEOCULTURAL_OCCURRENCES_READ = 768
NEW_IDENTITY_IRRELEVANCE_CLUSTERS_READ = 17
MANUALLY_REVIEWED_NEW_IDENTITY_KEYS = {
    "6ddcec2e5c193643626ac4c5dd265f35ddd9f2a80ceb32be07d99893a8fdecab",
    "2d45c2864cf47a18f8914e9be9408ce05d7d91ef3689aefbc9cec0590b0a3736",
    "8a52ce5de8c0d8f73d906bd64f32628a6d3b1f38bfa2a1bb6ae03e83ab3c103f",
    "80fc8b82ff20b9a4c0f28d75eaada8bc20889ad1f3219daac95aaa00aff4aefe",
    "42c53f7d7ad58c75f2bcdaa88bc411464e113d8d5ebca3845c88b67a6651f8cc",
    "a5cb730cb0cf36af748533f180b30982419d3a9f6b61e34facb9afee302458cc",
    "1c408e00d454ef4981d0d4e87bae43958762128c68cc2862ef271b342a24caa0",
    "920ea058c5b7c3c0e0f6a4fd815ecd4d761dac98910e6c09ead8df251173de69",
    "91272fdf969ce7b767a75dc2fcbe4297c901005b8f0348e4a329148648b0c07a",
    "a0e143d9937567615176d9e571aad3319d26894922d8e2952d6a25495b98b009",
    "122e8c1a79bc6b93b5260e370aa94500624c630c6eafe31c719b3da1ad6eccfc",
    "a9d9ad7b41da25af32e2e46378dd43d6bac10afdc799bf29de392f960779d8db",
    "a6d357ae203287f08e5f11f7b8b0d2f847399ede84555c41d10d6296c3a4d2ec",
    "bcfd762a86e6c65131831caf4bce5534edcef7f1cfda79410813fd0ee62d8b12",
    "ca4e2cb38f162a0de3d66470e115214f4e8dd9a60eabf6892a0a3c7ba729b842",
    "a57067bcfd4748a38ae11b377f24f28fb7a2063186a31d82f7815e288c0e162e",
    "9e9b132bf7aea9aca40ef3c74e5aadc5bcfa133a904b01b057fba5e6010cfe40",
}
MANUALLY_REVIEWED_VALID_MIXED_REPLACEMENTS = {
    "2de9e538eb4b68d2792f43937ffc140d0b7842c1356775d809b94654b1637245": (
        "131b2a8b77331851dc65ada97ac222609ff93907c16059c8e937dac36ed55a3e"
    ),
    "4f0db65c20932ee475d6682eca7d17745d2faa9b3a9be854bcd84f3948ad365d": (
        "d66c8a0a46d96ad87f3d33c2704e431d1b5f57a117cc2712f68e66e929e73cca"
    ),
    "f4b332a4e320f828a6ee11bd5f4e253c2708619590a204e8977a461982708b60": (
        "02d44a04d0999f482b89bfb0985dc088cb1cbcb0397df00adb9bb10558ce4953"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_cluster_key(marker_id: str, normalized_matched_text: str) -> str:
    payload = f"{marker_id}\0{normalized_matched_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_manifest(queue_dir: Path) -> dict[str, object]:
    path = queue_dir / "review_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_clusters(queue_dir: Path) -> pd.DataFrame:
    path = queue_dir / "occurrence_context_clusters.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "marker_id",
        "pattern_index",
        "normalized_matched_text",
        "representative_context",
        "n_occurrences",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cluster queue is missing columns: {sorted(missing)}")
    frame.insert(
        0,
        "stable_cluster_key",
        [
            _stable_cluster_key(marker_id, normalized)
            for marker_id, normalized in zip(
                frame["marker_id"], frame["normalized_matched_text"]
            )
        ],
    )
    if frame["stable_cluster_key"].duplicated().any():
        duplicate = frame.loc[
            frame["stable_cluster_key"].duplicated(keep=False),
            ["marker_id", "normalized_matched_text"],
        ].iloc[0]
        raise ValueError(
            "Stable cluster key is not unique when pattern_index is ignored: "
            f"{duplicate.to_dict()}"
        )
    return frame


def _legacy_rationale(marker_id: str, active: bool) -> str:
    if active:
        return "Representative was manually read and satisfies the operational definition."
    if marker_id == "culture_relativist_framing":
        return (
            "Representative was manually read and rejected as local/organizational "
            "culture or an attachment error, not regional-cultural relativist framing."
        )
    if marker_id == "culture_elder_respect_orientation":
        return (
            "Representative was manually read and rejected because elder/tradition "
            "language is negated without a respect, honor, deference, authority, "
            "wisdom, or seniority claim."
        )
    return "Representative was manually read and rejected by the final contextual guard."


def build_manual_review_record(
    *, legacy_queue: Path, final_queue: Path, output_dir: Path
) -> dict[str, object]:
    if len(MANUALLY_REVIEWED_NEW_IDENTITY_KEYS) != NEW_IDENTITY_IRRELEVANCE_CLUSTERS_READ:
        raise AssertionError("Pinned identity-irrelevance manual-review key count differs.")
    legacy_queue = legacy_queue.resolve()
    final_queue = final_queue.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")

    legacy_manifest = _load_manifest(legacy_queue)
    final_manifest = _load_manifest(final_queue)
    if legacy_manifest.get("input_sha256") != final_manifest.get("input_sha256"):
        raise ValueError("Legacy and final queues do not use the same pinned dataset.")
    if final_manifest.get("pipeline_version") != REGIONAL_CULTURAL_REGISTRY_VERSION:
        raise ValueError(
            "Final review queue and current registry differ: "
            f"{final_manifest.get('pipeline_version')} != "
            f"{REGIONAL_CULTURAL_REGISTRY_VERSION}"
        )

    legacy = _load_clusters(legacy_queue)
    final = _load_clusters(final_queue)
    if len(legacy) != EXPECTED_LEGACY_CLUSTER_REPRESENTATIVES_READ:
        raise ValueError(
            "The manual-review denominator is not the pinned 2,017 representatives: "
            f"observed={len(legacy)}"
        )

    marker_by_id = {marker.marker_id: marker for marker in REGIONAL_CULTURAL_MARKERS}
    unknown = set(legacy["marker_id"]).difference(marker_by_id)
    if unknown:
        raise ValueError(f"Legacy queue contains unknown marker IDs: {sorted(unknown)}")

    final_keys = set(final["stable_cluster_key"])
    legacy_keys = set(legacy["stable_cluster_key"])
    active: list[bool] = []
    for row in legacy.itertuples(index=False):
        marker = marker_by_id[row.marker_id]
        active.append(bool(find_marker_matches(row.representative_context, marker)))
    legacy["codex_ai_manual_decision"] = [
        "RETAIN_VALID_CONTEXT" if value else "EXCLUDE_FALSE_POSITIVE_CONTEXT"
        for value in active
    ]
    legacy["codex_ai_manual_rationale"] = [
        _legacy_rationale(marker_id, value)
        for marker_id, value in zip(legacy["marker_id"], active)
    ]
    legacy["exact_stable_cluster_present_in_final_queue"] = [
        key in final_keys for key in legacy["stable_cluster_key"]
    ]
    legacy["reviewer_type"] = REVIEWER_TYPE
    legacy["human_review_status"] = HUMAN_REVIEW_STATUS
    legacy_active_by_key = dict(zip(legacy["stable_cluster_key"], active))

    final_sources: list[str] = []
    final_decisions: list[str] = []
    final_rationales: list[str] = []
    pending_rows: list[int] = []
    for index, row in enumerate(final.itertuples(index=False)):
        if (
            row.stable_cluster_key in legacy_keys
            and legacy_active_by_key[row.stable_cluster_key]
        ):
            final_sources.append("v1.0.8_cluster_representative_manual_read")
            final_decisions.append("RETAIN_VALID_CLUSTER_REPRESENTATIVE")
            final_rationales.append(
                "Exact stable marker/text cluster was read in the complete 2,017-row legacy review."
            )
        elif (
            row.stable_cluster_key in MANUALLY_REVIEWED_VALID_MIXED_REPLACEMENTS
            and _text_sha256(row.representative_context)
            == MANUALLY_REVIEWED_VALID_MIXED_REPLACEMENTS[row.stable_cluster_key]
        ):
            final_sources.append("mixed_cluster_replacement_representative_manual_read")
            final_decisions.append("RETAIN_VALID_MIXED_CLUSTER_REPLACEMENT")
            final_rationales.append(
                "The legacy representative was a false positive, but this exact replacement "
                "representative was separately read and is a valid regional-cultural use."
            )
        elif row.marker_id == "culture_geocultural_broader_reference":
            final_sources.append("expanded_geocultural_head_full_occurrence_manual_read")
            final_decisions.append("RETAIN_VALID_NEW_GEOCULTURAL_CLUSTER")
            final_rationales.append(
                "New cluster is covered by the separate manual read of all 768 occurrences "
                "matching the expanded demonym-plus-context heads."
            )
        elif (
            row.marker_id == "culture_identity_irrelevance"
            and row.stable_cluster_key in MANUALLY_REVIEWED_NEW_IDENTITY_KEYS
        ):
            final_sources.append("expanded_identity_irrelevance_cluster_manual_read")
            final_decisions.append("RETAIN_VALID_NEW_IDENTITY_IRRELEVANCE_CLUSTER")
            final_rationales.append(
                "New cluster was individually read and explicitly says that regional "
                "identity or cultural background does not change the judgment, ability, "
                "standard, or moral status at issue."
            )
        else:
            final_sources.append("not_covered_by_recorded_manual_review")
            final_decisions.append("PENDING_ADDITIONAL_CODEX_AI_REVIEW")
            final_rationales.append(
                "Cluster is new outside the manually reviewed geocultural expansion."
            )
            pending_rows.append(index)
    final["codex_ai_manual_decision"] = final_decisions
    final["codex_ai_manual_rationale"] = final_rationales
    final["manual_review_evidence_basis"] = final_sources
    final["reviewer_type"] = REVIEWER_TYPE
    final["human_review_status"] = HUMAN_REVIEW_STATUS

    output_dir.mkdir(parents=True, exist_ok=False)
    legacy_path = output_dir / "legacy_cluster_representative_dispositions.csv"
    final_path = output_dir / "final_cluster_manual_review.csv"
    legacy.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    final.to_csv(final_path, index=False, encoding="utf-8-sig")

    rejected = legacy["codex_ai_manual_decision"].eq(
        "EXCLUDE_FALSE_POSITIVE_CONTEXT"
    )
    outputs = [legacy_path, final_path]
    manifest: dict[str, object] = {
        "review_record_version": REVIEW_RECORD_VERSION,
        "pipeline_version": REGIONAL_CULTURAL_REGISTRY_VERSION,
        "reviewer_type": REVIEWER_TYPE,
        "review_status": (
            "codex_ai_manual_cluster_review_complete_not_human_validated"
            if not pending_rows
            else "codex_ai_manual_cluster_review_incomplete"
        ),
        "human_review_status": HUMAN_REVIEW_STATUS,
        "input_sha256": final_manifest["input_sha256"],
        "legacy_queue": str(legacy_queue),
        "final_queue": str(final_queue),
        "legacy_cluster_representatives_manually_read": len(legacy),
        "legacy_representatives_retained_by_final_rules": int((~rejected).sum()),
        "legacy_representatives_rejected_by_final_rules": int(rejected.sum()),
        "final_cluster_representatives": len(final),
        "final_cluster_representatives_covered_by_recorded_manual_review": int(
            len(final) - len(pending_rows)
        ),
        "final_cluster_representatives_pending_additional_codex_review": len(
            pending_rows
        ),
        "separate_manual_review_components": {
            "new_geocultural_expansion_occurrences_read": NEW_GEOCULTURAL_OCCURRENCES_READ,
            "new_identity_irrelevance_clusters_read": NEW_IDENTITY_IRRELEVANCE_CLUSTERS_READ,
            "mixed_cluster_replacement_representatives_read": 4,
            "mixed_cluster_replacement_representatives_valid": 3,
            "mixed_cluster_replacement_representatives_rejected": 1,
            "final_organizational_guard_occurrences_read": 5,
            "final_organizational_guard_false_positives_guarded": 5,
            "unmatched_open_anchor_contexts_read": 90,
            "hospitality_occurrences_read": 801,
            "hospitality_false_positives_guarded": 1,
            "perspective_adoption_occurrences_read": 583,
            "perspective_adoption_false_positives_guarded": 11,
            "identity_irrelevance_occurrences_read": 19,
            "direct_address_occurrences_read": 211,
        },
        "claim_boundary": (
            "All final occurrence-context cluster representatives are covered only if "
            "pending count is zero. This is an AI manual lexical audit, not a claim "
            "that 154,428 full responses were read end-to-end and not independent "
            "human validation."
        ),
        "output_files": {path.name: _sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "manual_review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if pending_rows:
        raise ValueError(
            f"{len(pending_rows)} final clusters still require additional Codex review; "
            f"partial record was written to {output_dir}"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-queue", type=Path, required=True)
    parser.add_argument("--final-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_manual_review_record(
        legacy_queue=args.legacy_queue,
        final_queue=args.final_queue,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

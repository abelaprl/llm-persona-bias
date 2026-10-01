"""Exhaustive row-level pattern discovery audit for ETHICS Regional/Cultural.

The script scans every response exactly once and writes a new, write-once audit
bundle.  It inventories all registered patterns, broad open-vocabulary anchor
families, residual context signatures, and the already completed cluster audit.
It never changes or promotes a marker rule and never uses ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "reports" / "Regional_Cultural_Persona"
)
MARKER_DIR = (
    REPORT_ROOT
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16"
)
QUEUE_DIR = (
    REPORT_ROOT
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_exhaustive_review_queue_v1"
)
CLUSTER_REVIEW_DIR = (
    REPORT_ROOT
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_ai_manual_cluster_review_r2"
)
CODEX_AUDIT_DIR = (
    REPORT_ROOT
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_audit_r1"
)
DEFAULT_OUTPUT_DIR = (
    REPORT_ROOT
    / "linguistic_markers"
    / "audit"
    / "regional_cultural_ethics_pattern_coverage_v2_0_2_20260827"
)

EXPECTED_ROWS = 154_428
EXPECTED_TEXT_AVAILABLE = 151_827
EXPECTED_REGIONAL_MARKER_RESPONSES = 40_099
EXPECTED_OCCURRENCES = 204_660
EXPECTED_CLUSTERS = 1_876
EXPECTED_MARKERS = 23
EXPECTED_ANCHOR_FORMS = 132
EXPECTED_INPUT_SHA256 = (
    "09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5"
)

# These anchors deliberately over-generate.  A residual hit is a discovery
# candidate, not an automatic true positive.  This separates recall screening
# from rule promotion and avoids silently expanding the operational construct.
DISCOVERY_PATTERNS = {
    "regional_identity": re.compile(
        r"\b(?:north[- ]america(?:n)?s?|east[- ]asia(?:n)?s?|south[- ]asia(?:n)?s?|"
        r"sub[- ]saharan[- ]africa(?:n)?s?|africa(?:n)?s?|asia(?:n)?s?|america(?:n)?s?|"
        r"western|chinese|japanese|korean|indian|pakistani|bangladeshi|nigerian|"
        r"ghanaian|kenyan|south[- ]african)\b",
        re.IGNORECASE,
    ),
    "cultural_meta": re.compile(
        r"\b(?:cultur\w*|heritage|tradition\w*|custom\w*|norms?|worldviews?|upbringing)\b",
        re.IGNORECASE,
    ),
    "collective_individual": re.compile(
        r"\b(?:collectiv\w*|communal\w*|individualis\w*|autonom\w*|self[- ]reliance|"
        r"common good|social cohesion|group harmony|group consensus|ubuntu)\b",
        re.IGNORECASE,
    ),
    "family_community": re.compile(
        r"\b(?:famil(?:y|ies|ial)|kinship|communit(?:y|ies)|familial bonds?|kinship ties?)\b",
        re.IGNORECASE,
    ),
    "reciprocity_interdependence": re.compile(
        r"\b(?:reciproc\w*|interdepend\w*|interconnect\w*|mutual aid)\b",
        re.IGNORECASE,
    ),
    "hospitality_generosity": re.compile(
        r"\b(?:hospitality|generos\w*)\b", re.IGNORECASE
    ),
    "elder_respect": re.compile(
        r"\b(?:elders?|seniority|deference)\b", re.IGNORECASE
    ),
    "harmony_face": re.compile(
        r"\b(?:harmony|harmonious|face[- ]saving|save face|saving face|loss of face)\b",
        re.IGNORECASE,
    ),
    "hierarchy_authority": re.compile(
        r"\b(?:hierarch\w*|power distance|authority|seniority)\b", re.IGNORECASE
    ),
    "named_concept": re.compile(
        r"\b(?:filial piety|guanxi|ubuntu|dharma|karma|karmic|ahimsa|seva|namaste|"
        r"vasudhaiva kutumbakam|ujamaa|harambee|confucian\w*|taoism|daoism|zen|"
        r"feng shui|gongfu)\b",
        re.IGNORECASE,
    ),
    "prompt_practice_or_name": re.compile(
        r"\b(?:thanksgiving|ch[uū]nji[eé]|hong ?bao|red envelopes?|diwali|deepavali|"
        r"diyas?|mehendi|mehndi|henna ceremony|harvest (?:celebration|festival)|"
        r"matching fabric|tea ceremony|james mitchell|wei chen|priya kapoor|"
        r"kwame asante)\b",
        re.IGNORECASE,
    ),
    "identity_accommodation": re.compile(
        r"\b(?:cultural (?:background|identity|perspective|context|lens)|regional "
        r"(?:background|identity|perspective)|from (?:a|an|the|your) (?:north[- ]american|"
        r"east[- ]asian|south[- ]asian|sub[- ]saharan[- ]african) (?:perspective|lens|context))\b",
        re.IGNORECASE,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    return parser.parse_args()


def _normalise_context(text: str, start: int, end: int) -> tuple[str, str]:
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    snippet = text[left:right]
    local_start = start - left
    local_end = end - left
    signature = snippet[:local_start] + " <ANCHOR> " + snippet[local_end:]
    signature = re.sub(r"\d+(?:[.,]\d+)?", "<NUM>", signature.casefold())
    signature = re.sub(r"\s+", " ", signature).strip()
    raw = re.sub(r"\s+", " ", snippet).strip()
    return raw, signature


def _registry_inventory(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regional = registry[registry["registry_scope"].eq("regional_cultural_specific")]
    for marker in regional.itertuples(index=False):
        patterns = json.loads(marker.patterns)
        for index, pattern in enumerate(patterns):
            rows.append(
                {
                    "marker_id": marker.marker_id,
                    "family": marker.family,
                    "measurement": marker.measurement,
                    "analysis_role": marker.analysis_role,
                    "pattern_index": index,
                    "pattern": pattern["pattern"],
                    "pattern_note": pattern.get("note", ""),
                    "operational_definition": marker.operational_definition,
                    "exclusion_rule": marker.exclusion_rule,
                    "theoretical_source": marker.theoretical_source,
                    "validation_status": marker.validation_status,
                }
            )
    return pd.DataFrame(rows)


def _marker_coverage(registry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    regional_ids = registry.loc[
        registry["registry_scope"].eq("regional_cultural_specific"), "marker_id"
    ].tolist()
    usecols = [
        "model_id",
        "persona",
        "subset",
        "text_available",
        "word_count",
        *[f"{marker}__count" for marker in regional_ids],
        *[f"{marker}__present" for marker in regional_ids],
    ]
    frame = pd.read_csv(
        MARKER_DIR / "response_level_markers.csv.gz",
        usecols=usecols,
        low_memory=False,
    )
    for column in [c for c in frame if c.endswith(("__count", "__present"))]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for (model_id, persona, subset), group in frame.groupby(
        ["model_id", "persona", "subset"], observed=True, sort=False
    ):
        text_available = int(group["text_available"].astype(str).str.casefold().eq("true").sum())
        for marker in regional_ids:
            present = group[f"{marker}__present"]
            count = group[f"{marker}__count"]
            rows.append(
                {
                    "model_id": model_id,
                    "persona": persona,
                    "subset": subset,
                    "marker_id": marker,
                    "total": len(group),
                    "text_available": text_available,
                    "responses_present": int(present.fillna(0).gt(0).sum()),
                    "prevalence_all": float(present.fillna(0).gt(0).mean()),
                    "prevalence_visible": (
                        float(present.dropna().gt(0).mean()) if present.notna().any() else float("nan")
                    ),
                    "occurrences": int(count.fillna(0).sum()),
                }
            )
    any_present = frame[[f"{marker}__present" for marker in regional_ids]].fillna(0).gt(0).any(axis=1)
    totals = {
        "rows": len(frame),
        "text_available": int(frame["text_available"].astype(str).str.casefold().eq("true").sum()),
        "responses_with_any_regional_marker": int(any_present.sum()),
        "regional_occurrences": int(
            frame[[f"{marker}__count" for marker in regional_ids]].fillna(0).sum().sum()
        ),
        "word_count_sum_visible": int(pd.to_numeric(frame["word_count"], errors="coerce").fillna(0).sum()),
    }
    return pd.DataFrame(rows), totals


def scan_every_response(queue_path: Path, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    response_counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], tuple[str, str]] = {}
    aggregate: Counter[tuple[str, str, str, str, str]] = Counter()
    rows_scanned = 0
    text_rows = 0
    open_anchor_uncovered_rows = 0
    screen_labels: Counter[str] = Counter()
    for chunk in pd.read_csv(queue_path, chunksize=chunk_size, low_memory=False):
        for row in chunk.itertuples(index=False):
            rows_scanned += 1
            text = row.visible_response if isinstance(row.visible_response, str) else ""
            if text:
                text_rows += 1
            has_marker = isinstance(row.present_regional_markers, str) and bool(
                row.present_regional_markers.strip()
            )
            if isinstance(row.uncovered_anchor_forms, str) and row.uncovered_anchor_forms.strip():
                open_anchor_uncovered_rows += 1
            screen_labels[str(row.automated_screen_label)] += 1
            seen_signatures: set[tuple[str, str]] = set()
            seen_categories: set[str] = set()
            for category, pattern in DISCOVERY_PATTERNS.items():
                matches = list(pattern.finditer(text))
                if not matches:
                    continue
                seen_categories.add(category)
                for match in matches:
                    raw, signature = _normalise_context(text, match.start(), match.end())
                    key = (category, signature)
                    occurrence_counts[key] += 1
                    seen_signatures.add(key)
                    examples.setdefault(key, (row.response_key, raw))
            for key in seen_signatures:
                response_counts[key] += 1
            for category in DISCOVERY_PATTERNS:
                aggregate[(row.model_id, row.persona, row.subset, category, "total")] += 1
                if category in seen_categories:
                    aggregate[(row.model_id, row.persona, row.subset, category, "anchor_present")] += 1
                    if has_marker:
                        aggregate[(row.model_id, row.persona, row.subset, category, "anchor_and_registered_marker")] += 1
                    else:
                        aggregate[(row.model_id, row.persona, row.subset, category, "anchor_without_registered_marker")] += 1
        print(f"[scan] {rows_scanned:,}/{EXPECTED_ROWS:,} responses", flush=True)

    inventory_rows = []
    for key, occurrences in occurrence_counts.most_common():
        response_key, context = examples[key]
        inventory_rows.append(
            {
                "anchor_category": key[0],
                "normalized_context_signature": key[1],
                "occurrences": occurrences,
                "responses": response_counts[key],
                "example_response_key": response_key,
                "example_context": context,
                "interpretation": "discovery_candidate_not_automatic_marker",
            }
        )
    aggregate_rows = []
    keys = sorted({key[:4] for key in aggregate})
    for model_id, persona, subset, category in keys:
        prefix = (model_id, persona, subset, category)
        total = aggregate[prefix + ("total",)]
        present = aggregate[prefix + ("anchor_present",)]
        with_marker = aggregate[prefix + ("anchor_and_registered_marker",)]
        without_marker = aggregate[prefix + ("anchor_without_registered_marker",)]
        aggregate_rows.append(
            {
                "model_id": model_id,
                "persona": persona,
                "subset": subset,
                "anchor_category": category,
                "total": total,
                "anchor_present": present,
                "anchor_rate": present / total if total else float("nan"),
                "anchor_and_registered_marker": with_marker,
                "anchor_without_registered_marker": without_marker,
                "residual_is_not_automatically_a_miss": True,
            }
        )
    summary = {
        "rows_scanned_once": rows_scanned,
        "rows_with_visible_response_in_queue": text_rows,
        "discovery_anchor_categories": len(DISCOVERY_PATTERNS),
        "distinct_context_signatures": len(occurrence_counts),
        "discovery_anchor_occurrences": int(sum(occurrence_counts.values())),
        "rows_with_uncovered_registered_open_anchor": open_anchor_uncovered_rows,
        "automated_screen_labels": dict(screen_labels),
        "registered_open_anchor_potential_miss_rows": int(
            sum(count for label, count in screen_labels.items() if "POTENTIAL_MISS" in label)
        ),
    }
    return pd.DataFrame(inventory_rows), pd.DataFrame(aggregate_rows), summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Audit output is write-once: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    registry_path = MARKER_DIR / "marker_registry.csv"
    queue_path = QUEUE_DIR / "response_review_queue.csv.gz"
    forms_path = MARKER_DIR / "candidate_anchor_forms.csv"
    clusters_path = QUEUE_DIR / "occurrence_context_clusters.csv"
    final_review_path = CLUSTER_REVIEW_DIR / "final_cluster_manual_review.csv"
    canonical_path = REPORT_ROOT / "_combined" / "all_models_results_full.csv"
    for path in (
        registry_path,
        queue_path,
        forms_path,
        clusters_path,
        final_review_path,
        canonical_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(canonical_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("Canonical Regional/Cultural CSV hash changed.")

    registry = pd.read_csv(registry_path, low_memory=False)
    inventory = _registry_inventory(registry)
    regional_registry = registry[registry["registry_scope"].eq("regional_cultural_specific")]
    if len(regional_registry) != EXPECTED_MARKERS:
        raise ValueError(f"Expected {EXPECTED_MARKERS} regional markers; got {len(regional_registry)}")

    print("[1/4] Aggregate every registered marker by model/persona/subset", flush=True)
    marker_coverage, marker_totals = _marker_coverage(registry)
    print("[2/4] Scan all response texts and build broad-anchor concordance", flush=True)
    context_inventory, anchor_coverage, scan_summary = scan_every_response(
        queue_path, args.chunk_size
    )
    print("[3/4] Validate prior open-anchor and cluster coverage", flush=True)
    forms = pd.read_csv(forms_path, low_memory=False)
    clusters = pd.read_csv(clusters_path, low_memory=False)
    final_review = pd.read_csv(final_review_path, low_memory=False)
    if len(forms) != EXPECTED_ANCHOR_FORMS or len(clusters) != EXPECTED_CLUSTERS:
        raise ValueError("Prior form/cluster inventory count changed.")
    if len(final_review) != EXPECTED_CLUSTERS:
        raise ValueError("Final cluster review does not cover every cluster.")
    if final_review["codex_ai_manual_decision"].isna().any():
        raise ValueError("A final cluster lacks its AI review disposition.")

    contracts = {
        "rows_scanned_once": scan_summary["rows_scanned_once"] == EXPECTED_ROWS,
        "response_rows_match_marker_surface": marker_totals["rows"] == EXPECTED_ROWS,
        "text_available_match": marker_totals["text_available"] == EXPECTED_TEXT_AVAILABLE,
        "regional_marker_response_count_match": marker_totals[
            "responses_with_any_regional_marker"
        ] == EXPECTED_REGIONAL_MARKER_RESPONSES,
        "regional_occurrence_count_match": marker_totals["regional_occurrences"]
        == EXPECTED_OCCURRENCES,
        "registry_has_23_regional_markers": len(regional_registry) == EXPECTED_MARKERS,
        "every_registry_pattern_inventoried": inventory.groupby("marker_id").ngroups
        == EXPECTED_MARKERS,
        "open_anchor_forms_complete": len(forms) == EXPECTED_ANCHOR_FORMS,
        "registered_open_anchor_potential_miss_zero": scan_summary[
            "registered_open_anchor_potential_miss_rows"
        ] == 0,
        "all_occurrence_clusters_have_ai_disposition": len(final_review)
        == EXPECTED_CLUSTERS
        and final_review["codex_ai_manual_decision"].notna().all(),
        "human_review_still_pending": final_review["human_review_status"]
        .eq("pending_two_independent_human_reviewers")
        .all(),
        "ground_truth_used": False,
        "hidden_reasoning_used": False,
        "registry_modified": False,
    }
    contracts = {name: bool(value) for name, value in contracts.items()}
    if not all(value is False if key in {"ground_truth_used", "hidden_reasoning_used", "registry_modified"} else bool(value) for key, value in contracts.items()):
        raise ValueError(f"Pattern coverage contract failed: {contracts}")

    print("[4/4] Write write-once audit artifacts", flush=True)
    inventory.to_csv(output_dir / "registered_pattern_inventory.csv", index=False)
    marker_coverage.to_csv(
        output_dir / "marker_coverage_by_model_persona_subset.csv.gz",
        index=False,
        compression="gzip",
    )
    context_inventory.to_csv(
        output_dir / "broad_anchor_context_inventory.csv.gz",
        index=False,
        compression="gzip",
    )
    anchor_coverage.to_csv(
        output_dir / "broad_anchor_coverage_by_model_persona_subset.csv.gz",
        index=False,
        compression="gzip",
    )
    forms.to_csv(output_dir / "registered_open_anchor_forms_snapshot.csv", index=False)
    final_review.to_csv(
        output_dir / "final_cluster_ai_review_snapshot.csv.gz",
        index=False,
        compression="gzip",
    )
    coverage_note = """# Batas klaim coverage pola Regional/Cultural ETHICS

Seluruh 154.428 baris diproses satu per satu oleh scan programatik. Seluruh
regex pada 23 marker, 132 bentuk open-anchor, dan 1.876 representative cluster
diinventarisasi. Broad-anchor concordance sengaja over-generating untuk mencari
residual phrase yang belum masuk registry.

Residual broad-anchor bukan otomatis marker yang terlewat: banyak merupakan isi
skenario, kultur organisasi, hubungan keluarga biasa, atau penggunaan kata yang
tidak memenuhi definisi operasional. Tidak ada aturan baru dipromosikan dari
scan ini. Klaim yang sah adalah coverage penuh terhadap inventori dan anchor
terdaftar, bukan recall semantik 100% atas bahasa alami yang bersifat open-world.

Audit cluster tetap merupakan pembacaan AI, bukan dua reviewer manusia
independen. Karena itu registry dan seluruh inferensi tetap berstatus candidate.
"""
    (output_dir / "COVERAGE_CLAIM.md").write_text(coverage_note, encoding="utf-8")

    source_paths = {
        "canonical_csv": canonical_path,
        "marker_registry": registry_path,
        "response_review_queue": queue_path,
        "candidate_anchor_forms": forms_path,
        "occurrence_context_clusters": clusters_path,
        "final_cluster_manual_review": final_review_path,
        "codex_audit_manifest": CODEX_AUDIT_DIR / "audit_manifest.json",
    }
    summary = {
        "audit_version": "regional-cultural-ethics-pattern-coverage-v2.0.2",
        "status": "automated_and_ai_audit_complete_human_validation_pending",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "canonical_sha256": EXPECTED_INPUT_SHA256,
        "marker_totals": marker_totals,
        "scan_summary": scan_summary,
        "registered_markers": len(regional_registry),
        "registered_regex_patterns": len(inventory),
        "registered_open_anchor_forms": len(forms),
        "occurrence_context_clusters": len(clusters),
        "final_cluster_ai_dispositions": len(final_review),
        "contracts": contracts,
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "interpretation": {
            "every_response_programmatically_scanned": True,
            "semantic_recall_100_percent_claimed": False,
            "broad_anchor_residuals_are_automatic_misses": False,
            "human_validation_complete": False,
        },
    }
    (output_dir / "pattern_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        **summary,
        "output_dir": str(output_dir),
        "generated_file_sha256": hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

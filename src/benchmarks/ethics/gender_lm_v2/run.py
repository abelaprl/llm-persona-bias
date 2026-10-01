"""Reproducible candidate pipeline; every stage creates new artifacts only."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import time
import importlib.metadata

import pandas as pd
from . import VERSION
from .detectors import detect_markers, load_registry
from .references import annotate_references
from .sources import KEYS, load_source, reserve_output, sha256
from .surfaces import build_surfaces

DEFAULT_SOURCE = Path("result/ETHICS/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv")
BASELINE = Path("result/ETHICS/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0")
SURFACE_COLUMNS = ["raw_text", "visible_text", "explanation_text", "structure_text", "layout_fidelity", "offset_map", "surface_flags"]
SPAN_COLUMNS = ["response_id", *KEYS, "marker_id", "rule_id", "start", "end", "text", "surface_name", "flags", "evidence_strength", "rule_status", "reference_type"]


def check_mode(mode: str, audit_manifest: Path | None) -> None:
    if mode == "candidate":
        return
    if audit_manifest is None:
        raise ValueError("Validated mode requires an independent human audit manifest; candidate extraction remains available.")
    # A JSON claim is not a substitute for checked reference evidence.
    from .audit import evaluate_audit, promotion_decisions
    manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
    annotations_path = audit_manifest.parent / manifest["annotations_csv"]
    if sha256(annotations_path) != manifest.get("annotations_sha256"):
        raise ValueError("Annotation hash mismatch")
    annotations = pd.read_csv(annotations_path)
    decisions = promotion_decisions(annotations)
    summaries = evaluate_audit(annotations, load_registry())
    required_rules = {pattern["rule_id"] for marker in load_registry() for pattern in marker["patterns"]}
    if {decision.rule_id for decision in decisions} != required_rules:
        raise ValueError("Audit must cover every registered rule, not a selected subset")
    if not decisions or any(item.status != "accepted" for item in decisions):
        raise ValueError("Independent audit has not accepted every supplied rule")
    # Validated release requires separate reviewed instrumentation and parser evidence.
    if not manifest.get("parser_validation_verified") or not manifest.get("independent_reference_verified"):
        raise ValueError("Independent reference and parser validation evidence are incomplete")
    if "target_status" not in summaries or not summaries.target_status.eq("demonstrated_individual").all():
        raise ValueError("Per-marker precision/recall 99% has not been demonstrated by the audit evaluator")
    if manifest.get("registry_sha256") != sha256(Path(__file__).with_name("registry.json")):
        raise ValueError("Audit must refer to the current immutable registry")
    if manifest.get("instrument_hashes") != instrument_hashes():
        raise ValueError("Audit must bind the surface and detector code, not only the lexicon")


def instrument_hashes() -> dict:
    base = Path(__file__).parent
    names = ["surfaces.py", "detectors.py", "references.py", "contracts.py", "_baseline/extractor.py", "_baseline/registry.py"]
    return {name: sha256(base / name) for name in names}


def evidence_path(output: Path) -> Path:
    reviewed = output / "marker_spans_reviewed.csv.gz"
    return reviewed if reviewed.exists() else output / "marker_spans.csv.gz"


def finalize_evidence(output: Path) -> None:
    """Append a reviewed evidence artifact; retain the original stream unchanged."""
    if (output / "evidence_review_manifest.json").exists():
        return
    lookup = pd.read_csv(BASELINE / "scenario_lookup.csv", usecols=["question_id", "scenario"], keep_default_na=False)
    term_pattern = re.compile(r"\b(?:man|woman|men|women|male|female|gender)\b", re.I)
    terms = {row.question_id: set(term_pattern.findall(row.scenario.casefold())) for row in lookup.itertuples(index=False)}
    old_path = output / "marker_spans.csv.gz"
    new_path = output / "marker_spans_reviewed.csv.gz"
    changed = total = 0
    flag = "same_gender_term_in_scenario_not_coreference"
    with gzip.open(old_path, "rt", encoding="utf-8", newline="") as src, gzip.open(new_path, "xt", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if row["marker_id"].startswith("gender"):
                flags = json.loads(row["flags"])
                present = bool(terms.get(row["question_id"], set()).intersection(term_pattern.findall(row["text"].casefold())))
                if (flag in flags) != present:
                    changed += 1
                    flags = [item for item in flags if item != flag]
                    if present:
                        flags.append(flag)
                    row["flags"] = json.dumps(flags)
            writer.writerow(row)
    write_json_new(output / "evidence_review_manifest.json", {"rows": total, "overlap_flags_corrected": changed,
                   "original_artifact": old_path.name, "original_sha256": sha256(old_path),
                   "used_artifact": new_path.name, "used_sha256": sha256(new_path),
                   "counts_changed": False, "instrument_hashes": instrument_hashes(),
                   "reason": "Whole-word scenario overlap; no substring man-in-woman or male-in-female matching."})


def extract_batch(rows: list[dict]) -> list[tuple[dict, list[dict]]]:
    registry = load_registry()
    result = []
    for row in rows:
        surface = build_surfaces(row.get("response_content_full", ""))
        spans = annotate_references(surface, detect_markers(surface), row.get("scenario", ""))
        rid = hashlib.sha256("|".join(str(row[key]) for key in KEYS).encode()).hexdigest()[:24]
        key = {"response_id": rid, **{field: row[field] for field in KEYS}}
        record = dict(key, file=row.get("file", ""), text_status=surface.status,
                      text_available=surface.status == "substantive_explanation", word_count=surface.word_count,
                      raw_text=surface.raw_text, visible_text=surface.visible_text, explanation_text=surface.explanation_text,
                      structure_text=surface.structure_text, layout_fidelity=surface.layout_fidelity,
                      offset_map=surface.offset_map, surface_flags=json.dumps(surface.flags),
                      word_count_method="unicode_word_regex_v2_numeric_tokens_in_explanation", rule_status="candidate")
        counts = Counter(span.marker_id for span in spans)
        for item in registry:
            mid = item["marker_id"]
            count = counts[mid] if record["text_available"] else None
            record[mid + "__count"] = count
            record[mid + "__present"] = int(count > 0) if count is not None else None
            record[mid + "__density"] = 100 * count / surface.word_count if count is not None and surface.word_count else None
        evidence = []
        for span in spans:
            values = asdict(span)
            values["flags"] = json.dumps(values["flags"])
            evidence.append({**key, **values})
        result.append((record, evidence))
    return result


def write_json_new(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def extract(source: Path, output: Path, workers: int = 4, limit: int | None = None) -> dict:
    started = time.monotonic()
    extraction_instrument = instrument_hashes()
    extraction_registry_hash = sha256(Path(__file__).with_name("registry.json"))
    source_hash = sha256(source)
    config = json.loads(Path(__file__).with_name("source_config.json").read_text(encoding="utf-8"))
    if source.resolve() == Path(config["source"]).resolve() and source_hash != config["sha256"]:
        raise ValueError("Frozen source has changed; choose and audit a new source version")
    frame = load_source(source)
    if limit is None and source.resolve() == Path(config["source"]).resolve() and len(frame) != config["expected_rows"]:
        raise ValueError("Frozen source row count has changed")
    if limit is not None:
        frame = frame.head(limit)
    lookup_path = BASELINE / "scenario_lookup.csv"
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path, usecols=["question_id", "scenario"], keep_default_na=False)
        frame = frame.merge(lookup, on="question_id", how="left", validate="many_to_one")
    rows = frame.fillna("").to_dict("records")
    batches = [rows[start:start + 200] for start in range(0, len(rows), 200)]
    print(f"Extracting {len(rows):,} responses; {len(load_registry())} markers; workers={workers}", flush=True)
    statuses = Counter()
    with gzip.open(output / "response_markers.csv.gz", "xt", encoding="utf-8", newline="") as mf, gzip.open(output / "response_surfaces.csv.gz", "xt", encoding="utf-8", newline="") as sf, gzip.open(output / "marker_spans.csv.gz", "xt", encoding="utf-8", newline="") as ef:
        evidence_writer = csv.DictWriter(ef, fieldnames=SPAN_COLUMNS)
        evidence_writer.writeheader()
        marker_writer = surface_writer = None
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            iterator = pool.map(extract_batch, batches, chunksize=1) if pool else map(extract_batch, batches)
            processed = 0
            for batch in iterator:
                for record, evidence in batch:
                    metadata = {k: v for k, v in record.items() if k not in SURFACE_COLUMNS}
                    texts = {k: v for k, v in record.items() if k in ["response_id", *KEYS, *SURFACE_COLUMNS]}
                    if marker_writer is None:
                        marker_writer = csv.DictWriter(mf, fieldnames=list(metadata)); marker_writer.writeheader()
                        surface_writer = csv.DictWriter(sf, fieldnames=list(texts)); surface_writer.writeheader()
                    marker_writer.writerow(metadata); surface_writer.writerow(texts)
                    evidence_writer.writerows(evidence)
                    statuses[record["text_status"]] += 1
                processed += len(batch)
                if processed % 2000 == 0 or processed == len(rows):
                    print(f"Extracted {processed:,}/{len(rows):,}; elapsed={time.monotonic()-started:.0f}s", flush=True)
        finally:
            if pool:
                pool.shutdown()
    if instrument_hashes() != extraction_instrument:
        raise ValueError("Instrument code changed during extraction; incomplete artifacts must not be analyzed")
    if sha256(Path(__file__).with_name("registry.json")) != extraction_registry_hash:
        raise ValueError("Registry changed during extraction; incomplete artifacts must not be analyzed")
    manifest = {"status": "candidate", "pipeline_version": VERSION, "source": str(source.resolve()), "source_sha256": source_hash,
                "source_hash_unchanged": sha256(source) == source_hash, "rows": len(rows), "text_status": dict(statuses),
                "registry_sha256": extraction_registry_hash, "markers": len(load_registry()),
                "instrument_hashes_at_extraction": extraction_instrument,
                "limited_smoke_run": limit is not None, "workers": workers, "elapsed_seconds": time.monotonic()-started,
                "precision_recall_99_demonstrated": False, "normalization": "offset-preserving visible masks; hidden reasoning excluded"}
    if not manifest["source_hash_unchanged"]:
        raise ValueError("Source changed during extraction")
    write_json_new(output / "extraction_manifest.json", manifest)
    write_json_new(output / "registry.json", load_registry())
    return manifest


def read_records(output: Path, with_text: bool = True) -> pd.DataFrame:
    records = pd.read_csv(output / "response_markers.csv.gz")
    if with_text:
        surfaces = pd.read_csv(output / "response_surfaces.csv.gz", keep_default_na=False)
        records = records.merge(surfaces, on=["response_id", *KEYS], how="left", validate="one_to_one")
    return records


def verify_extraction(output: Path) -> dict:
    manifest = json.loads((output / "extraction_manifest.json").read_text(encoding="utf-8"))
    current_instrument = instrument_hashes()
    extraction_instrument = manifest.get("instrument_hashes_at_extraction")
    if extraction_instrument is not None and extraction_instrument != current_instrument:
        raise ValueError("Instrument code changed since extraction; choose a new versioned output")
    instrument_status = "verified_against_extraction" if extraction_instrument is not None else "historical_hash_unavailable"
    if manifest["registry_sha256"] != sha256(Path(__file__).with_name("registry.json")):
        raise ValueError("Registry changed after extraction; choose a new versioned output instead of relabeling old features")
    saved = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    if saved != load_registry():
        raise ValueError("Saved registry and current registry disagree")
    artifacts = {name: sha256(output / name) for name in ("response_markers.csv.gz", "response_surfaces.csv.gz", evidence_path(output).name)}
    prior = output / "analysis_input_provenance.json"
    if prior.exists():
        expected = json.loads(prior.read_text(encoding="utf-8"))["artifacts"]
        if artifacts != expected:
            raise ValueError("Extracted artifacts changed since analysis")
    return {"artifacts": artifacts, "source_sha256": manifest["source_sha256"], "registry_sha256": manifest["registry_sha256"],
            "packages": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "scipy", "matplotlib")},
            "analysis_config_sha256": sha256(Path(__file__).with_name("analysis_config.json")), "instrument_hashes": current_instrument,
            "instrument_hashes_at_extraction": extraction_instrument, "extraction_instrument_hash_status": instrument_status}


def baseline_delta(records: pd.DataFrame, output: Path) -> None:
    path = BASELINE / "response_level_markers.csv.gz"
    baseline = pd.read_csv(path, usecols=lambda c: c in KEYS or c.endswith(("__count", "__present", "__density")))
    joined = records.merge(baseline, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
    rows = []
    for marker in load_registry():
        mid = marker["marker_id"]
        for measurement in ("count", "present", "density"):
            old, new = f"{mid}__{measurement}_old", f"{mid}__{measurement}_new"
            if old not in joined:
                continue
            for key, group in joined.groupby(["model_id", "persona", "subset"], observed=True):
                paired = group[[old, new]].dropna()
                rows.append(dict(zip(["model_id", "persona", "subset"], key), marker_id=mid, measurement=measurement,
                                 total=len(group), old_eligible=int(group[old].notna().sum()), new_eligible=int(group[new].notna().sum()),
                                 old_sum=group[old].sum(min_count=1), new_sum=group[new].sum(min_count=1),
                                 changed_on_common=int(paired[old].ne(paired[new]).sum()), comparison_limitation="surface and rule changes combined; not a semantic error estimate"))
    pd.DataFrame(rows).to_csv(output / "baseline_delta.csv", index=False, mode="x")


def _compare_model(job):
    from .statistics import run_comparisons
    model_records, registry, resamples, permutations = job
    return run_comparisons(model_records, registry, bootstrap_resamples=resamples, availability_permutations=permutations,
                           interaction_permutations=permutations, seed=42, include_subsets=True)


def analyze(output: Path, resamples: int, permutations: int, workers: int = 4) -> None:
    from .statistics import make_contrast_specs
    from .audit import build_audit_package
    finalize_evidence(output)
    provenance = verify_extraction(output)
    write_json_new(output / "analysis_input_provenance.json", provenance)
    records = read_records(output)
    spans = pd.read_csv(evidence_path(output), keep_default_na=False)
    print("Preparing full-response audit packets", flush=True)
    source_config = json.loads(Path(__file__).with_name("source_config.json").read_text(encoding="utf-8"))
    previously_read = {tuple(pair) for pair in source_config.get("previously_read_clusters", [])}
    prior_samples = [Path("result/ETHICS/reports/gender_lm_plan_recon_20260915/ethics_response_reading/full_responses.json"),
                     Path("result/ETHICS/reports/gender_lm_99_readiness_20260915_r1/full_response_discovery_samples.json")]
    for path in prior_samples:
        for row in json.loads(path.read_text(encoding="utf-8")):
            previously_read.add((row["subset"], row["question_id"]))
    metadata = build_audit_package(records, spans, load_registry(), output / "audit", previously_read=previously_read)
    write_json_new(output / "audit_stage_manifest.json", metadata)
    print("Comparing 27 contrasts, overall and per subset", flush=True)
    numeric = records.drop(columns=[field for field in SURFACE_COLUMNS if field in records])
    jobs = [(group, load_registry(), resamples, permutations) for _, group in numeric.groupby("model_id", sort=True)]
    parts = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for model_tables in pool.map(_compare_model, jobs):
                parts.append(model_tables)
                print(f"Completed statistics {len(parts)}/{len(jobs)} models", flush=True)
    else:
        for job in jobs:
            parts.append(_compare_model(job))
            print(f"Completed statistics {len(parts)}/{len(jobs)} models", flush=True)
    stats = {name: pd.concat([part[name] for part in parts], ignore_index=True) for name in parts[0]}
    for name, table in stats.items():
        table.to_csv(output / f"{name}.csv", index=False, mode="x")
    write_json_new(output / "contrast_manifest.json", make_contrast_specs())
    baseline_delta(records, output)
    write_json_new(output / "analysis_manifest.json", {"status": "candidate", "bootstrap_resamples": resamples,
                                                       "permutations": permutations, "workers": workers,
                                                       "previously_read_question_clusters": len(previously_read),
                                                       "tables": {k: len(v) for k, v in stats.items()}})


def report(output: Path) -> None:
    from .reporting import build_report
    provenance = verify_extraction(output)
    recorded = json.loads((output / "analysis_input_provenance.json").read_text(encoding="utf-8"))
    if provenance["analysis_config_sha256"] != recorded["analysis_config_sha256"]:
        raise ValueError("Analysis configuration changed before reporting")
    if "instrument_hashes" in recorded and provenance["instrument_hashes"] != recorded["instrument_hashes"]:
        raise ValueError("Surface or detector code changed since analysis; a new versioned analysis is required")
    records = read_records(output)
    spans = pd.read_csv(evidence_path(output), keep_default_na=False)
    comparisons = pd.read_csv(output / "comparisons.csv")
    audit = pd.read_csv(output / "audit/marker_audit_summary.csv")
    extras = {"triplets": "neutral_triplet_comparisons", "interactions": "interaction_comparisons", "coverage": "pairing_coverage"}
    kwargs = {key: pd.read_csv(output / (name + ".csv")) for key, name in extras.items()}
    index = build_report(records, spans, comparisons, audit, output, registry=load_registry(), **kwargs)
    write_json_new(output / "report_stage_manifest.json", {"status": "candidate", "figures": len(index), "default_scope": "overall"})
    write_json_new(output / "manifest.json", {"status": "candidate_pending_independent_human_audit", "pipeline_version": VERSION,
                                             "source_stage": "extraction_manifest.json", "analysis_stage": "analysis_manifest.json",
                                             "report_stage": "report_stage_manifest.json", "precision_recall_99_demonstrated": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["candidate", "validated"], default="candidate")
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--stage", choices=["all", "extract", "analyze", "report"], default="all")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--limit", type=int, help="Explicit smoke subset; manifest will mark incomplete corpus")
    args = parser.parse_args()
    check_mode(args.mode, args.audit_manifest)
    if args.stage in ("all", "extract"):
        reserve_output(args.output_dir)
        extract(args.source, args.output_dir, args.workers, args.limit)
    elif not (args.output_dir / "extraction_manifest.json").exists():
        raise ValueError("Later stages require a completed extraction manifest")
    if args.stage in ("all", "analyze"):
        analyze(args.output_dir, args.bootstrap_resamples, args.permutations, args.workers)
    if args.stage in ("all", "report"):
        report(args.output_dir)


if __name__ == "__main__":
    main()

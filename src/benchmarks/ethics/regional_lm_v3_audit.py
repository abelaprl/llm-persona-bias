"""Audit sampling and release gates for Regional/Cultural ETHICS v3.

This module never scores an ETHICS answer and never consults ground truth.  It
operates on the response-level and hit-level artifacts emitted by the v3
extractor and writes auditable, blinded review packets.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SEED = 18222008
SUBSET_TARGETS = {"commonsense": 3, "deontology": 3, "justice": 2, "virtue": 2}
FORCED_DEVELOPMENT_QUESTIONS = {
    "ethics_deontology_0196",
    "ethics_virtue_0470",
}
TARGETED_PROBES = {
    "ask_or_clarify_plus_if": re.compile(r"(?is)\b(?:ask(?:ing|ed)?|clarif(?:y|ying|ied))\b.*?\bif\b"),
    "line_bullet_before_bold": re.compile(r"(?m)^\s*[-*+]\s+\*\*"),
    "line_number_before_bold": re.compile(r"(?m)^\s*\d{1,2}[.)]\s+\*\*"),
    "contracted_i_am_ai_identity": re.compile(r"(?i)\bI['’]m\s+(?:an?\s+)?(?:helpful\s+)?AI\b"),
    "custom_or_customs": re.compile(r"(?i)\bcustoms?\b"),
}


def _execution_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("rows", "responses", "items", "markers"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        return pd.DataFrame(payload)
    return pd.read_csv(path, low_memory=False)


def _column(frame: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"Kolom wajib tidak ditemukan; salah satu dari {names!r}")
    return None


def _standardize_responses(frame: pd.DataFrame) -> pd.DataFrame:
    model = _column(frame, "model_id", "model")
    persona = _column(frame, "persona", "persona_folder")
    question = _column(frame, "question_id")
    subset = _column(frame, "subset", "dataset")
    renamed = frame.rename(
        columns={model: "model_id", persona: "persona", question: "question_id", subset: "subset"}
    ).copy()
    for name in ("model_id", "persona", "question_id", "subset"):
        renamed[name] = renamed[name].astype("string")
    renamed["response_key"] = (
        renamed["model_id"] + "|" + renamed["persona"] + "|" + renamed["question_id"]
    )
    if renamed["response_key"].duplicated().any():
        examples = renamed.loc[renamed["response_key"].duplicated(), "response_key"].head().tolist()
        raise ValueError(f"response_key duplikat: {examples}")
    return renamed


def _resolve_artifact(path: Path, filename: str) -> Path | None:
    if path.is_dir():
        candidate = path / filename
    else:
        candidate = path.parent / filename
    return candidate if candidate.exists() else None


def _registry_rules(path: Path, hits: pd.DataFrame) -> pd.DataFrame:
    registry_path = _resolve_artifact(path, "registry.json")
    records: list[dict[str, Any]] = []
    if registry_path:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("markers", payload.get("registry", payload.get("rows", [])))
    else:
        csv_registry = _resolve_artifact(path, "marker_registry.csv")
        payload = _read_table(csv_registry).to_dict("records") if csv_registry else []
    if registry_path or payload:
        for marker in payload:
            patterns = marker.get("patterns") or []
            if isinstance(patterns, str):
                try:
                    patterns = json.loads(patterns)
                except json.JSONDecodeError:
                    patterns = [patterns]
            for ordinal, pattern in enumerate(patterns):
                pattern_index = pattern.get("pattern_index", ordinal) if isinstance(pattern, dict) else ordinal
                records.append(
                    {
                        "marker_id": str(marker["marker_id"]),
                        "rule_id": f'{marker["marker_id"]}:p{int(pattern_index)}',
                        "audit_class": str(marker.get("audit_class", marker.get("rule_type", "semantic"))).lower(),
                        "rule_version": marker.get("version", marker.get("rule_version", "unknown")),
                    }
                )
            for exclusion in marker.get("candidate_exclusions", []) or []:
                records.append(
                    {
                        "marker_id": str(marker["marker_id"]),
                        "rule_id": str(exclusion["rule_id"]),
                        "audit_class": str(exclusion.get("audit_class", "semantic")).lower(),
                        "rule_version": exclusion.get(
                            "rule_version",
                            marker.get("candidate_version", marker.get("version", "unknown")),
                        ),
                    }
                )
    if not records and not hits.empty:
        columns = [name for name in ("marker_id", "rule_id", "rule_version") if name in hits]
        records = hits[columns].drop_duplicates().to_dict("records")
    if not records:
        raise ValueError("Tidak ada rule di registry.json atau marker_hits.csv.gz")
    rules = pd.DataFrame(records)
    if "audit_class" not in rules:
        rules["audit_class"] = "semantic"
    if "rule_version" not in rules:
        rules["rule_version"] = "unknown"
    return rules.drop_duplicates("rule_id").sort_values("rule_id").reset_index(drop=True)


def inventory(source: Path, frozen: Path, out: Path) -> dict:
    """Join canonical response metadata and emit a dense cell-by-rule inventory."""

    source, frozen, out = Path(source), Path(frozen), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_df = _standardize_responses(_read_table(source))
    generated_path = out / "response_level.csv.gz"
    if generated_path.exists():
        response_input_path = generated_path
    elif frozen.is_dir() and (frozen / "response_level.csv.gz").exists():
        response_input_path = frozen / "response_level.csv.gz"
    elif frozen.is_dir() and (frozen / "response_level_markers.csv.gz").exists():
        response_input_path = frozen / "response_level_markers.csv.gz"
    else:
        response_input_path = frozen
    frozen_df = _standardize_responses(_read_table(response_input_path))
    source_keys, frozen_keys = set(source_df["response_key"]), set(frozen_df["response_key"])
    if source_keys != frozen_keys:
        raise ValueError(
            f"Himpunan response_key berbeda: source_only={len(source_keys-frozen_keys)}, "
            f"frozen_only={len(frozen_keys-source_keys)}"
        )

    metadata_columns = ["response_key"]
    for name in ("response_content_full", "answer_status", "extraction_confidence"):
        if name in source_df:
            metadata_columns.append(name)
    metadata = source_df[metadata_columns].rename(
        columns={
            "answer_status": "answer_status_source",
            "extraction_confidence": "extraction_confidence_source",
        }
    )
    response = frozen_df.merge(metadata, on="response_key", how="left", validate="one_to_one")
    if "answer_status_source" not in response:
        response["answer_status_source"] = response.get("answer_status", pd.NA)
    confidence = response.get(
        "extraction_confidence_source",
        response.get("extraction_confidence", pd.Series(pd.NA, index=response.index)),
    )
    response["analysis_answer_status"] = response["answer_status_source"]
    response.loc[confidence.astype("string").eq("semantic_unaudited"), "analysis_answer_status"] = "abstain"

    audit_response_path = out / "audit_response_index.csv.gz"
    if audit_response_path.exists():
        raise FileExistsError(f"Output inventory sudah ada: {audit_response_path}")
    response.to_csv(audit_response_path, index=False, compression="gzip")

    hits_path = (out / "marker_hits.csv.gz") if (out / "marker_hits.csv.gz").exists() else _resolve_artifact(response_input_path, "marker_hits.csv.gz")
    hits = _read_table(hits_path) if hits_path else pd.DataFrame()
    if not hits.empty:
        required = {"response_key", "marker_id", "rule_id"}
        missing = required - set(hits.columns)
        if missing:
            raise ValueError(f"marker_hits kehilangan kolom: {sorted(missing)}")
        if not set(hits["response_key"].dropna()).issubset(frozen_keys):
            raise ValueError("marker_hits memuat response_key di luar response_level")
    registry_context = out if (out / "registry.json").exists() or (out / "marker_registry.csv").exists() else frozen
    rules = _registry_rules(registry_context, hits)
    rules.to_csv(out / "rules.csv", index=False)

    cells = response[["model_id", "persona", "subset"]].drop_duplicates()
    dense = cells.merge(rules, how="cross")
    keyed = response[["response_key", "model_id", "persona", "subset"]]
    if hits.empty:
        hit_summary = pd.DataFrame(
            columns=[
                "model_id", "persona", "subset", "rule_id",
                "population_hit_count", "population_response_count",
                "accepted_hit_count", "rejected_hit_count",
            ]
        )
    else:
        joined_hits = hits.merge(keyed, on="response_key", how="inner", validate="many_to_one")
        if "exclusion_rule_id" in joined_hits:
            exclusions = joined_hits[
                joined_hits["exclusion_rule_id"].fillna("").astype(str).str.strip().ne("")
            ].copy()
            exclusions["rule_id"] = exclusions["exclusion_rule_id"].astype(str)
            joined_hits = pd.concat([joined_hits, exclusions], ignore_index=True)
        joined_hits["_decision"] = joined_hits.get(
            "decision", pd.Series("accepted", index=joined_hits.index)
        ).fillna("accepted").astype(str).str.lower()
        joined_hits["_accepted"] = joined_hits["_decision"].eq("accepted").astype(int)
        joined_hits["_rejected"] = joined_hits["_decision"].eq("rejected").astype(int)
        hit_summary = (
            joined_hits.groupby(["model_id", "persona", "subset", "rule_id"], dropna=False)
            .agg(
                population_hit_count=("rule_id", "size"),
                population_response_count=("response_key", "nunique"),
                accepted_hit_count=("_accepted", "sum"),
                rejected_hit_count=("_rejected", "sum"),
            )
            .reset_index()
        )
    dense = dense.merge(hit_summary, on=["model_id", "persona", "subset", "rule_id"], how="left")
    count_columns = [
        "population_hit_count", "population_response_count",
        "accepted_hit_count", "rejected_hit_count",
    ]
    dense[count_columns] = dense[
        count_columns
    ].fillna(0).astype("int64")

    summary_frame = response.copy()
    if "text_status" not in summary_frame:
        summary_frame["text_status"] = pd.NA
    summary_frame["_total"] = 1
    for status in ("committed", "abstain", "missing"):
        summary_frame[status] = summary_frame["answer_status_source"].eq(status).astype(int)
    summary_frame["technical_only"] = summary_frame["text_status"].eq("technical_only").astype(int)
    summary_frame["surface_unavailable"] = (
        summary_frame["text_status"].isin(["missing", "technical_only"])
        | summary_frame.get("text_available", pd.Series(True, index=summary_frame.index)).eq(False)
    ).astype(int)
    status_summary = (
        summary_frame.groupby(["model_id", "persona", "subset"], dropna=False)[
            ["_total", "committed", "abstain", "missing", "technical_only", "surface_unavailable"]
        ]
        .sum()
        .rename(columns={"_total": "population_response_total"})
        .reset_index()
    )
    dense = dense.merge(status_summary, on=["model_id", "persona", "subset"], how="left")
    inventory_path = out / "inventory.csv"
    dense.to_csv(inventory_path, index=False)
    return {
        "response_level_path": str(response_input_path),
        "audit_response_index_path": str(audit_response_path),
        "marker_hits_path": str(hits_path) if hits_path else None,
        "inventory_path": str(inventory_path),
        "rules_path": str(out / "rules.csv"),
        "n_responses": int(len(response)),
        "n_rules": int(len(rules)),
        "execution_code_sha256": _execution_code_sha256(),
    }


def _exposure_files(path: Path) -> list[Path]:
    accepted = [path] if path.is_file() else []
    roots = [path] if path.is_dir() else []
    regional_root = Path.cwd() / "result" / "ETHICS" / "reports" / "Regional_Cultural_Persona"
    is_repository_evidence = "result" in {part.lower() for part in path.resolve().parts}
    if is_repository_evidence and regional_root.exists() and regional_root not in roots:
        roots.append(regional_root)
    legacy_names = {
        "audit_precision_regional_hits.csv", "audit_recall_responses.csv",
        "audit_unmatched_anchor_contexts.csv", "final_cluster_ai_review_snapshot.csv.gz",
        "candidate_context_clusters_review_queue.csv.gz",
        "nonoverlap_response_ai_recall_triage_100.csv.gz",
        "custom_marker_ai_precision_full_text_51.csv.gz",
    }
    for root in roots:
      for candidate in root.rglob("*"):
        lowered = candidate.name.lower()
        accepted_format = lowered.endswith((".csv", ".csv.gz", ".json"))
        relevant = candidate.name in legacy_names or any(
            token in lowered for token in ("readback", "exposure")
        )
        if candidate.is_file() and accepted_format and relevant:
            accepted.append(candidate)
    return sorted(set(accepted))


def _build_exposure_ledger(path: Path) -> pd.DataFrame:
    ledger: list[dict[str, Any]] = []
    label_tokens = ("review_label", "manual_marker", "reviewer", "disposition", "decision", "readback")
    for file in _exposure_files(path):
        try:
            frame = _read_table(file)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if frame.empty:
            continue
        question_col = _column(frame, "question_id", required=False)
        key_col = _column(
            frame, "response_key", "representative_response_key", "example_response_key",
            required=False,
        )
        if question_col is None and key_col is None:
            continue
        model_col = _column(frame, "model_id", "model", required=False)
        persona_col = _column(frame, "persona", "persona_folder", required=False)
        for _, row in frame.iterrows():
            response_key = row.get(key_col) if key_col else pd.NA
            pieces = str(response_key).split("|") if pd.notna(response_key) else []
            question = row.get(question_col) if question_col else (pieces[2] if len(pieces) >= 3 else pd.NA)
            model = row.get(model_col) if model_col else (pieces[0] if len(pieces) >= 3 else pd.NA)
            persona = row.get(persona_col) if persona_col else (pieces[1] if len(pieces) >= 3 else pd.NA)
            if pd.isna(question):
                continue
            review_fields = [c for c in frame.columns if any(token in c.lower() for token in label_tokens)]
            reviewed = "readback" in file.name.lower() or any(
                pd.notna(row.get(c)) and str(row.get(c)).strip() for c in review_fields
            )
            ledger.append(
                {
                    "model_id": model,
                    "persona": persona,
                    "question_id": str(question),
                    "response_key": response_key,
                    "exposure_source": str(file),
                    "selected_in_historical_form": True,
                    "has_label_or_readback": bool(reviewed),
                    "clean_holdout": False,
                }
            )
    for question in sorted(FORCED_DEVELOPMENT_QUESTIONS):
        ledger.append(
            {
                "model_id": pd.NA,
                "persona": pd.NA,
                "question_id": question,
                "response_key": pd.NA,
                "exposure_source": "regional_v3_full_census_if_you_will",
                "selected_in_historical_form": True,
                "has_label_or_readback": True,
                "clean_holdout": False,
            }
        )
    return pd.DataFrame(ledger).drop_duplicates().reset_index(drop=True)


def _opaque_id(kind: str, key: str, seed: int, ordinal: int = 0) -> str:
    digest = hashlib.sha256(f"{seed}|{kind}|{key}|{ordinal}".encode()).hexdigest()[:16]
    return f"{kind[:3].upper()}-{digest}"


def _write_packet(rows: pd.DataFrame, kind: str, out: Path, seed: int) -> tuple[str, str]:
    rows = rows.reset_index(drop=True).copy()
    if "audit_id" not in rows:
        rows["audit_id"] = [
            _opaque_id(kind, str(row.get("response_key", row.get("question_id", i))), seed, i)
            for i, row in rows.iterrows()
        ]
    rows["packet_kind"] = kind
    rows["full_response_sha256"] = rows.get(
        "full_response", pd.Series("", index=rows.index)
    ).fillna("").astype(str).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    metadata = [
        c
        for c in (
            "audit_id", "packet_kind", "response_key", "model_id", "persona",
            "question_id", "subset", "audit_partition", "sampling_probability",
            "probe_id", "marker_id", "rule_id", "rule_version", "surface",
            "start", "end", "full_response_sha256",
            "raw_final_sha256",
        )
        if c in rows
    ]
    index = rows[metadata].copy()
    whitelists = {
        "precision": [
            "audit_id", "full_response", "marker_id", "rule_id", "rule_version",
            "surface", "measured_surface_text", "start", "end", "matched_text", "decision", "reason",
            "label", "reviewer_kind", "reviewer", "audit_note",
        ],
        "recall": [
            "audit_id", "full_response", "manual_marker_ids", "missed_span_notes",
            "reviewer_kind", "reviewer", "audit_note",
        ],
        "targeted": ["audit_id", "full_response", "probe_id", "diagnostic_label", "audit_note"],
        "eligibility": ["audit_id", "full_response", "text_status", "eligibility_label", "audit_note"],
    }
    front = rows[[c for c in whitelists[kind] if c in rows]].copy()
    packet_path = out / f"audit_{kind}.csv"
    index_path = out / f"audit_{kind}_index.csv"
    front.to_csv(packet_path, index=False)
    index.to_csv(index_path, index=False)
    return str(packet_path), str(index_path)


def make_audit_packets(source: Path, exposure: Path, out: Path, seed: int = SEED) -> dict:
    """Create deterministic blinded precision, recall, targeted and eligibility packets."""

    source, exposure, out = Path(source), Path(exposure), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "sample_manifest.json").exists() or (out / "audit_precision.csv").exists():
        raise FileExistsError(f"Audit packet sudah ada di {out}")
    if (out / "audit_response_index.csv.gz").exists():
        response_path = out / "audit_response_index.csv.gz"
        responses = _standardize_responses(_read_table(response_path))
    elif (out / "response_level.csv.gz").exists():
        response_path = out / "response_level.csv.gz"
        responses = _standardize_responses(_read_table(response_path))
        canonical = _standardize_responses(_read_table(source))
        text_name = _column(canonical, "response_content_full", "final_text", "visible_text")
        responses = responses.merge(
            canonical[["response_key", text_name]].rename(columns={text_name: "response_content_full"}),
            on="response_key",
            how="left",
            validate="one_to_one",
        )
    else:
        response_path = source / "response_level.csv.gz" if source.is_dir() else source
        responses = _standardize_responses(_read_table(response_path))
    text_col = _column(responses, "response_content_full", "final_text", "visible_text")
    responses = responses.rename(columns={text_col: "full_response"})
    hits_path = (out / "marker_hits.csv.gz") if (out / "marker_hits.csv.gz").exists() else _resolve_artifact(response_path, "marker_hits.csv.gz")
    hits = _read_table(hits_path) if hits_path else pd.DataFrame()
    rules = _registry_rules(out, hits)

    ledger = _build_exposure_ledger(exposure)
    ledger_path = out / "exposure_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    if ledger.empty:
        exposure_by_question = pd.DataFrame(
            columns=["question_id", "selected_in_historical_form", "has_label_or_readback"]
        )
    else:
        exposure_by_question = (
            ledger.dropna(subset=["question_id"])
            .assign(question_id=lambda frame: frame["question_id"].astype(str))
            .groupby("question_id", as_index=False)[
                ["selected_in_historical_form", "has_label_or_readback"]
            ]
            .any()
        )
    reviewed_questions = set(
        exposure_by_question.loc[
            exposure_by_question["has_label_or_readback"], "question_id"
        ]
    )
    questions = sorted(responses["question_id"].dropna().astype(str).unique())
    partition_rows = []
    for question in questions:
        digest = int(hashlib.sha256(f"{seed}|partition|{question}".encode()).hexdigest()[:16], 16)
        if question in reviewed_questions:
            partition, reason = "development", "recorded_review_or_readback"
        elif digest % 5 == 0:
            partition, reason = "development", "no_recorded_review_seeded_development"
        else:
            partition, reason = "evaluation", "no_recorded_review_seeded_evaluation"
        partition_rows.append({"question_id": question, "audit_partition": partition, "partition_reason": reason})
    partition = pd.DataFrame(partition_rows)
    partition_path = out / "question_partition.csv"
    partition.to_csv(partition_path, index=False)
    responses = responses.merge(partition, on="question_id", how="left", validate="many_to_one")
    evaluation = responses[responses["audit_partition"].eq("evaluation")].copy()

    rng_seed = int(seed)
    deficits: list[dict[str, Any]] = []
    precision_parts: list[pd.DataFrame] = []
    if not hits.empty:
        evaluation_hits = hits.merge(
            evaluation[["response_key", "full_response", "subset", "audit_partition"]],
            on="response_key",
            how="inner",
            validate="many_to_one",
        ).merge(rules[["rule_id", "audit_class"]], on="rule_id", how="left", validate="many_to_one")
        if "decision" in evaluation_hits:
            evaluation_hits = evaluation_hits[evaluation_hits["decision"].fillna("accepted").eq("accepted")]
        for ordinal, rule in rules.iterrows():
            population = evaluation_hits[evaluation_hits["rule_id"].eq(rule["rule_id"])].copy()
            quota = 50 if str(rule["audit_class"]).lower() == "explicit" else 100
            target = min(quota, len(population))
            if target:
                sample = population.sample(n=target, random_state=rng_seed + ordinal).copy()
                sample["sampling_probability"] = target / len(population)
                sample["label"] = ""
                sample["reviewer_kind"] = ""
                sample["reviewer"] = ""
                precision_parts.append(sample)
            if len(population) < quota:
                deficits.append({"packet": "precision", "stratum": rule["rule_id"], "target": quota, "actual": len(population), "reason": "population_below_quota"})
    precision = pd.concat(precision_parts, ignore_index=True) if precision_parts else pd.DataFrame(columns=["audit_id", "rule_id", "full_response", "label"])
    recall_parts: list[pd.DataFrame] = []
    available_mask = ~evaluation.get("text_status", pd.Series("available", index=evaluation.index)).isin(["missing", "technical_only"])
    if "text_available" in evaluation:
        available_mask &= evaluation["text_available"].fillna(False).astype(bool)
    recall_population = evaluation[available_mask].copy()
    for group_ordinal, ((model, persona), group) in enumerate(recall_population.groupby(["model_id", "persona"], sort=True)):
        for subset_ordinal, (subset, target) in enumerate(SUBSET_TARGETS.items()):
            cell = group[group["subset"].eq(subset)]
            actual = min(target, len(cell))
            if actual:
                sample = cell.sample(n=actual, random_state=rng_seed + 10000 + group_ordinal * 10 + subset_ordinal).copy()
                sample["sampling_probability"] = actual / len(cell)
                sample["manual_marker_ids"] = ""
                sample["missed_span_notes"] = ""
                recall_parts.append(sample)
            if actual < target:
                deficits.append({"packet": "recall", "stratum": f"{model}|{persona}|{subset}", "target": target, "actual": actual, "reason": "evaluation_population_shortfall"})
    recall = pd.concat(recall_parts, ignore_index=True) if recall_parts else pd.DataFrame(columns=["audit_id", "full_response", "manual_marker_ids", "missed_span_notes"])
    selected_keys = set(precision.get("response_key", pd.Series(dtype=str)).dropna().astype(str))
    selected_keys.update(recall.get("response_key", pd.Series(dtype=str)).dropna().astype(str))
    sampled_surfaces: dict[str, dict[str, str]] = {}
    sampled_raw_hashes: dict[str, str] = {}
    probe_matches: dict[str, list[tuple[str, str]]] = {probe: [] for probe in TARGETED_PROBES}
    surface_file = out / "surface_texts.jsonl.gz"
    if surface_file.exists():
        with gzip.open(surface_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = str(record["response_key"])
                surfaces = record.get("surfaces", {})
                visible = str(surfaces.get("visible", ""))
                if key in selected_keys:
                    sampled_surfaces[key] = surfaces
                    sampled_raw_hashes[key] = str(record.get("raw_final_sha256", ""))
                for probe_id, regex in TARGETED_PROBES.items():
                    if regex.search(visible):
                        probe_matches[probe_id].append((key, visible))
        if not precision.empty:
            precision["full_response"] = precision["response_key"].map(
                lambda key: sampled_surfaces.get(str(key), {}).get("visible", "")
            )
            precision["measured_surface_text"] = precision.apply(
                lambda row: sampled_surfaces.get(str(row["response_key"]), {}).get(str(row["surface"]), ""),
                axis=1,
            )
            span_ok = precision.apply(
                lambda row: str(row["measured_surface_text"])[int(row["start"]):int(row["end"])]
                == str(row["matched_text"]),
                axis=1,
            )
            if not span_ok.all():
                raise ValueError(f"Offset hit tidak cocok measured surface pada {int((~span_ok).sum())} baris")
            precision["raw_final_sha256"] = precision["response_key"].astype(str).map(sampled_raw_hashes)
        if not recall.empty:
            recall["full_response"] = recall["response_key"].map(
                lambda key: sampled_surfaces.get(str(key), {}).get("visible", "")
            )
            recall["raw_final_sha256"] = recall["response_key"].astype(str).map(sampled_raw_hashes)
    else:
        for probe_id, regex in TARGETED_PROBES.items():
            matched = responses[responses["full_response"].fillna("").astype(str).str.contains(regex, regex=True)]
            probe_matches[probe_id] = list(zip(matched["response_key"].astype(str), matched["full_response"].astype(str)))
        deficits.append({"packet": "surface", "stratum": "surface_texts", "target": 1, "actual": 0, "reason": "visible_surface_artifact_missing_raw_text_fallback"})

    precision_path, precision_index = _write_packet(precision, "precision", out, seed)
    recall_path, recall_index = _write_packet(recall, "recall", out, seed)

    targeted_parts = []
    response_metadata = responses.drop(columns=["full_response"], errors="ignore").set_index("response_key")
    for ordinal, probe_id in enumerate(TARGETED_PROBES):
        pairs = probe_matches[probe_id]
        matched = pd.DataFrame(pairs, columns=["response_key", "full_response"])
        if not matched.empty:
            matched = matched.join(response_metadata, on="response_key", how="left", validate="many_to_one")
        actual = min(50, len(matched))
        if actual:
            sample = matched.sample(n=actual, random_state=rng_seed + 20000 + ordinal)
            sample["probe_id"] = probe_id
            sample["sampling_probability"] = actual / len(matched)
            sample["diagnostic_label"] = ""
            targeted_parts.append(sample)
        if actual < 50:
            deficits.append({"packet": "targeted", "stratum": probe_id, "target": 50, "actual": actual, "reason": "probe_population_shortfall"})
    targeted = pd.concat(targeted_parts, ignore_index=True) if targeted_parts else pd.DataFrame(columns=["audit_id", "probe_id", "full_response", "diagnostic_label"])
    targeted_path, targeted_index = _write_packet(targeted, "targeted", out, seed)

    status = responses.get("text_status", pd.Series("available", index=responses.index))
    eligibility = responses[status.isin(["missing", "technical_only"])].copy()
    eligibility["eligibility_label"] = ""
    eligibility["sampling_probability"] = 1.0
    eligibility_path, eligibility_index = _write_packet(eligibility, "eligibility", out, seed)

    manifest = {
        "seed": seed,
        "execution_code_sha256": _execution_code_sha256(),
        "validation_scope": "internal_validation_after_discovery",
        "clean_holdout_claimed": False,
        "exposure_ledger": {
            "questions_selected_or_recorded": int(len(exposure_by_question)),
            "questions_with_recorded_review_or_readback": int(
                exposure_by_question["has_label_or_readback"].sum()
            ),
            "questions_selected_only_unfilled": int(
                (
                    exposure_by_question["selected_in_historical_form"]
                    & ~exposure_by_question["has_label_or_readback"]
                ).sum()
            ),
            "partition_ruling": (
                "Only recorded labels/readbacks force development; unfilled historical "
                "selections remain eligible for the seeded internal evaluation partition."
            ),
            "unfilled_selection_interpretation": "unknown_whether_ever_opened",
        },
        "n_question_groups": len(questions),
        "partition_counts": partition["audit_partition"].value_counts().to_dict(),
        "sampling": {
            "precision": "SRSWOR per rule from seeded internal evaluation groups without recorded prior review; quota 50 explicit/100 semantic or census when smaller; not a clean holdout",
            "recall": "SRSWOR per model/persona/subset from eligible evaluation responses; targets 3/3/2/2",
            "targeted": "diagnostic SRSWOR up to 50 per predeclared probe; not a population estimator",
            "eligibility": "census of missing and technical_only responses",
        },
        "population_counts": {
            "responses": int(len(responses)),
            "evaluation_responses": int(len(evaluation)),
            "recall_eligible": int(len(recall_population)),
            "rules": int(len(rules)),
            "if_you_will_exclusion_population": 2,
        },
        "if_you_will_exclusion": {
            "rule_id": "connective_conditional:exclude_if_you_will",
            "decision": "rejected",
            "reason": "idiomatic_if_you_will",
            "question_ids": sorted(FORCED_DEVELOPMENT_QUESTIONS),
            "audit_note": "full census of two known hits; development evidence, no independent human gold",
        },
        "deficits": deficits,
    }
    manifest_path = out / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "exposure_ledger_path": str(ledger_path),
        "partition_path": str(partition_path),
        "manifest_path": str(manifest_path),
        "precision_path": precision_path,
        "precision_index_path": precision_index,
        "recall_path": recall_path,
        "recall_index_path": recall_index,
        "targeted_path": targeted_path,
        "targeted_index_path": targeted_index,
        "eligibility_path": eligibility_path,
        "eligibility_index_path": eligibility_index,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def rule_gate(rows: list[dict], *, semantic: bool, population_hits: int | None = None) -> dict:
    """Evaluate one rule's audit evidence without treating AI labels as human gold."""

    frame = pd.DataFrame(rows)
    quota = 100 if semantic else 50
    blockers: list[str] = []
    if frame.empty:
        frame = pd.DataFrame(columns=["rule_id", "response_key", "start", "end", "label", "reviewer_kind", "reviewer", "audit_partition", "rule_version"])
    rule_ids = set(frame.get("rule_id", pd.Series(dtype=str)).dropna().astype(str))
    versions = set(frame.get("rule_version", pd.Series(dtype=str)).dropna().astype(str))
    if len(rule_ids) > 1:
        blockers.append("mixed_rules")
    if len(versions) > 1:
        blockers.append("mixed_rule_versions")
    evaluation = frame[frame.get("audit_partition", pd.Series("", index=frame.index)).eq("evaluation")].copy()
    evaluation["_label"] = evaluation.get("label", pd.Series("", index=evaluation.index)).fillna("").astype(str).str.strip().str.upper()
    if (evaluation["_label"] == "").any():
        blockers.append("blank_labels")
    hit_columns = [c for c in ("rule_id", "response_key", "start", "end") if c in evaluation]
    reviewer_columns = hit_columns + [c for c in ("reviewer_kind", "reviewer") if c in evaluation]
    duplicate_conflict = False
    if reviewer_columns:
        grouped_labels = evaluation.groupby(reviewer_columns, dropna=False)["_label"].nunique()
        duplicate_conflict = bool((grouped_labels > 1).any())
        evaluation = evaluation.drop_duplicates(reviewer_columns + ["_label"])
    if duplicate_conflict:
        blockers.append("duplicate_labels")
    valid = evaluation[evaluation["_label"].isin(["TP", "FP", "UNCERTAIN"])]
    if (valid["_label"] == "FP").any():
        blockers.append("false_positive")
    if (valid["_label"] == "UNCERTAIN").any():
        blockers.append("uncertain_labels")
    if hit_columns and not valid.empty:
        hit_labels = valid.groupby(hit_columns, dropna=False)["_label"].agg(set)
        n_fp = int(sum("FP" in labels for labels in hit_labels))
        n_uncertain = int(sum("UNCERTAIN" in labels for labels in hit_labels))
        n_tp = int(sum(labels == {"TP"} for labels in hit_labels))
        n_reviewed = int(len(hit_labels))
    else:
        n_tp = n_fp = n_uncertain = n_reviewed = 0
    population = int(population_hits) if population_hits is not None else None
    if population is None:
        required = quota
        quota_met = False
        blockers.append("population_hits_unknown")
    else:
        required = min(quota, population)
        quota_met = n_reviewed >= required and required > 0
    if not quota_met:
        blockers.append("quota_not_met")

    ai = valid[valid.get("reviewer_kind", pd.Series("", index=valid.index)).astype(str).str.lower().eq("ai")]
    human = valid[valid.get("reviewer_kind", pd.Series("", index=valid.index)).astype(str).str.lower().eq("human")]
    ai_status = "not_reviewed" if ai.empty else ("passed_zero_fp" if (ai["_label"] == "TP").all() else "failed_or_unresolved")
    if human.empty:
        human_status = "not_reviewed"
    elif not (human["_label"] == "TP").all():
        human_status = "failed_or_unresolved"
    else:
        coverage = human.groupby(hit_columns, dropna=False)["reviewer"].nunique() if hit_columns else pd.Series(dtype=int)
        covered_twice = int((coverage >= 2).sum())
        human_status = "passed_two_reviewers" if covered_twice >= required else "incomplete_two_reviewer_coverage"
    if human_status != "passed_two_reviewers":
        blockers.append("human_gate_not_met")
    ci_low, ci_high = _wilson_interval(n_tp, n_tp + n_fp)
    n_unique_responses = int(valid["response_key"].nunique()) if "response_key" in valid else 0
    return {
        "rule_id": next(iter(rule_ids), None),
        "n_hits_population": population,
        "n_reviewed": n_reviewed,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_uncertain": n_uncertain,
        "n_unique_responses": n_unique_responses,
        "quota": required,
        "quota_met": quota_met,
        "estimated_precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else None,
        "precision_ci_low": ci_low,
        "precision_ci_high": ci_high,
        "precision_ci_method": "naive_hit_binomial_descriptive_not_cluster_adjusted",
        "precision_ci_limitation": "multiple_hits_within_response_are_not_independent_bernoulli_trials",
        "zero_fp_rule_of_three_upper": min(1.0, 3 / n_reviewed) if n_reviewed and n_fp == 0 else None,
        "ai_review_status": ai_status,
        "human_review_status": human_status,
        "release_eligible": quota_met and not blockers,
        "blockers": sorted(set(blockers)),
    }


def validate_audit_input(path: Path) -> dict:
    """Validate an imported audit form without modifying it."""

    path = Path(path)
    frame = _read_table(path)
    required = {"rule_id", "response_key", "label", "reviewer_kind", "audit_partition", "rule_version"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Audit input kehilangan kolom: {missing}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "rows": int(len(frame)), "columns": list(frame.columns)}


def import_audit(path: Path, out: Path) -> dict:
    """Import a completed audit form after verifying packet identity and versions.

    The imported copy is immutable: callers must choose a new output directory
    for a corrected form.  This prevents reviewer labels from silently drifting
    away from the sampled packet.
    """

    path, out = Path(path), Path(out)
    frame = _read_table(path)
    forbidden = {
        "ground_truth",
        "is_correct",
        "gt_stance",
        "extracted_answer",
        "candidate_answer",
    }
    leaked = sorted(forbidden & set(frame.columns))
    if leaked:
        raise ValueError(f"Audit input memuat field ground-truth/prediksi terlarang: {leaked}")
    required = {
        "audit_id",
        "rule_id",
        "rule_version",
        "label",
        "reviewer_kind",
        "reviewer",
        "full_response",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Audit input kehilangan kolom: {missing}")
    if frame["audit_id"].isna().any():
        raise ValueError("audit_id kosong")
    reviewer_identity = ["audit_id", "reviewer_kind", "reviewer"]
    if all(column in frame for column in reviewer_identity) and frame.duplicated(reviewer_identity).any():
        raise ValueError("Pasangan audit_id/reviewer duplikat")
    labels = frame["label"].fillna("").astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(labels) - {"TP", "FP", "UNCERTAIN"})
    if invalid_labels:
        raise ValueError(f"Label audit kosong/tidak dikenal: {invalid_labels}")
    if frame["reviewer"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Reviewer wajib diisi")
    if not frame["reviewer_kind"].astype(str).str.lower().isin(["ai", "human"]).all():
        raise ValueError("reviewer_kind hanya boleh ai atau human")

    search_roots = [out] if out.exists() else []
    if out.parent not in search_roots:
        search_roots.append(out.parent)
    index_paths: list[Path] = []
    for root in search_roots:
        index_paths.extend(sorted(root.glob("audit_*_index.csv")))
    if not index_paths:
        raise ValueError("Indeks packet audit tidak ditemukan")
    valid_ids: set[str] = set()
    index_hashes: dict[str, str] = {}
    index_frames: list[pd.DataFrame] = []
    for index_path in index_paths:
        index_frame = _read_table(index_path)
        if "packet_kind" not in index_frame:
            index_frame["packet_kind"] = index_path.name.removeprefix("audit_").removesuffix("_index.csv")
        if "audit_id" in index_frame:
            valid_ids.update(index_frame["audit_id"].dropna().astype(str))
            index_frames.append(index_frame)
        index_hashes[index_path.name] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    invented = sorted(set(frame["audit_id"].astype(str)) - valid_ids)
    if invented:
        raise ValueError(f"audit_id tidak berasal dari packet: {invented[:5]}")
    packet_index = pd.concat(index_frames, ignore_index=True)
    selected_index = packet_index[packet_index["audit_id"].astype(str).isin(frame["audit_id"].astype(str))]
    if not selected_index["packet_kind"].eq("precision").all():
        raise ValueError("Label TP/FP/uncertain hanya dapat diimpor untuk packet precision")
    expected_text_hash = selected_index.drop_duplicates("audit_id").set_index("audit_id")["full_response_sha256"]
    submitted_text_hash = frame.set_index("audit_id")["full_response"].fillna("").astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    if not submitted_text_hash.eq(expected_text_hash.reindex(submitted_text_hash.index)).all():
        raise ValueError("Teks respons audit tidak cocok hash packet")
    binding_columns = [
        column
        for column in ("response_key", "rule_id", "rule_version", "start", "end", "audit_partition")
        if column in frame and column in selected_index
    ]
    binding = selected_index[["audit_id"] + binding_columns].drop_duplicates("audit_id")
    check = frame.merge(binding, on="audit_id", how="left", suffixes=("", "_packet"), validate="many_to_one")
    mismatched_fields = []
    for column in binding_columns:
        left = check[column].fillna("<NA>").astype(str)
        right = check[f"{column}_packet"].fillna("<NA>").astype(str)
        if not left.eq(right).all():
            mismatched_fields.append(column)
    if mismatched_fields:
        raise ValueError(f"Metadata audit tidak cocok packet index: {mismatched_fields}")

    rules_path = next((root / "rules.csv" for root in search_roots if (root / "rules.csv").exists()), None)
    if rules_path:
        rules = _read_table(rules_path)
        expected = {
            str(row["rule_id"]): str(row["rule_version"])
            for _, row in rules.drop_duplicates("rule_id").iterrows()
        }
        mismatches = [
            str(row["rule_id"])
            for _, row in frame.iterrows()
            if str(row["rule_id"]) not in expected
            or str(row["rule_version"]) != expected[str(row["rule_id"])]
        ]
        if mismatches:
            raise ValueError(f"rule_id/rule_version tidak cocok registry snapshot: {sorted(set(mismatches))[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    normalized = frame.copy()
    normalized["label"] = labels
    private = selected_index.drop_duplicates("audit_id").set_index("audit_id")
    for column in ("audit_partition", "response_key", "surface", "start", "end", "marker_id"):
        if column not in normalized and column in private:
            normalized[column] = normalized["audit_id"].map(private[column])
    audit_path = out / "audit_import.csv"
    manifest_path = out / "audit_import_manifest.json"
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Artefak audit import sudah ada di: {out}")
    normalized.to_csv(audit_path, index=False)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "source_path": str(path),
        "source_sha256": source_sha,
        "rows": int(len(normalized)),
        "packet_index_sha256": index_hashes,
        "rule_registry_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest() if rules_path else None,
        "execution_code_sha256": _execution_code_sha256(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summarize_audits(out)
    return {
        "audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        "source_sha256": source_sha,
        "rows": int(len(normalized)),
        "rule_audit_summary_path": summary["path"],
    }


def summarize_audits(out: Path) -> dict:
    """Write one gate row for every registry rule, including unreviewed zeros."""

    out = Path(out)
    rules_path = out / "rules.csv"
    if not rules_path.exists():
        raise ValueError(f"rules.csv tidak ditemukan di {out}")
    rules = _read_table(rules_path)
    inventory_path = out / "inventory.csv"
    if inventory_path.exists():
        inventory_frame = _read_table(inventory_path)
        population = (
            inventory_frame.groupby("rule_id", dropna=False)
            .agg(
                population_hit_count=("population_hit_count", "sum"),
                accepted_hit_count=("accepted_hit_count", "sum"),
                rejected_hit_count=("rejected_hit_count", "sum"),
            )
            .reset_index()
        )
    else:
        population = pd.DataFrame(
            columns=["rule_id", "population_hit_count", "accepted_hit_count", "rejected_hit_count"]
        )
    audit_path = out / "audit_import.csv"
    audits = _read_table(audit_path) if audit_path.exists() else pd.DataFrame()
    diagnostic_root = Path.cwd() / "result" / "ETHICS" / "reports" / "regional_lm_v3_ai_diagnostic_20260916_r1"
    development_rows: list[dict[str, Any]] = []
    diagnostic_path = (out / "diagnostic_labels.json") if (out / "diagnostic_labels.json").exists() else diagnostic_root / "diagnostic_labels.json"
    if diagnostic_path.exists():
        payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        development_rows.extend(payload.get("labels", []))
    idiom_path = (out / "idiom_census.json") if (out / "idiom_census.json").exists() else diagnostic_root / "idiom_census.json"
    if idiom_path.exists():
        payload = json.loads(idiom_path.read_text(encoding="utf-8"))
        development_rows.extend(payload.get("responses", []))
    development = pd.DataFrame(development_rows)
    rows = []
    for _, rule in rules.drop_duplicates("rule_id").sort_values("rule_id").iterrows():
        rule_id = str(rule["rule_id"])
        pop_row = population[population["rule_id"].astype(str).eq(rule_id)]
        accepted_population = int(pop_row["accepted_hit_count"].iloc[0]) if not pop_row.empty else None
        raw_population = int(pop_row["population_hit_count"].iloc[0]) if not pop_row.empty else None
        rejected_population = int(pop_row["rejected_hit_count"].iloc[0]) if not pop_row.empty else None
        is_exclusion_rule = rule_id == "connective_conditional:exclude_if_you_will"
        gate_population = rejected_population if is_exclusion_rule else accepted_population
        denominator_source = (
            "rejected_rule_decisions_development_census"
            if is_exclusion_rule
            else "accepted_detector_hits_inventory"
        )
        audit_rows = (
            audits[audits["rule_id"].astype(str).eq(rule_id)].to_dict("records")
            if not audits.empty and "rule_id" in audits
            else []
        )
        gate = rule_gate(
            audit_rows,
            semantic=str(rule.get("audit_class", "semantic")).lower() != "explicit",
            population_hits=gate_population,
        )
        if not development.empty and "rule_id" in development:
            dev = development[development["rule_id"].astype(str).eq(rule_id)].copy()
        else:
            dev = pd.DataFrame()
        dev_labels = dev.get("label", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
        development_n = int(len(dev))
        gate.update(
            {
                "rule_id": rule_id,
                "marker_id": rule.get("marker_id", rule_id.split(":p", 1)[0]),
                "audit_class": rule.get("audit_class", "semantic"),
                "rule_version": rule.get("rule_version", "unknown"),
                "population_hit_count": raw_population,
                "accepted_hit_count": accepted_population,
                "rejected_hit_count": rejected_population,
                "audit_denominator_source": denominator_source,
                "audit_sampling_design": "development_census_no_independent_gold" if is_exclusion_rule else "rule_stratified_srswor_evaluation",
                "development_n_reviewed": development_n,
                "development_n_tp": int((dev_labels == "TP").sum()),
                "development_n_fp": int((dev_labels == "FP").sum()),
                "development_n_uncertain": int((dev_labels == "UNCERTAIN").sum()),
                "development_evidence_source": "ai_diagnostic_artifacts" if development_n else "none",
                "development_independent_evaluation": False,
                "diagnostic_ai_n_reviewed": development_n,
                "diagnostic_ai_status": "development_only_not_counted_toward_release" if development_n else "not_reviewed",
                "execution_code_sha256": _execution_code_sha256(),
            }
        )
        gate["blockers"] = ";".join(gate["blockers"])
        rows.append(gate)
    summary = pd.DataFrame(rows)
    summary_path = out / "rule_audit_summary.csv"
    if summary_path.exists():
        raise FileExistsError(f"Rule audit summary sudah ada: {summary_path}")
    summary.to_csv(summary_path, index=False)
    return {"path": str(summary_path), "rows": int(len(summary))}

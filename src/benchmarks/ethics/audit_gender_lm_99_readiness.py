"""Read-only corpus audit; writes new artifacts, never promotes linguistic rules."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source = Path("result/ETHICS/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv")
    baseline = Path("result/ETHICS/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0")
    registry = pd.read_csv(baseline / "marker_registry.csv")
    marker_path = baseline / "response_level_markers.csv.gz"
    registry_path = baseline / "marker_registry.csv"
    paths = [source, marker_path, registry_path]
    hashes = {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in paths}
    print("Loading frozen marker records", flush=True)
    records = pd.read_csv(marker_path, usecols=lambda c: c in {
        "model_id", "persona", "subset", "question_id", "text_available"
    } or c.endswith("__count"))
    keys = ["model_id", "persona", "subset", "question_id"]
    if records.duplicated(keys).any():
        raise ValueError("Duplicate marker keys")
    audit = pd.read_csv(baseline / "audit_marker_hits.csv")
    extractor = Path("src/core/linguistic_markers/extractor.py")
    tree = ast.parse(extractor.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_semantic_match_is_valid")
    guarded = ast.literal_eval(fn.body[1].test.comparators[0])
    rows = []
    for _, marker in registry.iterrows():
        mid = marker.marker_id
        present = records[f"{mid}__count"].fillna(0).gt(0)
        hit_records = records.loc[present]
        samples = audit.loc[audit.marker_id.eq(mid)]
        rows.append({
            "marker_id": mid, "family": marker.family,
            "definition": marker.operational_definition,
            "exclusion_rule": marker.exclusion_rule,
            "responses_present": int(present.sum()),
            "total_occurrences": int(records[f"{mid}__count"].sum()),
            "unique_question_clusters_present": len(hit_records[["subset", "question_id"]].drop_duplicates()),
            "models_with_hit": int(hit_records.model_id.nunique()),
            "personas_with_hit": int(hit_records.persona.nunique()),
            "precision_template_rows": len(samples),
            "precision_template_labels_filled": int(samples.is_true_positive.notna().sum()),
            "context_guard_in_current_extractor": mid in guarded,
            "registry_status": marker.validation_status,
            "readiness_99": "not_demonstrated",
        })
    pd.DataFrame(rows).to_csv(output / "marker_readiness.csv", index=False)
    counts = [f"{mid}__count" for mid in registry.marker_id]
    long = records.melt(id_vars=keys, value_vars=counts, var_name="marker_id", value_name="count")
    long["marker_id"] = long.marker_id.str.removesuffix("__count")
    long["present"] = long["count"].gt(0)
    grouped = long.groupby(["model_id", "persona", "subset", "marker_id"], observed=True).agg(
        total=("question_id", "size"), eligible=("count", "count"),
        responses_present=("present", "sum"), total_occurrences=("count", "sum")
    ).reset_index()
    grouped.to_csv(output / "marker_by_model_persona_subset.csv", index=False)
    del long
    print("Loading response text without benchmark labels", flush=True)
    responses = pd.read_csv(source, usecols=["model", "persona_folder", "file", "question_id", "subset", "response_content_full"]).rename(
        columns={"model": "model_id", "persona_folder": "persona"}
    )
    if len(responses) != len(records) or responses.duplicated(keys).any():
        raise ValueError("Source row/key mismatch")
    texts = responses.response_content_full.fillna("").astype(str)
    probes = {
        "hypothetical_assuming_you": r"\bassuming\s+(?:that\s+)?you(?:'re|\s+are)\s+(?:a|an)\s+(?:man|woman|male|female)\b",
        "hypothetical_if_adverb_you": r"\bif\s+(?:indeed|really|actually)\s+you(?:'re|\s+are)\s+(?:a|an)\s+(?:man|woman|male|female)\b",
        "interrogative_if": r"\b(?:determine|decide|know|judge|assess|tell|wonder|see)\s+if\b",
        "quoted_identity": r"[\"\u201c]\s*I\s+am\s+a\s+(?:man|woman)\b",
        "negated_booster": r"\bnot\s+(?:necessarily\s+)?(?:clearly|obviously|definitely|certainly)\b",
        "if_you_want": r"\bif\s+you\s+(?:want|would\s+like|need)\b",
        "empathy_contraction": r"\bit['\u2019]s\s+(?:completely\s+|perfectly\s+)?understandable\s+(?:that|why|to\s+feel)\b",
        "self_correction_dash": r"\bwait\s*[\u2013\u2014-]\s*[A-Za-z]",
    }
    samples, frequencies = [], []
    used: set[int] = set()
    for probe_id, pattern in probes.items():
        hit = texts.str.contains(pattern, flags=re.I, regex=True)
        frequencies.append({"probe_id": probe_id, "pattern": pattern, "responses_present": int(hit.sum())})
        candidates = responses.loc[hit].copy()
        candidates["distance"] = (candidates.response_content_full.str.len() - 1200).abs()
        candidates = candidates.sort_values(["distance", "model_id", "persona", "question_id"])
        chosen_models: set[str] = set()
        for idx, row in candidates.iterrows():
            if idx in used or row.model_id in chosen_models:
                continue
            used.add(idx)
            chosen_models.add(row.model_id)
            record = row.drop(labels="distance").to_dict()
            record.update({"probe_id": probe_id, "source_row": int(idx), "reviewer_type": "agent", "review_status": "selected_not_reviewed"})
            samples.append(record)
            if len(chosen_models) >= 2:
                break
    pd.DataFrame(frequencies).to_csv(output / "discovery_probes.csv", index=False)
    with (output / "full_response_discovery_samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    for start in range(0, len(samples), 4):
        content = []
        for idx, sample in enumerate(samples[start:start + 4], start):
            content.append(f"## {idx}: {sample['probe_id']} | {sample['model_id']} | {sample['persona']} | {sample['question_id']}\n\n{sample['response_content_full']}\n")
        (output / f"full_samples_{start // 4 + 1}.md").write_text("\n".join(content), encoding="utf-8")
    bounds = []
    for n in [50, 100, 298, 299, 300, 368]:
        bounds.append({"n_zero_error_independent_trials": n, "one_sided_95_lower": .05 ** (1 / n), "two_sided_95_lower": .025 ** (1 / n)})
    pd.DataFrame(bounds).to_csv(output / "ideal_binomial_bounds.csv", index=False)
    summary = {
        "source_rows": len(responses), "markers": len(registry),
        "model_persona_subset_marker_groups": len(grouped), "samples_selected": len(samples),
        "audit_type": "agent_readiness_and_discovery_not_human_validation",
        "selection": "Two distinct models per literal probe when possible, nearest 1200 characters, without GT/correctness. Not random and not a precision/recall estimator.",
        "minimum_independent_zero_error_trials_one_sided_95_for_99": math.ceil(math.log(.05) / math.log(.99)),
        "minimum_independent_zero_error_trials_two_sided_95_for_99": math.ceil(math.log(.025) / math.log(.99)),
        "minimum_independent_zero_error_trials_bonferroni_82_endpoints": math.ceil(math.log(.05 / 82) / math.log(.99)),
        "input_hashes": hashes,
        "input_hashes_unchanged": all(hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest() == h for p, h in hashes.items()),
        "precision_recall_99_achieved": False,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Pool frozen Religion/Regional responses using the existing statistical engines."""
from __future__ import annotations

import argparse
from dataclasses import fields
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics.run_gender_ethics_lm_workspace import enable_marker_imports, sha256
enable_marker_imports()
import numpy as np
import pandas as pd

REPORTS = ROOT / "result/benchmarks/ethics/reports"
SOURCES = {
    "religion": REPORTS / "Religion_Persona/linguistic_markers/analysis/religion_ethics_linguistic_markers_v2_0_2_candidate_20260827",
    "regional": REPORTS / "Regional_Cultural_Persona/linguistic_markers/analysis/regional_lm_v2_0_1_complete_20260828",
}
SUBSETS = ("commonsense", "deontology", "justice", "virtue")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def engine_for(group):
    name = "religion" if group == "religion" else "regional_cultural"
    return importlib.import_module(f"src.benchmarks.ethics.{name}_ethics_linguistic_markers_v2")


def pool_frame(frame):
    keys = ["model_id", "persona", "subset", "question_id"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Invalid original response keys")
    result = frame.copy()
    result["source_subset"] = result.subset
    result["source_question_id"] = result.question_id.astype(str)
    result["question_id"] = [json.dumps([str(s), str(q)], separators=(",", ":")) for s, q in zip(result.subset, result.question_id)]
    result["subset"] = "ethics_all"
    if result.duplicated(["model_id", "persona", "question_id"]).any():
        raise ValueError("Pooled key collision")
    return result


def load_frame(group):
    source = SOURCES[group]
    engine = engine_for(group)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((source / "automated_validation.json").read_text(encoding="utf-8"))
    dictionary_path = source / "tables/outcome_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path).fillna({"raw_col": ""})
    names = [f.name for f in fields(engine.OutcomeSpec)]
    outcomes = tuple(engine.OutcomeSpec(**row) for row in dictionary[names].to_dict("records"))
    response_path = next((source / "dataset").glob("*.csv.gz"))
    head = pd.read_csv(response_path, nrows=0)
    metadata = ["model_id", "persona", "identity", "condition", "question_id", "subset", "text_available", "text_status",
                "validated_committed", "answer_status", "analysis_answer_status", "opening_category", "word_count", "avg_sentence_length", "ttr_300"]
    value_columns = [c for c in head if c.endswith(("__count", "__density", "__present"))]
    frame = pd.read_csv(response_path, usecols=list(dict.fromkeys(metadata + value_columns)), dtype={"question_id": str})
    frame["text_available"] = engine.bool_series(frame.text_available)
    frame["validated_committed"] = engine.bool_series(frame.validated_committed)
    # The existing pair builder requires this column; correctness is never selected for ethics_all.
    frame["analysis_is_correct"] = pd.NA
    assert len(frame) == manifest["total_responses"]
    assert set(frame.subset) == set(SUBSETS)
    assert frame.model_id.nunique() == 6 and frame.persona.nunique() == 17
    cols = list(dict.fromkeys(spec.value_col for spec in outcomes))
    assert frame.loc[~frame.text_available, cols].isna().all().all()
    for spec in outcomes:
        if spec.raw_col:
            available = frame.loc[frame.text_available]
            assert available[spec.raw_col].ge(0).all()
            if spec.outcome_scale != "binary":
                np.testing.assert_allclose(available[spec.value_col], available[spec.raw_col] / available.word_count * 100, rtol=1e-5, atol=1e-6)
    hashes = {str(p): sha256(p) for p in (source / "manifest.json", source / "automated_validation.json", dictionary_path, response_path, Path(engine.__file__))}
    return engine, frame, outcomes, dictionary, hashes, validation


def support_audit(frame, results):
    """Reconstruct every row denominator using original subset and question keys."""
    rows = []
    group_keys = ["model_id", "left_persona", "right_persona"]
    for (model, left_persona, right_persona), tests in results.groupby(group_keys, sort=False):
        source = frame.loc[frame.model_id.eq(model)]
        left = source.loc[source.persona.eq(left_persona)].set_index(["source_subset", "source_question_id"])
        right = source.loc[source.persona.eq(right_persona)].set_index(["source_subset", "source_question_id"])
        shared = left.index.intersection(right.index, sort=False)
        left, right = left.loc[shared], right.loc[shared]
        visible = left.text_available.to_numpy() & right.text_available.to_numpy()
        committed = left.validated_committed.to_numpy() & right.validated_committed.to_numpy()
        origins = shared.get_level_values("source_subset").to_numpy()
        for test in tests.itertuples(index=False):
            value_col = test.value_col
            keep = visible & left[value_col].notna().to_numpy() & right[value_col].notna().to_numpy()
            if test.population == "both_committed":
                keep &= committed
            assert int(keep.sum()) == test.n_pairs, (model, test.contrast_id, test.outcome_id)
            record = {"model_id": model, "left_persona": left_persona, "right_persona": right_persona,
                      "population": test.population, "outcome_id": test.outcome_id, "n_pairs": test.n_pairs}
            for subset in SUBSETS:
                record[f"n_{subset}"] = int((keep & (origins == subset)).sum())
            assert sum(record[f"n_{s}"] for s in SUBSETS) == test.n_pairs
            rows.append(record)
    return pd.DataFrame(rows)


def validate_results(engine, table, dictionary):
    keys = ["population", "model_id", "subset", "left_persona", "right_persona", "outcome_id"]
    assert not table.duplicated(keys).any()
    assert set(table.subset) == {"ethics_all"}
    assert not table.population.eq("both_correct").any()
    assert table.p_holm.dropna().between(0, 1).all()
    assert table.n_pairs.ge(table.n_nonzero).all()
    assert table.loc[table.n_pairs.eq(0), ["ci_low", "ci_high"]].isna().all().all()
    assert table.loc[table.n_pairs.gt(0), "ci_low"].le(table.loc[table.n_pairs.gt(0), "ci_high"]).all()
    groups = ["population", "model_id", "subset", "outcome_id", "multiplicity_family"]
    # Direct comparisons use one family of 24; within adds the identity key.
    if "identity" in table:
        groups += ["identity"]
    for key, family in table.groupby(groups, dropna=False):
        expected = engine.holm_adjust(family.p_raw.to_numpy(float))
        np.testing.assert_allclose(family.p_holm, expected, rtol=0, atol=0, equal_nan=True)
        expected_size = 24 if family.multiplicity_family.iloc[0] == "direct_identity_24" else (4 if family.multiplicity_family.iloc[0] == "neutral_sensitivity_4" else 3)
        assert len(family) == expected_size
    expected_outcomes = set(dictionary.outcome_id)
    assert set(table.loc[table.population.eq("all_visible"), "outcome_id"]) == expected_outcomes


def baseline_check(engine, original, outcomes, source, output):
    """Check all within-model p/effect/CI fields on fixed original-subset cases."""
    model = engine.MODELS[0]
    subset = "commonsense"
    selected = original.loc[original.model_id.eq(model) & original.subset.eq(subset)].copy()
    values = [c for spec in outcomes for c in (spec.value_col, spec.raw_col) if c]
    records = []
    baseline = pd.read_csv(source / "tables/paired_direct_identity_all_scales.csv.gz")
    first_pair = engine.DIRECT_IDENTITY_PAIRS[0]
    ids = f"{first_pair[0]}_vs_{first_pair[1]}"
    baseline = baseline.loc[baseline.population.eq("all_visible") & baseline.model_id.eq(model) & baseline.subset.eq(subset) & baseline.identity_contrast.eq(ids) & baseline.condition.eq("explicit")]
    assert len(baseline) == len(outcomes)
    left, right = (engine.IDENTITY_CONDITIONS[p]["explicit"] for p in first_pair)
    paired = engine._pair_frame(selected, left, right, values)
    for spec in outcomes:
        result = engine._outcome_pair_row(paired, engine._population_mask(paired, "all_visible"), spec, 10000,
            ("direct", "all_visible", model, subset, ids, "explicit", spec.outcome_id))
        expected = baseline.loc[baseline.outcome_id.eq(spec.outcome_id)].iloc[0]
        differences = {}
        for col in ["n_pairs", "n_nonzero", "effect", "effect_size", "ci_low", "ci_high", "p_raw"]:
            a, b = float(result[col]), float(expected[col])
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
            differences[col] = abs(a-b) if np.isfinite(a) and np.isfinite(b) else None
        records.append({"outcome_id": spec.outcome_id, **differences})
    pd.DataFrame(records).to_csv(output / "baseline_engine_comparison.csv", index=False)


def build(group, output):
    output.mkdir(parents=True, exist_ok=False)
    engine, original, outcomes, dictionary, source_hashes, prior_validation = load_frame(group)
    print(f"[{group}] {len(original):,} responses / {len(outcomes)} outcomes; validating engine", flush=True)
    baseline_check(engine, original, outcomes, SOURCES[group], output)
    frame = pool_frame(original)
    frame.to_csv(output / "pooled_response_level.csv.gz", index=False, compression="gzip")
    dictionary.to_csv(output / "outcome_dictionary.csv", index=False)
    coverage = frame.groupby(["model_id", "persona", "condition", "identity"], dropna=False).agg(
        total=("question_id", "size"), available=("text_available", "sum"), validated_committed=("validated_committed", "sum")).reset_index()
    coverage["text_availability_rate"] = coverage.available / coverage.total
    coverage.to_csv(output / "coverage.csv", index=False)
    weights = frame.groupby(["model_id", "persona", "source_subset"], dropna=False).agg(total=("question_id", "size"), available=("text_available", "sum")).reset_index()
    weights["response_share"] = weights.total / weights.groupby(["model_id", "persona"]).total.transform("sum")
    weights.to_csv(output / "subset_weights.csv", index=False)
    engine.build_descriptives(frame, outcomes).to_csv(output / "descriptives.csv", index=False)
    within, direct, openings, supports = [], [], [], []
    for index, model in enumerate(engine.MODELS, 1):
        print(f"[{group}] Model {index}/6 {model}: within identity", flush=True)
        part = frame.loc[frame.model_id.eq(model)].copy()
        w = engine.build_within_identity_pairs(part, outcomes, 10000)
        validate_results(engine, w, dictionary)
        print(f"[{group}] Model {index}/6: direct identity", flush=True)
        d = engine.build_direct_identity_pairs(part, outcomes, 10000)
        validate_results(engine, d, dictionary)
        opening, _ = engine.build_opening_pairs(part)
        results = pd.concat([w, d], ignore_index=True).merge(dictionary[["outcome_id", "value_col"]], on="outcome_id", validate="many_to_one")
        support = support_audit(part, results)
        checkpoint = output / f"model_{index:02d}"
        checkpoint.mkdir()
        for name, data in [("within", w), ("direct", d), ("opening", opening), ("pair_composition", support)]:
            data.to_csv(checkpoint / f"{name}.csv", index=False)
        write_json(checkpoint / "manifest.json", {"model": model, "rows": len(w)+len(d), "source_hashes": source_hashes,
                   "artifact_hashes": {p.name: sha256(p) for p in checkpoint.glob("*.csv")}})
        within.append(w); direct.append(d); openings.append(opening); supports.append(support)
        print(f"[{group}] Model {index}/6 saved: {len(w)+len(d):,} tests", flush=True)
    w, d = pd.concat(within, ignore_index=True), pd.concat(direct, ignore_index=True)
    w.to_csv(output / "within.csv", index=False)
    d.to_csv(output / "direct.csv", index=False)
    pd.concat(openings, ignore_index=True).to_csv(output / "opening.csv", index=False)
    pd.concat(supports, ignore_index=True).to_csv(output / "pair_composition.csv", index=False)
    for p, expected in source_hashes.items():
        assert sha256(Path(p)) == expected
    write_json(output / "manifest.json", {"group": group, "result_status": "candidate", "aggregation": "response_weighted",
        "source_bundle": str(SOURCES[group]), "source_hashes": source_hashes, "script_sha256": sha256(Path(__file__)),
        "bootstrap_resamples": 10000, "rows": len(frame), "within_tests": len(w), "direct_tests": len(d),
        "pairing": "model + source_subset + original_question_id", "hidden_reasoning_used": False,
        "ground_truth_used": False, "accuracy_pooled": False, "subsets": list(SUBSETS)})
    write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()})
    write_json(output / "validation.json", {"valid": True, "source_unchanged": True, "support_all_rows_checked": len(w)+len(d), "baseline_outcomes_checked": len(outcomes)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert output.is_relative_to(REPORTS.resolve())
    output.mkdir(parents=True, exist_ok=False)
    for group in args.groups:
        build(group, output / group)

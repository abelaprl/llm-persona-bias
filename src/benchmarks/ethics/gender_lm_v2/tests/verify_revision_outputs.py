"""Independent artifact verification; never imports the extraction/statistics modules."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


KEYS = ["model_id", "persona", "subset", "question_id"]
FEATURE_SUFFIXES = ("__count", "__present", "__density")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_new(output, name, payload):
    payload["validator_script_sha256"] = digest(__file__)
    payload["validation_kind"] = "independent_programmatic_artifact_check_not_human_marker_audit"
    target = output / name
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"evidence": str(target), "passed": True}, ensure_ascii=False), flush=True)


def validate_revision(parent, output):
    previous = json.loads((parent / "extraction_manifest.json").read_text(encoding="utf-8"))
    revised = json.loads((output / "extraction_manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((parent / "analysis_input_provenance.json").read_text(encoding="utf-8"))
    parent_hashes = {name: digest(parent / name) for name in provenance["artifacts"]}
    assert parent_hashes == provenance["artifacts"], "Parent artifacts differ from frozen provenance"
    assert previous["source_sha256"] == revised["source_sha256"] == digest(previous["source"])
    assert previous["registry_sha256"] == revised["registry_sha256"]
    assert json.loads((parent / "registry.json").read_text(encoding="utf-8")) == json.loads((output / "registry.json").read_text(encoding="utf-8"))
    instrument_before = provenance["instrument_hashes"]
    instrument_after = revised["instrument_hashes_at_extraction"]
    assert set(instrument_before) == set(instrument_after)
    assert all(instrument_before[k] == instrument_after[k] for k in instrument_before if k != "surfaces.py")
    changes, keys, changed_fields = {}, set(), Counter()
    old_status, new_status = Counter(), Counter()
    count = 0
    with gzip.open(parent / "response_markers.csv.gz", "rt", encoding="utf-8", newline="") as before_handle, gzip.open(output / "response_markers.csv.gz", "rt", encoding="utf-8", newline="") as after_handle:
        before_reader, after_reader = csv.DictReader(before_handle), csv.DictReader(after_handle)
        assert before_reader.fieldnames == after_reader.fieldnames
        features = [name for name in before_reader.fieldnames if name.endswith(FEATURE_SUFFIXES)]
        assert len(features) == 123
        for before, after in zip(before_reader, after_reader, strict=True):
            count += 1
            key = tuple(after[k] for k in KEYS)
            assert key == tuple(before[k] for k in KEYS) and key not in keys
            assert before["response_id"] == after["response_id"]
            keys.add(key)
            old_status[before["text_status"]] += 1
            new_status[after["text_status"]] += 1
            if before != after:
                assert (before["text_status"], after["text_status"]) == ("substantive_explanation", "verdict_only")
                assert after["text_available"] == "False" and after["word_count"] == "0"
                assert all(after[name] == "" for name in features), "Unavailable markers must be NA"
                permitted = set(features) | {"text_status", "text_available", "word_count"}
                differences = {name for name in before if before[name] != after[name]}
                assert differences <= permitted, differences - permitted
                changed_fields.update(differences)
                changes[after["response_id"]] = {k: after[k] for k in KEYS}
    assert count == previous["rows"] == revised["rows"] == 81756
    assert len(changes) == revised["changed_responses"] == 12
    assert dict(old_status) == previous["text_status"]
    assert dict(new_status) == revised["text_status"]
    assert new_status == Counter(substantive_explanation=65678, verdict_only=15531, missing=544, technical_only=3)
    print(f"Marker rows checked: {count}; changed: {len(changes)}", flush=True)
    surface_changes, surface_count, raw_changes = set(), 0, 0
    changed_raw = {}
    with gzip.open(parent / "response_surfaces.csv.gz", "rt", encoding="utf-8", newline="") as before_handle, gzip.open(output / "response_surfaces.csv.gz", "rt", encoding="utf-8", newline="") as after_handle:
        before_reader, after_reader = csv.DictReader(before_handle), csv.DictReader(after_handle)
        assert before_reader.fieldnames == after_reader.fieldnames
        for before, after in zip(before_reader, after_reader, strict=True):
            surface_count += 1
            assert before["response_id"] == after["response_id"]
            assert all(before[k] == after[k] for k in KEYS)
            raw_changes += before["raw_text"] != after["raw_text"]
            if before != after:
                rid = after["response_id"]
                surface_changes.add(rid)
                changed_raw[rid] = after["raw_text"]
                assert after["explanation_text"] == after["structure_text"] == ""
    assert surface_count == count and raw_changes == 0
    assert surface_changes == set(changes)
    with (output / "changed_responses.csv").open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == len(changes) and {row["response_id"] for row in ledger} == set(changes)
    assert all(row["raw_text"] == changed_raw[row["response_id"]] for row in ledger)
    reviewed_name = "marker_spans_reviewed.csv.gz"
    assert reviewed_name in provenance["artifacts"], "Evidence selected for reuse must be bound by parent provenance"
    span_counts = Counter()
    with gzip.open(parent / reviewed_name, "rt", encoding="utf-8", newline="") as before_handle, gzip.open(output / "marker_spans.csv.gz", "rt", encoding="utf-8", newline="") as after_handle:
        before_reader, after_reader = csv.DictReader(before_handle), csv.DictReader(after_handle)
        assert before_reader.fieldnames == after_reader.fieldnames

        def retained():
            for row in before_reader:
                span_counts["parent"] += 1
                if row["response_id"] in changes:
                    span_counts["removed"] += 1
                else:
                    yield row

        sentinel = object()
        for before, after in zip_longest(retained(), after_reader, fillvalue=sentinel):
            assert before is not sentinel and after is not sentinel, "Evidence rows differ after filtering revised responses"
            assert before == after, "Reused evidence was modified"
            span_counts["retained"] += 1
    assert span_counts["retained"] == revised["spans"]
    assert span_counts["removed"] == revised["removed_spans"]
    print(f"Evidence rows retained unchanged: {span_counts['retained']}; removed: {span_counts['removed']}", flush=True)
    config_path = Path(__file__).resolve().parents[1] / "source_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    clusters = {tuple(pair) for pair in config["previously_read_clusters"]}
    assert len(clusters) == len(config["previously_read_clusters"]) == 139
    evidence_clusters, evidence_hashes = set(), {}
    for path in config["previously_read_evidence"]:
        evidence_hashes[path] = digest(path)
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values["rows"]
        for row in values:
            evidence_clusters.add((row["subset"], row["question_id"]))
    assert clusters == evidence_clusters
    assert {(row["subset"], row["question_id"]) for row in changes.values()} <= clusters
    assert parent_hashes == {name: digest(parent / name) for name in parent_hashes}
    artifact_names = ["response_markers.csv.gz", "response_surfaces.csv.gz", "marker_spans.csv.gz", "changed_responses.csv"]
    summary = {"passed": True, "parent_output": str(parent), "revised_output": str(output),
               "rows_compared": count, "unique_keys": len(keys), "changed_response_count": len(changes),
               "unchanged_response_count": count - len(changes), "parent_status_counts": dict(old_status),
               "revised_status_counts": dict(new_status), "changed_columns": dict(changed_fields),
               "raw_text_changes": raw_changes, "changed_surface_rows": len(surface_changes),
               "evidence_rows": {key: span_counts[key] for key in ("parent", "removed", "retained")}, "reused_evidence_exactly_equal": True,
               "source_sha256": revised["source_sha256"], "parent_artifacts_unchanged": parent_hashes,
               "revised_artifact_hashes": {name: digest(output / name) for name in artifact_names},
               "changed_responses": [{"response_id": rid, **key} for rid, key in sorted(changes.items())],
               "previously_read_unique_clusters": len(clusters), "previously_read_union_matches_evidence": True,
               "all_changed_clusters_excluded_from_held_out": True,
               "source_config_sha256": digest(config_path), "previously_read_evidence_hashes": evidence_hashes,
               "limitation": "Verifies artifact consistency and the bounded revision; does not validate marker semantics or historical extraction-code provenance."}
    save_new(output, "independent_revision_validation.json", summary)


def validate_statistics(output):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest, rankdata, wilcoxon

    comparison_columns = ["model_id", "scope", "marker_id", "measurement", "contrast_id", "left", "right", "role",
                          "n_pairs", "n_changed", "n_left_available", "n_right_available", "n_left_total", "n_right_total",
                          "effect", "ci_low", "ci_high", "raw_p", "adjusted_p", "sign_raw_p", "sign_adjusted_p",
                          "matched_rank_biserial", "sign_effect", "low_support", "sample_scope", "effect_direction"]
    table_names = ["comparisons", "interaction_comparisons", "neutral_triplet_comparisons", "availability_tests"]
    hashes_before = {name + ".csv": digest(output / (name + ".csv")) for name in table_names}
    manifest = json.loads((output / "contrast_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len({row["contrast_id"] for row in manifest}) == 27
    assert sum(row["role"] == "primary" for row in manifest) == 12
    assert sum(row["right"] == "neutral" for row in manifest) == 8
    tables = {name: pd.read_csv(output / (name + ".csv"), usecols=comparison_columns) for name in table_names[:2]}
    tables["neutral_triplet_comparisons"] = pd.read_csv(output / "neutral_triplet_comparisons.csv")
    tables["availability_tests"] = pd.read_csv(output / "availability_tests.csv")
    expected_counts = dict(comparisons=88560, interaction_comparisons=11070, neutral_triplet_comparisons=14760, availability_tests=60)
    common = ["model_id", "scope", "marker_id", "measurement"]
    keys = {"comparisons": common + ["contrast_id"], "interaction_comparisons": common + ["contrast_id"],
            "neutral_triplet_comparisons": common + ["condition"], "availability_tests": ["model_id", "scope", "outcome"]}
    for name, table in tables.items():
        assert len(table) == expected_counts[name] and not table.duplicated(keys[name]).any(), name
        for column in ("raw_p", "adjusted_p", "sign_raw_p", "sign_adjusted_p"):
            if column in table:
                assert table[column].dropna().between(0, 1).all(), (name, column)
        if "adjusted_p" in table:
            valid = table[["raw_p", "adjusted_p"]].dropna()
            assert (valid.adjusted_p + 1e-12 >= valid.raw_p).all()
        if "n_pairs" in table:
            assert table.n_pairs.ge(0).all() and table.n_changed.between(0, table.n_pairs).all()
            assert table.n_left_available.ge(table.n_pairs).all() and table.n_right_available.ge(table.n_pairs).all()
            assert table.n_left_total.ge(table.n_left_available).all() and table.n_right_total.ge(table.n_right_available).all()
            assert table.loc[table.n_pairs.eq(0), ["effect", "raw_p", "ci_low", "ci_high"]].isna().all().all()
            assert table.low_support.eq(table.n_changed.lt(10)).all()
            bounds = table[["ci_low", "ci_high"]].dropna()
            assert bounds.ci_low.le(bounds.ci_high).all()
            assert table.effect_direction.eq("left_minus_right").all()
    triplets = tables["neutral_triplet_comparisons"]
    assert np.allclose(triplets.d_woman_neutral - triplets.d_man_neutral, triplets.d_woman_man, atol=1e-10, rtol=1e-10, equal_nan=True)
    for suffix in ("woman_neutral", "man_neutral", "woman_man"):
        bounds = triplets[["ci_low_" + suffix, "ci_high_" + suffix]].dropna()
        assert (bounds.iloc[:, 0] <= bounds.iloc[:, 1]).all()
    records = pd.read_csv(output / "response_markers.csv.gz")
    assert len(records) == 81756 and not records.duplicated(KEYS).any()
    assert records.text_available.eq(records.text_status.eq("substantive_explanation")).all()
    random = np.random.default_rng(42)
    sample_rows = []
    for _, group in tables["comparisons"].groupby(["model_id", "scope"], sort=True):
        sample_rows.extend(random.choice(group.index.to_numpy(), 12, replace=False).tolist())
    sampled = tables["comparisons"].loc[sample_rows]
    assert len(sampled) == 360
    suffixes = {"count": "count", "density": "density", "presence": "present"}
    group_cache = {}
    verified = []
    for row in sampled.itertuples(index=False):
        cache_key = (row.model_id, row.scope)
        if cache_key not in group_cache:
            block = records[records.model_id.eq(row.model_id)]
            if row.scope != "overall":
                block = block[block.subset.eq(row.scope)]
            group_cache[cache_key] = block
        block = group_cache[cache_key]
        feature = row.marker_id + "__" + suffixes[row.measurement]
        values = []
        for side in (row.left, row.right):
            side_frame = block[block.persona.eq(side)].copy()
            side_frame.loc[~side_frame.text_available, feature] = np.nan
            values.append(side_frame.set_index(["subset", "question_id"])[feature].rename(side))
        aligned = pd.concat(values, axis=1).dropna()
        a, b = aligned.iloc[:, 0].to_numpy(float), aligned.iloc[:, 1].to_numpy(float)
        difference = a - b
        nonzero = difference[difference != 0]
        positive = int((nonzero > 0).sum())
        assert row.n_pairs == len(aligned) and row.n_changed == len(nonzero)
        assert row.n_left_total == len(values[0]) and row.n_right_total == len(values[1])
        assert row.n_left_available == values[0].notna().sum() and row.n_right_available == values[1].notna().sum()
        expected_effect = float(difference.mean()) * (100 if row.measurement == "presence" else 1) if len(difference) else np.nan
        assert np.isclose(row.effect, expected_effect, atol=1e-10, rtol=1e-10, equal_nan=True)
        if len(nonzero):
            expected_sign_p = float(binomtest(positive, len(nonzero), p=0.5).pvalue)
            expected_p = expected_sign_p if row.measurement == "presence" else float(wilcoxon(difference, zero_method="wilcox", correction=False, alternative="two-sided", method="asymptotic").pvalue)
            ranks = rankdata(abs(nonzero), method="average")
            rank_effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
            sign_effect = (2 * positive - len(nonzero)) / len(nonzero)
        else:
            expected_p = expected_sign_p = 1.0 if len(difference) else np.nan
            rank_effect = sign_effect = np.nan
        for actual, expected in ((row.raw_p, expected_p), (row.sign_raw_p, expected_sign_p),
                                 (row.matched_rank_biserial, rank_effect), (row.sign_effect, sign_effect)):
            assert np.isclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
        verified.append({key: getattr(row, key) for key in common + ["contrast_id", "n_pairs", "effect"]})
    assert hashes_before == {name: digest(output / name) for name in hashes_before}
    save_new(output, "independent_statistics_validation.json",
             {"passed": True, "output": str(output), "artifact_hashes": hashes_before,
              "rows_checked": {name: len(table) for name, table in tables.items()},
              "unique_keys_valid": True, "p_value_and_ci_bounds_valid": True,
              "all_triplet_additive_identities_valid": True, "pairwise_samples_recalculated": len(verified),
              "sampling_seed": 42, "sampled_pairs": verified,
              "limitation": "Checks numerical consistency; does not validate marker semantics or independently reconstruct bootstrap intervals."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["revision", "statistics"])
    parser.add_argument("--parent", type=Path, default=Path("result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260915_r2"))
    parser.add_argument("--output", type=Path, default=Path("result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260916_r3"))
    args = parser.parse_args()
    if args.stage == "revision":
        validate_revision(args.parent, args.output)
    else:
        validate_statistics(args.output)

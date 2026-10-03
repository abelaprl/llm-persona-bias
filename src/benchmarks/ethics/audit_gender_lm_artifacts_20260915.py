"""Independently inspect frozen Gender ETHICS presentation without mutating it."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "result/benchmarks/ethics/reports/overall_presentation_20260913_r2/gender"


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BUNDLE / "source_manifest.json").read_text(encoding="utf-8"))
    notebook_path = ROOT / manifest["notebook_target"]
    tracked = {p: digest(p) for p in BUNDLE.rglob("*") if p.is_file()}
    tracked[notebook_path] = digest(notebook_path)
    checks = []

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
        print(name, bool(passed), flush=True)

    mismatches = [str(p) for p, h in manifest["source_hashes"].items() if digest(p) != h]
    check("source_hashes", not mismatches, files=len(manifest["source_hashes"]), mismatches=mismatches)
    check("backup_hash", digest(BUNDLE / "notebook_before.ipynb") == manifest["notebook_before_sha256"])
    audit = pd.read_csv(BUNDLE / "audit/visual_audit.csv")
    bad = []
    for row in audit.itertuples():
        for path, expected in [(row.figure, row.png_sha256), (row.svg, row.svg_sha256), (row.table, row.table_sha256)]:
            if digest(BUNDLE / path) != expected:
                bad.append(path)
    check("figure_hashes", not bad, figures=len(audit), artifacts=len(audit)*3, mismatches=bad)
    data = pd.read_csv(BUNDLE / "all_comparisons.csv")
    key = ["population", "model_id", "left_persona", "right_persona", "outcome_id", "scale"]
    check("comparison_keys", not data.duplicated(key).any(), rows=len(data), populations=data.population.value_counts().to_dict())
    errors = []
    compared = 0
    for source_file, rows in data.groupby("source_file"):
        source = pd.read_csv(ROOT / source_file).iloc[rows.source_row.to_numpy()]
        presence = "presence" in source_file
        mapping = {c: c for c in ["p_raw", "p_holm", "ci_low", "ci_high", "n_pairs"]}
        mapping.update(estimate="prevalence_delta" if presence else "median_paired_delta",
                       signed_effect="prevalence_delta" if presence else "rank_biserial",
                       effect_size="matched_odds_ratio" if presence else "rank_biserial",
                       n_changed="discordant" if presence else "n_nonzero")
        for target, original in mapping.items():
            same = np.isclose(rows[target].to_numpy(), source[original].to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
            compared += len(rows)
            if not same.all():
                errors.append(dict(file=source_file, column=target, mismatches=int((~same).sum())))
        for c in ["model_id", "subset", "left_persona", "right_persona", "outcome_id", "test_used"]:
            if not np.array_equal(rows[c].fillna("").to_numpy(), source[c].fillna("").to_numpy()):
                errors.append(dict(file=source_file, column=c))
    check("source_values", not errors, numeric_values=compared, tolerance=1e-12, mismatches=errors)
    expected_decision = np.where(data.p_holm.isna() | data.n_pairs.le(0), "not_testable", np.where(data.p_holm.lt(.05), "significant", "not_significant"))
    expected_direction = np.where(data.signed_effect.isna(), "unavailable", np.where(data.signed_effect.gt(0), "left_higher", np.where(data.signed_effect.lt(0), "right_higher", "zero")))
    check("decisions_and_directions", np.array_equal(data.decision, expected_decision) and np.array_equal(data.direction, expected_direction))
    coverage = pd.read_csv(BUNDLE / "figure_tables/coverage.csv")
    valid = coverage.total.eq(coverage.committed + coverage.abstain + coverage.missing).all()
    for status in ["committed", "abstain", "missing", "available"]:
        col = "commit_rate" if status == "committed" else status + "_rate"
        valid &= np.allclose(coverage[col], coverage[status] / coverage.total)
    check("coverage_arithmetic", valid, rows=len(coverage), totals=coverage[["total", "available", "committed", "abstain", "missing"]].sum().to_dict())
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    saved_errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    embedded = [hashlib.sha256(base64.b64decode(o["data"]["image/png"])).hexdigest() for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    check("saved_notebook", not saved_errors and all(c.get("execution_count") is not None for c in code), cells=len(notebook["cells"]), code_cells=len(code), errors=len(saved_errors))
    check("embedded_images", sorted(embedded) == sorted(audit.png_sha256), images=len(embedded))
    validation = json.loads((BUNDLE / "notebook_validation.json").read_text(encoding="utf-8"))
    check("active_notebook_hash", digest(notebook_path) == validation["notebook_sha256"])
    namespace = {}
    for index, cell in enumerate(code):
        exec(compile("".join(cell["source"]), f"notebook_code_cell_{index}", "exec"), namespace)
        namespace["display"] = lambda *a, **kw: None
    check("fresh_code_execution", True, code_cells=len(code), method="Python exec; displays suppressed after setup; no notebook save")
    changed = [str(p) for p, h in tracked.items() if digest(p) != h]
    source_changed = [str(p) for p, h in manifest["source_hashes"].items() if digest(p) != h]
    check("inputs_unchanged", not changed and not source_changed, presentation_files=len(tracked), mismatches=changed+source_changed)
    result = {"scope": "Artifact and presentation integrity, not human semantic validation", "checks": checks, "all_passed": all(c["passed"] for c in checks)}
    with (args.output_dir / "artifact_checks.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    data.groupby(["population", "test_used", "decision"]).size().rename("rows").reset_index().to_csv(args.output_dir / "test_decisions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()

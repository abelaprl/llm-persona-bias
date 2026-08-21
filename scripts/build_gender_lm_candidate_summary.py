"""Build a versioned, candidate-only Bab IV summary from a completed LM run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def _overview(
    frame: pd.DataFrame,
    group_columns: list[str],
    effect_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        significant = group[group["p_holm"] < 0.05]
        effects = pd.to_numeric(significant[effect_column], errors="coerce").abs().dropna()
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_planned_tests": len(group),
                "n_holm_significant": len(significant),
                "proportion_holm_significant": len(significant) / len(group),
                "median_n_pairs": float(pd.to_numeric(group["n_pairs"]).median()),
                f"median_abs_{effect_column}_significant": (
                    float(effects.median()) if len(effects) else None
                ),
                f"max_abs_{effect_column}_significant": (
                    float(effects.max()) if len(effects) else None
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Summary output already exists: {output_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("correctness_filter_applied") is not False:
        raise ValueError("Summary source must explicitly declare no correctness filter.")
    version = str(manifest.get("pipeline_version") or "")
    if "candidate" not in version:
        raise ValueError("This script only builds explicitly candidate-labelled summaries.")

    numeric = pd.read_csv(run_dir / "paired_numeric.csv")
    presence = pd.read_csv(run_dir / "paired_binary.csv")
    openings = pd.read_csv(run_dir / "paired_openings.csv")
    omnibus = pd.read_csv(run_dir / "omnibus.csv")
    descriptives = pd.read_csv(run_dir / "condition_descriptives.csv")
    quality = pd.read_csv(run_dir / "quality_control.csv")

    output_dir.mkdir(parents=True, exist_ok=False)
    numeric_overview = _overview(
        numeric,
        ["outcome_id", "outcome_family", "analysis_role"],
        "rank_biserial",
    )
    presence_overview = _overview(
        presence,
        ["outcome_id", "outcome_family", "analysis_role"],
        "prevalence_delta",
    )
    opening_overview = _overview(
        openings,
        ["opening_category"],
        "prevalence_delta",
    )
    _write(numeric_overview, output_dir / "numeric_outcome_overview.csv")
    _write(presence_overview, output_dir / "presence_outcome_overview.csv")
    _write(opening_overview, output_dir / "opening_outcome_overview.csv")

    primary_numeric = numeric[
        (numeric["analysis_role"] == "primary") & (numeric["p_holm"] < 0.05)
    ].sort_values(
        ["model_id", "identity", "outcome_id", "contrast_id"], kind="stable"
    )
    supplementary_numeric = numeric[
        (numeric["analysis_role"] != "primary") & (numeric["p_holm"] < 0.05)
    ].sort_values(
        ["analysis_role", "model_id", "identity", "outcome_id", "contrast_id"],
        kind="stable",
    )
    significant_presence = presence[presence["p_holm"] < 0.05].sort_values(
        ["model_id", "identity", "outcome_id", "contrast_id"], kind="stable"
    )
    significant_openings = openings[openings["p_holm"] < 0.05].sort_values(
        ["model_id", "identity", "opening_category", "contrast_id"], kind="stable"
    )
    significant_omnibus = omnibus[omnibus["p_holm"] < 0.05].sort_values(
        ["model_id", "identity", "outcome_type", "outcome_id"], kind="stable"
    )
    _write(primary_numeric, output_dir / "primary_numeric_holm_significant.csv")
    _write(
        supplementary_numeric,
        output_dir / "supplementary_numeric_holm_significant.csv",
    )
    _write(significant_presence, output_dir / "presence_holm_significant.csv")
    _write(significant_openings, output_dir / "openings_holm_significant.csv")
    _write(significant_omnibus, output_dir / "omnibus_holm_significant.csv")
    _write(descriptives, output_dir / "all_condition_descriptives.csv")
    _write(quality, output_dir / "quality_control.csv")

    table_rows = {
        "numeric_planned": len(numeric),
        "numeric_primary_holm_significant": len(primary_numeric),
        "numeric_supplementary_holm_significant": len(supplementary_numeric),
        "presence_planned": len(presence),
        "presence_holm_significant": len(significant_presence),
        "opening_planned": len(openings),
        "opening_holm_significant": len(significant_openings),
        "omnibus_planned": len(omnibus),
        "omnibus_holm_significant": len(significant_omnibus),
    }
    summary_manifest = {
        "status": "candidate_not_for_final_bab_iv_until_manual_audit",
        "pipeline_version": version,
        "source_run": str(run_dir),
        "source_manifest_sha256": _sha256(manifest_path),
        "correctness_filter_applied": False,
        "selection_rule": "p_holm < 0.05 only in explicitly named significant tables",
        "all_planned_tests_remain_in_source_run": True,
        "row_counts": table_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# Candidate Bab IV summary: {version}

Status: **candidate**. Do not treat these tables as final thesis findings until
`audit_marker_hits.csv` and `audit_recall_responses.csv` have been independently
annotated and agreement has been reported.

This bundle is derived from `{run_dir}`. Correctness was not used to select
responses or contrasts. Files whose names contain `holm_significant` are views
filtered at `p_holm < 0.05`; every planned contrast, including non-significant
ones, remains in the source run's paired tables.

Row counts:

```json
{json.dumps(table_rows, indent=2)}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[DONE] Candidate summary written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

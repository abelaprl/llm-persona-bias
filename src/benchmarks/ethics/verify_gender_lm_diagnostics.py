"""Independently verify presentation data, hashes, and optional installed notebook."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3"
KEY = ["model_id", "scope", "marker_id", "measurement", "contrast_id"]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(output, notebook=None):
    index = pd.read_csv(output / "diagnostic_figure_index.csv").fillna("")
    manifest = json.loads((output / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert index.figure_id.is_unique and len(index) == manifest["figure_count"]
    for name, expected in manifest["source_hashes"].items():
        assert digest(BUNDLE / name) == expected, name
    comparisons = pd.read_csv(BUNDLE / "comparisons.csv", float_precision="round_trip").set_index(KEY)
    registry = pd.DataFrame(json.loads((BUNDLE / "registry.json").read_text(encoding="utf-8")))
    exact_rows = 0
    tables = {}
    for row in index.itertuples():
        for name, expected in [(row.png_path, row.png_sha256), (row.svg_path, row.svg_sha256), (row.source_csv, row.source_sha256)]:
            assert digest(output / name) == expected, name
        table = pd.read_csv(output / row.source_csv, float_precision="round_trip")
        tables[row.figure_id] = table
        if row.figure_kind in {"subset_sensitivity", "normalization_sensitivity", "marker_pair_support"}:
            presented = table.set_index(KEY)
            expected = comparisons.loc[presented.index]
            # A fully empty text column is inferred as float in a per-marker CSV.
            pd.testing.assert_frame_equal(presented[expected.columns], expected, check_exact=True, check_dtype=False)
            exact_rows += len(table)
    reference = tables["diagnostic_gender_reference_context"]
    records = pd.read_csv(BUNDLE / "response_markers.csv.gz", usecols=["response_id", "model_id", "persona", "text_available"])
    spans = pd.read_csv(BUNDLE / "marker_spans_reviewed.csv.gz", usecols=["response_id", "marker_id", "reference_type", "flags"])
    gender_ids = set(registry.loc[registry.registry_scope.eq("gender_specific"), "marker_id"])
    spans = spans.loc[spans.marker_id.isin(gender_ids)]
    for row in reference.itertuples():
        group = records.loc[records.model_id.eq(row.model_id) & records.persona.eq(row.persona)]
        available = set(group.loc[group.text_available, "response_id"])
        candidate = spans.loc[spans.response_id.isin(available)]
        if row.indicator == "scenario_overlap":
            candidate = candidate.loc[candidate["flags"].str.contains('"same_gender_term_in_scenario_not_coreference"', regex=False, na=False)]
        elif row.indicator != "any_gender":
            candidate = candidate.loc[candidate.reference_type.eq(row.indicator)]
        count = candidate.response_id.nunique()
        assert (row.n_responses, row.n_available, row.total) == (count, len(available), len(group))
        assert abs(row.percent - 100*count/len(available)) < 1e-12 if available else pd.isna(row.percent)
    support = tables["diagnostic_pair_support_overview"]
    comp = comparisons.reset_index()
    primary_measure = registry.set_index("marker_id").measurement.replace({"binary": "presence"})
    direct = comp.loc[comp.scope.eq("overall") & comp.left.str.startswith("woman_") & comp.right.str.startswith("man_") &
                      comp.measurement.eq(comp.marker_id.map(primary_measure))]
    for row in support.itertuples():
        suffix = {"E":"explicit_1", "I1":"implicit_1", "I2":"implicit_2", "I3":"implicit_3"}[row.condition]
        selected = direct.loc[direct.model_id.eq(row.model_id) & direct.left.eq("woman_"+suffix) & direct.right.eq("man_"+suffix)]
        assert selected.n_pairs.nunique() == 1 and row.n_pairs == selected.n_pairs.iloc[0]
        assert row.marker_count == len(selected) == 41
        assert row.low_support_markers == int(selected.n_changed.lt(10).sum())
    if manifest["complete_registry"]:
        assert set(index.loc[index.figure_kind.eq("subset_sensitivity"), "marker_ids"]) == set(registry.marker_id)
        assert set(index.loc[index.figure_kind.eq("normalization_sensitivity"), "marker_ids"]) == set(registry.loc[registry.measurement.eq("density"), "marker_id"])
        support_ids = set(";".join(index.loc[index.figure_kind.eq("marker_pair_support"), "marker_ids"]).split(";"))
        assert support_ids == set(registry.marker_id)
    gallery = (output / "diagnostic_gallery.html").read_text(encoding="utf-8")
    for target in re.findall(r'(?:href|src)="([^"]+)"', gallery):
        if not target.startswith("#"):
            assert (output / html.unescape(target)).is_file(), target
    result = dict(status="pass", figures=len(index), exact_frozen_statistic_rows=exact_rows,
                  independently_recomputed_reference_cells=len(reference), independently_checked_support_cells=len(support),
                  complete_registry=manifest["complete_registry"], source_files_unchanged=True)
    if notebook:
        note = json.loads(notebook.read_text(encoding="utf-8"))
        code = [cell for cell in note["cells"] if cell["cell_type"] == "code"]
        outputs = [item for cell in code for item in cell["outputs"]]
        assert all(cell["execution_count"] is not None for cell in code)
        assert not any(item["output_type"] == "error" for item in outputs)
        combined = pd.concat([pd.read_csv(notebook.parent / "figure_index.csv"), index], ignore_index=True)
        expected_hashes = set(combined.png_sha256)
        embedded = []
        for item in outputs:
            data = item.get("data", {})
            if "image/png" in data:
                value = data["image/png"]
                embedded.append(hashlib.sha256(base64.b64decode(''.join(value) if isinstance(value, list) else value)).hexdigest())
        assert len(embedded) == 88 + note["metadata"]["diagnostic_default_pngs"]
        assert len(set(embedded)) == len(embedded) and set(embedded).issubset(expected_hashes)
        checked_links = 0
        for item in outputs:
            markup = item.get("data", {}).get("text/html", "")
            for target in re.findall(r'href="([^"]+)"', ''.join(markup)):
                if not target.startswith(("#", "http://", "https://")):
                    assert (notebook.parent / unquote(html.unescape(target))).is_file(), target
                    checked_links += 1
        for row in combined.itertuples():
            assert digest(notebook.parent / row.png_path) == row.png_sha256
        for row in index.itertuples():
            assert digest(notebook.parent / row.svg_path) == row.svg_sha256
            assert digest(notebook.parent / row.source_csv) == row.source_sha256
        result.update(code_cells=len(code), error_outputs=0, embedded_pngs=len(embedded),
                      local_png_hashes_matched=len(combined), checked_html_links=checked_links)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()
    verify(args.bundle, args.notebook)

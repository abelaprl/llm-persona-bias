"""Render regional GSM8K figures without axis text from frozen measurements."""
from __future__ import annotations

import ast
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from . import remove_regional_plot_status_label as regional
else:
    import remove_regional_plot_status_label as regional


def hide_axis_text(fig):
    removed = []
    for ax in fig.axes:
        artists = [ax.xaxis.label, ax.yaxis.label,
                   ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(),
                   *ax.get_xticklabels(), *ax.get_yticklabels(),
                   *ax.get_xticklabels(minor=True), *ax.get_yticklabels(minor=True)]
        for artist in artists:
            if artist.get_visible() and artist.get_text():
                removed.append(artist.get_text())
            artist.set_visible(False)
        ax.tick_params(axis="both", which="both", bottom=False, top=False,
                       left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
    for artist in [getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None)]:
        if artist is not None:
            removed.append(artist.get_text())
            artist.set_visible(False)
    return removed


def main():
    active = regional.BASE / "GSM8K"
    notebook = next(active.glob("*.ipynb"))
    cells = ["".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]]
    source = Path(json.loads((active / "LATEST_RUN.json").read_text(encoding="utf-8"))["run"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((source / "registry.json").read_text(encoding="utf-8"))
    mapping = dict(coverage="coverage.csv", pair_support="pair_support.csv",
                   statistics="paired_statistics.csv.gz", format_summary="format_diagnostics.csv",
                   sensitivity="sensitivity_statistics.csv.gz", records="response_records.csv.gz")
    inputs = {str(p): regional.sha(p) for p in [notebook, source / "registry.json",
               source / "manifest.json", *[source / f for f in mapping.values()]]}
    context = regional.setup("GSM8K", cells, registry, manifest)
    tables = {}
    for name, filename in mapping.items():
        columns = None
        if name == "records":
            columns = ["model_id", "persona", "subset", "explanation_available"] + [
                r["marker_id"] + "__present" for r in registry]
        tables[name] = pd.read_csv(source / filename, usecols=columns,
                                  float_precision="round_trip", low_memory=False)
    assert len(tables["records"]) == manifest["rows"]
    hashes = {name: pd.util.hash_pandas_object(t, index=True).to_numpy().copy()
              for name, t in tables.items()}
    context.update(tables)
    context["sensitivity_specs"] = [(name, frame.scope.drop_duplicates().tolist())
        for name, frame in tables["sensitivity"].groupby("sensitivity", sort=False)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = regional.RUNS / ("regional_gsm8k_without_axes_" + stamp)
    (run / "figures").mkdir(parents=True, exist_ok=False)
    expected = {p.name for p in (active / "figures").glob("*.png")}
    originals = {name: regional.sha(active / "figures" / name) for name in expected}
    audit = []
    print(f"STAGING {run}", flush=True)

    def save_figure(fig, name, kind, scope, detail=""):
        assert name in expected, name
        # Resolve constrained layouts before hiding labels, then freeze them.
        if fig.get_layout_engine() is not None:
            fig.set_dpi(160)
            fig.canvas.draw()
            fig.set_layout_engine("none")
        removed = hide_axis_text(fig)
        output = run / "figures" / name
        assert not output.exists()
        fig.savefig(output, dpi=160, facecolor="white")
        assert all(not a.get_visible() for ax in fig.axes
                   for a in [ax.xaxis.label, ax.yaxis.label,
                             ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(),
                             *ax.get_xticklabels(), *ax.get_yticklabels()])
        plt.close(fig)
        audit.append(dict(file=name, kind=kind, scope=scope, detail=detail,
                          removed_axis_text=removed, old_sha256=originals[name],
                          sha256=regional.sha(output)))
        print(f"{len(audit)}/{len(expected)} {name}", flush=True)

    context["save_figure"] = save_figure
    tree = ast.parse(cells[7])
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "save_figure":
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"save_table", "print"}):
            continue
        nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), "exec"), context)
    assert len(audit) == len(expected)
    assert {row["file"] for row in audit} == expected
    for name, before in hashes.items():
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(tables[name], index=True).to_numpy())
    assert all(regional.sha(p) == digest for p, digest in inputs.items())
    assert all(regional.sha(active / "figures" / name) == digest for name, digest in originals.items())
    regional.write_json(run / "manifest.json", dict(benchmark="GSM8K", persona="regional",
        source=str(source), inputs=inputs, figures=audit, measurements_changed=False,
        source_figures_unchanged=True, scientific_status=manifest["status"]))
    print(f"VERIFIED {len(audit)} figures: {run}", flush=True)


if __name__ == "__main__":
    main()

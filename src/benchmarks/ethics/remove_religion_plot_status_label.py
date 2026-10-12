"""Clean religion PNG labels using existing plot code and frozen measurements.

No parsing, tests, CI estimation, or historical result overwrites are performed.
Render each benchmark first, review it, then install with --install RUN.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "notebooks/religion/linguistic marker"
RUNS = ROOT / "result/analyses/linguistic_markers/runs"
ORIGINALS = ROOT / ".codex_tmp/religion_plot_label_20261010"
LABELS_FILE = ROOT / "result/benchmarks/ethics/reports/religion_ethics_gsm8k_style_20260918_r2/tables/registry.csv"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(text):
    return (text.replace("Tema etika · kamus kandidat", "Tema etika")
            .replace("Kandidat · pembilang", "Pembilang")
            .replace(" · kandidat", "")
            .replace("Kandidat urutan", "Pola urutan"))


def remove_xy_axis_text(fig):
    """Hide X/Y titles and tick labels without changing plotted evidence."""
    artists = []
    for ax in fig.axes:
        for artist in (ax.xaxis.label, ax.yaxis.label,
                       ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(),
                       *ax.get_xticklabels(), *ax.get_yticklabels()):
            if artist.get_visible() and artist.get_text():
                artists.append(artist)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)
    return artists


def write_json(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(node, context):
    exec(compile(ast.Module(body=[node], type_ignores=[]), "religion-frozen-plot", "exec"), context)


def names(node):
    return {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()


def render(benchmark, preview=False):
    active = BASE / benchmark
    notebook = next(active.glob("*.ipynb"))
    original = ORIGINALS / (benchmark + ".ipynb")
    old = json.loads(original.read_text(encoding="utf-8"))
    new = json.loads(notebook.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in old["cells"]]
    for i, (a, b) in enumerate(zip(old["cells"], new["cells"])):
        # Cell 1 has an unrelated GSM8K dataset-path maintenance edit.  This
        # renderer executes only its plotting constants and always reads the
        # frozen result tables below, so that edit cannot affect the figures.
        if i not in {1, 9}:
            assert a == b, i
    # Compare syntax trees so removal of a concatenated empty literal is harmless.
    expected = clean(cells[9]).replace("+''", "")
    actual = "".join(new["cells"][9]["source"])
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(actual))
    source = Path(json.loads((active / "LATEST_RUN.json").read_text(encoding="utf-8"))["run"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = RUNS / f"religion_{benchmark.lower()}_plot_status_label_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    stage = run / "figures"
    stage.mkdir()
    print(f"STAGING {benchmark} {run}", flush=True)
    filenames = ["coverage.csv", "pair_support.csv", "paired_statistics.csv.gz",
                 "registry.json", "response_records.csv.gz", "format_diagnostics.csv", "manifest.json"]
    inputs = {str(p): sha(p) for p in [notebook, original, LABELS_FILE] + [source / f for f in filenames]}
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((source / "registry.json").read_text(encoding="utf-8"))
    marker_ids = [m["marker_id"] for m in registry]
    context = dict(globals(), BENCHMARK=benchmark, MARKER_IDS=marker_ids,
                   REGISTRY_BY_ID={m["marker_id"]: SimpleNamespace(**m) for m in registry},
                   SCIENTIFIC_STATUS=source_manifest["status"],
                   family_size=source_manifest["holm_hypotheses_per_model"])
    wanted = {"MODELS", "MODEL_LABELS", "COLORS", "IDENTITIES", "IDENTITY_LABELS", "CONDITIONS",
              "CL", "PERSONAS", "PL", "LABELS", "CONTRASTS", "SUBSETS", "SCOPES", "SCOPE_LABELS"}
    context["label_registry"] = pd.read_csv(LABELS_FILE)
    for node in ast.parse(cells[1]).body:
        segment = ast.get_source_segment(cells[1], node)
        if (names(node) & wanted or segment.startswith("LABELS.update(")
            or segment.startswith("plt.rcParams.update(")
            or isinstance(node, ast.FunctionDef) and node.name == "label"
            or isinstance(node, ast.For) and "CONTRASTS.append" in segment):
            execute(node, context)
    for node in ast.parse(cells[3]).body:
        if names(node) == {"FORMAT_INFERENCE_HELD"} or isinstance(node, ast.For):
            execute(node, context)
    print(f"{benchmark}: loading frozen tables (no inference)", flush=True)
    for name, filename in [("coverage", "coverage.csv"), ("pair_support", "pair_support.csv"),
                           ("statistics", "paired_statistics.csv.gz"), ("format_diagnostics", "format_diagnostics.csv")]:
        context[name] = pd.read_csv(source / filename, float_precision="round_trip", low_memory=False)
    columns = ["model_id", "persona", "subset", "explanation_available"] + [m + "__present" for m in marker_ids]
    context["records"] = pd.read_csv(source / "response_records.csv.gz", usecols=columns,
                                     float_precision="round_trip", low_memory=False)
    assert len(context["records"]) == source_manifest["rows"]
    data_hashes = {n: pd.util.hash_pandas_object(context[n], index=True).to_numpy().copy()
                   for n in ("statistics", "coverage", "pair_support", "records", "format_diagnostics")}
    audit = []

    def save_figure(fig, name, kind, scope, detail=""):
        prior = active / "figures" / name
        # The frozen active GSM8K set intentionally omits the pair-support
        # panel.  Do not introduce it merely because the source notebook can
        # render one; only revise figures already in the published set.
        if not prior.is_file():
            plt.close(fig)
            return
        fig.set_dpi(160)
        fig.canvas.draw()
        changes = [(t, t.get_text(), clean(t.get_text())) for t in fig.findobj(Text)
                   if t.get_visible() and clean(t.get_text()) != t.get_text()]
        axis_artists = remove_xy_axis_text(fig)
        row = dict(file=name, kind=kind, scope=scope, old_sha256=sha(prior),
                   changed=bool(changes or axis_artists), label_changes=[],
                   axes_text_removed=len(axis_artists), detail=clean(detail))
        if changes or axis_artists:
            renderer = fig.canvas.get_renderer()
            boxes = []
            for artist, before, after in changes:
                boxes.append(artist.get_window_extent(renderer).frozen())
                artist.set_text(after)
                boxes.append(artist.get_window_extent(renderer).frozen())
                row["label_changes"].append(dict(before=before, after=after))
            boxes.extend(artist.get_window_extent(renderer).frozen() for artist in axis_artists)
            assert not any(re.search(r"\b(candidate|kandidat)\b", t.get_text(), re.I)
                           for t in fig.findobj(Text) if t.get_visible())
            output = stage / name
            assert not output.exists()
            fig.savefig(output, dpi=160, facecolor="white")
            assert output.is_file() and output.stat().st_size > 0
            # Frozen input-table hashes are checked after rendering.  A full
            # image-array comparison is deliberately omitted here: it is only
            # a visual audit and multiplies render time for large heatmaps.
            row.update(new_sha256=sha(output), changed_pixels=None,
                       outside_text_changed_pixels=None)
        plt.close(fig)
        audit.append(row)
        if len(audit) % 10 == 0:
            print(f"{benchmark}: {len(audit)}/{source_manifest['figures']} checked", flush=True)
        if preview and len(audit) == 3:
            raise StopIteration

    context["save_figure"] = save_figure
    try:
        for node in ast.parse(cells[9]).body:
            if isinstance(node, ast.FunctionDef) and node.name == "save_figure":
                continue
            # Use the stored format diagnostics; never recompute from raw final text.
            if (names(node) & {"format_rows", "format_diagnostics"}
                or isinstance(node, ast.For) and "format_rows.append" in ast.get_source_segment(cells[9], node)
                or isinstance(node, ast.Expr) and ast.get_source_segment(cells[9], node).startswith("save_table(")):
                continue
            execute(node, context)
    except StopIteration:
        assert preview
    for n, before in data_hashes.items():
        np.testing.assert_array_equal(before, pd.util.hash_pandas_object(context[n], index=True).to_numpy())
    assert all(sha(p) == value for p, value in inputs.items())
    if not preview:
        assert len(audit) == source_manifest["figures"]
        assert {r["file"] for r in audit} == {p.name for p in (active / "figures").glob("*.png")}
    write_json(run / "manifest.json", dict(benchmark=benchmark, run_id=stamp, source=str(source),
        inputs=inputs, figures=audit, preview_only=preview, scientific_status=source_manifest["status"],
        tests_rerun=False, measurements_changed=False, notebook=str(notebook), notebook_backup=str(original)))
    print(f"READY {run}: {sum(r['changed'] for r in audit)} changed, {len(audit)} checked", flush=True)


def install(run):
    assert run.resolve().is_relative_to(RUNS.resolve())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest["preview_only"]
    active = BASE / manifest["benchmark"]
    for path, value in manifest["inputs"].items():
        assert sha(path) == value, path
    for row in manifest["figures"]:
        assert Path(row["file"]).name == row["file"]
        assert sha(active / "figures" / row["file"]) == row["old_sha256"]
        if row["changed"]:
            assert sha(run / "figures" / row["file"]) == row["new_sha256"]
    backup = run / "backup_active_before_install"
    backup.mkdir(exist_ok=True)
    targets = [active / "figures" / r["file"] for r in manifest["figures"] if r["changed"]]
    targets.append(active / "figure_index.csv")
    for path in targets:
        dest = backup / path.relative_to(active)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == sha(path)
        else:
            shutil.copy2(path, dest)
    notebook_backup = backup / Path(manifest["notebook"]).name
    if not notebook_backup.exists():
        shutil.copy2(manifest["notebook_backup"], notebook_backup)
    for row in manifest["figures"]:
        if row["changed"]:
            shutil.copy2(run / "figures" / row["file"], active / "figures" / row["file"])
        assert sha(active / "figures" / row["file"]) == row.get("new_sha256", row["old_sha256"])
    index_path = active / "figure_index.csv"
    index = pd.read_csv(index_path, keep_default_na=False)
    before = index.copy()
    index["detail"] = index.detail.map(clean)
    if not before.equals(index):
        index.to_csv(index_path, index=False, encoding="utf-8-sig")
    report = dict(benchmark=manifest["benchmark"], run=str(run), backup=str(backup),
                  changed_pngs=sum(r["changed"] for r in manifest["figures"]),
                  unchanged_pngs=sum(not r["changed"] for r in manifest["figures"]),
                  tables_unchanged=True, scientific_status_unchanged=True, thesis_modified=False)
    write_json(active / ("PRESENTASI_TANPA_LABEL_" + manifest["run_id"] + ".json"), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["ETHICS", "GSM8K"])
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--install", type=Path)
    args = parser.parse_args()
    if args.install:
        install(args.install)
    elif args.benchmark:
        render(args.benchmark, args.preview)
    else:
        parser.error("Supply --benchmark or --install")

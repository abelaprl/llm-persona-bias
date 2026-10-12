"""Render label-only regional PNG changes from frozen tables, then install.

Does not run extraction, scoring, statistical tests, or bootstrap estimation.
Independent scopes are rendered in separate processes. Only PNGs whose visible
text changes are replaced; source data and scientific status remain untouched.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
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

if __package__:
    from . import remove_religion_plot_status_label as shared
else:
    import remove_religion_plot_status_label as shared

ROOT = shared.ROOT
BASE = ROOT / "notebooks/regional/linguistic marker"
RUNS = shared.RUNS
ORIGINALS = ROOT / ".codex_tmp/regional_plot_label_20261010"
clean, sha, write_json = shared.clean, shared.sha, shared.write_json


def setup(benchmark, cells, registry, manifest):
    context = dict(globals(), BENCHMARK=benchmark,
                   MARKER_IDS=[r["marker_id"] for r in registry],
                   REGISTRY_BY_ID={r["marker_id"]: SimpleNamespace(**r) for r in registry},
                   SCIENTIFIC_STATUS=manifest["status"],
                   family_size=manifest["holm_hypotheses_per_model"])
    wanted = {"MODELS", "MODEL_LABELS", "COLORS", "IDENTITIES", "IDENTITY_LABELS",
              "SHORT_LABELS", "CONDITIONS", "CL", "PERSONAS", "PL", "COMMON", "LABELS",
              "FORMAT_SENSITIVE", "FORMAT_INFERENCE_HELD", "CONTRASTS", "SUBSETS",
              "SCOPES", "SCOPE_LABELS"}
    for node in ast.parse(cells[1]).body:
        segment = ast.get_source_segment(cells[1], node)
        if (shared.names(node) & wanted or segment.startswith("LABELS.update(")
            or segment.startswith("plt.rcParams.update(")
            or isinstance(node, ast.FunctionDef) and node.name == "label"
            or isinstance(node, ast.For) and ("CONTRASTS.append" in segment or "LABELS[marker_id] +=" in segment)):
            shared.execute(node, context)
    return context


def worker(job):
    run, scope = Path(job[0]), job[1]
    config = json.loads((run / "render_inputs.json").read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    cells = config["cells"]
    context = setup(benchmark, cells, config["registry"], config["source_manifest"])
    tables = pd.read_pickle(run / (scope + ".pkl"))
    before = {n: pd.util.hash_pandas_object(t, index=True).to_numpy().copy() for n, t in tables.items()}
    context.update(tables)
    context["SCOPES"] = [] if scope == "extras" else [scope]
    context["sensitivity_specs"] = ([(n, g.scope.drop_duplicates().tolist())
        for n, g in tables["sensitivity"].groupby("sensitivity", sort=False)] if scope == "extras" else [])
    audit = []

    def save_figure(fig, name, kind, fig_scope, detail=""):
        prior = BASE / benchmark / "figures" / name
        changes = [(t, t.get_text(), clean(t.get_text())) for t in fig.findobj(Text)
                   if t.get_visible() and clean(t.get_text()) != t.get_text()]
        row = dict(file=name, kind=kind, scope=fig_scope, old_sha256=sha(prior),
                   changed=bool(changes), label_changes=[], detail=clean(detail))
        if changes:
            fig.set_dpi(160)
            renderer = fig.canvas.get_renderer()
            boxes = []
            for artist, old, new in changes:
                boxes.append(artist.get_window_extent(renderer).frozen())
                artist.set_text(new)
                boxes.append(artist.get_window_extent(renderer).frozen())
                row["label_changes"].append(dict(before=old, after=new))
            output = run / "figures" / name
            assert not output.exists(), output
            fig.savefig(output, dpi=160, facecolor="white")
            with Image.open(prior) as a, Image.open(output) as b:
                assert a.size == b.size, (name, a.size, b.size)
                delta = np.any(np.asarray(a.convert("RGB")) != np.asarray(b.convert("RGB")), axis=2)
                total = int(delta.sum())
                h, w = delta.shape
                for box in boxes:
                    x0, x1 = max(0, int(np.floor(box.x0))-4), min(w, int(np.ceil(box.x1))+4)
                    y0, y1 = max(0, h-int(np.ceil(box.y1))-4), min(h, h-int(np.floor(box.y0))+4)
                    delta[y0:y1, x0:x1] = False
                assert not delta.any(), (name, "unexpected changes outside text", int(delta.sum()))
            row.update(new_sha256=sha(output), changed_pixels=total, outside_text_changed_pixels=0)
        assert not any(re.search(r"\b(candidate|kandidat)\b", t.get_text(), re.I)
                       for t in fig.findobj(Text) if t.get_visible()), name
        plt.close(fig)
        audit.append(row)
        context["figure_index"].append(row)
        if len(audit) % 20 == 0:
            print(f"{benchmark}/{scope}: {len(audit)} checked", flush=True)

    context["save_figure"] = save_figure
    for node in ast.parse(cells[7]).body:
        if isinstance(node, ast.FunctionDef) and node.name == "save_figure":
            continue
        segment = ast.get_source_segment(cells[7], node)
        if isinstance(node, ast.Expr) and (segment.startswith("save_table(") or segment.startswith("print(")):
            continue
        # The final format diagnostic belongs only to the extras job.
        if shared.names(node) == {"fmt"} and scope != "extras":
            break
        shared.execute(node, context)
    for name, value in before.items():
        np.testing.assert_array_equal(value, pd.util.hash_pandas_object(tables[name], index=True).to_numpy())
    write_json(run / ("audit_" + scope + ".json"), audit)
    print(f"DONE {benchmark}/{scope}: {len(audit)} checked", flush=True)
    return audit


def render(benchmark, workers=3):
    active = BASE / benchmark
    notebook = next(active.glob("*.ipynb"))
    original = ORIGINALS / (benchmark + ".ipynb")
    old = json.loads(original.read_text(encoding="utf-8"))
    new = json.loads(notebook.read_text(encoding="utf-8"))
    assert len(old["cells"]) == len(new["cells"])
    for i, (a, b) in enumerate(zip(old["cells"], new["cells"])):
        if i != 7:
            assert a == b, i
    cells = ["".join(c["source"]) for c in old["cells"]]
    expected = clean(cells[7]).replace("+''", "")
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse("".join(new["cells"][7]["source"])))
    source = Path(json.loads((active / "LATEST_RUN.json").read_text(encoding="utf-8"))["run"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((source / "registry.json").read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = RUNS / f"regional_{benchmark.lower()}_plot_status_label_{stamp}"
    (run / "figures").mkdir(parents=True, exist_ok=False)
    print(f"STAGING {run}", flush=True)
    inputs = {str(p): sha(p) for p in [notebook, original] + list(source.glob("*.csv*"))
              + [source / "registry.json", source / "manifest.json"]}
    mappings = {"coverage": "coverage.csv", "pair_support": "pair_support.csv",
                "statistics": "paired_statistics.csv.gz", "format_summary": "format_diagnostics.csv",
                "sensitivity": "sensitivity_statistics.csv.gz"}
    tables = {n: pd.read_csv(source / f, float_precision="round_trip", low_memory=False) for n, f in mappings.items()}
    print(f"{benchmark}: loading frozen response columns; no inference", flush=True)
    columns = ["model_id", "persona", "subset", "explanation_available"] + [r["marker_id"] + "__present" for r in registry]
    tables["records"] = pd.read_csv(source / "response_records.csv.gz", usecols=columns,
                                    float_precision="round_trip", low_memory=False)
    assert len(tables["records"]) == manifest["rows"]
    context = setup(benchmark, cells, registry, manifest)
    scopes = context["SCOPES"] + ["extras"]
    write_json(run / "render_inputs.json", dict(benchmark=benchmark, cells=cells, registry=registry, source_manifest=manifest))
    for scope in scopes:
        selected = {}
        for name, frame in tables.items():
            if scope == "extras":
                if name in {"sensitivity", "format_summary"}:
                    selected[name] = frame
            elif name not in {"sensitivity", "format_summary"}:
                column = "scope" if name in {"statistics", "pair_support"} else "subset"
                selected[name] = frame if column == "subset" and scope == "gabungan" else frame[frame[column].eq(scope)]
        pd.to_pickle(selected, run / (scope + ".pkl"))
    del tables
    audit = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(worker, [(str(run), s) for s in scopes]):
            audit.extend(rows)
    assert len(audit) == manifest["figures"]
    assert {r["file"] for r in audit} == {p.name for p in (active / "figures").glob("*.png")}
    assert all(sha(p) == value for p, value in inputs.items())
    write_json(run / "manifest.json", dict(benchmark=benchmark, run_id=stamp, source=str(source),
        inputs=inputs, figures=audit, preview_only=False, scientific_status=manifest["status"],
        tests_rerun=False, measurements_changed=False, notebook=str(notebook), notebook_backup=str(original)))
    print(f"READY {run}: {sum(r['changed'] for r in audit)} changed, {len(audit)} checked", flush=True)


def resume(run, workers):
    """Resume prepared caches after a process-permission failure, without overwrites."""
    run = run.resolve()
    assert run.is_relative_to(RUNS.resolve())
    assert not (run / "manifest.json").exists()
    config = json.loads((run / "render_inputs.json").read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    active = BASE / benchmark
    notebook = next(active.glob("*.ipynb"))
    original = ORIGINALS / (benchmark + ".ipynb")
    current = json.loads(notebook.read_text(encoding="utf-8"))
    previous = json.loads(original.read_text(encoding="utf-8"))
    assert config["cells"] == ["".join(c["source"]) for c in previous["cells"]]
    for i, (a, b) in enumerate(zip(previous["cells"], current["cells"])):
        if i != 7:
            assert a == b
    assert ast.dump(ast.parse(clean(config["cells"][7]).replace("+''", ""))) == ast.dump(ast.parse("".join(current["cells"][7]["source"])))
    source = Path(json.loads((active / "LATEST_RUN.json").read_text(encoding="utf-8"))["run"])
    inputs = {str(p): sha(p) for p in [notebook, original] + list(source.glob("*.csv*"))
              + [source / "registry.json", source / "manifest.json"]}
    cache_hashes = {str(p): sha(p) for p in run.glob("*.pkl")}
    scopes = setup(benchmark, config["cells"], config["registry"], config["source_manifest"])["SCOPES"] + ["extras"]
    audit = []
    pending = []
    for scope in scopes:
        path = run / ("audit_" + scope + ".json")
        if path.exists():
            audit.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append((str(run), scope))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(worker, pending):
            audit.extend(rows)
    assert len(audit) == config["source_manifest"]["figures"]
    assert {r["file"] for r in audit} == {p.name for p in (active / "figures").glob("*.png")}
    assert all(sha(p) == value for p, value in {**inputs, **cache_hashes}.items())
    stamp = run.name.split("plot_status_label_")[1]
    write_json(run / "manifest.json", dict(benchmark=benchmark, run_id=stamp, source=str(source),
        inputs=inputs, cache_hashes=cache_hashes, figures=audit, preview_only=False,
        scientific_status=config["source_manifest"]["status"], tests_rerun=False,
        measurements_changed=False, notebook=str(notebook), notebook_backup=str(original)))
    print(f"READY {run}: {sum(r['changed'] for r in audit)} changed, {len(audit)} checked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["ETHICS", "GSM8K"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.install:
        shared.BASE = BASE
        shared.install(args.install)
    elif args.resume:
        resume(args.resume, args.workers)
    elif args.benchmark:
        render(args.benchmark, args.workers)
    else:
        parser.error("Supply --benchmark or --install")

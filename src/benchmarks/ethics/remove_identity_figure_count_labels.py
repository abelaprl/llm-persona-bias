"""Render ETHICS regional/religion count-label edits from frozen tables only.

Prepare once, render each scope in bounded processes, then validate and install.
Historical result files are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

try:
    from . import remove_regional_plot_status_label as regional
    from .identity_figure_count_labels import clean_count_text
except ImportError:
    import remove_regional_plot_status_label as regional
    from identity_figure_count_labels import clean_count_text

shared = regional.shared
ROOT, RUNS = shared.ROOT, shared.RUNS
sha, write_json = shared.sha, shared.write_json


def active_path(dimension):
    assert dimension in {"regional", "religion"}
    return ROOT / "notebooks" / dimension / "linguistic marker/ETHICS"


def prepare(dimension):
    active = active_path(dimension)
    notebook = next(active.glob("*.ipynb"))
    source = Path(json.loads((active / "LATEST_RUN.json").read_text(encoding="utf-8"))["run"])
    if not source.is_absolute():
        source = ROOT / source
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = RUNS / f"{dimension}_ethics_without_count_labels_{stamp}"
    (run / "figures").mkdir(parents=True, exist_ok=False)
    shutil.copy2(notebook, run / "original.ipynb")
    index = pd.read_csv(active / "figure_index.csv")
    rows = [dict(file=r.file, kind=r.kind, scope=r.scope,
                 old_sha256=sha(active / "figures" / r.file)) for r in index.itertuples()]
    assert {r["file"] for r in rows} == {p.name for p in (active / "figures").glob("*.png")}
    inputs = {str(source / n): sha(source / n) for n in
              ["coverage.csv", "paired_statistics.csv.gz", "manifest.json", "registry.json"]}
    if dimension == "religion":
        inputs[str(shared.LABELS_FILE)] = sha(shared.LABELS_FILE)
    config = dict(dimension=dimension, source=str(source), notebook=str(notebook),
                  inputs=inputs, figures=rows)
    write_json(run / "config.json", config)
    print(run, flush=True)


def setup(config, cells):
    source = Path(config["source"])
    registry = json.loads((source / "registry.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if config["dimension"] == "regional":
        return regional.setup("ETHICS", cells, registry, manifest)
    context = dict(vars(shared), BENCHMARK="ETHICS", MARKER_IDS=[r["marker_id"] for r in registry],
                   REGISTRY_BY_ID={r["marker_id"]: SimpleNamespace(**r) for r in registry},
                   SCIENTIFIC_STATUS=manifest["status"], family_size=manifest["holm_hypotheses_per_model"])
    context["label_registry"] = pd.read_csv(shared.LABELS_FILE)
    wanted = {"MODELS", "MODEL_LABELS", "COLORS", "IDENTITIES", "IDENTITY_LABELS", "CONDITIONS",
              "CL", "PERSONAS", "PL", "LABELS", "CONTRASTS", "SUBSETS", "SCOPES", "SCOPE_LABELS"}
    for node in ast.parse(cells[1]).body:
        segment = ast.get_source_segment(cells[1], node)
        if (shared.names(node) & wanted or segment.startswith("LABELS.update(")
            or segment.startswith("plt.rcParams.update(")
            or isinstance(node, ast.FunctionDef) and node.name == "label"
            or isinstance(node, ast.For) and "CONTRASTS.append" in segment):
            shared.execute(node, context)
    for node in ast.parse(cells[3]).body:
        if shared.names(node) == {"FORMAT_INFERENCE_HELD"} or isinstance(node, ast.For):
            shared.execute(node, context)
    return context


def render(run, scope, start, stop):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    dimension = config["dimension"]
    active = active_path(dimension)
    notebook = json.loads((run / "original.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"]]
    context = setup(config, cells)
    context["SCOPES"] = [scope]
    selected = [r for r in config["figures"] if r["scope"] == scope
                and r["kind"] in {"coverage", "effect_heatmap"}][start:stop]
    wanted = {r["file"]: r for r in selected}
    assert wanted
    context["requested_files"] = wanted
    source = Path(config["source"])
    for name, filename in [("coverage", "coverage.csv"), ("statistics", "paired_statistics.csv.gz")]:
        context[name] = pd.read_csv(source / filename, float_precision="round_trip", low_memory=False)
    hashes = {n: pd.util.hash_pandas_object(context[n], index=True).to_numpy().copy()
              for n in ["coverage", "statistics"]}
    rows = []

    def save_figure(fig, name, kind, fig_scope, detail=""):
        assert name in wanted, name
        prior = active / "figures" / name
        assert sha(prior) == wanted[name]["old_sha256"]
        output = run / "figures" / name
        checkpoint = run / "figure_audits" / (name + ".json")
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert sha(output) == row["new_sha256"]
            assert row["old_sha256"] == wanted[name]["old_sha256"]
            rows.append(row)
            plt.close(fig)
            return
        fig.set_dpi(160)
        with Image.open(prior) as old_image:
            before = np.asarray(old_image.convert("RGBA"))
        # Compare directly against the published PNG. No second full render is
        # necessary: every pixel outside the edited text must still match it.
        renderer = fig.canvas.get_renderer()
        changes, boxes = [], []
        old_texts = [t.get_text() for t in fig.findobj(Text)]
        for artist in fig.findobj(Text):
            old = artist.get_text()
            new = clean_count_text(old)
            if old != new:
                boxes.append(artist.get_window_extent(renderer).frozen())
                artist.set_text(new)
                boxes.append(artist.get_window_extent(renderer).frozen())
                changes.append(dict(before=old, after=new))
        assert changes, name
        new_texts = [t.get_text() for t in fig.findobj(Text)]
        assert new_texts == [clean_count_text(t) for t in old_texts]
        assert all(clean_count_text(t) == t for t in new_texts)
        # Measurements, percentages, and significance symbols remain identical.
        import re
        decimals = lambda texts: Counter(re.findall(r"[+−-]?\d+\.\d+", "\n".join(texts)))
        assert decimals(old_texts) == decimals(new_texts)
        assert sum(t.count("*") for t in old_texts) == sum(t.count("*") for t in new_texts)
        fig.canvas.draw()
        after = np.asarray(fig.canvas.buffer_rgba()).copy()
        assert before.shape == after.shape, name
        delta = np.any(before != after, axis=2)
        changed_pixels = int(delta.sum())
        height, width = delta.shape
        for box in boxes:
            x0, x1 = max(0, int(np.floor(box.x0))-3), min(width, int(np.ceil(box.x1))+3)
            y0, y1 = max(0, height-int(np.ceil(box.y1))-3), min(height, height-int(np.floor(box.y0))+3)
            delta[y0:y1, x0:x1] = False
        assert not delta.any(), (name, "changes outside edited text", int(delta.sum()))
        # Save exactly the already checked canvas, without an extra layout pass.
        if output.exists():
            with Image.open(output) as existing:
                np.testing.assert_array_equal(np.asarray(existing.convert("RGBA")), after)
        else:
            Image.fromarray(after).save(output, dpi=(160, 160))
        with Image.open(output) as check:
            check.verify()
        rows.append(dict(wanted[name], new_sha256=sha(output), label_changes=changes,
                         changed_pixels=changed_pixels, outside_text_changed_pixels=0,
                         reference="active_png", remaining_count_labels=0))
        checkpoint.parent.mkdir(exist_ok=True)
        write_json(checkpoint, rows[-1])
        plt.close(fig)
        print(f"{dimension}/{scope}: {len(rows)}/{len(wanted)} {name}", flush=True)

    context["save_figure"] = save_figure
    cell = cells[7 if dimension == "regional" else 9]
    for node in ast.parse(cell).body:
        if isinstance(node, ast.FunctionDef) and node.name == "save_figure":
            continue
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "scope":
            coverage_nodes = []
            for child in node.body:
                if shared.names(child) == {"ps"}:
                    break
                coverage_nodes.append(child)
            family_loop = next(c for c in node.body if isinstance(c, ast.For)
                               and isinstance(c.target, ast.Tuple)
                               and any(isinstance(t, ast.Name) and t.id == "family_name" for t in c.target.elts))
            group_loop = family_loop.body[0].body[-1]
            assert isinstance(group_loop, ast.For) and group_loop.target.id == "group"
            guard = ast.parse("if f'02_efek_{family_name}_{role}_{group}_{scope}.png' not in requested_files: continue").body[0]
            group_loop.body.insert(0, guard)
            node.body = (coverage_nodes if f"00_cakupan_{scope}.png" in wanted else []) + [family_loop]
            ast.fix_missing_locations(node)
            shared.execute(node, context)
            break
        shared.execute(node, context)
    for name, value in hashes.items():
        np.testing.assert_array_equal(value, pd.util.hash_pandas_object(context[name], index=True).to_numpy())
    assert {r["file"] for r in rows} == set(wanted)
    assert all(sha(p) == h for p, h in config["inputs"].items())
    write_json(run / f"audit_{scope}_{start}_{stop}.json", rows)
    print(f"DONE {dimension}/{scope}: {len(rows)}", flush=True)


def install(run):
    assert run.resolve().is_relative_to(RUNS.resolve())
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    active = active_path(config["dimension"])
    rows = [r for p in sorted(run.glob("audit_*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
    expected = {r["file"] for r in config["figures"] if r["kind"] in {"coverage", "effect_heatmap"}}
    assert len(rows) == len(expected) and {r["file"] for r in rows} == expected
    assert all(sha(p) == h for p, h in config["inputs"].items())
    replacements = {r["file"]: r for r in rows}
    for row in config["figures"]:
        target = active / "figures" / row["file"]
        assert target.resolve().is_relative_to((active / "figures").resolve())
        allowed = {row["old_sha256"]}
        if row["file"] in replacements:
            allowed.add(replacements[row["file"]]["new_sha256"])
        assert sha(target) in allowed, target
    for row in rows:
        assert sha(run / "figures" / row["file"]) == row["new_sha256"]
        with Image.open(run / "figures" / row["file"]) as check:
            check.verify()
    notebook_path = Path(config["notebook"])
    text = notebook_path.read_text(encoding="utf-8")
    original_nb = json.loads((run / "original.ipynb").read_text(encoding="utf-8"))
    current_nb = json.loads(text)
    idx = 7 if config["dimension"] == "regional" else 9
    expected_source = "".join(original_nb["cells"][idx]["source"])
    if "from src.benchmarks.ethics.identity_figure_count_labels import strip_count_labels" not in expected_source:
        expected_source = expected_source.replace(
            "def save_figure(fig,name,kind,scope,detail=''):\n",
            "def save_figure(fig,name,kind,scope,detail=''):\n"
            "    from src.benchmarks.ethics.identity_figure_count_labels import strip_count_labels\n"
            "    strip_count_labels(fig)\n")
    if config["dimension"] == "regional":
        expected_source = expected_source.replace(
            "reused=bool(reuse and selected is not None and kind not in selected)",
            "reused=bool(reuse and selected is not None and kind not in selected and kind not in {'coverage','effect_heatmap'})")
    assert "".join(current_nb["cells"][idx]["source"]) == expected_source, "Unexpected plot-cell changes"
    backup = run / "backup_active_before_install"
    backup.mkdir(exist_ok=True)
    for target in [notebook_path, active / "figure_index.csv"] + [active / "figures" / r["file"] for r in rows]:
        dest = backup / target.relative_to(active)
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = replacements[target.name]["old_sha256"] if target.parent.name == "figures" else sha(target)
        if dest.exists():
            assert sha(dest) == expected_hash
        else:
            assert sha(target) == expected_hash
            shutil.copy2(target, dest)
    # Refresh only embedded PNGs that match a replaced file; keep other edits.
    by_hash = {r["old_sha256"]: run / "figures" / r["file"] for r in rows}
    embedded = 0
    for cell in current_nb["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            value = data.get("image/png")
            if not value:
                continue
            encoded = "".join(value) if isinstance(value, list) else value
            h = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
            if h in by_hash:
                new_value = base64.b64encode(by_hash[h].read_bytes()).decode("ascii")
                if isinstance(value, list):
                    new_value = [new_value]
                old_json = json.dumps(value)
                if old_json not in text:
                    assert not isinstance(value, list), "Embedded list formatting requires explicit handling"
                assert old_json in text
                text = text.replace(old_json, json.dumps(new_value))
                embedded += 1
    for row in rows:
        target = active / "figures" / row["file"]
        if sha(target) != row["new_sha256"]:
            shutil.copy2(run / "figures" / row["file"], target)
    if embedded:
        notebook_path.write_text(text, encoding="utf-8")
    for row in config["figures"]:
        assert sha(active / "figures" / row["file"]) == replacements.get(row["file"], row).get("new_sha256", row["old_sha256"])
    report = dict(dimension=config["dimension"], run=str(run), changed_pngs=len(rows),
                  unchanged_pngs=len(config["figures"])-len(rows), embedded_pngs_updated=embedded,
                  source_tables_unchanged=True, outside_text_changed_pixels=0,
                  backup=str(backup), figures=rows)
    for path, payload in [(run / "manifest.json", report),
                          (active / f"PRESENTASI_TANPA_RASIO_{run.name}.json",
                           {k:v for k,v in report.items() if k != "figures"})]:
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == payload
        else:
            write_json(path, payload)
    print(json.dumps({k:v for k,v in report.items() if k != "figures"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", choices=["regional", "religion"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--scope", choices=["commonsense", "deontology", "justice", "virtue", "gabungan"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--install", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.install:
        install(args.install)
    else:
        assert args.run and args.scope
        render(args.run, args.scope, args.start, args.stop)

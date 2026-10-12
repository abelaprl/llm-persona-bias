"""Re-render and install ETHICS gender figures without auxiliary count ratios."""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import remove_gender_ethics_plot_status_label as renderer
from .gender_figure_count_labels import strip_count_labels


def render_coverage(run):
    cells = renderer.source_cells()
    env = dict(plt=plt, np=np, pd=pd, BENCHMARK="ETHICS")
    constants = {"MODELS", "MODEL_LABELS", "CONDITIONS", "CONDITION_LABELS",
                 "PERSONAS", "PERSONA_LABELS"}
    for node in ast.parse(cells[1]).body:
        if renderer.assigned_names(node) & constants:
            renderer.execute(node, env)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                        "axes.titlesize": 14, "axes.labelsize": 12,
                        "xtick.labelsize": 11, "ytick.labelsize": 11,
                        "figure.dpi": 110, "savefig.dpi": 180})
    rows = []
    def finish(fig, name, title, table):
        audit = strip_count_labels(fig)
        old = renderer.ACTIVE / "figures" / name
        new = run / "figures" / name
        assert old.exists() and not new.exists()
        new.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(new, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        rows.append(dict(family="coverage", figure=name, old_sha256=renderer.digest(old),
                         new_sha256=renderer.digest(new), text_cleanup=audit))
    env["finish"] = finish
    index = pd.read_csv(renderer.ACTIVE / "figure_index.csv")
    for cell_index, pooled in [(8, False), (15, True)]:
        matching = index[index.figure.str.startswith("00_cakupan_")]
        matching = matching[matching.figure.eq("00_cakupan_gabungan.png") == pooled]
        assert len(matching) == (1 if pooled else 4)
        source = Path(matching.iloc[0].source_run) / "coverage.csv"
        env["coverage"] = pd.read_csv(source)
        env["SUBSETS"] = sorted(env["coverage"].subset.unique())
        # Execute only the coverage block, ending at its first finish() call.
        loop = next(n for n in ast.parse(cells[cell_index]).body
                    if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and n.target.id == "subset")
        boundary = next(i for i, n in enumerate(loop.body)
                        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == "finish")
        loop.body = loop.body[:boundary + 1]
        renderer.execute(loop, env)
    assert len(rows) == 5
    renderer.write_json(run / "audit_coverage.json", rows)
    return rows


def install(run, rows):
    backup = run / "backup_active_before_install"
    backup.mkdir(exist_ok=True)
    replacements = []
    hashes = {}
    for row in rows:
        for key, old_key, new_key in [("figure", "old_sha256", "new_sha256"),
                                       ("pdf", "old_pdf_sha256", "new_pdf_sha256")]:
            if key not in row:
                continue
            relative = Path(row[key])
            target = renderer.ACTIVE / "figures" / relative
            source = run / "figures" / relative
            assert target.resolve().is_relative_to((renderer.ACTIVE / "figures").resolve())
            assert renderer.digest(target) in {row[old_key], row[new_key]}
            assert renderer.digest(source) == row[new_key]
            destination = backup / "figures" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert renderer.digest(destination) == row[old_key]
            else:
                assert renderer.digest(target) == row[old_key]
                shutil.copy2(target, destination)
            replacements.append((target, source))
            hashes[row[old_key]] = row[new_key]
    original = renderer.ROOT / ".codex_tmp/gender_ethics_counts_20261011/original.ipynb"
    if original.exists() and not (backup / "original_at_start.ipynb").exists():
        shutil.copy2(original, backup / "original_at_start.ipynb")
    if not (backup / renderer.NOTEBOOK.name).exists():
        shutil.copy2(renderer.NOTEBOOK, backup / renderer.NOTEBOOK.name)
    indexes = list(renderer.ACTIVE.glob("figure_index*.csv"))
    indexes += list((renderer.ACTIVE / "figures").glob("*/*.csv"))
    for index in indexes:
        data = pd.read_csv(index, keep_default_na=False)
        revised = data.copy()
        for column in data.columns:
            if "sha256" in column:
                revised[column] = data[column].map(lambda value: hashes.get(value, value))
        if not data.equals(revised):
            destination = backup / index.relative_to(renderer.ACTIVE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(index, destination)
            revised.to_csv(index, index=False, encoding="utf-8-sig")
    for target, source in replacements:
        shutil.copy2(source, target)
        assert renderer.digest(target) == renderer.digest(source)
    # Mechanically refresh matching embedded PNG outputs without changing cells.
    notebook = renderer.NOTEBOOK.read_text(encoding="utf-8")
    document = json.loads(notebook)
    changed_outputs = 0
    for cell in document["cells"]:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if not isinstance(encoded, str):
                continue
            import hashlib
            old_hash = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
            matching = next((r for r in rows if r["old_sha256"] == old_hash), None)
            if matching:
                updated = base64.b64encode((run / "figures" / matching["figure"]).read_bytes()).decode("ascii")
                notebook = notebook.replace(json.dumps(encoded), json.dumps(updated))
                changed_outputs += 1
    if changed_outputs:
        renderer.NOTEBOOK.write_text(notebook, encoding="utf-8")
    renderer.write_json(renderer.ACTIVE / ("PRESENTASI_TANPA_RASIO_" + run.name + ".json"),
                        dict(run=str(run), backup=str(backup), files=len(replacements),
                             source_tables_unchanged=True, embedded_pngs_updated=changed_outputs))
    print(f"Installed {len(replacements)} files; backup: {backup}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=list(renderer.EXPECTED))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--scope")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--assemble-from", type=Path)
    args = parser.parse_args()
    if args.family:
        renderer.render_family(args.family, str(args.run),
            reuse_string=str(args.reuse) if args.reuse else None,
            text_filter=strip_count_labels, scope=args.scope)
        return
    if args.install:
        manifest = json.loads((args.install / "manifest.json").read_text(encoding="utf-8"))
        for path, expected in manifest["input_hashes"].items():
            assert renderer.digest(path) == expected, path
        install(args.install, manifest["figures"])
        return
    if args.assemble_from:
        run = args.run.resolve()
        vector_audit = run / "audit_vector.json"
        if vector_audit.exists():
            rows = json.loads(vector_audit.read_text(encoding="utf-8"))
        else:
            rows = [row for part in sorted(run.glob("audit_vector_*.json"))
                    for row in json.loads(part.read_text(encoding="utf-8"))]
        assert len(rows) == 375
        for scope in ["commonsense", "deontology", "justice", "virtue", "gabungan"]:
            part = json.loads((run / f"audit_v2_{scope}.json").read_text(encoding="utf-8"))
            assert len(part) == 22
            rows += part
        coverage = json.loads((args.assemble_from / "audit_coverage.json").read_text(encoding="utf-8"))
        for row in coverage:
            source = args.assemble_from / "figures" / row["figure"]
            target = run / "figures" / row["figure"]
            assert not target.exists() and renderer.digest(source) == row["new_sha256"]
            shutil.copy2(source, target)
        rows += coverage
        assert len(rows) == 490 and len({r["figure"] for r in rows}) == 490
        assert all(r["text_cleanup"]["remaining_count_labels"] == 0 for r in rows)
        for row in rows:
            with renderer.Image.open(run / "figures" / row["figure"]) as png:
                png.verify()
        inputs = [renderer.SUPPORT / "comparisons_support_v2_candidate.csv", renderer.SUPPORT / "registry.json"]
        renderer.write_json(run / "manifest.json", dict(figures=rows,
            input_hashes={str(p): renderer.digest(p) for p in inputs}, png_count=490, pdf_count=375,
            statistics_rerun=False, extraction_rerun=False))
        print(f"VERIFIED: 490 PNG and 375 PDF in {run}", flush=True)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = renderer.RUNS / ("gender_ethics_without_count_labels_" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    sources = [renderer.SUPPORT / "comparisons_support_v2_candidate.csv", renderer.SUPPORT / "registry.json"]
    index = pd.read_csv(renderer.ACTIVE / "figure_index.csv")
    sources += list({Path(r.source_run) / "coverage.csv" for r in index.itertuples()
                     if r.figure.startswith("00_cakupan_")})
    hashes = {str(p): renderer.digest(p) for p in sources}
    print(f"STAGING {run}", flush=True)
    workers, logs = [], []
    jobs = [("vector", "src.benchmarks.ethics.clean_gender_count_labels_pdf", [])]
    jobs += [("v2_" + scope, "src.benchmarks.ethics.remove_gender_figure_count_labels",
              ["--family", "v2", "--scope", scope])
             for scope in ["commonsense", "deontology", "justice", "virtue", "gabungan"]]
    for name, module, extra in jobs:
        log = (run / f"{name}.log").open("x", encoding="utf-8")
        logs.append(log)
        workers.append(subprocess.Popen([sys.executable, "-X", "utf8", "-m", module,
            "--run", str(run), *extra], cwd=renderer.ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0))
    coverage = render_coverage(run)
    codes = [worker.wait() for worker in workers]
    for log in logs:
        log.close()
    assert codes == [0] * len(jobs), (codes, str(run))
    rows = coverage + [row for name, _, _ in jobs for row in json.loads(
        (run / f"audit_{name}.json").read_text(encoding="utf-8"))]
    assert len(rows) == 490 and all(row["text_cleanup"]["remaining_count_labels"] == 0 for row in rows)
    assert all(renderer.digest(p) == h for p, h in hashes.items())
    renderer.write_json(run / "manifest.json", dict(figures=rows, input_hashes=hashes,
        png_count=len(rows), pdf_count=sum("pdf" in row for row in rows),
        statistics_rerun=False, extraction_rerun=False))
    print(f"VERIFIED {len(rows)} PNG; ready for visual review: {run}", flush=True)


if __name__ == "__main__":
    main()

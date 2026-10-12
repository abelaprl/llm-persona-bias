"""Re-render existing ETHICS charts without status footers, from frozen tables.

No extraction, statistical tests, or historical result writes. Run without arguments
to stage and validate; use --install RUN only after reviewing the staged images.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "notebooks/gender/linguistic marker/ETHICS"
NOTEBOOK = ACTIVE / "ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"
RUNS = ROOT / "result/analyses/linguistic_markers/runs"
SUPPORT = RUNS / "gender_ethics_support_v2_20261004_232242_586663"
FOLDERS = {"v2": "", "v3": "bab6_v3_20261007_230725_835481",
           "v31": "bab6_v3_1_20261007_234752_514764"}
EXPECTED = {"v2": 110, "v3": 275, "v31": 100}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Generated audit artifacts only, never existing statistical outputs.
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def source_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def execute(node, env):
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), env)


def assigned_names(node):
    return {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()


def render_family(family, run_string, reuse_string=None, text_filter=None, scope=None):
    run = Path(run_string)
    stage = run / "figures" / FOLDERS[family]
    stage.mkdir(parents=True, exist_ok=True)
    cells = source_cells()
    if family != "v2":
        # Cache reusable embedded font bytes; glyph outlines and chart text do not change.
        from matplotlib.backends import _backend_pdf_ps
        from fontTools.ttLib import TTFont
        original_subset = _backend_pdf_ps.get_glyphs_subset
        font_cache = {}
        common_characters = set("".join(cells)) | set(map(chr, range(32, 127)))
        def cached_subset(fontfile, characters):
            key = (fontfile, "".join(sorted(common_characters | set(characters))))
            if key not in font_cache:
                with original_subset(*key) as font:
                    font_cache[key] = _backend_pdf_ps.font_as_file(font).getvalue()
            return TTFont(io.BytesIO(font_cache[key]))
        _backend_pdf_ps.get_glyphs_subset = cached_subset
    data = pd.read_csv(SUPPORT / "comparisons_support_v2_candidate.csv", float_precision="round_trip")
    before = pd.util.hash_pandas_object(data, index=True).to_numpy().copy()
    registry = json.loads((SUPPORT / "registry.json").read_text(encoding="utf-8"))
    env = dict(globals(), RV_BENCHMARK="ETHICS", RV_NEW=data, RV_REGISTRY=registry,
               RV_MARKERS=[r["marker_id"] for r in registry],
               RV_MEASURES={r["marker_id"]: r["measurement"] for r in registry},
               RV_PREFIX="v2_dukungan_20261004_232242_586663_",
               CP_BENCHMARK="ETHICS", CP_FAMILY_SIZE=3960, CP_CM=1/2.54,
               CP_REG={r["marker_id"]: r for r in registry},
               CP_DATA=data.assign(source_row_id=np.arange(len(data))),
               CP_LAYOUT=[], CP_HEAT_ROWS=[], CP_CHAR_ROWS=[], Z_ZOOM_AUDIT=[])
    rows = []
    def reused(stem):
        if not reuse_string:
            return False
        rel = Path(FOLDERS[family]) / (stem + ".png")
        prior = Path(reuse_string) / "figures" / rel
        prior_pdf = prior.with_suffix(".pdf")
        if not prior.is_file() or not prior_pdf.is_file():
            return False
        # These are completed exports from this command's earlier run, never external images.
        if not prior_pdf.read_bytes().rstrip().endswith(b"%%EOF"):
            return False
        old = ACTIVE / "figures" / rel
        with Image.open(old) as a, Image.open(prior) as b:
            assert a.size == b.size
            cutoff = int(a.height * .80)
            assert np.array_equal(np.asarray(a.convert("RGB"))[:cutoff], np.asarray(b.convert("RGB"))[:cutoff])
        out = stage / prior.name
        assert not out.exists() and not out.with_suffix(".pdf").exists()
        shutil.copy2(prior, out)
        shutil.copy2(prior_pdf, out.with_suffix(".pdf"))
        rows.append(dict(family=family, figure=str(rel), old_sha256=digest(old),
            new_sha256=digest(out), plot_area_changed_pixels=0, visible_status_labels=0,
            pdf=str(rel.with_suffix(".pdf")), old_pdf_sha256=digest(old.with_suffix(".pdf")),
            new_pdf_sha256=digest(out.with_suffix(".pdf")), reused_from=str(prior)))
        return True
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.dpi": 110, "savefig.dpi": 180})

    def finish(fig, stem, *args, **kwargs):
        filename = env["RV_PREFIX"] + stem if family == "v2" else stem
        old = ACTIVE / "figures" / FOLDERS[family] / (filename + ".png")
        assert old.is_file(), old
        texts = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
        assert not any(re.search(r"\b(candidate|kandidat)\b", t, re.I) for t in texts), filename
        png = stage / old.name
        if text_filter is not None and reuse_string is not None and family == "v2":
            prior = png if png.exists() else Path(reuse_string) / "figures" / old.name
            valid = False
            if prior.exists():
                try:
                    with Image.open(prior) as check:
                        check.verify()
                    valid = True
                except (OSError, SyntaxError):
                    pass
            if valid:
                if prior != png:
                    shutil.copy2(prior, png)
                rows.append(dict(family=family, figure=str(png.relative_to(run / "figures")),
                    old_sha256=digest(old), new_sha256=digest(png),
                    text_cleanup=dict(remaining_count_labels=0, reused_from=str(prior))))
                plt.close(fig)
                return
        # A resumed run may contain its own truncated last PNG after interruption.
        assert not png.exists() or (text_filter is not None and reuse_string is not None and family == "v2")
        options = {"bbox_inches": "tight"} if family == "v2" else {"dpi": 240}
        text_audit = text_filter(fig) if text_filter is not None else None
        fig.savefig(png, facecolor="white", **options)
        # Compare the entire plot area, excluding only the footer strip.
        with Image.open(old) as a, Image.open(png) as b:
            if text_filter is not None:
                changed = None
            else:
                assert a.size == b.size, (filename, a.size, b.size)
                cutoff = int(a.height * .80)
                aa = np.asarray(a.convert("RGB"))[:cutoff]
                bb = np.asarray(b.convert("RGB"))[:cutoff]
                changed = int(np.any(aa != bb, axis=2).sum())
        row = dict(family=family, figure=str(png.relative_to(run / "figures")),
                   old_sha256=digest(old), new_sha256=digest(png),
                   plot_area_changed_pixels=changed, visible_status_labels=0)
        if text_audit is not None:
            row["text_cleanup"] = text_audit
        # Strict pixel equality is required before any active installation.
        if changed:
            raise AssertionError((filename, "plot area changed", changed))
        if family != "v2":
            pdf = png.with_suffix(".pdf")
            old_pdf = old.with_suffix(".pdf")
            assert not pdf.exists()
            fig.savefig(pdf, facecolor="white", metadata={"Title": args[4],
                        "Subject": "Gender linguistic markers; frozen support v2"})
            row.update(pdf=str(pdf.relative_to(run / "figures")),
                       old_pdf_sha256=digest(old_pdf), new_pdf_sha256=digest(pdf))
        plt.close(fig)
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"{family}: {len(rows)}/{EXPECTED[family]} passed", flush=True)

    # Only label/model constants from the legacy setup; never execute inference.
    for node in ast.parse(cells[18]).body:
        if assigned_names(node) & {"RV_MODELS", "RV_MODEL_LABELS", "RV_COLORS", "RV_CONDITIONS", "RV_SHORT"}:
            execute(node, env)
    env["rv_finish"] = finish
    v2_nodes = ast.parse(cells[20]).body
    if family == "v2":
        # v2 was originally executed standalone, with default axis label sizing.
        plt.rcdefaults()
        for node in v2_nodes:
            if isinstance(node, ast.FunctionDef) and node.name == "rv_finish":
                continue
            if assigned_names(node) & {"RV_INDEX", "RV_EXPECTED"}:
                break
            execute(node, env)
            if scope is not None and assigned_names(node) == {"RV_SCOPES"}:
                assert scope in env["RV_SCOPES"]
                env["RV_SCOPES"] = [scope]
    else:
        for node in v2_nodes:
            if assigned_names(node) == {"RV_LABELS"}:
                execute(node, env)
        env["CP_LABELS"] = env["RV_LABELS"].copy()
        allowed = {"CP_MODELS", "CP_NAMES", "CP_COLORS", "CP_CONDITIONS", "CP_SHORT",
                   "CP_ROLES", "CP_CHARACTER", "CP_GROUPS", "CP_SCOPES"}
        for node in ast.parse(cells[23]).body:
            segment = ast.get_source_segment(cells[23], node)
            if (assigned_names(node) & allowed or segment.startswith("CP_LABELS.update(")
                    or segment.startswith("CP_GROUPS +=")
                    or (isinstance(node, ast.If) and "CP_GROUPS" in segment)):
                execute(node, env)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
            "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 240, "hatch.linewidth": .35})
        for node in ast.parse(cells[24]).body:
            if ((isinstance(node, ast.FunctionDef) and node.name != "cp_finish")
                    or assigned_names(node) & {"CP_CHAR_LABELS", "CP_CHAR_SHORT"}):
                execute(node, env)
        if family == "v31":
            # Reuse the notebook's exact v3.1 renderer substitutions, no setup writes.
            env.update(z_render_source=cells[24], z_tree=ast.parse(cells[24]), Z_PATH=NOTEBOOK)
            for node in ast.parse(cells[27]).body:
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "z_node":
                    execute(node, env)
            for node in ast.parse(cells[28]).body:
                if isinstance(node, ast.FunctionDef) and node.name == "z_word_count_zoom":
                    execute(node, env)
        env["cp_finish"] = finish
        for scope in env["CP_SCOPES"]:
            for role, title, conditions in env["CP_ROLES"]:
                prefix = f"{scope}__{env['CP_ROLES'].index((role,title,conditions))+1:02d}_{role}__"
                if family == "v3":
                    if not reused(prefix + "00_cakupan"):
                        env["cp_coverage"](scope, role, title, conditions)
                for marker in env["CP_CHARACTER"]:
                    if not reused(prefix + "00_" + marker):
                        env["cp_characteristic"](scope, role, title, conditions, marker)
                if family == "v31":
                    if not reused(prefix + "00_word_count_zoom"):
                        env["z_word_count_zoom"](scope, role, title, conditions)
                else:
                    for group, group_title, markers, status in env["CP_GROUPS"]:
                        if not reused(prefix + group):
                            env["cp_heatmap"](scope, role, title, conditions, group, group_title, markers, status)
    np.testing.assert_array_equal(before, pd.util.hash_pandas_object(data, index=True).to_numpy())
    assert len(rows) == (EXPECTED[family] // 5 if scope is not None else EXPECTED[family])
    suffix = f"_{scope}" if scope is not None else ""
    write_json(run / f"audit_{family}{suffix}.json", rows)
    return rows


def install(run):
    assert run.is_relative_to(RUNS.resolve()), run
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["png_count"] == 485 and manifest["pdf_count"] == 375
    for path, expected in manifest["input_hashes"].items():
        assert digest(path) == expected, path
    for path, expected in manifest["unchanged_png_hashes"].items():
        assert digest(path) == expected, path
    rows = manifest["figures"]
    replacements = []
    for row in rows:
        for key, old_key, new_key in [("figure", "old_sha256", "new_sha256"),
                                       ("pdf", "old_pdf_sha256", "new_pdf_sha256")]:
            if key not in row:
                continue
            rel = Path(row[key])
            target, source = ACTIVE / "figures" / rel, run / "figures" / rel
            assert target.resolve().is_relative_to((ACTIVE / "figures").resolve())
            assert source.resolve().is_relative_to((run / "figures").resolve())
            assert digest(target) == row[old_key] and digest(source) == row[new_key]
            replacements.append((target, source))
    backup = run / "backup_active_before_install"
    backup.mkdir(parents=True, exist_ok=True)
    indexes = list(ACTIVE.glob("figure_index*.csv")) + list((ACTIVE / "figures").glob("*/*.csv"))
    for target, _ in replacements:
        dest = backup / target.relative_to(ACTIVE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert digest(dest) == digest(target), ("Existing backup mismatch", dest)
        else:
            shutil.copy2(target, dest)
    # Back up active indexes before refreshing only their image hash columns.
    for target in indexes:
        dest = backup / target.relative_to(ACTIVE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert digest(dest) == digest(target), ("Existing index backup mismatch", dest)
        else:
            shutil.copy2(target, dest)
    original_notebook = ROOT / ".codex_tmp/ethics_gender_candidate_label_20261010/original.ipynb"
    if original_notebook.exists():
        notebook_backup = backup / NOTEBOOK.name
        if notebook_backup.exists():
            assert digest(notebook_backup) == digest(original_notebook)
        else:
            shutil.copy2(original_notebook, notebook_backup)
    changes = {digest(target): digest(source) for target, source in replacements}
    for target, source in replacements:
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
    for path in indexes:
        frame = pd.read_csv(path, keep_default_na=False)
        updated = frame.copy()
        for column in frame.columns:
            if "sha256" in column:
                updated[column] = frame[column].map(lambda v: changes.get(v, v))
        if not frame.equals(updated):
            updated.to_csv(path, index=False, encoding="utf-8-sig")
    report = dict(run=str(run), backup=str(backup), png_count=485, pdf_count=375,
                  source_tables_unchanged=True, scientific_status_unchanged=True,
                  thesis_files_modified=False, scope="visual status label only")
    assert all(digest(p) == h for p, h in manifest["unchanged_png_hashes"].items())
    write_json(ACTIVE / ("PRESENTASI_TANPA_LABEL_" + manifest["run_id"] + ".json"), report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--family", choices=list(EXPECTED))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--assemble", type=Path, nargs=3, metavar=("V2", "V3", "V31"))
    args = parser.parse_args()
    if args.family:
        if not args.run:
            parser.error("--family requires --run")
        render_family(args.family, str(args.run), str(args.reuse) if args.reuse else None)
        return
    if args.install:
        install(args.install.resolve())
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = RUNS / ("gender_ethics_plot_status_label_" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    inputs = {str(p): digest(p) for p in [NOTEBOOK, SUPPORT / "comparisons_support_v2_candidate.csv",
                                         SUPPORT / "registry.json"]}
    print(f"STAGING {run}", flush=True)
    # Independent interpreters avoid Windows multiprocessing named-pipe access.
    if args.assemble:
        rows = []
        for family, part in zip(EXPECTED, args.assemble):
            audit = json.loads((part / f"audit_{family}.json").read_text(encoding="utf-8"))
            assert len(audit) == EXPECTED[family]
            for row in audit:
                assert row["plot_area_changed_pixels"] == row["visible_status_labels"] == 0
                for field, hash_field in [("figure", "new_sha256"), ("pdf", "new_pdf_sha256")]:
                    if field not in row:
                        continue
                    src, dst = part / "figures" / row[field], run / "figures" / row[field]
                    assert digest(src) == row[hash_field] and not dst.exists()
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            rows.extend(audit)
    else:
        workers, logs = [], []
        for family in EXPECTED:
            log = (run / f"render_{family}.log").open("x", encoding="utf-8")
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--family", family, "--run", str(run)], stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0))
        codes = [worker.wait() for worker in workers]
        for log in logs:
            log.close()
        assert codes == [0, 0, 0], codes
        rows = [r for family in EXPECTED for r in json.loads(
            (run / f"audit_{family}.json").read_text(encoding="utf-8"))]
    assert all(digest(p) == h for p, h in inputs.items())
    manifest = dict(run_id=stamp, input_hashes=inputs, png_count=len(rows),
                    pdf_count=sum("pdf" in r for r in rows), figures=rows,
                    statistical_tests_rerun=False, extraction_rerun=False,
                    scientific_status="candidate_not_validated_for_thesis")
    # Ensure the staged inventory exactly covers every affected active family.
    expected_paths = {str(p.relative_to(ACTIVE / "figures")) for p in
                      (ACTIVE / "figures").glob("v2_dukungan_20261004_232242_586663_*.png")}
    for folder in (FOLDERS["v3"], FOLDERS["v31"]):
        expected_paths.update(str(p.relative_to(ACTIVE / "figures")) for p in
                              (ACTIVE / "figures" / folder).glob("*.png"))
    assert {r["figure"] for r in rows} == expected_paths
    for row in rows:
        old = ACTIVE / "figures" / row["figure"]
        new = run / "figures" / row["figure"]
        with Image.open(old) as a, Image.open(new) as b:
            assert a.size == b.size
            cutoff = int(a.height * .85)
            assert np.array_equal(np.asarray(a.convert("RGB"))[:cutoff],
                                  np.asarray(b.convert("RGB"))[:cutoff]), row["figure"]
    manifest["full_plot_area_pixel_equality"] = True
    manifest["unchanged_png_hashes"] = {str(p): digest(p) for p in
        (ACTIVE / "figures").rglob("*.png") if str(p.relative_to(ACTIVE / "figures")) not in expected_paths}
    write_json(run / "manifest.json", manifest)
    print(f"CHECKLIST / TESTING passed. Review then install: {run}", flush=True)


if __name__ == "__main__":
    main()

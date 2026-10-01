"""Add response-weighted ETHICS marker summaries without changing subset results."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarks.ethics.run_gender_ethics_lm_workspace import (
    ROOT, REPORTS, WORKSPACE, cached_pair_reader, enable_marker_imports,
    sha256, validate_measurements, write_json,
)

enable_marker_imports()
import numpy as np
import pandas as pd
from src.benchmarks.ethics import build_gender_ethics_linguistic_markers_v1_4 as core
from src.benchmarks.ethics import build_gender_ethics_linguistic_markers_v1_5 as integrated
from src.core.linguistic_markers import statistics as stats

BASE = REPORTS / "linguistic_markers/analysis/gender_ethics_lm_workspace_v1_5_20260909_r2"
SOURCE = REPORTS / "linguistic_markers_gender_candidate_20260821_v1_6_0"
OUTPUT = REPORTS / "linguistic_markers/analysis/gender_ethics_lm_gabungan_20260909"
NOTEBOOK = WORKSPACE / "ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"
SECTION_ID = "ethics-gabungan"


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Artifact changed: {directory / name}")


def pool_responses(frame):
    key = ["model_id", "persona", "subset", "question_id"]
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError("Missing or duplicate response key")
    result = frame.copy()
    result["source_subset"] = result["subset"]
    result["question_id"] = [json.dumps([str(s), str(q)], ensure_ascii=False)
                             for s, q in zip(frame.subset, frame.question_id)]
    result["subset"] = "ethics_all"
    result["benchmark"] = "ethics"
    if result.duplicated(["model_id", "persona", "question_id"]).any():
        raise ValueError("Pooling caused a key collision")
    return result


def describe_markers(frame, marker_ids):
    rows = []
    selected = frame.loc[frame.text_available]
    for marker in marker_ids:
        count, density, present = [f"{marker}__{suffix}" for suffix in ("count", "density", "present")]
        grouped = selected.groupby(["model_id", "persona"]).agg(
            n_text=(count, "size"), raw_count=(count, "sum"),
            mean_density=(density, "mean"), median_density=(density, "median"),
            responses_with_marker=(present, "sum")).reset_index()
        grouped["prevalence"] = grouped.responses_with_marker / grouped.n_text
        grouped["marker_id"] = marker
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def pair_composition(frame, comparisons, committed=False, outcome_id="available"):
    columns = ["model_id", "persona", "question_id", "source_subset", "text_available", "answer_status"]
    if outcome_id != "available":
        columns.append(outcome_id)
    frame = frame[columns]
    keys = ["model_id", "left_persona", "right_persona"]
    rows = []
    for item in comparisons[keys].drop_duplicates().itertuples(index=False):
        group = frame.loc[frame.model_id.eq(item.model_id)]
        keep = group.text_available & (group.answer_status.eq("committed") if committed else True)
        if outcome_id != "available":
            keep &= pd.to_numeric(group[outcome_id], errors="coerce").notna()
        cols = ["question_id", "source_subset"]
        left = group.loc[keep & group.persona.eq(item.left_persona), cols]
        right = group.loc[keep & group.persona.eq(item.right_persona), ["question_id"]]
        matched = left.merge(right, on="question_id", validate="one_to_one")
        counts = matched.source_subset.value_counts()
        for subset in core.SUBSETS:
            n = int(counts.get(subset, 0))
            rows.append({**item._asdict(), "source_subset": subset, "n_pairs": n,
                         "outcome_id": outcome_id,
                         "pair_share": n / len(matched) if len(matched) else np.nan,
                         "pair_filter": "committed" if committed else "all_text"})
    return pd.DataFrame(rows)


def validate_tables(tables, composition):
    tested = 0
    for name, table in tables.items():
        if "p_raw" not in table:
            continue
        ids = [c for c in ("model_id", "subset", "identity", "outcome_id", "contrast_id") if c in table]
        if table.duplicated(ids).any() or set(table.subset) != {"ethics_all"}:
            raise ValueError(f"Invalid keys: {name}")
        for column in ("p_raw", "p_holm"):
            if not table[column].notna().all() or not table[column].between(0, 1).all():
                raise ValueError(f"Invalid {column}: {name}")
        if not table.p_holm.ge(table.p_raw - 1e-14).all():
            raise ValueError(f"Invalid Holm adjustment: {name}")
        mode = "committed" if name.endswith("_committed") else "all_text"
        mix = composition.loc[composition.pair_filter.eq(mode)]
        support = mix.groupby(["model_id", "left_persona", "right_persona", "outcome_id"]).n_pairs.sum()
        for item in table.itertuples(index=False):
            outcome = getattr(item, "outcome_id", "available")
            support_outcome = outcome if outcome in core.SURFACE_OUTCOMES else "available"
            expected = support.loc[(item.model_id, item.left_persona, item.right_persona, support_outcome)]
            if item.n_pairs != expected:
                raise ValueError(f"Pair support mismatch: {name}/{outcome}: {item.n_pairs} != {expected}")
        if "rank_biserial" in table and not table.rank_biserial.between(-1, 1).all():
            raise ValueError(f"Invalid effect size: {name}")
        if "ci_low" in table:
            point = "median_paired_delta" if "median_paired_delta" in table else "prevalence_delta"
            supported = table.loc[table.n_pairs.gt(0)]
            if not (supported.ci_low.le(supported[point]) & supported.ci_high.ge(supported[point])).all():
                raise ValueError(f"Invalid confidence interval: {name}")
        tested += len(table)
    return {"valid": True, "tested_rows": tested, "tables": len(tables)}


def gender_plot(table, contrasts, labels, title, destination):
    import matplotlib.pyplot as plt
    from src.benchmarks.ethics.build_gender_ethics_direct_thesis_visuals_v1_2 import GENDER_MARKER_LABELS

    ids = integrated.GENDER_MARKER_IDS
    fig, axes = plt.subplots(1, len(contrasts), figsize=(6 * len(contrasts), 11), sharex=True, sharey=True)
    limit = max(1.0, 100 * table.prevalence_delta.abs().max()) * 1.12
    for ax, contrast, label in zip(np.atleast_1d(axes), contrasts, labels):
        panel = table.loc[table.contrast_id.eq(contrast)]
        for point in panel.itertuples(index=False):
            marker = "^" if getattr(point, "identity", "man") == "woman" else "o"
            ax.scatter(100 * point.prevalence_delta, len(ids) - 1 - ids.index(point.outcome_id),
                       color=core.MODEL_COLORS[point.model_id], marker=marker, s=48,
                       alpha=0.95 if point.p_holm < 0.05 else 0.4)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(ids)), [GENDER_MARKER_LABELS.get(i, i) for i in reversed(ids)])
    handles = core._model_legend()
    if "identity" in table:
        from matplotlib.lines import Line2D
        handles += [Line2D([], [], marker="o", linestyle="", color="gray", label="Man"),
                    Line2D([], [], marker="^", linestyle="", color="gray", label="Woman")]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(title)
    fig.supxlabel("Selisih prevalensi (poin persentase)", y=0.08)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    core._save(fig, destination.parent, destination.name)


def build(output, notebook, checkpoint_from=None):
    if not output.is_relative_to(ROOT / "result") or output.exists():
        raise ValueError("Choose a new output directory inside result/")
    old_hashes = json.loads((BASE / "workspace_artifact_hashes.json").read_text(encoding="utf-8"))
    verify_files(BASE, old_hashes)
    source_hashes = json.loads((BASE / "workspace_provenance.json").read_text(encoding="utf-8"))["source_sha256"]
    for name in ("response_level_markers.csv.gz", "marker_registry.csv"):
        relative = str((SOURCE / name).relative_to(ROOT))
        if sha256(SOURCE / name) != source_hashes[relative]:
            raise ValueError(f"Frozen source changed: {name}")
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    shutil.copy2(notebook, output / "notebook_before.ipynb")
    registry = pd.read_csv(SOURCE / "marker_registry.csv")
    cols = ["model_id", "persona", "subset", "question_id", "answer_status", "text_status",
            "text_available", "opening_category", *core.SURFACE_OUTCOMES]
    cols += [f"{marker}__{suffix}" for marker in registry.marker_id for suffix in ("count", "density", "present")]
    print("[1/5] Reading frozen response measurements", flush=True)
    source = pd.read_csv(SOURCE / "response_level_markers.csv.gz", usecols=cols, low_memory=False)
    source.text_available = core._bool_series(source.text_available)
    checks = validate_measurements(source, registry.marker_id)
    if len(source) != 81756 or set(source.subset) != set(core.SUBSETS):
        raise ValueError("Unexpected source coverage")
    frame = pool_responses(source)
    del source
    general, gender = core._selected_markers(), integrated._selected_gender_markers()
    density_gender = tuple(m for m in gender if m.measurement == "density")
    numeric_markers = (*general, *density_gender)
    tables = {"marker_descriptives": describe_markers(frame, registry.marker_id)}
    status = frame[["model_id", "persona", "source_subset", "answer_status", "text_available"]].copy()
    status["total"] = 1
    for state in ("committed", "abstain", "missing"):
        status[state] = status.answer_status.eq(state).astype(int)
    status["available"] = status.text_available.astype(int)
    counts = ["total", "committed", "abstain", "missing", "available"]
    coverage = status.groupby(["model_id", "persona"])[counts].sum().reset_index()
    for state, rate in (("committed", "commit_rate"), ("abstain", "abstain_rate"),
                        ("missing", "missing_rate"), ("available", "visible_text_rate")):
        coverage[rate] = coverage[state] / coverage.total
    tables["coverage"] = coverage
    mix = status.groupby(["model_id", "persona", "source_subset"])[counts].sum().reset_index()
    mix["response_share"] = mix.total / mix.groupby(["model_id", "persona"]).total.transform("sum")
    mix["visible_share"] = mix.available / mix.groupby(["model_id", "persona"]).available.transform("sum")
    tables["subset_weights"] = mix
    stats._paired_values = cached_pair_reader(stats._paired_values)
    compositions = []
    for mode in ("all_text", "committed"):
        print(f"[2/5] Paired statistics: {mode}", flush=True)
        cached = checkpoint_from / ("checkpoint_" + mode) if checkpoint_from else None
        if cached is not None and (cached / "ready.json").is_file():
            checkpoint = json.loads((cached / "ready.json").read_text(encoding="utf-8"))
            if checkpoint["code_sha256"] != sha256(Path(__file__)) or checkpoint["source_sha256"] != sha256(SOURCE / "response_level_markers.csv.gz"):
                raise ValueError("Checkpoint code/source mismatch")
            verify_files(cached, checkpoint["hashes"])
            for name in checkpoint["tables"]:
                tables[name] = pd.read_csv(cached / (name + ".csv"))
            compositions.append(pd.read_csv(cached / "pair_composition.csv"))
            print(f"[CHECKPOINT] Reused {mode}", flush=True)
            continue
        data = frame if mode == "all_text" else core._committed_frame(frame, (*general, *gender))
        within = core._attach_raw_count_summaries(stats.build_paired_numeric(data, 2000, markers=numeric_markers), data, numeric_markers)
        direct = core._attach_raw_count_summaries(stats.build_paired_gender_numeric(data, 2000, markers=numeric_markers), data, numeric_markers)
        binary = stats.build_paired_binary(data, 2000, markers=gender)
        binary_direct = stats.build_paired_gender_binary(data, 2000, markers=gender)
        for stem, ids in (("numeric", core.OUTCOME_IDS), ("gender_density", tuple(m.marker_id for m in density_gender))):
            for contrast, selected in (("vs_neutral", core.NEUTRAL_CONTRASTS), ("vs_explicit", core.EXPLICIT_CONTRASTS)):
                tables[f"{stem}_{contrast}_{mode}"] = within.loc[within.outcome_id.isin(ids) & within.contrast_id.isin(selected)].copy()
            tables[f"{stem}_direct_{mode}"] = direct.loc[direct.outcome_id.isin(ids)].copy()
        for contrast, selected in (("vs_neutral", core.NEUTRAL_CONTRASTS), ("vs_explicit", core.EXPLICIT_CONTRASTS)):
            tables[f"gender_presence_{contrast}_{mode}"] = binary.loc[binary.contrast_id.isin(selected)].copy()
        tables[f"gender_presence_direct_{mode}"] = binary_direct
        tables[f"opening_within_{mode}"] = stats.build_paired_opening_multinomial(data)
        tables[f"opening_direct_{mode}"] = stats.build_paired_gender_opening_multinomial(data)
        comparisons = pd.concat([within, direct], ignore_index=True)
        mode_composition = pd.concat([
            pair_composition(frame, comparisons, mode == "committed", outcome)
            for outcome in ("available", *core.SURFACE_OUTCOMES)], ignore_index=True)
        compositions.append(mode_composition)
        checkpoint = output / ("checkpoint_" + mode)
        checkpoint.mkdir()
        checkpoint_tables = [name for name in tables if name.endswith("_" + mode)]
        for name in checkpoint_tables:
            tables[name].to_csv(checkpoint / (name + ".csv"), index=False)
        mode_composition.to_csv(checkpoint / "pair_composition.csv", index=False)
        write_json(checkpoint / "ready.json", {
            "tables": checkpoint_tables, "code_sha256": sha256(Path(__file__)),
            "source_sha256": sha256(SOURCE / "response_level_markers.csv.gz"),
            "hashes": {p.name: sha256(p) for p in checkpoint.glob("*.csv")}})
        print(f"[CHECKPOINT] Saved {mode}", flush=True)
    tables["pair_composition"] = pd.concat(compositions, ignore_index=True)
    sensitivity = []
    for stem in ("numeric", "gender_density", "gender_presence"):
        for contrast in ("vs_neutral", "vs_explicit", "direct"):
            left = tables[f"{stem}_{contrast}_all_text"]
            right = tables[f"{stem}_{contrast}_committed"]
            ids = [c for c in ("model_id", "identity", "outcome_id", "contrast_id") if c in left]
            value = "prevalence_delta" if stem == "gender_presence" else "rank_biserial"
            joined = left[ids + ["n_pairs", value, "p_holm"]].merge(right[ids + ["n_pairs", value, "p_holm"]], on=ids, suffixes=("_all", "_committed"), validate="one_to_one")
            joined["family"] = stem
            joined["comparison"] = contrast
            joined["pair_retention"] = joined.n_pairs_committed / joined.n_pairs_all.replace(0, np.nan)
            joined["direction_reversed"] = joined[value + "_all"] * joined[value + "_committed"] < 0
            sensitivity.append(joined)
    tables["sensitivity"] = pd.concat(sensitivity, ignore_index=True)
    print("[3/5] Validating pooled tables", flush=True)
    validation = validate_tables(tables, tables["pair_composition"])
    expected = {"numeric": (1008, 756, 504), "gender_density": (192, 144, 96), "gender_presence": (528, 396, 264)}
    for stem, sizes in expected.items():
        for contrast, size in zip(("vs_neutral", "vs_explicit", "direct"), sizes):
            for mode in ("all_text", "committed"):
                if len(tables[f"{stem}_{contrast}_{mode}"]) != size:
                    raise ValueError(f"Unexpected row count: {stem}/{contrast}/{mode}")
    for name, table in tables.items():
        if any("accuracy" in c or "is_correct" in c for c in table):
            raise ValueError("Pooled accuracy is forbidden")
        table = table.assign(result_status="candidate", aggregation="response_weighted")
        table.to_csv(output / f"{name}.csv", index=False)
    registry.to_csv(output / "marker_registry.csv", index=False)
    print("[4/5] Rendering six pooled figures", flush=True)
    core._style()
    core.SUBSET_LABELS["ethics_all"] = "ETHICS Gabungan"
    core._plot_within_effects(tables["numeric_vs_neutral_all_text"], "ethics_all", core.NEUTRAL_CONTRASTS, images, "01_terhadap_netral.png", "Terhadap Netral")
    core._plot_within_effects(tables["numeric_vs_explicit_all_text"], "ethics_all", core.EXPLICIT_CONTRASTS, images, "02_implisit_eksplisit.png", "Implisit dan Eksplisit")
    direct_plot = tables["numeric_direct_all_text"].copy()
    core._plot_direct_effects(direct_plot, "ethics_all", images, "03_laki_laki_perempuan.png")
    for index, contrast in enumerate(("vs_neutral", "vs_explicit", "direct"), start=4):
        data = tables[f"gender_presence_{contrast}_all_text"]
        contrasts = list(dict.fromkeys(data.contrast_id))
        labels = [core.CONTRAST_LABELS.get(c, c.replace("man_vs_woman_", "").replace("implicit_", "I").replace("explicit", "E")) for c in contrasts]
        title = {"vs_neutral": "Terhadap Netral", "vs_explicit": "Implisit dan Eksplisit", "direct": "Laki-laki dan Perempuan"}[contrast]
        gender_plot(data, contrasts, labels, f"ETHICS Gabungan — Marker Gender: {title}", images / f"{index:02d}_gender_{contrast}.png")
    verify_files(BASE, old_hashes)
    write_json(output / "validation.json", {**validation, "measurements": checks, "old_artifacts_unchanged": True})
    write_json(output / "manifest.json", {
        "result_status": "candidate", "aggregation": "response_weighted", "rows": len(frame),
        "questions": frame.question_id.nunique(), "models": frame.model_id.nunique(),
        "personas": frame.persona.nunique(), "available": int(frame.text_available.sum()),
        "source_sha256": sha256(SOURCE / "response_level_markers.csv.gz"),
        "script_sha256": sha256(Path(__file__)), "baseline": str(BASE.relative_to(ROOT)),
        "code_sha256": {str(Path(module.__file__).relative_to(ROOT)): sha256(Path(module.__file__))
                        for module in (core, integrated, stats)},
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "subset_weighting": "Observed response/pair shares, not equal subset weights",
        "bootstrap_resamples": 2000, "numeric_ci": "median paired delta, not rank-biserial",
        "holm": "Within: 7 contrasts per model/identity/outcome; direct: 4 per model/outcome; openings: 7 or 4",
        "figures": [p.relative_to(output).as_posix() for p in sorted(images.glob("*.png"))],
        "old_notebook_sha256": sha256(output / "notebook_before.ipynb"),
        "old_bundle_hashes": old_hashes,
    })
    write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(f"[5/5] Complete: {output}", flush=True)


def save_generated_notebook(notebook, path, expected_hash):
    """Validate a staged generated notebook before atomically replacing its target."""
    import nbformat

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".ipynb",
                                         prefix=".pooled-", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            nbformat.write(notebook, handle)
            handle.flush()
            os.fsync(handle.fileno())
        generated = nbformat.read(temporary, as_version=4)
        nbformat.validate(generated)
        if generated != notebook or sha256(path) != expected_hash:
            raise ValueError("Generated notebook differs or target changed")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_template(output, notebook_path):
    """Mechanically merge a patched source template into the generated notebook."""
    import nbformat

    original = nbformat.read(output / "notebook_before.ipynb", as_version=4)
    notebook = nbformat.read(notebook_path, as_version=4)
    if any(cell.id == SECTION_ID for cell in original.cells):
        raise ValueError("Use a notebook without the pooled section as the starting point")
    if notebook != original:
        raise ValueError("Notebook changed since backup; refusing to append")
    expected_hash = sha256(notebook_path)
    template = Path(__file__).parent / "templates/ethics_gender_pooled_cells.json"
    addition = json.loads(template.read_text(encoding="utf-8"))
    for cell in addition:
        cell["source"] = "".join(cell["source"]).replace(
            "__POOLED_RELATIVE__", output.relative_to(ROOT).as_posix())
        notebook.cells.append(nbformat.from_dict(cell))
    if notebook.cells[:len(original.cells)] != original.cells:
        raise ValueError("Existing cells changed")
    nbformat.validate(notebook)
    save_generated_notebook(notebook, notebook_path, expected_hash)


def execute_added_cells(output, notebook_path):
    """Execute only the appended section; retain all earlier cells verbatim."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec

    original = nbformat.read(output / "notebook_before.ipynb", as_version=4)
    notebook = nbformat.read(notebook_path, as_version=4)
    n = len(original.cells)
    if notebook.cells[:n] != original.cells or notebook.cells[n].id != SECTION_ID:
        raise ValueError("Original cells changed or appended section missing")
    before_hash = sha256(notebook_path)
    addition = nbformat.v4.new_notebook(cells=copy.deepcopy(notebook.cells[n:]))
    manager = KernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="Python 3", language="python")
    NotebookClient(addition, km=manager, timeout=600, resources={"metadata": {"path": str(notebook_path.parent)}}).execute()
    notebook.cells[n:] = addition.cells
    if notebook.cells[:n] != original.cells or sha256(notebook_path) != before_hash:
        raise ValueError("Notebook changed during execution")
    nbformat.validate(notebook)
    # Only generated execution outputs/metadata in the appended section change.
    save_generated_notebook(notebook, notebook_path, before_hash)
    saved = nbformat.read(notebook_path, as_version=4)
    if saved.cells[:n] != original.cells:
        raise ValueError("Notebook preservation check failed")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    verify_files(BASE, manifest["old_bundle_hashes"])
    summary = {"old_cells_preserved": n, "added_cells": len(addition.cells),
               "added_code_cells_executed": sum(c.cell_type == "code" and c.execution_count is not None for c in addition.cells),
               "errors": sum(o.output_type == "error" for c in addition.cells if c.cell_type == "code" for o in c.outputs),
               "notebook_sha256": sha256(notebook_path), "old_artifacts_unchanged": True}
    write_json(output / "notebook_execution.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--execute-added", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--checkpoint-from", type=Path)
    args = parser.parse_args()
    if args.append:
        append_template(args.output_dir.resolve(), args.notebook.resolve())
    if args.execute_added:
        execute_added_cells(args.output_dir.resolve(), args.notebook.resolve())
    elif not args.append:
        build(args.output_dir.resolve(), args.notebook.resolve(),
              args.checkpoint_from.resolve() if args.checkpoint_from else None)

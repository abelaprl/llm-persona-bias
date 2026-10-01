"""Audit frozen LM availability without scoring answers or modifying old data.

Read only response text and marker metadata; count bare binary responses and
check frozen NA/density invariants. Normalize only short responses by default.
Write evidence exclusively into a new, previously nonexistent directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    output.relative_to(root / "result")
    output.mkdir(parents=True, exist_ok=False)
    base = root / "result/ETHICS/reports/Gender_Persona"
    marker_dir = base / "linguistic_markers_gender_candidate_20260821_v1_6_0"
    print("[1/4] Reading frozen marker table", flush=True)
    markers = pd.read_csv(marker_dir / "response_level_markers.csv.gz")
    print(f"[2/4] Reading response text for {len(markers):,} marker rows", flush=True)
    source = pd.read_csv(
        base / "_combined/parser_refinement_20260820/all_models_results_full.csv",
        usecols=["model", "persona_folder", "question_id", "file", "response_content_full"],
        keep_default_na=False,
    ).rename(columns={"model": "model_id", "persona_folder": "persona"})
    frame = markers.merge(source, on=["model_id", "persona", "question_id"], validate="one_to_one")
    frame["bare_binary"] = frame.response_content_full.str.strip().isin(["0", "1"])
    frame["think_tag"] = frame.response_content_full.str.contains(r"</?think\b", case=False)
    frame["reason_channel_tag"] = frame.response_content_full.str.contains(
        r"<\|(?:channel|im_start|im_sep)\|>|</?reasoning>", case=False
    )
    summary = frame.groupby(["model_id", "persona"]).agg(
        total=("question_id", "size"), available=("text_available", "sum"),
        word_median=("word_count", "median"), bare_binary=("bare_binary", "sum"),
        think_tags=("think_tag", "sum"), channel_tags=("reason_channel_tag", "sum"),
    ).reset_index()
    summary.to_csv(output / "surface_by_model_persona.csv", index=False)
    count_cols = [c for c in frame if c.endswith("__count")]
    all_marker_cols = [c for c in frame if c.endswith(("__count", "__density", "__present"))]
    density_errors = {}
    for count in count_cols:
        density = count.removesuffix("__count") + "__density"
        expected = 100 * frame[count] / frame.word_count
        density_errors[count] = int((~np.isclose(expected, frame[density], equal_nan=True)).sum())
    stats = {
        "total": len(frame), "bare_binary": int(frame.bare_binary.sum()),
        "bare_binary_available": int(frame.loc[frame.bare_binary, "text_available"].sum()),
        "bare_binary_marker_hits": frame.loc[frame.bare_binary, count_cols].sum().to_dict(),
        "unavailable_non_null_marker_cells": int(frame.loc[~frame.text_available, all_marker_cols].notna().sum().sum()),
        "available_null_count_cells": int(frame.loc[frame.text_available, count_cols].isna().sum().sum()),
        "density_formula_mismatch_rows_by_marker": density_errors,
        "status_by_answer_status": pd.crosstab(frame.answer_status, frame.text_status).to_dict(),
    }
    (output / "surface_checks.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[3/4] Bare binary responses: {stats['bare_binary']:,}; normalizing short responses", flush=True)
    spec = importlib.util.spec_from_file_location("surface_audit", root / "src/core/linguistic_markers/surface.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    selected = frame.loc[frame.word_count.le(1), ["model_id", "persona", "question_id", "file", "response_content_full", "text_status", "word_count"]].copy()
    statuses = []
    for i, raw in enumerate(selected.response_content_full, 1):
        statuses.append(module.prepare_text_surface(raw, "ethics").text_status)
        if i % 2000 == 0:
            print(f"[normalize] {i:,}/{len(selected):,}", flush=True)
    selected["recomputed_status"] = statuses
    selected.to_csv(output / "short_response_diagnosis.csv.gz", index=False, compression="gzip")
    print(f"[4/4] Evidence saved: {output}; {len(selected):,} short responses normalized", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

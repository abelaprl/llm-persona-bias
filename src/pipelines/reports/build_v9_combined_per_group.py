"""Build per-(persona-group, model) detail CSVs for ETHICS v9_pilot.

Output layout: split per group, then per model (one results.csv each, 18 files):
    result/archive/reports_mixed_legacy/v9_pilot/<group>/<model>/results.csv

Columns (results.csv style, adapted to ETHICS binary task):
    model, persona_folder, file, question_id, dataset, ground_truth,
    response_content_full, extracted_answer, answer_status, parse_source, is_correct,
    finish_reason, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
    truncated, text_tail

Usage (from D:\\TugasAkhir):
    python src/pipelines/reports/build_v9_combined_per_group.py --group neutral     # validate (small)
    python src/pipelines/reports/build_v9_combined_per_group.py                      # all groups, all models
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path("src/pipelines/rebuild").resolve()))
from rebuild_ethics_csv_reparse import build_row  # noqa: E402

BASE = Path("result/benchmarks/ethics/runs/pilot/v9")
OUT_BASE = Path("result/archive/reports_mixed_legacy/v9_pilot")
GROUP_FOLDER = {"gender": "Gender Persona", "neutral": "Neutral Persona", "religion": "Religion Persona"}

FIELDNAMES = [
    "model", "persona_folder", "file", "question_id", "dataset", "ground_truth",
    "response_content_full", "extracted_answer", "answer_status", "parse_source", "is_correct",
    "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
    "truncated", "text_tail",
]


def extract_meta(d: dict):
    """Return (finish_reason, total_tokens) from the raw API response."""
    resp = d.get("response", {}) or {}
    rar = resp.get("raw_api_response", {}) or {}
    ch = rar.get("choices") or []
    finish = ch[0].get("finish_reason") if ch and isinstance(ch[0], dict) else None
    total = (rar.get("usage", {}) or {}).get("total_tokens")
    return finish, total


def answer_status(row: dict, d: dict) -> str:
    if row["parsed_answer"] in ("0", "1"):
        return "committed"
    raw = str((d.get("prediction") or {}).get("raw", "") or "")
    return "abstain" if raw.strip() else "missing"


def build_group_model(group_slug: str, model: str) -> int:
    """Write result/archive/reports_mixed_legacy/v9_pilot/<group>/<model>/results.csv for one (group, model)."""
    gdir_name = GROUP_FOLDER[group_slug]
    mdir = BASE / gdir_name / model
    out_dir = OUT_BASE / group_slug / model
    out_path = out_dir / f"results_{group_slug}_{model}.csv"
    if not mdir.exists():
        print(f"  [{group_slug}/{model}] (no folder)", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for jf in sorted(mdir.rglob("*.json")):
            if "warmup" in jf.parts:
                continue
            try:
                d = json.load(open(jf, encoding="utf-8"))
            except Exception:
                continue
            row, _src = build_row("v9_pilot", d, jf, mdir)
            meta = d.get("metadata", {}) or {}
            raw = str((d.get("prediction") or {}).get("raw", "") or "")
            finish, total = extract_meta(d)
            w.writerow({
                "model": model,
                "persona_folder": row["persona"],
                "file": jf.name,
                "question_id": d.get("id"),
                "dataset": d.get("subset"),
                "ground_truth": row["ground_truth_label"],
                "response_content_full": raw,
                "extracted_answer": row["parsed_answer"],
                "answer_status": answer_status(row, d),
                "parse_source": row["parse_source"],
                "is_correct": row["is_correct"],
                "finish_reason": finish,
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "reasoning_tokens": meta.get("reasoning_tokens"),
                "total_tokens": total,
                "truncated": finish == "length",
                "text_tail": raw[-200:],
            })
            cnt += 1
    print(f"  [{group_slug}/{model}] {cnt} rows -> {out_path}", flush=True)
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUP_FOLDER), default=None)
    ap.add_argument("--model", default=None, help="Single model slug (default: all). Use to run in parallel.")
    args = ap.parse_args()

    # discover models present
    found = set()
    for gf in GROUP_FOLDER.values():
        gd = BASE / gf
        if gd.exists():
            found.update(p.name for p in gd.iterdir() if p.is_dir())
    models = [args.model] if args.model else sorted(found)
    print(f"Models: {len(models)} -> {models}", flush=True)

    groups = [args.group] if args.group else list(GROUP_FOLDER)
    for g in groups:
        print(f"\n=== GROUP: {g} ===", flush=True)
        for model in models:
            build_group_model(g, model)


if __name__ == "__main__":
    main()

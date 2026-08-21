"""Assemble the per-persona JSONL cache into one CSV per model.

Usage (from D:\\TugasAkhir):

    python "src/MAIN_PARSING/ETHICS/build_csv.py" --group gender
    python "src/MAIN_PARSING/ETHICS/build_csv.py" --all

Output, mirroring the shape of result/reports/v9_pilot/:

    result/reports/v9_main/<slug>/<model>/results_<slug>_<model>.csv
        one row per question per persona - Gender = 8 x 1,514 = 12,112 rows
    result/reports/v9_main/<slug>/<model>/summary_<slug>_<model>.csv
        one row per (persona x subset)
    result/reports/v9_main/summary_all_models.csv

Accuracy is reported per subset only. There is no pooled ETHICS accuracy,
because `deontology` and `virtue` do not measure moral judgement (findings T2
and T3) and averaging them into one number hides that.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import (  # noqa: E402
    CSV_COLUMNS,
    GROUPS,
    PARSE_CACHE_ROOT,
    REPORT_ROOT,
    STANCE_ACCEPTABLE,
    STANCE_WRONG,
    SUMMARY_COLUMNS,
)


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.4f}" if denominator else ""


def _read_cache(model_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for jsonl in sorted(model_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, columns, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def summarise(rows: list[dict]) -> list[dict]:
    """One row per (persona x subset). No cross-subset aggregate."""
    buckets: dict[tuple, Counter] = defaultdict(Counter)
    meta: dict[tuple, dict] = {}

    for row in rows:
        key = (row["model"], row["persona_group"], row["persona_folder"], row["subset"])
        meta.setdefault(key, {"construct_valid": row.get("construct_valid")})
        bucket = buckets[key]
        bucket["total"] += 1
        bucket[row["answer_status"]] += 1
        if row["answer_status"] != "committed":
            continue
        if row.get("is_correct") is True:
            bucket["n_correct"] += 1
        if row.get("numeric_semantic_conflict"):
            bucket["n_conflict"] += 1
            if row.get("is_correct") is True:
                bucket["n_correct_conflict"] += 1
        if row.get("extracted_stance") == STANCE_WRONG:
            bucket["stance_wrong"] += 1
        elif row.get("extracted_stance") == STANCE_ACCEPTABLE:
            bucket["stance_acceptable"] += 1

    out: list[dict] = []
    for key in sorted(buckets):
        model, group, persona, subset = key
        b = buckets[key]
        total = b["total"]
        committed = b["committed"]
        clean = committed - b["n_conflict"]
        out.append({
            "model": model,
            "persona_group": group,
            "persona": persona,
            "subset": subset,
            "construct_valid": meta[key]["construct_valid"],
            "total": total,
            "committed": committed,
            "abstain": b["abstain"],
            "missing": b["missing"],
            "commit_rate": _rate(committed, total),
            "abstain_rate": _rate(b["abstain"], total),
            "missing_rate": _rate(b["missing"], total),
            "n_correct": b["n_correct"],
            "accuracy_committed": _rate(b["n_correct"], committed),
            "accuracy_committed_excl_conflict": _rate(
                b["n_correct"] - b["n_correct_conflict"], clean
            ),
            "n_conflict": b["n_conflict"],
            # Answer distribution stays meaningful even where accuracy does
            # not, so deontology and virtue remain comparable across personas.
            "pct_answer_wrong": _rate(b["stance_wrong"], committed),
            "pct_answer_acceptable": _rate(b["stance_acceptable"], committed),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ETHICS v9 CSV reports from the parse cache")
    ap.add_argument("--group", default="gender", help="gender | religion | regional | neutral")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--no-global-summary",
        action="store_true",
        help="write selected model reports without replacing summary_all_models.csv",
    )
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    if args.all:
        folders = list(GROUPS.keys())
    else:
        if args.group not in slug_to_folder:
            ap.error(f"--group must be one of {sorted(slug_to_folder)}")
        folders = [slug_to_folder[args.group]]

    combined: list[dict] = []

    for folder in folders:
        slug = GROUPS[folder]
        group_cache = PARSE_CACHE_ROOT / folder
        if not group_cache.is_dir():
            print(f"[SKIP] cache belum ada untuk {folder} - jalankan run_parse.py dulu")
            continue

        print(f"\n{'=' * 78}\n  {folder}\n{'=' * 78}")
        model_dirs = [d for d in sorted(group_cache.iterdir()) if d.is_dir()]
        if args.model:
            model_dirs = [d for d in model_dirs if d.name == args.model]

        for model_dir in model_dirs:
            rows = _read_cache(model_dir)
            if not rows:
                print(f"  [SKIP] {model_dir.name}: cache kosong")
                continue

            out_dir = REPORT_ROOT / slug / model_dir.name
            main_csv = out_dir / f"results_{slug}_{model_dir.name}.csv"
            _write_csv(main_csv, CSV_COLUMNS, rows)

            summary = summarise(rows)
            summary_csv = out_dir / f"summary_{slug}_{model_dir.name}.csv"
            _write_csv(summary_csv, SUMMARY_COLUMNS, summary)
            combined.extend(summary)

            statuses = Counter(r["answer_status"] for r in rows)
            personas = len({r["persona_folder"] for r in rows})
            print(f"\n  {model_dir.name}")
            print(f"    {len(rows)} baris dari {personas} persona -> {main_csv.name}")
            print(f"    committed {statuses['committed']} | abstain {statuses['abstain']} "
                  f"| missing {statuses['missing']}")
            for line in summary:
                if line["persona"] != summary[0]["persona"]:
                    continue
                flag = "" if line["construct_valid"] == "True" or line["construct_valid"] is True else "  [konstruk tidak valid]"
                print(f"      {line['persona']:<22} {line['subset']:<12} "
                      f"acc={line['accuracy_committed'] or 'n/a':<7} "
                      f"coverage={line['commit_rate'] or 'n/a'}{flag}")

    if combined and not args.no_global_summary:
        path = REPORT_ROOT / "summary_all_models.csv"
        _write_csv(path, SUMMARY_COLUMNS, combined)
        print(f"\n{'=' * 78}")
        print(f"  Ringkasan gabungan: {path} ({len(combined)} baris)")
        print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()

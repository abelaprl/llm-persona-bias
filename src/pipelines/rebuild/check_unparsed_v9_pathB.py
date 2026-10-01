"""Scan the v9_pilot result tree (Path B layout) and report how many results
still have no final 0/1 answer — both as-stored (prediction.parsed_answer) and
after applying the reparse engine from rebuild_ethics_csv_reparse.py.

Layout scanned:
    result/ETHICS/v9_pilot/<Group Persona>/<model_slug>/<persona>/*.json

Groups: "Gender Persona", "Neutral Persona", "Religion Persona".

Usage (from D:\\TugasAkhir):
    python src/pipelines/rebuild/check_unparsed_v9_pathB.py
    python src/pipelines/rebuild/check_unparsed_v9_pathB.py --model deepseek_deepseek-r1-distill-qwen-32b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

# Import the reparse engine (same one used to build the CSVs).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_ethics_csv_reparse import reparse_ethics_answer  # noqa: E402

BASE = Path("result/ETHICS/v9_pilot")
GROUPS = ["Gender Persona", "Neutral Persona", "Religion Persona"]


def scan_model(model: str):
    # per-group, per-persona stats
    group_stats = defaultdict(lambda: Counter())
    persona_unparsed = Counter()
    persona_total = Counter()
    still_none_examples = []

    for group in GROUPS:
        gdir = BASE / group / model
        if not gdir.exists():
            continue
        for jf in sorted(gdir.rglob("*.json")):
            if "warmup" in jf.parts:
                continue
            try:
                d = json.load(open(jf, encoding="utf-8"))
            except Exception:
                group_stats[group]["read_error"] += 1
                continue
            persona = d.get("persona", jf.parent.name)
            pred = d.get("prediction", {}) or {}
            orig = str(pred.get("parsed_answer", "") or "").strip()

            group_stats[group]["total"] += 1
            persona_total[persona] += 1

            if orig in ("0", "1"):
                group_stats[group]["orig_parsed"] += 1
                continue

            # original failed -> try reparse engine
            group_stats[group]["orig_failed"] += 1
            ans, src = reparse_ethics_answer(pred.get("raw", ""))
            if ans in ("0", "1"):
                group_stats[group]["reparse_recovered"] += 1
            else:
                group_stats[group]["still_none"] += 1
                persona_unparsed[persona] += 1
                if len(still_none_examples) < 8:
                    raw = str(pred.get("raw", "") or "")
                    still_none_examples.append((persona, d.get("id"), raw[-260:]))

    return group_stats, persona_total, persona_unparsed, still_none_examples


def print_report(model, group_stats, persona_total, persona_unparsed, examples):
    print(f"\n{'='*70}\nMODEL: {model}\n{'='*70}")
    grand = Counter()
    for group in GROUPS:
        c = group_stats.get(group)
        if not c:
            print(f"  [{group}]  (no data)")
            continue
        grand.update(c)
        tot = c["total"]
        none = c["still_none"]
        pct = f"{100*none/tot:.1f}%" if tot else "n/a"
        print(f"  [{group}]")
        print(f"      total={tot}  orig_parsed={c['orig_parsed']}  "
              f"reparse_recovered={c['reparse_recovered']}  still_none={none} ({pct})")
    tot = grand["total"]
    none = grand["still_none"]
    pct = f"{100*none/tot:.1f}%" if tot else "n/a"
    print(f"  ── TOTAL: {tot} | orig_parsed={grand['orig_parsed']} "
          f"reparse_recovered={grand['reparse_recovered']} STILL_NONE={none} ({pct})")

    if persona_unparsed:
        print(f"  ── still-none per persona (top 15):")
        for p, n in persona_unparsed.most_common(15):
            t = persona_total[p]
            print(f"      {n:>5} / {t:<5} ({100*n/t:4.1f}%)  {p}")

    if examples:
        print(f"  ── sample still-none (tail 260 chars):")
        for persona, qid, tail in examples[:4]:
            print(f"      [{persona} | {qid}] ...{tail!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Single model slug (default: all found)")
    args = ap.parse_args()

    if args.model:
        models = [args.model]
    else:
        found = set()
        for group in GROUPS:
            gdir = BASE / group
            if gdir.exists():
                found.update(p.name for p in gdir.iterdir() if p.is_dir())
        models = sorted(found)

    print(f"Scanning {len(models)} model(s): {', '.join(models)}", flush=True)
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] scanning {model} ...", flush=True)
        gs, pt, pu, ex = scan_model(model)
        print_report(model, gs, pt, pu, ex)
        sys.stdout.flush()


if __name__ == "__main__":
    main()

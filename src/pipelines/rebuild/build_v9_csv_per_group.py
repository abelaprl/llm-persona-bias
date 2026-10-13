"""Build one CSV per (model, persona-group) from the v9_pilot result tree (Path B).

Layout read:
    result/benchmarks/ethics/runs/pilot/v9/<Group Persona>/<model_slug>/<persona>/*.json

Output per model+group:
    result/benchmarks/ethics/runs/<model_slug>/result_ethics_v9_<group>_<model_slug>.csv

Each row also carries an ``answer_status`` column (Jalur B coding scheme):
    committed : a 0/1 verdict was extracted (original or reparsed) -> used for accuracy
    abstain   : model responded with text but gave NO binary verdict
                (hedged "it depends", advice, conditional) -> NOT scored
    missing   : response was empty/truncated/error (no usable text) -> NOT scored

Why this matters: the unparsed cases are NOT a parser deficiency — analysis of
12,369 textual none-cases showed any regex recovery scores ~50% (coin-flip), so
forcing a 0/1 would inject noise. Instead we report accuracy ONLY on committed
answers and surface the abstention rate as a finding (see
docs/ETHICS_v9_deepseek_parse_failure_diagnosis.md).

Summary outputs (per model):
    result/benchmarks/ethics/runs/<model_slug>/result_ethics_v9_<model_slug>_status_summary.csv
        -> per (group, persona): total / committed / abstain / missing,
           accuracy_committed, commit_rate, abstain_rate, missing_rate

Usage (from D:\\TugasAkhir):
    python src/pipelines/rebuild/build_v9_csv_per_group.py
    python src/pipelines/rebuild/build_v9_csv_per_group.py --model deepseek_deepseek-r1-distill-qwen-32b
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_ethics_csv_reparse import build_row, CSV_FIELDNAMES, RESULT_ROOT  # noqa: E402

BASE = Path("result/benchmarks/ethics/runs/pilot/v9")
GROUP_SLUG = {
    "Gender Persona": "gender",
    "Neutral Persona": "neutral",
    "Religion Persona": "religion",
}

EXTRA_FIELDNAMES = [
    "answer_status", "prompt_tokens", "completion_tokens", "reasoning_tokens", "raw_answer",
]
STATUS_FIELDNAMES = CSV_FIELDNAMES + EXTRA_FIELDNAMES
SUMMARY_FIELDNAMES = [
    "group", "persona", "total", "committed", "abstain", "missing", "correct",
    "accuracy_committed", "commit_rate", "abstain_rate", "missing_rate",
]
# Combined cross-model summary carries an extra leading "model" column.
COMBINED_FIELDNAMES = ["model"] + SUMMARY_FIELDNAMES
REPORTS_DIR = RESULT_ROOT / "reports"


def classify_status(row: dict, d: dict) -> str:
    """committed (has 0/1) | abstain (text but no verdict) | missing (empty/error)."""
    if row["parsed_answer"] in ("0", "1"):
        return "committed"
    raw = str((d.get("prediction") or {}).get("raw", "") or "")
    return "abstain" if raw.strip() else "missing"


def build_model_group(model: str, group: str):
    """Return (rows, per_persona_stats). per_persona_stats[persona] is a Counter
    with keys: total, committed, abstain, missing, correct."""
    gdir = BASE / group / model
    rows: list[dict] = []
    per_persona: dict[str, Counter] = defaultdict(Counter)
    if not gdir.exists():
        return rows, per_persona
    for jf in sorted(gdir.rglob("*.json")):
        if "warmup" in jf.parts:
            continue
        try:
            d = json.load(open(jf, encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] {jf.name}: {e}")
            continue
        row, _source = build_row("v9_pilot", d, jf, gdir)
        status = classify_status(row, d)
        row["answer_status"] = status
        # enrich with token counts + full raw answer (no reasoning text)
        meta = d.get("metadata", {}) or {}
        row["prompt_tokens"] = meta.get("prompt_tokens")
        row["completion_tokens"] = meta.get("completion_tokens")
        row["reasoning_tokens"] = meta.get("reasoning_tokens")
        row["raw_answer"] = str((d.get("prediction") or {}).get("raw", "") or "")
        rows.append(row)

        p = row["persona"] or "?"
        st = per_persona[p]
        st["total"] += 1
        st[status] += 1
        if status == "committed" and row["is_correct"] is True:
            st["correct"] += 1
    return rows, per_persona


def _rate(n: int, d: int) -> str:
    return f"{n/d:.4f}" if d else ""


def summary_rows_for_group(group_slug: str, per_persona: dict[str, Counter]) -> list[dict]:
    out = []
    for persona in sorted(per_persona):
        st = per_persona[persona]
        total = st["total"]
        committed = st["committed"]
        out.append({
            "group": group_slug,
            "persona": persona,
            "total": total,
            "committed": committed,
            "abstain": st["abstain"],
            "missing": st["missing"],
            "correct": st["correct"],
            "accuracy_committed": _rate(st["correct"], committed),
            "commit_rate": _rate(committed, total),
            "abstain_rate": _rate(st["abstain"], total),
            "missing_rate": _rate(st["missing"], total),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Single model slug (default: all)")
    args = ap.parse_args()

    if args.model:
        models = [args.model]
    else:
        found = set()
        for group in GROUP_SLUG:
            gdir = BASE / group
            if gdir.exists():
                found.update(p.name for p in gdir.iterdir() if p.is_dir())
        models = sorted(found)

    print(f"Building CSVs for {len(models)} model(s) x {len(GROUP_SLUG)} groups", flush=True)
    combined: list[dict] = []
    for model in models:
        print(f"\n{'='*70}\nMODEL: {model}\n{'='*70}", flush=True)
        out_dir = RESULT_ROOT / model
        out_dir.mkdir(parents=True, exist_ok=True)

        all_summary: list[dict] = []
        for group, slug in GROUP_SLUG.items():
            rows, per_persona = build_model_group(model, group)
            if not rows:
                print(f"  [{slug}] (no data)", flush=True)
                continue

            csv_path = out_dir / f"result_ethics_v9_{slug}_{model}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

            # group-level tallies
            g = Counter()
            for st in per_persona.values():
                for k in ("total", "committed", "abstain", "missing", "correct"):
                    g[k] += st[k]
            acc = f"{100*g['correct']/g['committed']:.2f}%" if g["committed"] else "n/a"
            print(f"  [{slug}] {len(rows)} rows -> {csv_path.name}", flush=True)
            print(f"      committed={g['committed']} abstain={g['abstain']} missing={g['missing']} "
                  f"| acc(committed)={acc} abstain_rate={100*g['abstain']/g['total']:.1f}% "
                  f"missing_rate={100*g['missing']/g['total']:.1f}%", flush=True)

            all_summary.extend(summary_rows_for_group(slug, per_persona))

        if all_summary:
            sum_path = out_dir / f"result_ethics_v9_{model}_status_summary.csv"
            with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_summary)
            print(f"\n  Summary -> {sum_path.name} ({len(all_summary)} persona rows)", flush=True)
            for r in all_summary:
                combined.append({"model": model, **r})

    # One combined cross-model summary (all 6 models in a single CSV).
    if combined:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        comb_path = REPORTS_DIR / "result_ethics_v9_status_summary_ALL_models.csv"
        with open(comb_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(combined)
        print(f"\n{'='*70}\nCOMBINED summary ({len(combined)} rows, "
              f"{len({r['model'] for r in combined})} models) -> {comb_path}", flush=True)


if __name__ == "__main__":
    main()

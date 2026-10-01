"""Paired persona comparison on identical questions.

    python src/benchmarks/ethics/analyze_persona_gap.py --group gender
    python src/benchmarks/ethics/analyze_persona_gap.py --group gender --csv

Why paired: every persona receives the same 1,514 items, so comparing two
personas item-by-item removes item difficulty as a confound and sidesteps the
selection problem entirely. Marginal accuracy comparisons cannot do that -
different personas answer different numbers of questions, so their accuracies
are computed over different populations.

Two things are reported, and they answer different questions:

  flip_rate  - on how many shared items did the moral stance change when the
               persona changed? This is a DESCRIPTIVE measure of persona
               influence and needs no null hypothesis.

  McNemar    - among items where exactly one persona was correct, is the split
               away from 50/50? This asks whether the persona makes the model
               more or less ACCURATE, which is a different claim.

Only `commonsense` and `justice` are scored. `deontology` and `virtue` are
excluded because their labels do not encode a right/wrong judgement (see
docs/plan_parsing_ethics.md, findings T2 and T3).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import GROUPS, PARSE_CACHE_ROOT, REPORT_ROOT  # noqa: E402
from scoring import construct_valid  # noqa: E402

#: Pairs compared per group. Left member is the reference in the A/B counts.
PAIRS: dict[str, list[tuple[str, str]]] = {
    "gender": [
        ("man_explicit_1", "woman_explicit_1"),
        ("man_implicit_1", "woman_implicit_1"),
        ("man_implicit_2", "woman_implicit_2"),
        ("man_implicit_3", "woman_implicit_3"),
    ],
}

MIN_PAIRED = 30
ALPHA = 0.05


def _load(model_dir: Path) -> dict[str, dict[str, dict]]:
    by_persona: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in model_dir.glob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if construct_valid(row.get("subset", "")):
                    by_persona[row["persona_folder"]][row["question_id"]] = row
    return by_persona


def _pairs_for(group_slug: str, personas: list[str]) -> list[tuple[str, str]]:
    if group_slug in PAIRS:
        return PAIRS[group_slug]
    # Fall back to comparing everything against the neutral baseline.
    return [("neutral", p) for p in sorted(personas) if p != "neutral"]


def compare(a_rows: dict, b_rows: dict) -> dict | None:
    from scipy.stats import binomtest

    shared = [
        q for q in a_rows
        if q in b_rows
        and a_rows[q]["answer_status"] == "committed"
        and b_rows[q]["answer_status"] == "committed"
    ]
    if len(shared) < MIN_PAIRED:
        return None

    flips = sum(1 for q in shared if a_rows[q]["extracted_stance"] != b_rows[q]["extracted_stance"])
    a_only = sum(1 for q in shared if a_rows[q]["is_correct"] and not b_rows[q]["is_correct"])
    b_only = sum(1 for q in shared if b_rows[q]["is_correct"] and not a_rows[q]["is_correct"])
    discordant = a_only + b_only
    p_value = binomtest(a_only, discordant, 0.5).pvalue if discordant else 1.0

    return {
        "n_paired": len(shared),
        "n_flip": flips,
        "flip_rate": flips / len(shared),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "p_mcnemar": p_value,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired persona comparison")
    ap.add_argument("--group", default="gender")
    ap.add_argument("--csv", action="store_true", help="also write a CSV next to the reports")
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    folder = slug_to_folder.get(args.group)
    if folder is None:
        ap.error(f"--group must be one of {sorted(slug_to_folder)}")

    group_cache = PARSE_CACHE_ROOT / folder
    if not group_cache.is_dir():
        ap.error(f"cache belum ada: {group_cache} - jalankan run_parse.py dulu")

    out_rows: list[dict] = []
    for model_dir in sorted(d for d in group_cache.iterdir() if d.is_dir()):
        by_persona = _load(model_dir)
        if not by_persona:
            continue
        pairs = _pairs_for(args.group, list(by_persona))

        print("=" * 84)
        print(f"  {model_dir.name}")
        print("=" * 84)
        print(f"{'pasangan':<38}{'n':>7}{'beda jawaban':>15}{'A/B benar':>13}{'p':>9}")
        for a, b in pairs:
            stats = compare(by_persona.get(a, {}), by_persona.get(b, {}))
            if stats is None:
                continue
            mark = " *" if stats["p_mcnemar"] < ALPHA else ""
            label = f"{a} vs {b}"
            split = f"{stats['a_only_correct']}/{stats['b_only_correct']}"
            print(f"{label[:37]:<38}{stats['n_paired']:>7}"
                  f"{100 * stats['flip_rate']:>14.1f}%"
                  f"{split:>13}"
                  f"{stats['p_mcnemar']:>9.3f}{mark}")
            out_rows.append({"model": model_dir.name, "persona_a": a, "persona_b": b, **stats})
        print()

    n_tests = len(out_rows)
    n_sig = sum(1 for r in out_rows if r["p_mcnemar"] < ALPHA)
    bonferroni = ALPHA / n_tests if n_tests else ALPHA
    n_sig_corr = sum(1 for r in out_rows if r["p_mcnemar"] < bonferroni)

    print("-" * 84)
    print(f"  uji dijalankan            : {n_tests}")
    print(f"  signifikan p<{ALPHA}        : {n_sig}  (harapan kebetulan: {ALPHA * n_tests:.1f})")
    print(f"  bertahan Bonferroni p<{bonferroni:.4f} : {n_sig_corr}")
    if out_rows:
        rates = [r["flip_rate"] for r in out_rows]
        print(f"  beda jawaban (flip rate)  : {100 * min(rates):.1f}% - {100 * max(rates):.1f}% "
              f"(median {100 * sorted(rates)[len(rates) // 2]:.1f}%)")
    print("-" * 84)
    print("  flip rate = deskriptif, tidak perlu koreksi.")
    print("  McNemar   = klaim akurasi; laporkan dengan koreksi multiple comparison.")

    if args.csv and out_rows:
        path = REPORT_ROOT / args.group / f"persona_gap_{args.group}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n  CSV: {path}")


if __name__ == "__main__":
    main()
